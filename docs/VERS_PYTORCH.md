# Code PyTorch depuis un schéma

> **Prototype.** Le code produit est un **squelette** dont la structure suit le schéma : les
> couches, leur enchaînement, les branches, les additions et concaténations, les blocs répétés.
> Ce que le schéma ne dit pas (nombre de canaux absent, taille de sortie…) reçoit une valeur par
> défaut marquée `TODO`. À relire avant de s'en servir.

C'est l'inverse de [l'import PyTorch](DEPUIS_PYTORCH.md) : on part du dessin et on obtient un fichier
`.py` avec une classe `nn.Module` par réseau, prête à être exécutée. Aucun LLM, aucun accès réseau :
le même schéma donne toujours le même code.

## Démarrage rapide

Dans l'éditeur : **Fichier > Exporter en code PyTorch…**, puis choisir où écrire le `.py`.

Dans un terminal (ni Qt ni torch ne sont nécessaires pour générer le code) :

```bash
python outils/vers_pytorch.py schema.dgraph            # écrit schema.py à côté
python outils/vers_pytorch.py schema.dgraph --essai    # et l'exécute sur des entrées factices
python outils/vers_pytorch.py modeles/*.dgraph -d code # un .py par schéma
```

Le fichier se termine par un essai : `python schema.py` crée le modèle, lui donne des entrées
aléatoires de forme plausible et affiche la forme de la sortie et le nombre de paramètres.

## Ce qui est traduit

| dans le schéma | dans le code |
|---|---|
| bloc de la bibliothèque (Conv2D, Batch Norm, ReLU, Linear, LSTM, Multi-Head Attention…) | la couche PyTorch correspondante |
| texte du bloc : `Conv 7×7, 64, /2`, `FC 4096`, `Linear 256 → K`, `Dropout 0.1` | ses réglages (noyau 7, 64 canaux, pas 2…) |
| texte composé : `BN + ReLU`, `2× (Conv 3×3 + ReLU)` + `64 canaux` | un `nn.Sequential` |
| ligne de formes (import PyTorch) : `3×224×224 → 64×112×112` | canaux de sortie, padding exact |
| flèche | le passage d'un tenseur ; une jonction = un tenseur réutilisé |
| ⊕ / ⊗ / ⊙ / − / concaténation | `a + b`, `a * b`, `a - b`, `torch.cat(…)` |
| flèche venant du vide, ou texte au départ d'une flèche (`x`, `Q`, `Inputs`) | argument du `forward`, nommé d'après le texte |
| flèche vers le vide, ou texte à l'arrivée | valeur renvoyée |
| cadre `N×` (ou `2× Nom`) | bloc répété : `nn.ModuleList` + boucle, ou `nn.Sequential` de N copies |
| cadre nommé (`Bloc 1`, `couche cachée`) autour d'une chaîne | `nn.Sequential` du même nom |
| cadre en pointillés titré (`Basic Block`), ou schéma à part avec un titre en gras | une classe, utilisée par les blocs qui portent ce titre (`Basic Block` + `128, /2` → `BasicBlock(64, 128, stride=2)`, 64 étant les canaux reçus) |
| flèches de couleur expliquées par une légende (flèche + texte : `Max Pool 2×2`, `Up-conv 2×2`, `copie + concaténation`) | l'opération sur chaque flèche de cette couleur (U-Net) |
| cellules récurrentes dessinées dépliées dans le temps (LSTM → LSTM → LSTM) | un seul module, appelé à chaque pas, l'état passant de l'un à l'autre |
| `Positional Encoding` ou `[class] token + Position Embedding` posé à côté d'un ⊕ | module d'encodage de position (sinusoïdal ou appris) |
| bloc sans contenu connu (`Encodeur`, `Générateur G`) | une classe vide à compléter, qui laisse passer son entrée |
| textes libres (dimensions, notes, référence) | recopiés en tête du fichier |

Le titre en gras du schéma devient le nom de la classe : `ResNet-18` → `ResNet18`,
`Autoencodeur variationnel (VAE)` → `VAE`.

## Choix par défaut

- Les couches sont des couches **classiques** (`nn.Conv2d(64, 128, …)`, `nn.Linear(4096, 1000)`,
  `nn.BatchNorm2d(128)`). Le nombre de canaux **d'entrée**, que les schémas ne donnent presque jamais,
  est suivi de couche en couche depuis l'entrée du réseau : sortie d'une couche = entrée de la suivante,
  canaux additionnés par une concaténation. Changer une taille de sortie demande donc de changer aussi
  l'entrée de la couche qui suit.
- L'entrée du réseau devient un argument du constructeur : `in_channels` pour une image (lu dans le
  texte : `224×224×3`, `RGB`, `niveaux de gris` ; sinon 3, marqué `TODO`), `in_features` pour un vecteur.
- Une `Linear` après un `Flatten` reçoit `C * H * W` (`nn.Linear(512 * 7 * 7, 4096)`), calculé à partir de
  la taille de l'image d'entrée. Si cette taille ou le padding des convolutions est supposé, un `TODO`
  le signale.
- Un bloc détaillé dans un cadre (`Basic Block`) reçoit sa taille d'entrée en premier argument :
  `BasicBlock(64, 128, stride=2)`, avec un raccourci Conv 1×1 quand le pas ou le nombre de canaux change.
  Un bloc répété dont les canaux changent est construit en conséquence
  (`Bloc(64 if i == 0 else 128)`, `Bloc(64 + i * 32)` pour une croissance à la DenseNet).
- Une taille impossible à déduire reçoit 64 et un `TODO`.
- Convolution : padding « same » (`k // 2`, ou calculé à partir des formes si elles sont écrites).
  Pooling sans pas indiqué : pas 2 pour un noyau 2×2, sinon pas 1.
- Une `Linear` après des convolutions reçoit un `nn.Flatten()`.
- Dernière couche sans taille : `num_classes` (ou `vocab_size` s'il y a des embeddings).
- Réglages globaux (`d_model`, `num_heads`, `d_ff`, `vocab_size`, `hidden_size`, `N`…) : arguments du
  constructeur, avec une valeur par défaut à vérifier.
- Bloc `Softmax` final : gardé, avec un rappel que `nn.CrossEntropyLoss` attend des logits.

## Dessiner pour un meilleur code

- **Écrire les réglages dans le texte des blocs** : `Conv 3×3, 64, /2` plutôt que `Conv`.
- **Écrire la taille de l'entrée** : `Image 224×224×3` donne `in_channels` et la taille exacte avant un
  `Flatten` ; sans elle, l'essai suppose 3 canaux et 64×64 (marqué `TODO`).
- **Entrées d'une attention** (bas du bloc, de gauche à droite) : **V, K, Q**, comme dans la figure du
  Transformer. Ou étiqueter les entrées `Q`, `K`, `V` : les noms l'emportent sur la position.
- **Détailler un bloc** : cadre en pointillés titré du nom du bloc, une flèche qui entre (étiquetée `x`)
  et une qui sort. Une taille écrite avec une lettre (`Conv 3×3, C`) devient un argument (`C`),
  rempli par chaque bloc qui l'utilise (`Basic Block` / `128`).
- **Plusieurs schémas dans un fichier** : chacun devient une classe, nommée par le titre en gras le
  plus proche.

## Blocs personnels

Dans la fenêtre **Nouveau bloc / Modifier le bloc**, le champ **Code PyTorch** (facultatif) donne le
module à utiliser pour ce bloc, par exemple `nn.Sequential(nn.Conv2d({in}, 32, 1), nn.Sigmoid())`.
`{in}` est remplacé par la taille d'entrée du bloc, suivie comme pour les autres couches ; les canaux de
sortie sont lus dans la dernière `Conv` ou `Linear` de l'expression. Il est enregistré dans
`mes_blocs.yaml` (clé `pytorch`). Sans lui, DeepGraph lit le texte du bloc, ou crée une classe vide à
compléter.

## Aller-retour avec l'import PyTorch

Un schéma produit par `outils/depuis_pytorch.py` se retraduit en code, ce qui permet de vérifier les
deux outils l'un par l'autre. Par exemple, `torchvision.models:resnet18 --detail BasicBlock` puis
`vers_pytorch.py --essai` redonne un ResNet-18 de 11,69 M paramètres (torchvision : 11,69 M, l'écart
étant les biais des convolutions, que le schéma n'indique pas).

## Limites connues

- Le schéma montre une architecture, pas tout le calcul : remises en forme (`view`, `permute`),
  masques, choix du jeton de classe, initialisation… restent à écrire.
- Un bloc dont le texte n'est pas une couche connue (`Encodeur`, `Fusion`) devient une classe vide.
- Les copies empilées (`×h` des têtes d'attention) sont signalées en commentaire, pas découpées en têtes.
- Une flèche dessinée pour l'explication (« réinjectée en entrée ») est prise pour un vrai passage de
  tenseur : la retirer ou la mettre sur un calque masqué avant l'export.
- Un texte court posé juste à droite d'un cadre est pris pour son nom (`conv2_x`).
