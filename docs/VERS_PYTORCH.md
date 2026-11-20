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
| texte du bloc : `Conv 7×7, 64, /2`, `Conv1d k7, s2, p3`, `FC 4096`, `Linear 256 → K`, `Dropout 0.1`, `GroupNorm (16)` | ses réglages (noyau, canaux, pas, padding, groupes…) |
| `Linear C → 4C` | `nn.Linear(d, 4 * d)` : la lettre de gauche désigne la taille reçue |
| attention : `Multi-Head Attention` + `256 canaux · 8 têtes` | `nn.MultiheadAttention(256, 8)` ; sur une image ou un signal (B, C, …), passage en séquence (B, positions, C) et retour |
| texte composé : `BN + ReLU`, `2× (Conv 3×3 + ReLU)` + `64 canaux`, ou couches sur deux lignes (`Conv1d k7, 32` / `BN + ReLU`) | un `nn.Sequential` |
| taille symbolique divisée : `FC C/16` | `nn.Linear(C, C // 16)` |
| `LSTM 128, 2 couches`, `BiGRU 64`, `Embedding 1000 → 128`, `Dropout 0.3` | taille de l'état, `num_layers`, `bidirectional`, vocabulaire, probabilité |
| nom en première ligne d'un bloc : `mu` / `Linear 400 → 20` | `self.mu = nn.Linear(400, 20)`, `mu = self.mu(x)` |
| note générale : `ReLU après chaque convolution et FC`, `activations : tanh`, `Conv : pas 1, padding 1, ReLU`, `FC 4096 : ReLU + Dropout`, `chaque convolution est suivie de BN + ReLU` | la couche ajoutée après chaque couche visée (après sa BatchNorm si elle en a une, avant une LRN), sauf si le schéma la dessine déjà et sauf après la dernière couche ; les réglages complètent les couches qui ne les précisent pas |
| taille écrite à côté d'un bloc : `55×55×96`, `C1 : 6@28×28`, `224×224` | forme de sa sortie : padding exact des convolutions et poolings ; à côté de l'entrée (`1@32×32`) : ses canaux et sa taille. Les convolutions sans taille suivent l'habitude du schéma (padding 0 si les autres n'en ont pas) |
| `Sélection [:, 0]`, `[:, -1]` | indexation `x[:, 0]` (dernier pas d'un RNN, jeton de classe) |
| bloc `Noise` (`ε ~ N(0, I)`) qui alimente un ⊗ ou un ⊕ | `eps = torch.randn_like(…)`, tiré dans le forward |
| vecteur (B, C) combiné à une image par ⊗ ou ⊕ (SE, CBAM, embedding de temps) | `w[:, :, None, None] * x` |
| paramètre appris posé à côté d'un opérateur (`class_token` / `1×1×768`, `layer_scale` / `96×1×1`) | `nn.Parameter` de cette forme |
| ligne de formes (import PyTorch) : `3×224×224 → 64×112×112` | canaux de sortie, padding exact |
| flèche | le passage d'un tenseur ; une jonction = un tenseur réutilisé |
| ⊕ / ⊗ / ⊙ / − / concaténation | `a + b`, `a * b`, `a - b`, `torch.cat(…)` |
| flèche venant du vide, ou texte au départ d'une flèche (`x`, `Q`, `Inputs`) | argument du `forward`, nommé d'après le texte |
| flèche vers le vide, ou texte à l'arrivée | valeur renvoyée |
| texte court posé le long d'une flèche (`skip`, `Q`, `K, V`) | nom de la valeur qu'elle apporte : ordre des entrées d'une attention, argument du même nom d'un bloc détaillé (`forward(self, x, skip)`) |
| cadre `N×` (ou `2× Nom`) | bloc répété : `nn.ModuleList` + boucle, ou `nn.Sequential` de N copies |
| cadre nommé (`Bloc 1`, `couche cachée`) autour d'une chaîne | `nn.Sequential` du même nom |
| cadre en pointillés titré (`Basic Block`), ou schéma à part avec un titre en gras | une classe, utilisée par les blocs qui portent ce titre (`Basic Block` + `128, /2` → `BasicBlock(64, 128, stride=2)`, 64 étant les canaux reçus) ; si la définition n'écrit aucune taille et que les blocs disent `64 → 128`, ses convolutions sortent `C` canaux, fourni par chaque bloc |
| flèches de couleur expliquées par une légende (flèche + texte : `Max Pool 2×2`, `Up-conv 2×2`, `copie + concaténation`) | l'opération sur chaque flèche de cette couleur (U-Net) |
| cellules récurrentes dessinées dépliées dans le temps (LSTM → LSTM → LSTM) | un seul module, appelé à chaque pas, l'état passant de l'un à l'autre |
| `Positional Encoding` ou `[class] token + Position Embedding` posé à côté d'un ⊕ | module d'encodage de position (sinusoïdal ou appris) ; une `MLP Head` qui suit lit le jeton de classe (`x[:, 0]`) |
| échantillonnage `z = μ + σ ⊙ ε` | le forward renvoie aussi μ et log σ², nécessaires à la perte KL |
| bloc sans contenu connu (`Encodeur`, `Générateur G`) | une classe vide à compléter, qui laisse passer son entrée ; si son texte dit `256 → 128`, une projection provisoire (Conv 1×1 ou Linear) aux bonnes tailles, pour que le reste s'exécute |
| convolution transposée suivie d'une concaténation | appelée avec `output_size=` l'autre branche : la longueur suit le skip (2L ou 2L − 1) |
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
- Une `Linear` après des convolutions reçoit un `nn.Flatten()`, sauf si sa taille d'entrée est celle des
  canaux (`Linear 96 → 384` sur une image 96×56×56) : elle agit alors en chaque position et s'écrit
  `nn.Conv2d(96, 384, kernel_size=1)` (même calcul). Une `LayerNorm` sur une image normalise ses canaux
  (`ChannelLayerNorm`, comme ConvNeXt). Un RNN ou une attention après des convolutions reçoit la
  séquence des positions (`x.transpose(1, 2)`).
- Un bloc détaillé qui réduit la taille (`/2`) reçoit `stride` sur chaque branche : la première
  convolution de noyau > 1 du chemin principal, la Conv 1×1 d'un raccourci dessiné.
- Une définition dessinée pour un exemple qui garde la forme (`256×14×14` en entrée et en sortie)
  exprime ses tailles internes par rapport à C : `C // 4` (Bottleneck), `4 * C` (MLP de ConvNeXt).
- Dernière couche sans taille : `num_classes` (ou `vocab_size` s'il y a des embeddings).
- Réglages globaux (`d_model`, `num_heads`, `d_ff`, `vocab_size`, `hidden_size`, `N`…) : arguments du
  constructeur, avec une valeur par défaut à vérifier.
- Bloc `Softmax` final : gardé, avec un rappel que `nn.CrossEntropyLoss` attend des logits.

## Dessiner pour un meilleur code

- **Écrire les réglages dans le texte des blocs** : `Conv 3×3, 64, /2` plutôt que `Conv`.
- **Écrire la taille de l'entrée** : `Image 224×224×3` donne `in_channels` et la taille exacte avant un
  `Flatten` ; sans elle, l'essai suppose 3 canaux et 64×64 (marqué `TODO`). Pour un signal 1D,
  `1 × 1000` se lit canaux × longueur.
- **Écrire les tailles intermédiaires** à droite des blocs (`55×55×96`, `C1 : 6@28×28`) : le padding des
  convolutions en est déduit (LeNet-5, AlexNet).
- **Une note pour ce qui se répète** : `ReLU après chaque convolution et FC` plutôt qu'un bloc ReLU
  après chaque couche.
- **Nommer les flèches qui comptent** : un texte court le long d'une flèche (`skip`, `Q`, `K, V`) dit
  quelle entrée elle apporte ; sans lui, l'ordre vient de la position des flèches.
- **Entrées d'une attention** (bas du bloc, de gauche à droite) : **V, K, Q**, comme dans la figure du
  Transformer. Ou étiqueter les entrées `Q`, `K`, `V` : les noms l'emportent sur la position.
- **Détailler un bloc** : cadre en pointillés titré du nom du bloc, une flèche qui entre (étiquetée `x`)
  et une qui sort. Une taille écrite avec une lettre (`Conv 3×3, C`) devient un argument (`C`),
  rempli par chaque bloc qui l'utilise (`Basic Block` / `128`).
- **Plusieurs schémas dans un fichier** : chacun devient une classe, nommée par le titre en gras le
  plus proche.

## Où sont les TODO

- **Au bout de la ligne concernée** : réglage d'une couche (`# Conv 3×3 — TODO : préciser le nombre de
  canaux…`), taille d'entrée du réseau (au bout de la première ligne qui utilise `in_channels`). Pour une
  couche écrite sur plusieurs lignes (`nn.Sequential(…)`), sur la ligne de la parenthèse fermante.
- **Dans le corps d'une classe vide** : bloc que le schéma ne détaille pas (`# TODO : couches de « … »`).
- **Dans `__main__`** : les formes des entrées de l'essai.

## Blocs personnels

Dans la fenêtre **Nouveau bloc / Modifier le bloc**, le champ **Code PyTorch** (facultatif) donne le
module à utiliser pour ce bloc, par exemple `nn.Sequential(nn.Conv2d({in}, 32, 1), nn.Sigmoid())`.
`{in}` est remplacé par la taille d'entrée du bloc, suivie comme pour les autres couches ; les canaux de
sortie sont lus dans la dernière `Conv` ou `Linear` de l'expression. Il est enregistré dans
`mes_blocs.yaml` (clé `pytorch`). Sans lui, DeepGraph lit le texte du bloc, ou crée une classe vide à
compléter.

## Aller-retour avec l'import PyTorch

Un schéma produit par `outils/depuis_pytorch.py` se retraduit en code, ce qui permet de vérifier les
deux outils l'un par l'autre. Les blocs répétés détaillés en encart (`--detail`) donnent les classes ;
le nombre de paramètres du code obtenu, comparé au modèle d'origine :

| modèle | options de l'import | paramètres du code / d'origine |
|---|---|---|
| ResNet-18, ResNet-50 | `--detail BasicBlock`, `--detail Bottleneck` | 11,69 M / 11,69 M ; 25,58 M / 25,56 M |
| ConvNeXt-T | `--detail CNBlock` | 28,59 M / 28,59 M |
| EfficientNet-B0 | `--detail MBConv` | 5,32 M / 5,29 M |
| DenseNet-121 | `--deplier _DenseBlock --detail _DenseLayer` | 7,99 M / 7,98 M |
| ViT-B/16 | `--deplier Encoder --detail EncoderBlock` | 86,57 M / 86,57 M |
| MobileNetV2 | `--detail InvertedResidual` | 4,25 M / 3,50 M (raccourci Conv 1×1 ajouté là où MobileNet n'en met pas) |
| U-Net, GPT, VAE, SE-Net, LSTM (petits modèles d'essai) | `--detail` des blocs répétés | identiques |

Les petits écarts viennent des biais des convolutions, que le schéma n'indique pas.

## Limites connues

- Le schéma montre une architecture, pas tout le calcul : remises en forme (`view`, `permute`),
  masques, initialisation… restent à écrire.
- Un schéma qui décrit un entraînement plutôt qu'un calcul (GAN : D appliqué tour à tour à G(z) et à x)
  donne un forward qui combine les deux entrées : à réécrire.
- Une définition dessinée pour un exemple ne connaît pas les variantes : un bloc résiduel dont la forme
  change reçoit un raccourci Conv 1×1 (ResNet), même si le modèle d'origine n'en a pas (MobileNet).
- Un bloc dont le texte n'est pas une couche connue (`Encodeur`, `Fusion`) devient une classe vide.
- Les copies empilées (`×h` des têtes d'attention) sont signalées en commentaire, pas découpées en têtes.
- Une flèche dessinée pour l'explication (« réinjectée en entrée ») est prise pour un vrai passage de
  tenseur : la retirer ou la mettre sur un calque masqué avant l'export.
- Un texte court posé juste à droite d'un cadre est pris pour son nom (`conv2_x`).
