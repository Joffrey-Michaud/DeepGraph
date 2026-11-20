# Brouillon de schéma depuis un modèle PyTorch

> **Prototype.** L'outil donne un *brouillon* juste sur la structure, à finir dans l'éditeur
> (noms, annotations, ajustements). Ses règles de mise en page ont été réglées sur une poignée de
> modèles : les architectures très ramifiées peuvent donner un résultat moins propre.

`outils/depuis_pytorch.py` fabrique un fichier `.dgraph` à partir de **ton code PyTorch** : il
importe la classe, crée le modèle, l'exécute **une fois** sur des entrées aléatoires de la forme
indiquée, et observe ce qui se passe. Aucun LLM, aucun accès réseau : la même commande redonne
toujours le même schéma.

## Démarrage rapide

Dans un terminal, **depuis la racine de ton projet** (là d'où tu lances habituellement ton code),
avec l'environnement Python qui contient torch :

```bash
python ~/chemin/vers/deep_graphics/outils/depuis_pytorch.py \
    models/mon_modele.py:MaClasse -e x:1,3,224,224 -o brouillon.dgraph
```

Puis **Fichier > Ouvrir** `brouillon.dgraph` dans DeepGraph.

- `models/mon_modele.py:MaClasse` : le fichier qui définit la classe, `:`, le nom de la classe.
  Pour un paquet importable : `monpaquet.models:MaClasse` (ou `torchvision.models:resnet18`).
- `-e nom:forme` : **une option par argument du `forward`**, dans l'ordre, dimension du batch comprise.

## Options

| option | effet |
|---|---|
| `-e nom:forme[:type]` | une entrée du `forward`, ex. `-e x:1,3,224,224`. Type : `float` (défaut), `long` / `int` (indices, ex. tokens : `-e tokens:1,32:long`), `bool` |
| `--args '{…}'` | arguments du constructeur en JSON, ex. `--args '{"channels": [32, 64, 128], "dropout": 0.1}'` |
| `-p N`, `--profondeur N` | niveau de détail (1 par défaut) : 1 = les sous-modules directs du modèle, 2 = un niveau de plus… |
| `--detail Classe` | dessine **l'intérieur** de cette classe en **encart**, à droite de la vue d'ensemble (répétable) |
| `--deplier Classe` | montre l'intérieur de cette classe directement dans la vue d'ensemble |
| `--replier Classe` | garde cette classe en un seul bloc, même avec une grande profondeur |
| `--formes entree-sortie` | dans chaque bloc, forme d'entrée → forme de sortie (défaut) |
| `--formes sortie` / `aucune` | seulement la forme de sortie / aucune forme |
| `--sans-repetitions` | ne regroupe pas les blocs identiques en « N× » |
| `--resume graphe.json` | écrit aussi le graphe brut (nœuds, liaisons, formes), pour vérifier ce qui a été vu ou le donner à un LLM |
| `-o fichier.dgraph` | fichier à écrire (obligatoire) |

`python outils/depuis_pytorch.py -h` rappelle la liste.

L'opération inverse (schéma → code PyTorch) est décrite dans [VERS_PYTORCH.md](VERS_PYTORCH.md).

## Recettes

**Vue d'ensemble + détail des blocs** (le plus proche d'une figure d'article) :

```bash
python …/depuis_pytorch.py models/unet.py:MonUNet -e x:1,3,256,256 \
    --detail EncoderBlock --detail DecoderBlock -o unet.dgraph
```

**Un modèle « enveloppe »** (`self.encoder`, `self.decoder` et rien d'autre) : la profondeur 1 ne
montre que deux blocs. Utiliser `-p 2`, ou `--detail` sur les classes de l'encodeur et du décodeur.

**Un Transformer** : `-p 2` montre l'intérieur d'un bloc, et les N blocs identiques sont regroupés
dans un seul cadre « N× Block ».

```bash
python …/depuis_pytorch.py torchvision.models:vit_b_16 -e image:1,3,224,224 -p 2 -o vit.dgraph
```

**Un modèle à tokens** : `-e tokens:1,64:long`.

**Aperçu sans ouvrir l'éditeur** : `python outils/exporter.py brouillon.dgraph -f png`.

## Lire le résultat

- **Un bloc** = un module appelé pendant l'exécution. Texte : nom de la classe (ou réglages pour les
  modules de `torch.nn` : « Conv2d k3 s2 », « Linear 128 → 64 », « Multi-Head Attention (8 têtes) »),
  puis les formes **sans la première dimension** (le batch). Une petite suite de couches de `torch.nn`
  regroupées dans une classe (`Conv2dNormActivation`…) s'écrit « Conv2d k3 s2 + BatchNorm + ReLU6 ».
  Quand deux blocs auraient le même texte, leur nom d'attribut s'ajoute en première ligne
  (`mu` / `logvar`, `q_proj` / `k_proj`), sauf s'il est générique (`norm1`, `fc2`…).
- **⊕ / concaténation / ⊗** : les `+`, `torch.cat`, `*` écrits dans un `forward` entre deux tenseurs.
  `view`, `transpose`, `reshape`, `x * 0.5`… sont transparents (pas de bloc), comme les modules
  `Identity`, `Permute`, `Rearrange`…
- **« Noise ε ~ 𝒩(0, I) »** : un tirage aléatoire (`torch.randn_like(mu)`, `torch.randn(…)`), dessiné comme
  une source à part (l'astuce de reparamétrisation d'un VAE donne `logvar ⊗ bruit`, puis `⊕ mu`).
- **Petit bloc gris à côté** : un paramètre utilisé directement (`pos_embedding`, `class_token`…).
- **Cadre plein** (titre en haut à gauche) : une liste de modules nommée (`self.encoder = nn.ModuleList(…)`,
  `nn.Sequential`) — son nom d'attribut sert de titre (pas de titre si c'est un numéro : `features.3`).
- **Cadre pointillé** : un module dont on montre l'intérieur ; « 6× Block » s'il est répété.
- **Cadre « N× » autour d'un bloc** : N blocs identiques (mêmes réglages) enchaînés.
- **Sorties** : « Output », ou « Output 1, 2… » si le `forward` en rend plusieurs ; la clé quand il rend
  un dictionnaire (`{"out": …}`).
- **Forme en U** : choisie automatiquement quand de longues liaisons emboîtées relient le début et la
  fin du réseau (U-Net) ; les skip connections deviennent horizontales.
- Chaque élément garde le **nom du module** d'origine (`encoder.conv1.0`) dans sa référence interne.

Les blocs et cadres se retouchent ensuite comme n'importe quel schéma : renommer (« Cross-Attention
32 canaux · 4 têtes »), ajouter des annotations (Q, K, V, « skip »), déplacer, recolorer.

## Comment ça marche (sans LLM)

1. **Exécution observée** : des *hooks* PyTorch signalent chaque appel de module (entrées, sorties),
   et un `TorchFunctionMode` montre les opérations écrites dans les `forward` (`cat`, `+`…). En suivant
   chaque tenseur de qui le produit à qui l'utilise, on obtient le graphe exact.
   `torch.fx` n'est pas utilisé : il échoue dès qu'un `forward` contient une vérification
   (`if x.ndim != 3: raise …`).
2. **Niveau de détail** : la hiérarchie de tes modules ; `ModuleList`, `ModuleDict` et `Sequential` ne
   comptent pas comme un niveau. Une classe qui en hérite (bloc dense de DenseNet,
   `Conv2dNormActivation`) est un vrai module, avec son nom.
3. **Factorisation** : deux blocs sont identiques s'ils ont la même classe, les mêmes réglages et les
   mêmes sous-modules. Les taux de régularisation (`Dropout`, *stochastic depth*) ne comptent pas :
   ils changent souvent d'un bloc à l'autre (Swin, ConvNeXt, EfficientNet).
4. **Mise en page** : colonne principale = plus long chemin entrée → sortie ; repli en U si utile ;
   les autres chaînes forment des colonnes ; l'ordre des colonnes minimise les longueurs et les
   traversées ; les résiduelles passent par le côté ; le reste suit le tracé automatique des flèches.

Ce sont des règles fixes : elles voient la structure, pas l'intention (elles ne savent pas qu'une
entrée joue le rôle de Q ou de K).

## Si ça ne marche pas

| message | solution |
|---|---|
| `ModuleNotFoundError: No module named 'models'` | lancer la commande depuis la racine du projet (le dossier qui contient `models/`) |
| `PyTorch est nécessaire` | activer l'environnement qui contient torch |
| `TypeError: __init__() missing … argument` | donner les arguments du constructeur avec `--args '{…}'` |
| `Expected tensor … Long, Int` | l'entrée est un indice (tokens) : ajouter `:long` à `-e` |
| erreur de forme dans le `forward` | vérifier l'ordre et la forme des `-e` (batch compris) |
| le constructeur charge des poids | mettre l'argument à `null` dans `--args` s'il est facultatif |
| schéma réduit à 2 ou 3 blocs | modèle « enveloppe » : `-p 2` ou `--detail` |
| schéma trop chargé / emmêlé | baisser `-p`, et montrer les détails avec `--detail` plutôt qu'en profondeur |

## Limites connues

- Seul le chemin **réellement exécuté** est vu : un `if` qui dépend des données n'en montre qu'une
  branche.
- Une **grande profondeur** sur un modèle complexe donne un schéma emmêlé : préférer `--detail`.
- La première dimension est toujours considérée comme le batch, même quand le code y a replié le
  temps (`T×…` au lieu de `1×T×…`).
- Le symbole de concaténation de la bibliothèque ressemble à ⊕.
- Il faut pouvoir **instancier** le modèle et que torch soit installé.
