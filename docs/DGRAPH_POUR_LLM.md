# Générer un schéma DeepGraph (`.dgraph`) — instructions pour un LLM

Tu vas recevoir le code (ou la description) d'un modèle de réseau de neurones. Ta tâche est de
produire **un fichier `.dgraph`** que l'éditeur DeepGraph ouvrira directement et qui dessinera
l'architecture dans le style des figures d'articles (« Attention is all you need ») : blocs
arrondis colorés, flèches orthogonales, opérateurs dans des cercles, cadres gris pour les
groupes répétés.

Un `.dgraph` est un fichier **JSON**. Ce document décrit le **format simplifié**, conçu pour être
écrit à la main : l'éditeur calcule lui-même les tailles, les couleurs des blocs de bibliothèque,
les cadres des groupes et les côtés des flèches.

---

## 1. Ce que tu dois rendre

- Un seul bloc de code ```json contenant le fichier complet, valide (pas de commentaires, pas de
  virgule finale), à enregistrer sous `nom_du_modele.dgraph`.
- Avant le JSON, deux ou trois phrases maximum : ce que montre le schéma et les simplifications faites.
- L'utilisateur l'ouvre avec **Fichier > Ouvrir**. Un fichier mal formé affiche un message
  d'erreur précis (ex. « flèche n° 4 : élément « enc3 » introuvable ») : corrige alors le JSON.

## 2. Structure du fichier

```json
{
  "format": "deepgraph",
  "nodes":  [ ... éléments (blocs, opérateurs, textes, jonctions, groupes) ... ],
  "arrows": [ ... flèches ... ]
}
```

`"format": "deepgraph"` est **obligatoire**. Tout le reste est facultatif.

### Repère
- Unités : pixels. **x vers la droite, y vers le bas** (y = 100 est sous y = 0).
- Origine libre ; partir de (0, 0) en haut à gauche ou au centre du schéma.
- Coordonnées toujours **absolues**, y compris pour les éléments placés dans un groupe.
- Position conseillée : `"cx"`, `"cy"` = **centre** de l'élément. (`"x"`, `"y"` = coin haut-gauche,
  aussi accepté.)

## 3. Éléments (`nodes`)

Chaque élément a un **`"id"` unique** (texte court et lisible : `"enc1"`, `"mha_dec"`, `"plus2"`).
L'**ordre de la liste compte** : un élément placé plus loin dans la liste est dessiné devant.

### 3.1 Bloc de bibliothèque (cas le plus courant)

```json
{"id": "c1", "lib": "Conv2D", "text": "Conv 3×3, 64", "cx": 0, "cy": 60}
```

- `"lib"` : identifiant d'un bloc de la bibliothèque (table en **annexe**). Il fournit la couleur,
  la taille par défaut, le texte par défaut **et la description** affichée dans l'éditeur.
  Utilise-le chaque fois qu'un bloc correspond (Conv, Norm, activation, attention…).
- `"text"` : remplace le texte affiché (garde la couleur du bloc). Retour à la ligne : `\n`.
  Mets-y les hyperparamètres utiles : `"Conv1d k5, s2\n32 → 64"`.
- Un `"lib"` inconnu ne fait pas échouer le chargement : il devient un bloc blanc portant ce nom.
- Les opérateurs ronds sont aussi des `"lib"` : `op_plus`, `op_concat`, `op_times`, `op_dot`, `op_minus`.

### 3.2 Bloc libre (pas d'équivalent en bibliothèque)

```json
{"id": "bb", "type": "block", "text": "Bottleneck\n256 → 128", "color": "#EFE6D2", "w": 150, "h": 40, "cx": 300, "cy": 500}
```

Propriétés d'un bloc (`"type": "block"`, toutes facultatives) :

| clé | type | défaut | rôle |
|---|---|---|---|
| `text` | texte | `"Bloc"` | texte centré, `\n` pour aller à la ligne |
| `color` | `"#RRGGBB"` | `"#FFFFFF"` | couleur de fond |
| `w`, `h` | nombre | 110, 28 | largeur, hauteur (px) |
| `font_size` | entier | 13 | taille du texte (px) |
| `bold`, `italic` | booléen | false | style du texte |
| `text_color` | `"#RRGGBB"` | `"#000000"` | couleur du texte |
| `stack` | entier | 0 | nombre de **copies empilées derrière** (effet « ×h » des têtes d'attention) |
| `stack_dx`, `stack_dy` | nombre | 4, −4 | décalage entre deux copies (4, −4 = vers le haut à droite) |
| `stack_fade` | booléen | true | copies de plus en plus transparentes |
| `opacity_pct` | 0–100 | 100 | opacité du bloc |
| `border_width` | nombre ou null | null | épaisseur du bord (null = valeur globale) |

**Taille du texte** : compte environ 7 px par caractère à 13 px. Élargis (`"w"`) ou passe à la
ligne si le texte dépasse ; pour 2 lignes, `"h": 40` ; 3 lignes, `"h": 56`.

### 3.3 Opérateur (cercle)

```json
{"id": "add1", "lib": "op_plus", "cx": 0, "cy": 260}
```
ou `{"id": "add1", "type": "operator", "symbol": "plus", "cx": 0, "cy": 260}`.
Symboles : `plus` (⊕ addition / résiduelle), `concat` (concaténation), `times` (⊗ produit),
`dot` (⊙ produit terme à terme), `minus`. Diamètre 18 px (`"w"` pour changer).

### 3.4 Texte libre (étiquettes : entrées, sorties, Q/K/V, dimensions)

```json
{"id": "lbl_x", "type": "text", "text": "x", "cx": 0, "cy": 0, "italic": true, "font_size": 13}
```
Clés : `text`, `font_size` (13), `bold`, `italic`, `color` (`"#000000"`), `align` (`"center"`,
`"left"`, `"right"`). La taille du cadre est calculée automatiquement.
Pour une petite annotation discrète : `"font_size": 10, "color": "#555555", "italic": true`.

### 3.5 Jonction (point d'embranchement invisible)

```json
{"id": "j1", "type": "junction", "cx": 0, "cy": 25}
```
Sert à faire partir **plusieurs flèches d'un même point** : connexion résiduelle, signal envoyé à
Q, K et V, skip connection… La flèche qui arrive sur une jonction a en général `"head": "none"`.
`"show_dot": true` dessine un point noir.

### 3.6 Groupe (cadre gris arrondi)

```json
{"id": "enc", "type": "group", "children": ["mha", "an1", "ff", "an2", "j1", "j2"], "label": "N×", "label_pos": "left"}
```

- `"children"` : identifiants des éléments contenus (déclarés **ailleurs** dans `nodes`, avec leurs
  coordonnées absolues). **Ne donne pas de position au groupe** : son cadre est ajusté
  automatiquement autour de ses enfants (+ marges).
- Un groupe peut contenir d'autres groupes (déclare le groupe intérieur dans `children`).
- Un élément ne peut appartenir qu'à un seul groupe direct.

| clé | défaut | valeurs |
|---|---|---|
| `label` | `""` | texte, ex. `"N×"`, `"L×"`, `"Encoder"` |
| `label_pos` | `"left"` | `left`, `right`, `top`, `bottom`, `inside`, `none` |
| `label_size` | 15 | taille du texte de l'étiquette |
| `fill` | `"#F2F2F2"` | couleur de fond (`"#FFFFFF"` pour un cadre blanc) |
| `border_style` | `"solid"` | `solid`, `dashed`, `none` |
| `margin` | 15 | marge autour du contenu (4 côtés) |
| `margin_top`, `margin_bottom`, `margin_left`, `margin_right` | = `margin` | marge d'un côté |

Usages : bloc répété N fois (`"label": "N×"` à gauche, comme le Transformer), branche d'un
réseau (`"label": "Encoder"`, `label_pos: "top"`), panneau de détail (`fill: "#FFFFFF"`,
`border_style: "dashed"`).

## 4. Flèches (`arrows`)

```json
{"from": "c1", "to": "bn1"}
```

`from` / `to` acceptent trois formes :

1. **Un identifiant** : `"c1"`. Les côtés sont choisis automatiquement (bas → haut si la cible est
   plutôt en dessous, droite → gauche si elle est plutôt à droite…). Une jonction est toujours
   reliée par son centre.
2. **Un côté imposé** : `{"node": "mha", "side": "bottom", "at": 0.2}`
   - `side` : `top`, `bottom`, `left`, `right`, `center`, `top-left`, `top-right`, `bottom-left`, `bottom-right` ;
   - `at` (0 → 1, facultatif) : position le long de ce côté (0 = gauche / haut, 1 = droite / bas).
     Indispensable pour les entrées multiples (V, K, Q à `at` 0.2 / 0.5 / 0.8 en bas d'une attention).
3. **Un point libre** : `{"x": 0, "y": -40}` (flèche qui entre depuis le vide ou sort vers une étiquette).

Options :

| clé | défaut | valeurs |
|---|---|---|
| `style` | `"straight"` | `straight` (segment droit), `ortho1` (1 angle), `ortho2` (2 angles) |
| `first` | `"auto"` | `h` ou `v` : direction du **premier** segment pour `ortho1` / `ortho2` |
| `mid` | null | `ortho2` : position du segment du milieu, **décalage en px depuis le départ** le long de `first` (ex. `first: "h", mid: 80` → part de 80 px vers la droite ; `-80` vers la gauche) |
| `head` | `"end"` | `end`, `start`, `both`, `none` |
| `dashed` | false | pointillés |
| `color` | `"#000000"` | couleur |

Choisir le style :
- éléments **alignés** (même `cx` ou même `cy`) → `straight` ;
- décalés en x et en y → `ortho1` (`first: "v"` = d'abord vertical puis horizontal) ;
- **connexion résiduelle / contournement** : `ortho2` avec `first: "h"` et `mid` (±70 à ±110 px)
  depuis une jonction jusqu'au côté de l'opérateur `+` ;
  ex. `{"from": "j1", "to": {"node": "add1", "side": "right"}, "style": "ortho2", "first": "h", "mid": 90}` ;
- branches parallèles depuis un bloc large : `straight` avec `at` différents sur les deux blocs.

Une flèche droite entre deux éléments non alignés est dessinée **en biais** : aligne les centres
ou utilise `ortho1` / `ortho2`.

## 5. Méthode de mise en page

1. **Lire le code** : `__init__` donne les couches, `forward` donne le **flux de données** (c'est lui
   qui définit les flèches, les concaténations, additions et skip connections).
2. **Choisir le niveau de détail** : schéma principal = blocs de haut niveau (un sous-module = un
   bloc avec ses dimensions). Les sous-modules non triviaux peuvent être détaillés à côté, dans
   un panneau séparé (groupe à fond blanc et bord en tirets, ou simple titre en gras au-dessus).
3. **Choisir le sens du flux** et s'y tenir :
   - vertical **haut → bas** (CNN, encodeurs) ou **bas → haut** (style Transformer) ;
   - horizontal gauche → droite (séquences, RNN déroulés, pipelines) ;
   - U-Net : encodeur qui descend à gauche, décodeur qui remonte à droite, skips horizontaux.
4. **Grille** : toutes les coordonnées en multiples de **5** ; colonnes alignées sur un même `cx`.
   - Pas vertical entre deux blocs d'une chaîne : `cy` + **45 à 60** (bloc de 28 px) ou
     + **70 à 95** (blocs de 40–56 px). Il faut ≥ 15 px de vide pour voir la pointe de flèche.
   - Écart horizontal entre deux colonnes : ≥ largeur d'un bloc + **60 px** (plus si des
     étiquettes ou des flèches passent entre elles).
   - Laisser **25 px** entre un élément et le bord du cadre d'un groupe voisin.
5. **Branches parallèles** (Inception, Q/K/V, multi-entrées) : même `cy`, `cx` espacés
   régulièrement.
6. **Dimensions** : dans le texte des blocs (`"Conv1d k3\n64 → 128"`) ou en petite annotation à
   droite (`type: text`, `font_size: 10`, `color: "#555555"`, `align: "left"`, `cx` ≈ bord droit + 40).
7. **Titre** : texte gras 16 px sous le schéma, et une ligne de référence en italique 10 px gris.

### Couleurs conseillées (cohérentes avec la bibliothèque)

| rôle | couleur |
|---|---|
| entrée / sortie, données | `#F0F0F0` (gris) ou `#F9D9DB` (rose, embeddings / signaux) |
| convolution | `#CFE8CC` (vert) |
| pooling | `#E4E4E4` |
| linéaire / FC | `#DCDDF2` (lavande) |
| MLP / feed-forward | `#C6E6F5` (bleu clair) |
| normalisation | `#F2F5C4` (jaune pâle) |
| activation | `#FFE3C4` (orange pâle) |
| attention | `#FCDDB8` (orange) |
| récurrent | `#CDEBE6` (vert d'eau) |
| décodeur / sur-échantillonnage | `#E3DAF2` (mauve) |
| bloc composite d'un étage | `#D6E6F7` (bleu) |

Les blocs `"lib"` ont déjà la bonne couleur : ne la redéfinis pas.

## 6. Exemple complet

Bloc résiduel (ResNet) répété 2 fois, flux haut → bas :

```json
{
  "format": "deepgraph",
  "nodes": [
    {"id": "x",    "type": "text", "text": "x", "italic": true, "cx": 0, "cy": 0},
    {"id": "j",    "type": "junction", "cx": 0, "cy": 35},
    {"id": "c1",   "lib": "Conv2D", "text": "Conv 3×3, 64", "cx": 0, "cy": 80},
    {"id": "bn1",  "lib": "Batch Norm", "cx": 0, "cy": 130},
    {"id": "relu", "lib": "ReLU", "w": 110, "cx": 0, "cy": 180},
    {"id": "c2",   "lib": "Conv2D", "text": "Conv 3×3, 64", "cx": 0, "cy": 230},
    {"id": "bn2",  "lib": "Batch Norm", "cx": 0, "cy": 280},
    {"id": "add",  "lib": "op_plus", "cx": 0, "cy": 330},
    {"id": "relu2","lib": "ReLU", "w": 110, "cx": 0, "cy": 375},
    {"id": "blk",  "type": "group", "label": "2×", "label_pos": "left",
     "children": ["j", "c1", "bn1", "relu", "c2", "bn2", "add", "relu2"], "margin": 15, "margin_right": 110},
    {"id": "note", "type": "text", "text": "identité", "font_size": 10, "color": "#555555",
     "italic": true, "align": "left", "cx": 120, "cy": 180},
    {"id": "title","type": "text", "text": "Bloc résiduel", "bold": true, "font_size": 16, "cx": 0, "cy": 470}
  ],
  "arrows": [
    {"from": "x", "to": "j", "head": "none"},
    {"from": "j", "to": "c1"},
    {"from": "c1", "to": "bn1"},
    {"from": "bn1", "to": "relu"},
    {"from": "relu", "to": "c2"},
    {"from": "c2", "to": "bn2"},
    {"from": "bn2", "to": "add"},
    {"from": "j", "to": {"node": "add", "side": "right"}, "style": "ortho2", "first": "h", "mid": 95},
    {"from": "add", "to": "relu2"},
    {"from": "relu2", "to": {"x": 0, "y": 425}}
  ]
}
```

Attention à plusieurs entrées (V, K, Q en bas d'un bloc, flux bas → haut) :

```json
{"from": "lin_v", "to": {"node": "sdpa", "side": "bottom", "at": 0.2}},
{"from": "lin_k", "to": {"node": "sdpa", "side": "bottom", "at": 0.5}},
{"from": "lin_q", "to": {"node": "sdpa", "side": "bottom", "at": 0.8}}
```
(avec `lin_v`, `lin_k`, `lin_q` placés sous `sdpa` aux `cx` correspondant à ces fractions de sa largeur).

## 7. Vérifications avant de rendre le JSON

- [ ] `"format": "deepgraph"` présent ; JSON valide (guillemets doubles, pas de virgule finale).
- [ ] Chaque `id` est unique ; chaque `from` / `to` / `children` désigne un `id` existant.
- [ ] Les groupes n'ont pas de `cx` / `cy` ; leurs enfants ont des coordonnées absolues.
- [ ] Aucune paire d'éléments ne se chevauche (centre ± moitié de la taille, en comptant ~7 px
      par caractère pour les textes).
- [ ] Les flèches `straight` relient des éléments alignés ; sinon `ortho1` / `ortho2`.
- [ ] Le flux (`forward`) est respecté : chaque tenseur réutilisé part d'une **jonction** ; chaque
      `torch.cat` → `op_concat`, chaque addition résiduelle → `op_plus`.
- [ ] Les dimensions (canaux, longueur, résolution) apparaissent là où elles aident à lire.

## 8. Format complet (pour information)

L'éditeur enregistre une version développée du même fichier : `uid` entiers, `x`/`y` = coin
haut-gauche (relatif au groupe parent pour un enfant, avec `"parent": uid`), flèches avec
`{"node": uid, "fx": 0–1, "fy": 0–1}`, plus `"settings"` (épaisseurs, police) et `"layers"`
(calques). Les deux formes sont acceptées à l'ouverture ; **écris toujours la forme simplifiée**.

---

## Annexe — blocs disponibles pour `"lib"`

Table générée depuis `blocks.yaml` (la régénérer après avoir ajouté des blocs :
`python outils/liste_blocs.py`). Le texte affiché peut toujours être remplacé par `"text"`.

Taille par défaut d'un bloc : 110 × 28 px ; opérateur : diamètre 18 px.

| lib | texte affiché | catégorie | couleur | taille (l × h) |
|---|---|---|---|---|
| `Input` | Input | Entrée / Sortie | #F0F0F0 | 110 × 28 |
| `Output` | Output | Entrée / Sortie | #F0F0F0 | 110 × 28 |
| `Embedding` | Embedding | Entrée / Sortie | #F9D9DB | 110 × 28 |
| `Positional Encoding` | Positional Encoding | Entrée / Sortie | #E9E3F3 | 110 × 44 |
| `Patch Embedding` | Patch Embedding | Entrée / Sortie | #F9D9DB | 110 × 44 |
| `Linear` | Linear | Dense / MLP | #DCDDF2 | 110 × 22 |
| `MLP` | MLP | Dense / MLP | #C6E6F5 | 110 × 44 |
| `Feed Forward` | Feed Forward | Dense / MLP | #C6E6F5 | 110 × 44 |
| `Dropout` | Dropout | Dense / MLP | #EDEDED | 110 × 28 |
| `Flatten` | Flatten | Dense / MLP | #EDEDED | 110 × 28 |
| `Conv1D` | Conv1D | Convolution | #CFE8CC | 110 × 28 |
| `Conv2D` | Conv2D | Convolution | #CFE8CC | 110 × 28 |
| `Conv3D` | Conv3D | Convolution | #CFE8CC | 110 × 28 |
| `Conv 3×3` | Conv 3×3 | Convolution | #CFE8CC | 110 × 28 |
| `Conv 1×1` | Conv 1×1 | Convolution | #CFE8CC | 110 × 28 |
| `Transposed Conv` | Transposed Conv | Convolution | #BFE0C5 | 110 × 28 |
| `Depthwise Conv` | Depthwise Conv | Convolution | #DDEFD6 | 110 × 28 |
| `Max Pooling` | Max Pooling | Convolution | #E4E4E4 | 110 × 28 |
| `Avg Pooling` | Avg Pooling | Convolution | #E4E4E4 | 110 × 28 |
| `Upsample` | Upsample | Convolution | #D6E6F7 | 110 × 28 |
| `Batch Norm` | Batch Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Layer Norm` | Layer Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Group Norm` | Group Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Instance Norm` | Instance Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `RMS Norm` | RMS Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `AdaIN` | AdaIN | Normalisation | #F2F5C4 | 110 × 28 |
| `AdaLN` | AdaLN | Normalisation | #F2F5C4 | 110 × 28 |
| `Weight Norm` | Weight Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Spectral Norm` | Spectral Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `L2 Norm` | L2 Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Local Response Norm` | Local Response Norm | Normalisation | #F2F5C4 | 110 × 28 |
| `Activation` | Activation | Activations | #FFE3C4 | 110 × 28 |
| `ReLU` | ReLU | Activations | #FFE3C4 | 80 × 28 |
| `Leaky ReLU` | Leaky ReLU | Activations | #FFE3C4 | 80 × 28 |
| `PReLU` | PReLU | Activations | #FFE3C4 | 80 × 28 |
| `ELU` | ELU | Activations | #FFE3C4 | 80 × 28 |
| `SELU` | SELU | Activations | #FFE3C4 | 80 × 28 |
| `GELU` | GELU | Activations | #FFE3C4 | 80 × 28 |
| `SiLU / Swish` | SiLU / Swish | Activations | #FFE3C4 | 80 × 28 |
| `Mish` | Mish | Activations | #FFE3C4 | 80 × 28 |
| `Sigmoid` | Sigmoid | Activations | #FFE3C4 | 80 × 28 |
| `Tanh` | Tanh | Activations | #FFE3C4 | 80 × 28 |
| `Hard Sigmoid` | Hard Sigmoid | Activations | #FFE3C4 | 80 × 28 |
| `Hard Swish` | Hard Swish | Activations | #FFE3C4 | 80 × 28 |
| `Softplus` | Softplus | Activations | #FFE3C4 | 80 × 28 |
| `GLU` | GLU | Activations | #FFE3C4 | 80 × 28 |
| `SwiGLU` | SwiGLU | Activations | #FFE3C4 | 80 × 28 |
| `Softmax` | Softmax | Activations | #CFEBD2 | 110 × 22 |
| `Self-Attention` | Self-Attention | Attention | #FCDDB8 | 110 × 44 |
| `Cross-Attention` | Cross-Attention | Attention | #FCDDB8 | 110 × 44 |
| `Multi-Head Attention` | Multi-Head Attention | Attention | #FCDDB8 | 110 × 44 |
| `Masked Multi-Head Attention` | Masked / Multi-Head / Attention | Attention | #FCDDB8 | 110 × 56 |
| `Scaled Dot-Product Attention` | Scaled Dot-Product Attention | Attention | #CBC1E6 | 180 × 44 |
| `RNN` | RNN | Récurrent | #CDEBE6 | 110 × 28 |
| `LSTM` | LSTM | Récurrent | #CDEBE6 | 110 × 28 |
| `GRU` | GRU | Récurrent | #CDEBE6 | 110 × 28 |
| `BiLSTM` | BiLSTM | Récurrent | #B9E0D9 | 110 × 28 |
| `Encoder` | Encoder | Génératif | #D6E6F7 | 110 × 44 |
| `Decoder` | Decoder | Génératif | #E3DAF2 | 110 × 44 |
| `Bottleneck` | Bottleneck | Génératif | #EFE6D2 | 110 × 28 |
| `Denoising` | Denoising | Génératif | #D9EEDB | 110 × 44 |
| `Noise` | Noise | Génératif | #EDEDED | 110 × 28 |
| `Sampling` | Sampling | Génératif | #F9E3C8 | 110 × 28 |
| `Generator` | Generator | Génératif | #D5ECF3 | 110 × 44 |
| `Discriminator` | Discriminator | Génératif | #F6D3D3 | 110 × 44 |
| `Input Embedding` | Input Embedding | Transformer (papier) | #F9D9DB | 110 × 44 |
| `Output Embedding` | Output Embedding | Transformer (papier) | #F9D9DB | 110 × 44 |
| `Add & Norm` | Add & Norm | Transformer (papier) | #F2F5C4 | 110 × 22 |
| `MatMul` | MatMul | Transformer (papier) | #CBC1E6 | 80 × 22 |
| `Scale` | Scale | Transformer (papier) | #FBF3BF | 80 × 22 |
| `Mask (opt.)` | Mask (opt.) | Transformer (papier) | #F7CEDB | 80 × 22 |
| `softmax_attention` | SoftMax | Transformer (papier) | #CFEBD2 | 80 × 22 |
| `Concat` | Concat | Transformer (papier) | #FCEFBF | 80 × 22 |
| `linear_proj` | Linear | Transformer (papier) | #E4EDE9 | 60 × 22 |
| `op_concat` | Concaténation | Opérateurs | #FFFFFF | cercle « concat » |
| `op_plus` | Addition | Opérateurs | #FFFFFF | cercle « plus » |
| `op_minus` | Différence | Opérateurs | #FFFFFF | cercle « minus » |
| `op_times` | Multiplication | Opérateurs | #FFFFFF | cercle « times » |
| `op_dot` | Produit terme à terme | Opérateurs | #FFFFFF | cercle « dot » |
