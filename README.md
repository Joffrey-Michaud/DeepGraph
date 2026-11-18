# DeepGraph

Éditeur de schémas de réseaux de neurones dans le style de *Attention is all you need*.

```bash
pip install -r requirements.txt     # PyQt5 + PyYAML
python main.py                      # ou : python main.py mon_schema.dgraph
python main.py --update             # installe la dernière release
```

> **Modifier DeepGraph : une branche par modification, jamais de push sur `main`.** Chaque
> modification se fait dans une branche nommée `<type>/<sujet>` (ex. `fix/police-italique`,
> `feat/export-cli`) puis passe par une pull request ; l'administrateur décide de la fusionner.
> Règles complètes : [CONTRIBUTING.md](CONTRIBUTING.md).

## Utilisation

| Action | Comment |
|---|---|
| Ajouter un bloc | glisser depuis la bibliothèque (ou double-clic dessus) ; boutons − / + à côté de la recherche pour replier / déplier toutes les catégories |
| Modifier le texte | double-clic sur le bloc (ou F2), ou panneau Propriétés |
| Formules | entre `$…$`, syntaxe LaTeX simplifiée : `$x_t$`, `$W^Q$`, `$\frac{QK^\top}{\sqrt{d_k}}$`, `$\hat{x}_0$`, `$\alpha$`, `$\mathbb{R}^{d}$`, `$d \to 4d$`… (blocs, textes libres, étiquettes de groupe ; sans installer LaTeX). Hors des `$` le texte est inchangé (`conv2_x` reste `conv2_x`) ; `\$` pour un dollar. Liste complète : Aide > Documentation pour LLM |
| Tracer une flèche | **A** (mode flèche), glisser d'un bloc à un autre ; **V** / Échap pour revenir |
| Type de flèche | **1** droite · **2** un angle · **3** deux angles · **4** automatique (contourne les blocs, se recalcule quand on les déplace) · **5** coudes libres (s'applique aussi aux flèches sélectionnées) |
| Embranchement (résiduelle…) | en mode flèche, partir d'une flèche existante : une jonction est insérée |
| Ajuster une flèche | la sélectionner, puis tirer ses extrémités ou le **carré bleu** : 1 angle → le coude bascule du côté où on le tire ; 2 angles → le segment du milieu passe par la souris (tirer horizontalement = segment vertical, et inversement ; en dehors des blocs = forme en U) ; automatique / coudes libres → un carré bleu par segment intérieur (pas sur le premier ni le dernier, attachés aux blocs), à tirer perpendiculairement ; une flèche automatique devient alors « coudes libres » et garde sa forme. **Double-clic sur la flèche** (ou clic droit > « Ajouter un coude ici ») : le segment est coupé en deux morceaux, chacun avec son carré bleu ; tirer l'un des deux crée la marche. Clic droit > « Supprimer ce segment », « Tracé automatique ». Une flèche tracée directement en coudes libres part du tracé automatique |
| Marges d'un groupe | Propriétés : « Marges » (les 4 côtés) puis haut / bas / gauche / droite, « Défaut » pour revenir à la valeur commune ; le cadre suit son contenu (« Ajuster au contenu ») ; la poignée de taille règle les marges droite / bas |
| Empiler (stack) | clic droit sur un bloc > **Empiler (stack)…** (ou menu Édition) : nombre total de blocs, direction (↖ ↗ ↙ ↘), espacements horizontal / vertical, opacité dégressive ; aperçu en direct, Confirmer / Annuler. Les copies n'ont pas de texte et suivent le bloc. Réglable aussi dans Propriétés (« Copies empilées derrière », décalages) |
| Opacité d'un bloc | Propriétés > « Opacité (%) » |
| Ordre d'affichage | boutons **Propriétés / Ordre** (fin de la barre d'outils, ou menu Affichage) pour basculer le panneau de droite. « Ordre » liste les éléments du calque sélectionné (haut = premier plan) : glisser-déposer pour réordonner, les groupes se déplient pour ordonner leur contenu, survol = surbrillance, double-clic = centrer. Aussi : clic droit > Ordre, menu Disposition, **Ctrl+Page préc. / suiv.** (avancer / reculer), avec **Maj** (premier plan / arrière-plan) |
| Grouper | sélectionner puis **Ctrl+G** (cadre gris « N× ») ; **Ctrl+Maj+G** pour dégrouper |
| Ajouter à un groupe | déposer le bloc dans le cadre ; un groupe se saisit par son bord ou son étiquette |
| Aligner | menu Disposition (Ctrl+Maj+H : même axe vertical) |
| Calques | panneau « Calques » (haut de la liste = premier plan). **Seuls les éléments du calque sélectionné sont modifiables** (ou de tous les calques d'un groupe sélectionné). Survoler un calque met ses éléments en surbrillance orange. Cases Vis. / Verr., curseur d'opacité, glisser-déposer pour réordonner ou ranger dans un groupe, double-clic pour renommer |
| Menu d'un calque | clic droit : nouveau calque / groupe, dupliquer, copier / couper / coller le contenu, sélectionner le contenu, envoyer la sélection ici, grouper / dégrouper, fusionner avec le calque du dessous, masquer, verrouiller, afficher uniquement celui-ci, renommer, supprimer |
| Groupes de calques | sélectionner plusieurs calques (Ctrl / Maj + clic) puis « Grouper » ; masquer / verrouiller le groupe s'applique à tous ses calques, les opacités se multiplient |
| Modèles types | **Fichier > Modèles types** : LeNet-5, AlexNet, VGG-16, Inception, ResNet-18, U-Net, Transformer, attention (fig. 2 de Vaswani et al.), ViT, VAE, GAN, MLP, seq2seq. Le modèle s'ouvre comme un nouveau document sans titre (jamais écrasé). Tout `.dgraph` déposé dans [modeles/](modeles/) apparaît aussi dans le menu ; `python outils/generer_modeles.py` régénère les modèles fournis |
| Copier comme image | **Ctrl+Maj+C** (ou clic droit) : la sélection, ou tout le diagramme, dans le presse-papiers en PNG + SVG + PDF ; coller directement dans Word, LibreOffice, PowerPoint, Inkscape… (même échelle et même fond transparent que l'export PNG) |
| Exporter | **Ctrl+E** → PDF / SVG / PNG ; **Ctrl+Maj+E** pour la sélection seule ; sans l'éditeur : `python outils/exporter.py` (voir plus bas) |
| Police | barre d'outils « Police » (à côté de Bords / Flèches / Texte) : s'applique à tout le diagramme et aux exports, enregistrée dans le `.dgraph` ; « Défaut » = police de `blocks.yaml` ; « Autre police… » pour choisir parmi toutes les polices installées |
| Épaisseurs | barre d'outils « Bords » / « Flèches » (tout le diagramme) ; par élément dans Propriétés, bouton « Défaut » pour revenir au global |
| Blocs ajustés au texte | case de la barre d'outils (cochée pour les nouveaux diagrammes, décochée à l'ouverture d'un ancien fichier) : un bloc s'agrandit, en gardant son centre, dès que son texte déborderait (texte modifié, taille du texte, police) et ne peut pas être redimensionné plus petit que son texte. **Réduire aussi** (cochée par défaut) : quand le texte rétrécit, il revient vers sa taille d'origine (celle de la bibliothèque, ou la dernière taille donnée en le redimensionnant), jamais en dessous ; décochée, il ne fait que grandir. Clic droit > **Ajuster la taille au texte** (Ctrl+Maj+T) le met juste à la taille du texte |
| Taille du texte | barre d'outils « Texte » (tout le diagramme : blocs, textes libres, étiquettes de groupe à +2 px) ; un élément peut garder sa propre taille (Propriétés, bouton « Défaut » pour revenir au global) |
| Guides d'alignement | lignes roses pendant un déplacement (bords et centres) ; menu Affichage pour les couper |
| Sans magnétisme | maintenir **Maj** (grille et guides) |
| Se retrouver | **Recentrer** (touche Origine) : zoom 100 % centré sur le diagramme ; **Tout afficher** (Ctrl+1) |
| Mode sombre | Affichage > Mode sombre (Ctrl+Maj+D) ; les exports restent sur fond blanc |

Les diagrammes sont enregistrés en `.dgraph` (JSON lisible).

## Exporter en ligne de commande

Même rendu que **Ctrl+E**, sans ouvrir l'éditeur (utile pour régénérer les figures d'un article) :

```bash
python outils/exporter.py schema.dgraph                      # schema.pdf, .svg et .png à côté
python outils/exporter.py schema.dgraph -f pdf -o figs/arch  # figs/arch.pdf seulement
python outils/exporter.py modeles/*.dgraph -f pdf -d /tmp/x  # un PDF par fichier dans /tmp/x
python outils/exporter.py schema.dgraph --font "TeX Gyre Termes" --scale 6
```

PDF et SVG sont vectoriels (texte sélectionnable, polices intégrées). `--font` remplace la police du
diagramme, `--scale` règle la résolution du PNG (4 par défaut), `--transparent` enlève le fond blanc.

## Mises à jour

Au démarrage, DeepGraph cherche une release plus récente sur GitHub et **propose de l'installer** (à chaque
démarrage tant qu'elle n'est pas installée) ; **Aide > Rechercher des mises à jour…** le fait à la demande.

- **En fenêtre** : vérification en tâche de fond, boîte « Mise à jour disponible » avec les notes de version,
  puis proposition de redémarrer.
- **Sans fenêtre** (`outils/exporter.py`, éditeur lancé avec `QT_QPA_PLATFORM=offscreen`) : question dans le
  terminal s'il est interactif, sinon simple message. `python main.py --update` installe directement.
- **Installation** : dans un clone git, `main` avance jusqu'au tag de la release (seulement sur `main` sans
  modification locale, sinon rien n'est touché) ; dans une copie (.zip), l'archive de la release remplace les
  fichiers, un `blocks.yaml` modifié est conservé (la nouvelle version est posée à côté, `blocks.yaml.nouveau`).
- **Accès** : le dépôt est privé. La vérification utilise un jeton (`DEEPGRAPH_TOKEN`, `GITHUB_TOKEN` ou
  `GH_TOKEN`), sinon GitHub CLI connecté (`gh auth login`), sinon `git` pour un clone. Sans aucun de ces moyens,
  un avertissement explique quoi installer.
- **Désactiver** : `--no-update-check` ou `DEEPGRAPH_NO_UPDATE=1`.

## Générer un schéma avec un LLM

Menu **Aide > Documentation pour LLM…** : « Copier tout », coller dans la conversation, puis donner le code du modèle (même texte que [docs/DGRAPH_POUR_LLM.md](docs/DGRAPH_POUR_LLM.md), avec la liste des blocs toujours à jour) : il produit un `.dgraph`
au **format simplifié** (identifiants lisibles, positions par le centre, blocs de bibliothèque via `"lib"`,
groupes déclarés par leurs enfants, côtés des flèches automatiques), que l'éditeur ouvre directement.
Ce guide est aussi affiché par **Aide > Guide d'utilisation** (F1).

## Ajouter ses propres blocs

Éditer [blocks.yaml](blocks.yaml) (le format est documenté en tête du fichier) :

```yaml
  - name: Mon bloc
    color: "#C6E6F5"
    description: Ce que fait le bloc
    category: Ma catégorie
    height: 44
```

Le fichier est relu automatiquement à chaque enregistrement.

## Contribuer

Une branche `<type>/<sujet>` par modification, puis une pull request vers `main` ; on ne pousse jamais
directement sur `main`. Détails, types de branches et vérifications avant de pousser :
[CONTRIBUTING.md](CONTRIBUTING.md).
