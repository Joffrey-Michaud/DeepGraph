# DeepGraph

Éditeur de schémas de réseaux de neurones dans le style de *Attention is all you need*.

```bash
pip install -r requirements.txt     # PyQt5 + PyYAML
python main.py                      # ou : python main.py mon_schema.dgraph
```

## Utilisation

| Action | Comment |
|---|---|
| Ajouter un bloc | glisser depuis la bibliothèque (ou double-clic dessus) ; boutons − / + à côté de la recherche pour replier / déplier toutes les catégories |
| Modifier le texte | double-clic sur le bloc (ou F2), ou panneau Propriétés |
| Tracer une flèche | **A** (mode flèche), glisser d'un bloc à un autre ; **V** / Échap pour revenir |
| Type de flèche | **1** droite · **2** un angle · **3** deux angles (s'applique aussi aux flèches sélectionnées) |
| Embranchement (résiduelle…) | en mode flèche, partir d'une flèche existante : une jonction est insérée |
| Ajuster une flèche | la sélectionner, puis tirer ses extrémités ou le **carré bleu** : 1 angle → le coude bascule du côté où on le tire ; 2 angles → le segment du milieu passe par la souris (tirer horizontalement = segment vertical, et inversement ; en dehors des blocs = forme en U) |
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
| Exporter | **Ctrl+E** → PDF / SVG / PNG ; **Ctrl+Maj+E** pour la sélection seule |
| Police | barre d'outils « Police » (à côté de Bords / Flèches) : s'applique à tout le diagramme et aux exports, enregistrée dans le `.dgraph` ; « Défaut » = police de `blocks.yaml` ; « Autre police… » pour choisir parmi toutes les polices installées |
| Épaisseurs | barre d'outils « Bords » / « Flèches » (tout le diagramme) ; par élément dans Propriétés, bouton « Défaut » pour revenir au global |
| Guides d'alignement | lignes roses pendant un déplacement (bords et centres) ; menu Affichage pour les couper |
| Sans magnétisme | maintenir **Maj** (grille et guides) |
| Se retrouver | **Recentrer** (touche Origine) : zoom 100 % centré sur le diagramme ; **Tout afficher** (Ctrl+1) |
| Mode sombre | Affichage > Mode sombre (Ctrl+Maj+D) ; les exports restent sur fond blanc |

Les diagrammes sont enregistrés en `.dgraph` (JSON lisible).

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
