"""Fenêtre principale : menus, barre d'outils, fichiers, export, configuration."""
import json
import os

from PyQt5.QtCore import Qt, QFileSystemWatcher, QSettings, QTimer, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QKeySequence, QPalette
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QColorDialog, QComboBox, QDialog,
                             QDialogButtonBox, QDockWidget, QFormLayout, QLineEdit, QStackedWidget,
                             QDoubleSpinBox, QFileDialog, QInputDialog, QLabel, QMainWindow,
                             QMenu, QMessageBox, QSpinBox, QStyle, QCheckBox)

from .arrow import STYLES, ArrowItem
from .items import BlockItem, NodeItem, OperatorItem
from .config import CFG
from .export import copy_scene_image, export_scene
from .layers import LayersPanel
from .library import LibraryPanel
from .properties import PropertiesPanel
from .scene import FORMAT, DiagramScene
from .stack_dialog import StackDialog
from .order_panel import OrderPanel
from .help_dialog import DocDialog
from . import __version__, docs
from .view import DiagramView

FILE_FILTER = "Diagramme DeepGraph (*.dgraph);;Tous les fichiers (*)"
_LIGHT_PALETTE = None


def apply_ui_theme(dark):
    """Palette sombre / claire de l'interface (style Fusion)."""
    global _LIGHT_PALETTE
    app = QApplication.instance()
    if _LIGHT_PALETTE is None:
        _LIGHT_PALETTE = QPalette(app.palette())
    if not dark:
        app.setPalette(_LIGHT_PALETTE)
        return
    p = QPalette()
    base, win, text = QColor("#1E1F22"), QColor("#2B2D31"), QColor("#E3E3E3")
    for role, col in ((QPalette.Window, win), (QPalette.WindowText, text), (QPalette.Base, base),
                      (QPalette.AlternateBase, QColor("#26282C")), (QPalette.ToolTipBase, win),
                      (QPalette.ToolTipText, text), (QPalette.Text, text), (QPalette.Button, QColor("#35373C")),
                      (QPalette.ButtonText, text), (QPalette.BrightText, QColor("#FF6B6B")),
                      (QPalette.Link, QColor("#6AA9FF")), (QPalette.Highlight, QColor("#2F6FD0")),
                      (QPalette.HighlightedText, QColor("#FFFFFF")), (QPalette.Light, QColor("#45474D")),
                      (QPalette.Midlight, QColor("#3A3C41")), (QPalette.Mid, QColor("#2A2B2F")),
                      (QPalette.Dark, QColor("#18191B")), (QPalette.Shadow, QColor("#000000")),
                      (QPalette.PlaceholderText, QColor("#8A8A8A"))):
        p.setColor(role, col)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor("#77797E"))
    app.setPalette(p)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = DiagramScene(self)
        self.view = DiagramView(self.scene)
        self.setCentralWidget(self.view)

        self.library = LibraryPanel()
        dock = QDockWidget("Bibliothèque", self)
        dock.setObjectName("library")
        dock.setWidget(self.library)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.props = PropertiesPanel(self.scene)
        self.order_panel = OrderPanel(self.scene, self.view)
        self.right_stack = QStackedWidget()          # panneau de droite : Propriétés ou Ordre
        self.right_stack.addWidget(self.props)
        self.right_stack.addWidget(self.order_panel)
        dock2 = QDockWidget("Propriétés", self)
        dock2.setObjectName("properties")
        dock2.setWidget(self.right_stack)
        self.right_dock = dock2
        self.addDockWidget(Qt.RightDockWidgetArea, dock2)
        self.layers_panel = LayersPanel(self.scene)
        dock3 = QDockWidget("Calques", self)
        dock3.setObjectName("layers")
        dock3.setWidget(self.layers_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock3)
        self.splitDockWidget(dock2, dock3, Qt.Vertical)
        self.resizeDocks([dock, dock2], [240, 290], Qt.Horizontal)
        self.resizeDocks([dock2, dock3], [560, 220], Qt.Vertical)

        self.current_file = None
        self.saved_state = self.scene.current_state()
        self.png_scale = 4.0

        self._make_actions()
        self._make_menus()
        self._make_toolbar()
        self.view.context_actions = [self._bend_menu, self.a_edit, self._stack_entry, self.a_fit_text, self._lib_entry, None, self.a_copy, self.a_copy_img, self.a_dup, self.a_delete, None,
                                     self.a_group, self.a_ungroup, self._order_menu, self._layer_menu, None, self.a_export_sel]

        self.zoom_label = QLabel("100 %")
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))

        self.library.activated.connect(self.add_at_center)
        self.library.reloadRequested.connect(lambda: self.reload_config(verbose=True))
        self.library.openFileRequested.connect(self.open_yaml)
        self.library.changed.connect(lambda: self.reload_config(verbose=False))
        self.scene.modified.connect(self._on_modified)
        self.scene.modeChanged.connect(self._on_mode)

        # rechargement automatique quand blocks.yaml est modifié
        self.watcher = QFileSystemWatcher(self)
        self._watch_config()
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(300)
        self._reload_timer.timeout.connect(lambda: self.reload_config(verbose=False))
        self.watcher.fileChanged.connect(lambda _: self._reload_timer.start())

        self._set_custom_colors()
        self.settings = QSettings("DeepGraph", "DeepGraph")
        self.a_dark.setChecked(self.settings.value("dark", False, type=bool))
        self._on_modified()
        self._on_mode("select")
        self.resize(1450, 880)

    # --------------------------------------------------------------- actions
    def _act(self, text, slot, shortcut=None, tip=None, checkable=False, icon=None):
        a = QAction(text, self)
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        if shortcut:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts([QKeySequence(s) for s in shortcut])
            else:
                a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        if checkable:
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda checked=False, f=slot: f())
        self.addAction(a)
        return a

    def _make_actions(self):
        sc = self.scene
        self.a_new = self._act("Nouveau", self.new_file, QKeySequence.New, icon=QStyle.SP_FileIcon)
        self.a_open = self._act("Ouvrir…", self.open_dialog, QKeySequence.Open, icon=QStyle.SP_DialogOpenButton)
        self.a_save = self._act("Enregistrer", self.save, QKeySequence.Save, icon=QStyle.SP_DialogSaveButton)
        self.a_save_as = self._act("Enregistrer sous…", self.save_as, "Ctrl+Shift+S")
        self.a_export = self._act("Exporter…", lambda: self.export(False), "Ctrl+E",
                                  "Exporter tout le diagramme en PDF / SVG / PNG")
        self.a_export_sel = self._act("Exporter la sélection…", lambda: self.export(True), "Ctrl+Shift+E")
        self.a_png_transp = self._act("PNG : fond transparent", lambda on: None, checkable=True)
        self.a_export_py = self._act("Exporter en code PyTorch…", self.export_pytorch, None,
                                     "Squelette de code PyTorch (nn.Module) qui suit le schéma")
        self.a_quit = self._act("Quitter", self.close, QKeySequence.Quit)

        self.a_undo = self._act("Annuler", sc.undo, QKeySequence.Undo, icon=QStyle.SP_ArrowBack)
        self.a_redo = self._act("Rétablir", sc.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], icon=QStyle.SP_ArrowForward)
        self.a_copy = self._act("Copier", sc.copy_selection, QKeySequence.Copy)
        self.a_copy_img = self._act("Copier comme image", self.copy_image, "Ctrl+Shift+C",
                                    "Copier la sélection (ou tout le diagramme) comme image, "
                                    "à coller dans un document, une présentation, Inkscape…")
        self.a_fit_text = self._act("Ajuster la taille au texte", self.fit_selected_to_text, "Ctrl+Shift+T",
                                    "Met les blocs sélectionnés juste à la taille de leur texte")
        self.a_cut = self._act("Couper", sc.cut_selection, QKeySequence.Cut)
        self.a_paste = self._act("Coller", sc.paste, QKeySequence.Paste)
        self.a_dup = self._act("Dupliquer", sc.duplicate, "Ctrl+D")
        self.a_delete = self._act("Supprimer", sc.delete_selected, [QKeySequence.Delete, "Backspace"])
        self.a_edit = self._act("Modifier le texte", sc.edit_selected, "F2")
        self.a_all = self._act("Tout sélectionner", self._select_all, QKeySequence.SelectAll)

        self.a_select = self._act("Sélection", lambda on: on and sc.set_mode("select"), "V",
                                  "Mode sélection / déplacement (V)", checkable=True)
        self.a_arrow = self._act("Flèche", lambda on: on and sc.set_mode("arrow"), "A",
                                 "Mode flèche : glisser d'un bloc à un autre (A)", checkable=True)
        grp = QActionGroup(self)
        grp.addAction(self.a_select)
        grp.addAction(self.a_arrow)
        self.a_select.setChecked(True)
        self.a_styles = []
        for i, (key, lab) in enumerate(STYLES):
            a = self._act(f"Flèche : {lab.lower()}", lambda k=key: self._set_arrow_style(k), str(i + 1))
            self.a_styles.append(a)

        self.a_group = self._act("Grouper", sc.group_selected, "Ctrl+G")
        self.a_ungroup = self._act("Dégrouper", sc.ungroup_selected, "Ctrl+Shift+G")
        self.a_align = {}
        for key, lab, sh in [("left", "Aligner à gauche", None), ("hcenter", "Centrer sur un axe vertical", "Ctrl+Shift+H"),
                             ("right", "Aligner à droite", None), ("top", "Aligner en haut", None),
                             ("vcenter", "Centrer sur un axe horizontal", "Ctrl+Shift+V"),
                             ("bottom", "Aligner en bas", None),
                             ("hdist", "Répartir horizontalement", None), ("vdist", "Répartir verticalement", None)]:
            self.a_align[key] = self._act(lab, lambda k=key: sc.align(k), sh)
        self.a_same_w = self._act("Même largeur", lambda: sc.same_size("w"))
        self.a_same_h = self._act("Même hauteur", lambda: sc.same_size("h"))

        self.a_snap = self._act("Magnétisme (grille)", self._toggle_snap, checkable=True)
        self.a_snap.setChecked(True)
        self.a_grid = self._act("Afficher la grille", self._toggle_grid, checkable=True)
        self.a_grid.setChecked(True)
        self.a_guides = self._act("Guides d'alignement", lambda on: setattr(self.scene, "guides_enabled", on),
                                  checkable=True,
                                  tip="Afficher des repères (et aimanter) quand un bloc s'aligne sur un autre")
        self.a_guides.setChecked(True)
        self.a_zoom_in = self._act("Zoom +", lambda: self.view.zoom_by(1.25), ["Ctrl++", "Ctrl+="])
        self.a_zoom_out = self._act("Zoom −", lambda: self.view.zoom_by(0.8), "Ctrl+-")
        self.a_zoom_1 = self._act("Zoom 100 %", self.view.zoom_reset, "Ctrl+0")
        self.a_fit = self._act("Tout afficher", self.view.fit, "Ctrl+1",
                               "Zoomer pour voir tout le diagramme (Ctrl+1)", icon=QStyle.SP_TitleBarMaxButton)
        self.a_recenter = self._act("Recentrer", self.view.recenter, "Home",
                                    "Revenir à la vue de départ : zoom 100 %, centrée sur le diagramme (Origine)",
                                    icon=QStyle.SP_BrowserReload)
        self.a_dark = self._act("Mode sombre", self._toggle_dark, "Ctrl+Shift+D", checkable=True,
                                tip="Interface et fond sombres (les exports restent sur fond blanc)")

        self.a_reload = self._act("Recharger la configuration", lambda: self.reload_config(True), "F5")
        self.a_cfg_edit = self._act("Ouvrir blocks.yaml (avancé)", self.open_config_file)
        self.a_cfg_choose = self._act("Choisir un autre fichier de configuration…", self.choose_config)
        self.a_panel_props = self._act("Propriétés", lambda: self.show_right_panel("props"), checkable=True,
                                       tip="Panneau de droite : propriétés de la sélection")
        self.a_panel_order = self._act("Ordre", lambda: self.show_right_panel("order"), checkable=True,
                                       tip="Panneau de droite : ordre d'affichage (avant / arrière) "
                                           "des éléments du calque sélectionné")
        grp = QActionGroup(self)
        grp.setExclusive(True)
        for a in (self.a_panel_props, self.a_panel_order):
            grp.addAction(a)
        self.a_panel_props.setChecked(True)
        self.a_restack = {}
        for how, txt, sc_ in (("front", "Mettre au premier plan", "Ctrl+Shift+PgUp"),
                              ("up", "Avancer", "Ctrl+PgUp"),
                              ("down", "Reculer", "Ctrl+PgDown"),
                              ("back", "Mettre à l'arrière-plan", "Ctrl+Shift+PgDown")):
            self.a_restack[how] = self._act(txt, lambda checked=False, h=how: self.scene.restack_selected(h), sc_)
        self.a_help_llm = self._act("Documentation pour LLM…", self.show_llm_doc,
                                    tip="Format .dgraph expliqué à un LLM, pour lui faire générer le schéma "
                                        "d'un modèle")
        self.a_help_guide = self._act("Guide d'utilisation…", self.show_user_guide, "F1")
        self.a_help_pytorch = self._act("Schéma depuis un modèle PyTorch…", self.show_pytorch_doc,
                                        tip="Fabriquer un brouillon de schéma à partir du code PyTorch d'un modèle "
                                            "(outils/depuis_pytorch.py)")
        self.a_help_vers_pytorch = self._act("Code PyTorch depuis un schéma…", self.show_vers_pytorch_doc,
                                             tip="Ce que fait Fichier > Exporter en code PyTorch, et comment "
                                                 "écrire le schéma pour en tirer le plus")
        self.a_help_update = self._act("Rechercher des mises à jour…", self.check_updates)
        self.a_stack = self._act("Empiler (stack)…", self.stack_dialog,
                                 tip="Ajoute derrière le bloc des copies sans texte (style « ×h »)")
        self.a_new_block = self._act("Nouveau bloc de bibliothèque…", self.new_library_block,
                                     tip="Créer un bloc dans la bibliothèque (nom, couleur, catégorie…)")
        self.a_add_to_lib = self._act("Ajouter à la bibliothèque…", self.add_selection_to_library,
                                      tip="Enregistrer ce bloc (couleur, taille, texte…) comme bloc de "
                                          "bibliothèque réutilisable")
        self.a_reapply = self._act("Réappliquer les couleurs de la config aux blocs", self.scene.reapply_config)

    def _make_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Fichier")
        for a in (self.a_new, self.a_open):
            m.addAction(a)
        self.templates_menu = m.addMenu("Modèles types")
        self.templates_menu.setToolTipsVisible(True)
        self.templates_menu.aboutToShow.connect(self._fill_templates_menu)
        self.my_models_menu = m.addMenu("Mes modèles")
        self.my_models_menu.setToolTipsVisible(True)
        self.my_models_menu.aboutToShow.connect(self._fill_my_models_menu)
        for a in (self.a_save, self.a_save_as, None, self.a_export,
                  self.a_export_sel, self.a_png_transp, self.a_export_py, None, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Édition")
        for a in (self.a_undo, self.a_redo, None, self.a_cut, self.a_copy, self.a_copy_img, self.a_paste, self.a_dup,
                  self.a_delete, None, self.a_edit, self.a_stack, self.a_all):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Outils")
        m.addAction(self.a_select)
        m.addAction(self.a_arrow)
        m.addSeparator()
        for a in self.a_styles:
            m.addAction(a)
        m = mb.addMenu("&Disposition")
        m.addAction(self.a_group)
        m.addAction(self.a_ungroup)
        m.addSeparator()
        for k in ("left", "hcenter", "right"):
            m.addAction(self.a_align[k])
        m.addSeparator()
        for k in ("top", "vcenter", "bottom"):
            m.addAction(self.a_align[k])
        m.addSeparator()
        m.addAction(self.a_align["hdist"])
        m.addAction(self.a_align["vdist"])
        m.addSeparator()
        m.addAction(self.a_same_w)
        m.addAction(self.a_same_h)
        m.addSeparator()
        for a in self.a_restack.values():
            m.addAction(a)
        m = mb.addMenu("&Affichage")
        for a in (self.a_recenter, self.a_fit, self.a_zoom_in, self.a_zoom_out, self.a_zoom_1, None,
                  self.a_grid, self.a_snap, self.a_guides, None, self.a_dark, None,
                  self.a_panel_props, self.a_panel_order):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Configuration")
        for a in (self.a_new_block, None, self.a_reload, self.a_cfg_edit, self.a_cfg_choose, None, self.a_reapply):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("Ai&de")
        m.addAction(self.a_help_guide)
        m.addAction(self.a_help_pytorch)
        m.addAction(self.a_help_vers_pytorch)
        m.addAction(self.a_help_llm)
        m.addSeparator()
        m.addAction(self.a_help_update)
        a = m.addAction(f"À propos de DeepGraph {__version__}")
        a.triggered.connect(lambda: QMessageBox.about(
            self, "À propos", f"<b>DeepGraph {__version__}</b><br>Éditeur de schémas de réseaux de neurones."))

    def _make_toolbar(self):
        tb = self.addToolBar("Principal")
        tb.setObjectName("main_toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_new, self.a_open, self.a_save):
            tb.addAction(a)
        tb.addAction(self.a_export)
        tb.addSeparator()
        tb.addAction(self.a_undo)
        tb.addAction(self.a_redo)
        tb.addSeparator()
        tb.addAction(self.a_select)
        tb.addAction(self.a_arrow)
        self.style_combo = QComboBox()
        for key, lab in STYLES:
            self.style_combo.addItem(lab, key)
        self.style_combo.setToolTip("Tracé des nouvelles flèches (touches 1 à 5)")
        self.style_combo.currentIndexChanged.connect(
            lambda i: setattr(self.scene, "arrow_style", self.style_combo.itemData(i)))
        tb.addWidget(self.style_combo)
        tb.addSeparator()
        tb.addAction(self.a_group)
        tb.addAction(self.a_ungroup)
        tb.addSeparator()
        tb.addAction(self.a_align["hcenter"])
        tb.addAction(self.a_align["vcenter"])
        tb.addSeparator()
        tb.addAction(self.a_snap)
        tb.addSeparator()
        tb.addAction(self.a_panel_props)
        tb.addAction(self.a_panel_order)

        self.addToolBarBreak()
        tb2 = self.addToolBar("Épaisseurs")
        tb2.setObjectName("widths_toolbar")
        self.width_boxes = {}
        for key, lab, tip in (("border_width", "Bords", "Épaisseur globale des bords des blocs, opérateurs et groupes"),
                              ("arrow_width", "Flèches", "Épaisseur globale des flèches")):
            sb = QDoubleSpinBox()
            sb.setRange(0.1, 12)
            sb.setSingleStep(0.1)
            sb.setDecimals(2)
            sb.setValue(getattr(self.scene, key))
            sb.setToolTip(tip + " (n'affecte pas la taille du texte ; un élément peut avoir sa propre épaisseur)")
            sb.valueChanged.connect(lambda v, k=key: self.scene.set_global(k, v))
            l = QLabel(f"  {lab} : ")
            l.setToolTip(sb.toolTip())
            tb2.addWidget(l)
            tb2.addWidget(sb)
            self.width_boxes[key] = sb
        sb = QSpinBox()
        sb.setRange(5, 80)
        sb.setSuffix(" px")
        sb.setValue(self.scene.font_size)
        sb.setToolTip("Taille globale du texte des blocs et des textes libres (étiquettes de groupe : +2 px).\n"
                      "Un élément peut avoir sa propre taille (panneau Propriétés, bouton « Défaut » pour revenir "
                      "à la taille globale).")
        sb.valueChanged.connect(lambda v: self.scene.set_global("font_size", v))
        l = QLabel("  Texte : ")
        l.setToolTip(sb.toolTip())
        tb2.addWidget(l)
        tb2.addWidget(sb)
        self.width_boxes["font_size"] = sb
        self.fit_box = QCheckBox("Blocs ajustés au texte")
        self.fit_box.setToolTip("Les blocs s'agrandissent pour que leur texte ne déborde jamais "
                                "(« Réduire aussi » : ils reviennent ensuite à leur taille d'origine ;\nclic droit > "
                                "« Ajuster la taille au texte » pour les mettre juste à la taille du texte).")
        self.fit_box.setChecked(self.scene.fit_text)
        self.fit_box.toggled.connect(lambda v: self.scene.set_global("fit_text", v))
        tb2.addWidget(QLabel("  "))
        tb2.addWidget(self.fit_box)
        self.shrink_box = QCheckBox("Réduire aussi")
        self.shrink_box.setToolTip("Quand le texte rétrécit (taille du texte, texte plus court…), les blocs "
                                   "agrandis reviennent vers leur taille d'origine\n(celle de la bibliothèque, "
                                   "ou celle donnée en redimensionnant le bloc), sans descendre sous le texte.")
        self.shrink_box.setChecked(self.scene.fit_shrink)
        self.shrink_box.setEnabled(self.scene.fit_text)
        self.shrink_box.toggled.connect(lambda v: self.scene.set_global("fit_shrink", v))
        tb2.addWidget(self.shrink_box)
        self.scene.settingsChanged.connect(self._sync_widths)

        tb2.addSeparator()
        self.font_combo = QComboBox()
        self.font_combo.setToolTip("Police de tout le diagramme (blocs, textes, étiquettes, exports).\n"
                                   "Enregistrée dans le fichier .dgraph.")
        self.font_combo.setMaxVisibleItems(30)
        self.font_combo.setMinimumContentsLength(16)
        self._fill_fonts()
        self.font_combo.activated.connect(self._on_font_chosen)
        l = QLabel("  Police : ")
        l.setToolTip(self.font_combo.toolTip())
        tb2.addWidget(l)
        tb2.addWidget(self.font_combo)
        self.scene.settingsChanged.connect(self._sync_font)

        tb3 = self.addToolBar("Vue")
        tb3.setObjectName("view_toolbar")
        tb3.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb3.addAction(self.a_recenter)
        tb3.addAction(self.a_fit)
        tb3.addAction(self.a_dark)

    # ----------------------------------------------------------------- état
    def _sync_widths(self):
        for box, val in ((self.fit_box, self.scene.fit_text), (self.shrink_box, self.scene.fit_shrink)):
            if box.isChecked() != val:
                box.blockSignals(True)
                box.setChecked(val)
                box.blockSignals(False)
        self.shrink_box.setEnabled(self.scene.fit_text)
        for key, sb in self.width_boxes.items():
            v = getattr(self.scene, key)
            if abs(sb.value() - v) > 1e-9:
                sb.blockSignals(True)
                sb.setValue(v)
                sb.blockSignals(False)

    # Polices proposées (seules celles installées apparaissent)
    FONT_GROUPS = [
        ("Sans empattement", ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arial", "Liberation Sans",
                              "DejaVu Sans", "Latin Modern Sans", "Linux Biolinum O", "Roboto", "Open Sans",
                              "Lato", "Noto Sans", "Fira Sans", "Source Sans Pro", "Ubuntu", "Cantarell"]),
        ("Avec empattement", ["Latin Modern Roman", "CMU Serif", "TeX Gyre Termes", "Times New Roman",
                              "Nimbus Roman", "Liberation Serif", "TeX Gyre Pagella", "Linux Libertine O",
                              "STIX", "STIX Two Text", "Noto Serif", "DejaVu Serif", "EB Garamond", "Charis SIL"]),
        ("Chasse fixe", ["Latin Modern Mono", "DejaVu Sans Mono", "Fira Code", "Liberation Mono"]),
    ]
    OTHER_FONT = "__other__"

    def _fill_fonts(self):
        cb = self.font_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItem(f"Défaut ({CFG.default_family().split(' [')[0]})", None)
        seen = set()
        for title, names in self.FONT_GROUPS:
            fams = []
            for n in names:
                f = CFG.resolve_family(n)
                if f and f not in seen:
                    seen.add(f)
                    fams.append(f)
            if not fams:
                continue
            cb.insertSeparator(cb.count())
            cb.addItem(f"— {title} —", "__title__")
            cb.model().item(cb.count() - 1).setEnabled(False)
            for f in fams:
                cb.addItem(f.split(" [")[0], f)
                cb.setItemData(cb.count() - 1, QFont(f, 10), Qt.FontRole)
        cur = CFG.family_override
        if cur and cur not in seen:              # police d'un fichier ouvert, hors liste
            cb.insertSeparator(cb.count())
            cb.addItem(cur if CFG.resolve_family(cur) else f"{cur} (non installée)", cur)
        cb.insertSeparator(cb.count())
        cb.addItem("Autre police…", self.OTHER_FONT)
        cb.blockSignals(False)
        self._sync_font()

    def _sync_font(self):
        cb = self.font_combo
        cur = CFG.family_override
        i = cb.findData(cur) if cur else 0
        if cur and i < 0:
            self._fill_fonts()                   # ajoute l'entrée puis se resynchronise
            return
        cb.blockSignals(True)
        cb.setCurrentIndex(max(0, i))
        cb.blockSignals(False)

    def _on_font_chosen(self, i):
        data = self.font_combo.itemData(i)
        if data == self.OTHER_FONT:
            fams = [f for f in QFontDatabase().families() if not QFontDatabase().isPrivateFamily(f)]
            cur = CFG.font_family()
            name, ok = QInputDialog.getItem(self, "Autre police", "Police du diagramme :", fams,
                                            max(0, fams.index(cur)) if cur in fams else 0, False)
            if ok and name:
                self.scene.set_font_family(name)
            self._sync_font()
            return
        if data == "__title__":
            self._sync_font()
            return
        self.scene.set_font_family(data)

    def show_right_panel(self, which):
        order = which == "order"
        self.right_stack.setCurrentWidget(self.order_panel if order else self.props)
        self.right_dock.setWindowTitle("Ordre" if order else "Propriétés")
        (self.a_panel_order if order else self.a_panel_props).setChecked(True)
        self.right_dock.show()
        self.right_dock.raise_()

    def _order_menu(self, parent):
        if not any(isinstance(i, NodeItem) for i in self.scene.selectedItems()):
            return None
        m = QMenu("Ordre", parent)
        for a in self.a_restack.values():
            m.addAction(a)
        return m

    def _show_doc(self, key, title, text, hint, save_name):
        dlg = getattr(self, key, None)
        if dlg is not None and dlg.isVisible():
            dlg.close()
        dlg = DocDialog(title, text, self, hint=hint, save_name=save_name)
        setattr(self, key, dlg)
        dlg.show()
        dlg.raise_()

    def show_llm_doc(self):
        self._show_doc("_llm_doc", "Documentation pour LLM — format .dgraph", docs.llm_doc(),
                       "Pour obtenir le schéma d'un modèle : « Copier tout », coller dans la conversation avec le "
                       "LLM, puis donner le code du modèle. Enregistrer sa réponse JSON dans un fichier .dgraph "
                       "et l'ouvrir (Fichier > Ouvrir). La liste des blocs (annexe) reflète la configuration "
                       "actuelle.", "DGRAPH_POUR_LLM.md")

    def check_updates(self):
        from .update_ui import check_in_background
        check_in_background(self, manual=True)

    def show_pytorch_doc(self):
        tool = os.path.join(docs.ROOT, "outils", "depuis_pytorch.py")
        self._show_doc("_pytorch_doc", "Schéma depuis un modèle PyTorch", docs.pytorch_doc(),
                       f"Commande à lancer dans un terminal, depuis la racine du projet du modèle : "
                       f"python {tool} fichier.py:Classe -e x:1,3,224,224 -o brouillon.dgraph",
                       "DEPUIS_PYTORCH.md")

    def show_vers_pytorch_doc(self):
        self._show_doc("_vers_pytorch_doc", "Code PyTorch depuis un schéma", docs.vers_pytorch_doc(),
                       "Menu Fichier > Exporter en code PyTorch…, ou dans un terminal : "
                       f"python {os.path.join(docs.ROOT, 'outils', 'vers_pytorch.py')} schema.dgraph --essai",
                       "VERS_PYTORCH.md")

    def show_user_guide(self):
        self._show_doc("_guide", "Guide d'utilisation — DeepGraph", docs.user_guide(), "", "DeepGraph_guide.md")

    def _selected_blocks(self):
        return [i for i in self.scene.selectedItems() if isinstance(i, BlockItem)]

    def _bend_menu(self, parent):
        """Clic droit sur une flèche : actions sur les coudes, au premier niveau du menu."""
        arrows = [i for i in self.scene.selectedItems() if isinstance(i, ArrowItem)]
        if len(arrows) != 1:
            return None
        a = arrows[0]
        q = getattr(self.scene, "context_pos", None) or a.handle_points().get("src")

        def do(fn, *args):
            fn(*args)
            self.scene.checkpoint()
            self.scene.itemsEdited.emit()

        acts = []

        def act(text, fn, enabled=True, tip=None):
            x = QAction(text, parent)
            x.triggered.connect(fn)
            x.setEnabled(enabled)
            if tip:
                x.setStatusTip(tip)
            acts.append(x)

        act("Ajouter un coude ici (ou double-clic)", lambda: do(a.add_bend, q))
        act("Supprimer ce segment", lambda: do(a.remove_bend, q), a.style == "free" and len(a.bends) > 1)
        act("Tracé automatique (contourne les blocs)", lambda: do(a.set_prop, "style", "auto"), a.style != "auto")
        sep = QAction(parent)
        sep.setSeparator(True)
        return acts + [sep]

    def _stack_entry(self, parent):
        return self.a_stack if self._selected_blocks() else None

    def stack_dialog(self):
        blocks = self._selected_blocks()
        if not blocks:
            self.statusBar().showMessage("Sélectionner d'abord un ou plusieurs blocs", 3000)
            return
        old = getattr(self, "_stack_dlg", None)
        if old is not None and old.isVisible():
            old.reject()                          # une seule fenêtre à la fois
        dlg = StackDialog(blocks, self, self.view)
        self._stack_dlg = dlg

        def done(result):
            if result:
                self.scene.checkpoint()
                self.scene.itemsEdited.emit()
            self._stack_dlg = None
            dlg.deleteLater()

        dlg.finished.connect(done)
        dlg.place_beside_blocks()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _layer_menu(self, parent):
        if not self.scene.selectedItems():
            return None
        m = QMenu("Envoyer vers le calque", parent)
        for L in reversed(self.scene.layers):
            a = m.addAction(L["name"])
            a.triggered.connect(lambda checked=False, lid=L["id"]: self.scene.move_selection_to_layer(lid))
        m.addSeparator()
        a = m.addAction("Nouveau calque…")
        a.triggered.connect(self._send_to_new_layer)
        return m

    def _send_to_new_layer(self):
        sel = self.scene.selectedItems()
        lid = self.scene.add_layer()          # la sélection est perdue en changeant de calque
        for it in sel:
            it.setFlag(it.ItemIsSelectable, True)
            it.setSelected(True)
        self.scene.move_selection_to_layer(lid)

    def _set_arrow_style(self, key):
        i = self.style_combo.findData(key)
        self.style_combo.setCurrentIndex(i)
        self.scene.set_style_of_selected_arrows(key)
        self.statusBar().showMessage(f"Nouvelles flèches : {self.style_combo.currentText()}", 2500)

    def _select_all(self):
        for it in self.scene.items():
            if hasattr(it, "PROPS"):
                it.setSelected(True)

    def _toggle_snap(self, on):
        self.scene.snap_enabled = on

    def _toggle_dark(self, on):
        apply_ui_theme(on)
        self.scene.dark = on
        self.scene.update()
        self.view.viewport().update()
        self.library.rebuild()
        if hasattr(self, "settings"):
            self.settings.setValue("dark", on)

    def _toggle_grid(self, on):
        self.view.show_grid = on
        self.view.viewport().update()

    def _on_mode(self, mode):
        (self.a_arrow if mode == "arrow" else self.a_select).setChecked(True)
        if mode == "arrow":
            self.statusBar().showMessage(
                "Mode flèche : glisser d'un bloc à un autre · clic sur une flèche = embranchement · "
                "Maj = sans aimantation · Échap = quitter")
        else:
            self.statusBar().clearMessage()

    def is_dirty(self):
        return self.scene.current_state() != self.saved_state

    def _on_modified(self):
        name = os.path.basename(self.current_file) if self.current_file else "Sans titre"
        self.setWindowTitle(f"{'• ' if self.is_dirty() else ''}{name} — DeepGraph")
        self.a_undo.setEnabled(self.scene.can_undo())
        self.a_redo.setEnabled(self.scene.can_redo())

    def add_at_center(self, bid):
        self.scene.set_mode("select")
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene.create_from_def(bid, c)
        self.view.setFocus()

    # -------------------------------------------------------------- fichiers
    def _dir(self):
        return os.path.dirname(self.current_file) if self.current_file else os.getcwd()

    def maybe_save(self):
        self.scene.checkpoint()
        if not self.is_dirty():
            return True
        r = QMessageBox.question(self, "Modifications non enregistrées",
                                 "Enregistrer les modifications avant de continuer ?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def new_file(self):
        if not self.maybe_save():
            return
        self.scene.load({"nodes": [], "arrows": [], "settings": {"fit_text": True}})
        self.current_file = None
        self.saved_state = self.scene.current_state()
        self.view.zoom_reset()
        self.view.centerOn(0, 0)
        self._on_modified()

    def open_dialog(self):
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un diagramme", self._dir(), FILE_FILTER)
        if path:
            self.open_file(path)

    # ------------------------------------------------------- modèles types
    TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modeles")

    def _templates(self):
        """[(catégorie, titre, chemin, référence)] : modeles/index.json, puis les autres .dgraph du dossier."""
        d = self.TEMPLATES_DIR
        out, seen = [], set()
        try:
            with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
                for e in json.load(f):
                    path = os.path.join(d, e["file"])
                    if os.path.isfile(path):
                        out.append((e.get("category", ""), e.get("title", e["file"]), path, e.get("ref", "")))
                        seen.add(os.path.abspath(path))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):         # schémas ajoutés à la main dans modeles/
                path = os.path.abspath(os.path.join(d, fn))
                if fn.endswith(".dgraph") and path not in seen:
                    out.append(("Autres", os.path.splitext(fn)[0], path, ""))
        return out

    def _fill_templates_menu(self):
        m = self.templates_menu
        m.clear()
        subs = {}
        for cat, title, path, ref in self._templates():
            sub = m if not cat else subs.get(cat) or subs.setdefault(cat, m.addMenu(cat))
            self._model_submenu(sub, title, path, ref)
        if not subs and m.isEmpty():
            m.addAction("(aucun modèle)").setEnabled(False)
        m.addSeparator()
        a = m.addAction("Ouvrir le dossier des modèles")
        a.setToolTip("Tout fichier .dgraph déposé dans ce dossier apparaît dans ce menu")
        a.triggered.connect(lambda: (os.makedirs(self.TEMPLATES_DIR, exist_ok=True),
                                     QDesktopServices.openUrl(QUrl.fromLocalFile(self.TEMPLATES_DIR))))

    def _model_submenu(self, parent, title, path, tip="", user=False):
        """Sous-menu d'un modèle : l'ouvrir comme nouveau fichier, ou l'insérer dans le document."""
        sub = parent.addMenu(title)
        if tip:
            sub.menuAction().setToolTip(tip)
            sub.menuAction().setStatusTip(tip)
        for label, mode, hint in (
                ("Ouvrir dans un nouveau fichier", "new", "Le modèle s'ouvre comme un document sans titre"),
                ("Insérer dans un nouveau calque", "layer",
                 "Ajouté au document, dans un calque à son nom, au centre de la vue"),
                ("Insérer dans le calque actuel", "current",
                 "Ajouté au document, dans le calque sélectionné, au centre de la vue")):
            a = sub.addAction(label)
            a.setStatusTip(hint)
            a.triggered.connect(lambda checked=False, p=path, t=title, md=mode: self.use_model(p, t, md))
        if user:
            sub.addSeparator()
            sub.addAction("Renommer ou changer de catégorie…").triggered.connect(
                lambda checked=False, p=path: self.rename_my_model(p))
            sub.addAction("Supprimer de mes modèles…").triggered.connect(
                lambda checked=False, p=path: self.delete_my_model(p))
        return sub

    def use_model(self, path, title, mode):
        """mode « new » : nouveau document ; « layer » / « current » : insertion (Ctrl+Z pour annuler)."""
        if mode == "new":
            return self.open_template(path, title)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format") != FORMAT:
                raise ValueError("ce n'est pas un fichier DeepGraph")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Insertion impossible", f"{path}\n\n{e}")
            return False
        center = self.view.mapToScene(self.view.viewport().rect().center())
        created = self.scene.insert_diagram(data, mode, title, center)
        frames = [c.scene_frame() for c in created if isinstance(c, NodeItem) and c.parentItem() is None]
        if frames:                                    # la vue suit le modèle inséré
            r = frames[0]
            for f in frames[1:]:
                r = r.united(f)
            self.view.centerOn(r.center())
        where = "dans un nouveau calque" if mode == "layer" else "dans le calque actuel"
        self.statusBar().showMessage(f"Modèle « {title} » inséré {where} ({len(created)} éléments, "
                                     "sélectionnés) — Ctrl+Z pour annuler", 6000)
        return True

    # ------------------------------------------------------- mes modèles
    MY_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mes_modeles")

    def _my_models(self):
        """[(catégorie, titre, chemin)] : mes_modeles/*.dgraph, et un niveau de sous-dossiers (catégories)."""
        d, out = self.MY_MODELS_DIR, []
        if not os.path.isdir(d):
            return out
        for fn in sorted(os.listdir(d), key=str.lower):
            path = os.path.join(d, fn)
            if fn.endswith(".dgraph") and os.path.isfile(path):
                out.append(("", os.path.splitext(fn)[0], path))
            elif os.path.isdir(path) and not fn.startswith("."):
                for g in sorted(os.listdir(path), key=str.lower):
                    if g.endswith(".dgraph") and os.path.isfile(os.path.join(path, g)):
                        out.append((fn, os.path.splitext(g)[0], os.path.join(path, g)))
        return out

    def _fill_my_models_menu(self):
        m = self.my_models_menu
        m.clear()
        models = self._my_models()
        subs = {}
        for cat, title, path in sorted(models, key=lambda t: (t[0] == "", t[0].lower(), t[1].lower())):
            sub = m if not cat else subs.get(cat) or subs.setdefault(cat, m.addMenu(cat))
            self._model_submenu(sub, title, path, user=True)
        if not models:
            m.addAction("(vide : ajouter un schéma ci-dessous)").setEnabled(False)
        m.addSeparator()
        a = m.addAction("Ajouter le schéma à mes modèles…")
        a.setStatusTip("Enregistre tout le schéma dans « Mes modèles »")
        a.triggered.connect(lambda: self.add_to_my_models(selection=False))
        a = m.addAction("Ajouter la sélection à mes modèles…")
        a.setStatusTip("Enregistre seulement les éléments sélectionnés dans « Mes modèles »")
        a.setEnabled(bool(self.scene.selectedItems()))
        a.triggered.connect(lambda: self.add_to_my_models(selection=True))
        m.addSeparator()
        a = m.addAction("Ouvrir le dossier de mes modèles")
        a.setToolTip("Un .dgraph déposé dans ce dossier (ou dans un sous-dossier = une catégorie) apparaît ici")
        a.triggered.connect(lambda: (os.makedirs(self.MY_MODELS_DIR, exist_ok=True),
                                     QDesktopServices.openUrl(QUrl.fromLocalFile(self.MY_MODELS_DIR))))

    @staticmethod
    def _clean_name(name):
        """Nom saisi -> morceau de chemin sûr (« Catégorie/Nom » : un niveau de sous-dossier)."""
        parts = []
        for p in str(name).replace("\\", "/").split("/")[:2]:
            p = "".join("-" if c in '<>:"|?*' or ord(c) < 32 else c for c in p).strip().strip(".")
            if p:
                parts.append(p)
        return parts

    def _my_categories(self):
        d = self.MY_MODELS_DIR
        if not os.path.isdir(d):
            return []
        return sorted((f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f)) and not f.startswith(".")),
                      key=str.lower)

    def _ask_model_name(self, title, name, category=""):
        """Fenêtre nom + catégorie (existante ou nouvelle) -> [catégorie?, nom] nettoyés, ou [] si annulé."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        ed = QLineEdit(name)
        ed.selectAll()
        form.addRow("Nom :", ed)
        cb = QComboBox()
        cb.setEditable(True)
        cb.addItems([""] + self._my_categories())
        cb.lineEdit().setPlaceholderText("(aucune)")
        cb.setCurrentText(category)
        form.addRow("Catégorie :", cb)
        hint = QLabel("Choisir une catégorie existante ou en taper une nouvelle : elle devient un sous-menu "
                      "de Fichier > Mes modèles. Vide : le modèle est rangé à la racine du menu.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        form.addRow(hint)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Cancel).setText("Annuler")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        dlg.resize(460, dlg.sizeHint().height())
        if dlg.exec_() != QDialog.Accepted:
            return []
        parts = self._clean_name(ed.text())          # « Catégorie/Nom » tapé dans le nom : accepté aussi
        if not parts:
            return []
        if len(parts) == 1:
            cat = self._clean_name(cb.currentText().replace("/", "-"))
            parts = cat[:1] + parts
        return parts

    def add_to_my_models(self, selection=False):
        sel = self.scene.selectedItems() if selection else None
        if selection and not sel:
            QMessageBox.information(self, "Mes modèles", "Aucun élément sélectionné.")
            return False
        default = os.path.splitext(os.path.basename(self.current_file))[0] \
            if self.current_file and not selection else "Mon modèle"
        parts = self._ask_model_name("Ajouter à mes modèles", default)
        if not parts:
            return False
        folder = os.path.join(self.MY_MODELS_DIR, *parts[:-1])
        path = os.path.join(folder, parts[-1] + ".dgraph")
        if os.path.exists(path):
            r = QMessageBox.question(self, "Mes modèles", f"« {parts[-1]} » existe déjà. Le remplacer ?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return False
        self.scene.checkpoint()
        data = self.scene.to_dict(sel) if selection else self.scene.to_dict()
        data.setdefault("settings", self.scene.settings())
        try:
            os.makedirs(folder, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as e:
            QMessageBox.critical(self, "Enregistrement impossible", str(e))
            return False
        what = "La sélection" if selection else "Le schéma"
        self.statusBar().showMessage(f"{what} est dans Fichier > Mes modèles > {' > '.join(parts)}", 6000)
        return True

    def rename_my_model(self, path):
        """Renommer et / ou changer de catégorie (déplacer dans un autre sous-dossier de mes_modeles)."""
        old = os.path.splitext(os.path.basename(path))[0]
        folder = os.path.dirname(path)
        root = os.path.abspath(self.MY_MODELS_DIR)
        old_cat = "" if os.path.abspath(folder) == root else os.path.basename(folder)
        parts = self._ask_model_name("Renommer ou changer de catégorie", old, old_cat)
        if not parts:
            return False
        new_folder = os.path.join(self.MY_MODELS_DIR, *parts[:-1])
        new = os.path.join(new_folder, parts[-1] + ".dgraph")
        if os.path.abspath(new) == os.path.abspath(path):
            return False
        if os.path.exists(new):
            QMessageBox.warning(self, "Mes modèles", f"« {' > '.join(parts)} » existe déjà.")
            return False
        os.makedirs(new_folder, exist_ok=True)
        os.rename(path, new)
        if os.path.abspath(folder) != root and not os.listdir(folder):
            os.rmdir(folder)                          # catégorie devenue vide
        self.statusBar().showMessage(f"Modèle rangé dans Fichier > Mes modèles > {' > '.join(parts)}", 5000)
        return True

    def delete_my_model(self, path):
        title = os.path.splitext(os.path.basename(path))[0]
        r = QMessageBox.question(self, "Mes modèles", f"Supprimer « {title} » de mes modèles ?\n"
                                 "Le fichier est effacé du dossier mes_modeles.",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return False
        os.remove(path)
        folder = os.path.dirname(path)
        if os.path.abspath(folder) != os.path.abspath(self.MY_MODELS_DIR) and not os.listdir(folder):
            os.rmdir(folder)                          # catégorie vide
        return True

    def open_template(self, path, title=None):
        """Ouvre un modèle comme un nouveau document sans titre (le modèle n'est jamais écrasé)."""
        if not self.maybe_save():
            return False
        if not self.open_file(path):
            return False
        self.current_file = None
        self._on_modified()
        self.statusBar().showMessage(f"Modèle « {title or os.path.basename(path)} » ouvert — "
                                     "« Enregistrer » demandera un nom de fichier", 6000)
        return True

    def open_file(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format") != FORMAT:
                raise ValueError("ce n'est pas un fichier DeepGraph")
            self.scene.load(data)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ouverture impossible", f"{path}\n\n{e}")
            return False
        self.current_file = os.path.abspath(path)
        self.saved_state = self.scene.current_state()
        self._on_modified()
        QTimer.singleShot(0, self.view.fit)
        return True

    def save(self):
        if not self.current_file:
            return self.save_as()
        return self._write(self.current_file)

    def save_as(self):
        base = self.current_file or os.path.join(self._dir(), "diagramme.dgraph")
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le diagramme", base, FILE_FILTER)
        if not path:
            return False
        if not os.path.splitext(path)[1]:
            path += ".dgraph"
        return self._write(path)

    def _write(self, path):
        self.scene.checkpoint()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.scene.to_dict(), f, indent=1, ensure_ascii=False)
        except OSError as e:
            QMessageBox.critical(self, "Enregistrement impossible", str(e))
            return False
        self.current_file = os.path.abspath(path)
        self.saved_state = self.scene.current_state()
        self._on_modified()
        self.statusBar().showMessage(f"Enregistré : {path}", 4000)
        return True

    def fit_selected_to_text(self):
        if not self.scene.fit_selected_to_text():
            self.statusBar().showMessage("Sélectionnez un ou plusieurs blocs", 4000)

    def copy_image(self):
        sel = self.scene.selectedItems() or None
        try:
            copy_scene_image(self.scene, only=sel, scale=self.png_scale,
                             transparent=self.a_png_transp.isChecked())
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Copie impossible", str(e))
            return
        what = "La sélection" if sel else "Le diagramme"
        self.statusBar().showMessage(f"{what} est copié comme image (PNG + SVG + PDF) : "
                                     "collez-le dans votre document", 5000)

    def export(self, selection=False):
        only = None
        if selection:
            only = self.scene.selectedItems()
            if not only:
                QMessageBox.information(self, "Export", "Aucun élément sélectionné.")
                return
        base = os.path.splitext(self.current_file)[0] if self.current_file else os.path.join(self._dir(), "diagramme")
        path, filt = QFileDialog.getSaveFileName(self, "Exporter", base + ".pdf",
                                                 "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext not in ("pdf", "svg", "png"):
            ext = "svg" if "SVG" in filt else "png" if "PNG" in filt else "pdf"
            path += "." + ext
        if ext == "png":
            s, ok = QInputDialog.getDouble(self, "Export PNG",
                                           "Facteur d'échelle (4 ≈ 300 dpi à taille réelle) :",
                                           self.png_scale, 0.5, 20, 1)
            if not ok:
                return
            self.png_scale = s
        try:
            export_scene(self.scene, path, ext, scale=self.png_scale,
                         transparent=self.a_png_transp.isChecked(), only=only)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export impossible", str(e))
            return
        self.statusBar().showMessage(f"Exporté : {path}", 5000)

    def export_pytorch(self):
        """Squelette PyTorch du schéma (deepgraph/codegen.py, voir docs/VERS_PYTORCH.md)."""
        from .codegen import generate
        base = os.path.splitext(self.current_file)[0] if self.current_file else os.path.join(self._dir(), "modele")
        path, _ = QFileDialog.getSaveFileName(self, "Exporter en code PyTorch", base + ".py", "Python (*.py)")
        if not path:
            return
        if not path.lower().endswith(".py"):
            path += ".py"
        defs = {bid: bd.pytorch for bid, bd in CFG.blocks.items() if bd.pytorch}
        title = os.path.splitext(os.path.basename(self.current_file))[0] if self.current_file else ""
        try:
            code, info = generate(self.scene.to_dict(), defs, title)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export PyTorch impossible", str(e))
            return
        if not info["main"]:
            QMessageBox.warning(self, "Code PyTorch", "Aucun réseau trouvé : le schéma doit contenir des blocs "
                                "reliés par des flèches.")
            return
        todo = (f"<p>{info['todo']} point(s) à compléter ou à vérifier, marqué(s) <b>TODO</b> dans le code.</p>"
                if info["todo"] else "<p>Aucun réglage manquant.</p>")
        box = QMessageBox(self)
        box.setWindowTitle("Code PyTorch")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<p><b>{os.path.basename(path)}</b> : {', '.join(info['classes'])}.</p>{todo}"
                    f"<p>Brouillon qui suit le schéma, à relire. Pour l'essayer sur des entrées factices :<br>"
                    f"<code>python {os.path.basename(path)}</code></p>")
        b_open = box.addButton("Ouvrir le fichier", QMessageBox.ActionRole)
        b_help = box.addButton("Aide…", QMessageBox.HelpRole)
        box.addButton("Fermer", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is b_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        elif box.clickedButton() is b_help:
            self.show_vers_pytorch_doc()
        self.statusBar().showMessage(f"Code PyTorch : {path}", 5000)

    def closeEvent(self, e):
        if self.maybe_save():
            e.accept()
        else:
            e.ignore()

    # --------------------------------------------------------- configuration
    def _watch_config(self):
        files = self.watcher.files()
        if files:
            self.watcher.removePaths(files)
        for p in (CFG.path, CFG.user_path):
            if p and os.path.exists(p):
                self.watcher.addPath(p)

    def _set_custom_colors(self):
        seen = []
        for bd in CFG.blocks.values():
            c = bd.color.upper()
            if c not in seen and c != "#FFFFFF":
                seen.append(c)
        for i, c in enumerate(seen[:16]):
            QColorDialog.setCustomColor(i, QColor(c))

    def reload_config(self, verbose=True):
        CFG.load()
        self._watch_config()  # les éditeurs remplacent souvent le fichier
        self.library.rebuild()
        self.scene.refresh_style()
        self._fill_fonts()                 # le « Défaut » dépend de la config
        self._set_custom_colors()
        if CFG.errors:
            if verbose:
                QMessageBox.warning(self, "Configuration", "\n".join(CFG.errors[:40]))
            else:
                self.statusBar().showMessage(f"Config : {CFG.errors[0]}", 8000)
        else:
            self.statusBar().showMessage(f"Configuration rechargée ({len(CFG.blocks)} blocs)", 3000)

    def open_config_file(self):
        self.open_yaml(CFG.path)

    def open_yaml(self, path):
        if not os.path.exists(path):              # mes_blocs.yaml pas encore créé
            try:
                CFG._write_user([])
            except OSError as e:
                QMessageBox.warning(self, "Configuration", str(e))
                return
            self._watch_config()
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def new_library_block(self):
        self.library.new_block()

    def add_selection_to_library(self):
        from .block_dialog import values_from_item
        its = [i for i in self.scene.selectedItems() if isinstance(i, (BlockItem, OperatorItem))]
        if len(its) != 1:
            return
        bid = self.library.new_block(values=values_from_item(its[0]))
        if bid:
            its[0].def_id = bid          # le bloc du schéma est désormais rattaché au nouveau modèle
            self.statusBar().showMessage("Bloc ajouté à la bibliothèque", 4000)

    def _lib_entry(self, menu):
        its = self.scene.selectedItems()
        if len(its) == 1 and isinstance(its[0], (BlockItem, OperatorItem)):
            return self.a_add_to_lib
        return None

    def choose_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Fichier de configuration", os.path.dirname(CFG.path),
                                              "YAML (*.yaml *.yml)")
        if path:
            CFG.path = path
            self.reload_config(True)
