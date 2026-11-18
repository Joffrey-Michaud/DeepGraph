"""Fenêtre principale : menus, barre d'outils, fichiers, export, configuration."""
import json
import os

from PyQt5.QtCore import Qt, QFileSystemWatcher, QSettings, QTimer, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QKeySequence, QPalette
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QColorDialog, QComboBox, QDockWidget,
                             QStackedWidget,
                             QDoubleSpinBox, QFileDialog, QInputDialog, QLabel, QMainWindow,
                             QMenu, QMessageBox, QSpinBox, QStyle)

from .arrow import STYLES
from .items import BlockItem, NodeItem
from .config import CFG
from .export import export_scene
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
        self.view.context_actions = [self.a_edit, self._stack_entry, None, self.a_copy, self.a_dup, self.a_delete, None,
                                     self.a_group, self.a_ungroup, self._order_menu, self._layer_menu, None, self.a_export_sel]

        self.zoom_label = QLabel("100 %")
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))

        self.library.activated.connect(self.add_at_center)
        self.library.reloadRequested.connect(lambda: self.reload_config(verbose=True))
        self.library.editRequested.connect(self.open_config_file)
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
        self.a_quit = self._act("Quitter", self.close, QKeySequence.Quit)

        self.a_undo = self._act("Annuler", sc.undo, QKeySequence.Undo, icon=QStyle.SP_ArrowBack)
        self.a_redo = self._act("Rétablir", sc.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], icon=QStyle.SP_ArrowForward)
        self.a_copy = self._act("Copier", sc.copy_selection, QKeySequence.Copy)
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
        self.a_cfg_edit = self._act("Ouvrir le fichier de configuration", self.open_config_file)
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
        self.a_help_update = self._act("Rechercher des mises à jour…", self.check_updates)
        self.a_stack = self._act("Empiler (stack)…", self.stack_dialog,
                                 tip="Ajoute derrière le bloc des copies sans texte (style « ×h »)")
        self.a_reapply = self._act("Réappliquer les couleurs de la config aux blocs", self.scene.reapply_config)

    def _make_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Fichier")
        for a in (self.a_new, self.a_open):
            m.addAction(a)
        self.templates_menu = m.addMenu("Modèles types")
        self.templates_menu.setToolTipsVisible(True)
        self.templates_menu.aboutToShow.connect(self._fill_templates_menu)
        for a in (self.a_save, self.a_save_as, None, self.a_export,
                  self.a_export_sel, self.a_png_transp, None, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Édition")
        for a in (self.a_undo, self.a_redo, None, self.a_cut, self.a_copy, self.a_paste, self.a_dup,
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
        for a in (self.a_reload, self.a_cfg_edit, self.a_cfg_choose, None, self.a_reapply):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("Ai&de")
        m.addAction(self.a_help_guide)
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
        self.style_combo.setToolTip("Tracé des nouvelles flèches (1 / 2 / 3)")
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

    def show_user_guide(self):
        self._show_doc("_guide", "Guide d'utilisation — DeepGraph", docs.user_guide(), "", "DeepGraph_guide.md")

    def _selected_blocks(self):
        return [i for i in self.scene.selectedItems() if isinstance(i, BlockItem)]

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
        self.scene.load({"nodes": [], "arrows": []})
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
                    out.append(("Mes modèles", os.path.splitext(fn)[0], path, ""))
        return out

    def _fill_templates_menu(self):
        m = self.templates_menu
        m.clear()
        subs = {}
        for cat, title, path, ref in self._templates():
            sub = m if not cat else subs.get(cat) or subs.setdefault(cat, m.addMenu(cat))
            a = sub.addAction(title)
            if ref:
                a.setToolTip(ref)
                a.setStatusTip(ref)
            a.triggered.connect(lambda checked=False, p=path, t=title: self.open_template(p, t))
        if not subs and m.isEmpty():
            m.addAction("(aucun modèle)").setEnabled(False)
        m.addSeparator()
        a = m.addAction("Ouvrir le dossier des modèles")
        a.setToolTip("Tout fichier .dgraph déposé dans ce dossier apparaît dans ce menu")
        a.triggered.connect(lambda: (os.makedirs(self.TEMPLATES_DIR, exist_ok=True),
                                     QDesktopServices.openUrl(QUrl.fromLocalFile(self.TEMPLATES_DIR))))

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
        if os.path.exists(CFG.path):
            self.watcher.addPath(CFG.path)

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
        QDesktopServices.openUrl(QUrl.fromLocalFile(CFG.path))

    def choose_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Fichier de configuration", os.path.dirname(CFG.path),
                                              "YAML (*.yaml *.yml)")
        if path:
            CFG.path = path
            self.reload_config(True)
