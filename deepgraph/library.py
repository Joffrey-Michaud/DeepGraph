"""Panneau de gauche : bibliothèque de blocs issue du fichier YAML."""
from PyQt5.QtCore import Qt, QMimeData, QSize, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QFont
from PyQt5.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
                             QPushButton, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .config import CFG, description_html
from .items import draw_symbol
from .view import LIB_MIME

TOOLS_CATEGORY = "Outils"
BUILTINS = [
    ("__text__", "Texte libre", "Texte sans cadre (« Inputs », titres, légendes…)."),
    ("__junction__", "Jonction", "Point d'embranchement pour faire partir plusieurs flèches. "
                                 "Invisible à l'export. Astuce : en mode flèche, cliquer sur une flèche "
                                 "existante crée automatiquement une jonction."),
    ("__group__", "Groupe (cadre)", "Cadre gris arrondi avec étiquette (ex. « N× »). Les blocs déposés "
                                    "dedans en font partie. Raccourci : sélectionner puis Ctrl+G."),
]


def make_icon(kind, color="#FFFFFF", symbol="plus"):
    pm = QPixmap(36, 22)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    border = QColor(CFG.style["border_color"])
    p.setPen(QPen(border, 1.2))
    if kind == "circle":
        c, r = QPointF(18, 11), 8.5
        p.setBrush(QColor(color))
        p.drawEllipse(c, r, r)
        f = QFont()
        f.setPixelSize(10)
        draw_symbol(p, c, r, symbol, border, 1.2, border, f)
    elif kind == "text":
        f = QFont()
        f.setPixelSize(14)
        p.setFont(f)
        p.drawText(QRectF(0, 0, 36, 22), int(Qt.AlignCenter), "Aa")
    elif kind == "junction":
        p.setBrush(border)
        p.drawEllipse(QPointF(12, 11), 2.5, 2.5)
        p.drawLine(QPointF(2, 11), QPointF(12, 11))
        p.drawLine(QPointF(12, 11), QPointF(32, 4))
        p.drawLine(QPointF(12, 11), QPointF(32, 18))
    elif kind == "group":
        p.setBrush(QColor(CFG.style["group_fill"]))
        p.drawRoundedRect(QRectF(2, 1, 32, 20), 5, 5)
    else:
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(2, 4, 32, 14), 3, 3)
    p.end()
    return QIcon(pm)


class LibraryTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setIconSize(QSize(36, 22))
        self.setIndentation(12)
        self.setAnimated(True)

    def mimeTypes(self):
        return [LIB_MIME]

    def mimeData(self, items):
        md = QMimeData()
        if items:
            bid = items[0].data(0, Qt.UserRole)
            if bid:
                md.setData(LIB_MIME, bid.encode("utf-8"))
        return md


class LibraryPanel(QWidget):
    activated = pyqtSignal(str)
    reloadRequested = pyqtSignal()
    openFileRequested = pyqtSignal(str)     # ouvrir un fichier YAML dans l'éditeur de texte
    changed = pyqtSignal()                  # mes_blocs.yaml réécrit : recharger la config

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher un bloc…")
        self.search.setClearButtonEnabled(True)
        self.tree = LibraryTree()
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:gray; padding:4px;")
        self.desc.setMinimumHeight(48)
        self.desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        row = QHBoxLayout()
        b_new = QPushButton("＋ Nouveau bloc…")
        b_new.setToolTip("Créer un bloc : nom, couleur, catégorie… Clic droit sur un bloc de la "
                         "bibliothèque pour le modifier.")
        b_more = QToolButton()
        b_more.setText("⋯")
        b_more.setToolTip("Recharger, fichiers de configuration (utilisateurs avancés)")
        b_more.setPopupMode(QToolButton.InstantPopup)
        more = QMenu(b_more)
        more.addAction("Recharger la bibliothèque (F5)", self.reloadRequested.emit)
        more.addSeparator()
        more.addAction("Ouvrir mes_blocs.yaml (mes blocs)",
                       lambda: self.openFileRequested.emit(CFG.user_path))
        more.addAction("Ouvrir blocks.yaml (blocs fournis, style)",
                       lambda: self.openFileRequested.emit(CFG.path))
        b_more.setMenu(more)
        row.addWidget(b_new, 1)
        row.addWidget(b_more)
        top = QHBoxLayout()
        top.setSpacing(2)
        top.addWidget(self.search, 1)
        for txt, tip, slot in (("−", "Replier toutes les catégories", self.collapse_all),
                               ("+", "Déplier toutes les catégories", self.expand_all)):
            b = QToolButton()
            b.setText(txt)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setFixedWidth(24)
            f = b.font()
            f.setBold(True)
            b.setFont(f)
            b.clicked.connect(slot)
            top.addWidget(b)
        v.addLayout(top)
        v.addWidget(self.tree, 1)
        v.addWidget(self.desc)
        v.addLayout(row)
        from . import __version__
        self.version_label = QLabel(f"DeepGraph {__version__}")
        self.version_label.setToolTip("Version installée — Aide > Rechercher des mises à jour")
        self.version_label.setStyleSheet("color: palette(mid);")
        v.addWidget(self.version_label)

        self.search.textChanged.connect(self.apply_filter)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.tree.currentItemChanged.connect(self._on_current)
        b_new.clicked.connect(lambda: self.new_block())
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.rebuild()

    def _category(self, name, cats):
        if name not in cats:
            it = QTreeWidgetItem(self.tree, [name])
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
            it.setFlags(Qt.ItemIsEnabled)
            cats[name] = it
        return cats[name]

    def _leaf(self, parent, bid, name, desc, icon):
        it = QTreeWidgetItem(parent, [name])
        it.setData(0, Qt.UserRole, bid)
        it.setData(0, Qt.UserRole + 1, desc)
        it.setIcon(0, icon)
        it.setToolTip(0, "<div style='max-width:420px'>" + description_html(desc, name) + "</div>"
                      if desc else name)
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
        return it

    def collapse_all(self):
        self.tree.collapseAll()

    def expand_all(self):
        self.tree.expandAll()

    def rebuild(self):
        # garde les catégories repliées d'un rechargement à l'autre
        collapsed = {self.tree.topLevelItem(i).text(0) for i in range(self.tree.topLevelItemCount())
                     if not self.tree.topLevelItem(i).isExpanded()}
        self.tree.clear()
        cats = {}
        for bd in CFG.blocks.values():
            parent = self._category(bd.category, cats)
            it = self._leaf(parent, bd.id, bd.name, bd.description,
                            make_icon(bd.shape, bd.color, bd.symbol))
            if bd.origin != "base":            # ajouté ou modifié par l'utilisateur
                f = it.font(0)
                f.setItalic(bd.origin == "override")
                it.setFont(0, f)
                it.setToolTip(0, it.toolTip(0) + ("<br><i>Bloc modifié (clic droit > Rétablir l'original)</i>"
                                                  if bd.origin == "override" else "<br><i>Mon bloc</i>"))
        tools = self._category(TOOLS_CATEGORY, cats)
        for bid, name, desc in BUILTINS:
            kind = {"__text__": "text", "__junction__": "junction", "__group__": "group"}[bid]
            self._leaf(tools, bid, name, desc, make_icon(kind))
        for name, it in cats.items():
            it.setExpanded(name not in collapsed)
        self.apply_filter(self.search.text())

    def apply_filter(self, text):
        text = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(cat.childCount()):
                ch = cat.child(j)
                ok = not text or text in ch.text(0).lower() or text in (ch.data(0, Qt.UserRole + 1) or "").lower()
                ch.setHidden(not ok)
                any_visible |= ok
            cat.setHidden(not any_visible)
            if text and any_visible:
                cat.setExpanded(True)

    def _on_double(self, item, col):
        bid = item.data(0, Qt.UserRole)
        if bid:
            self.activated.emit(bid)

    def _on_current(self, cur, prev):
        if cur is None or not cur.data(0, Qt.UserRole):
            self.desc.setText("")
            return
        d = cur.data(0, Qt.UserRole + 1) or ""
        self.desc.setText(description_html(d, cur.text(0), summary_only=True))

    # ------------------------------------------------------- blocs utilisateur
    def _context_menu(self, pos):
        it = self.tree.itemAt(pos)
        menu = QMenu(self)
        bid = it.data(0, Qt.UserRole) if it is not None else None
        bd = CFG.blocks.get(bid) if bid else None
        if bd is not None:
            menu.addAction("Modifier…", lambda: self.edit_block(bd.id))
            menu.addAction("Dupliquer…", lambda: self.duplicate_block(bd.id))
            menu.addSeparator()
            if bd.origin == "user":
                menu.addAction("Supprimer de la bibliothèque", lambda: self.delete_block(bd.id))
            elif bd.origin == "override":
                menu.addAction("Rétablir l'original", lambda: self.revert_block(bd.id))
            menu.addSeparator()
            menu.addAction(f"Nouveau bloc dans « {bd.category} »…", lambda: self.new_block(bd.category))
        elif it is not None and not bid and it.text(0) != TOOLS_CATEGORY:     # une catégorie
            cat = it.text(0)
            menu.addAction(f"Nouveau bloc dans « {cat} »…", lambda: self.new_block(cat))
        else:
            menu.addAction("Nouveau bloc…", lambda: self.new_block())
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _run_dialog(self, values, mode="new", origin="user"):
        from .block_dialog import BlockDialog
        dlg = BlockDialog(self.window(), values, mode, origin)
        if dlg.exec_() != dlg.Accepted:
            return None
        raw = dlg.result_raw()
        try:
            CFG.save_user_block(raw, dlg.edit_id)
        except OSError as e:
            QMessageBox.warning(self, "Bibliothèque", f"Impossible d'enregistrer {CFG.user_path} :\n{e}")
            return None
        bid = str(raw.get("id", raw["name"]))
        self.changed.emit()
        self.select(bid)
        return bid

    def new_block(self, category=None, values=None):
        v = dict(values or {})
        if category:
            v["category"] = category
        return self._run_dialog(v)

    def edit_block(self, bid):
        from .block_dialog import values_from_def
        bd = CFG.blocks.get(bid)
        if bd is not None:
            return self._run_dialog(values_from_def(bd), "edit", bd.origin)

    def duplicate_block(self, bid):
        from .block_dialog import values_from_def
        bd = CFG.blocks.get(bid)
        if bd is not None:
            v = values_from_def(bd)
            v.pop("id", None)
            v["name"] = bd.name + " (copie)"
            if "label" not in v:
                v["label"] = bd.label
            return self._run_dialog(v)

    def delete_block(self, bid):
        bd = CFG.blocks.get(bid)
        if bd is None:
            return
        r = QMessageBox.question(self, "Supprimer le bloc",
                                 f"Retirer « {bd.name} » de la bibliothèque ?\n"
                                 "Les blocs déjà posés dans les schémas ne changent pas.")
        if r == QMessageBox.Yes:
            self._remove(bid)

    def revert_block(self, bid):
        bd = CFG.base_blocks.get(bid)
        if bd is None:
            return
        r = QMessageBox.question(self, "Rétablir l'original",
                                 f"Annuler tes modifications de « {bd.name} » ?")
        if r == QMessageBox.Yes:
            self._remove(bid)
            self.select(bid)

    def _remove(self, bid):
        try:
            CFG.remove_user_block(bid)
        except OSError as e:
            QMessageBox.warning(self, "Bibliothèque", f"Impossible d'enregistrer {CFG.user_path} :\n{e}")
            return
        self.changed.emit()

    def select(self, bid):
        """Sélectionne (et montre) le bloc bid dans l'arbre."""
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            for j in range(cat.childCount()):
                ch = cat.child(j)
                if ch.data(0, Qt.UserRole) == bid:
                    cat.setExpanded(True)
                    self.tree.setCurrentItem(ch)
                    self.tree.scrollToItem(ch)
                    return
