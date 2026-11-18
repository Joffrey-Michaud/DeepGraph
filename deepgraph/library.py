"""Panneau de gauche : bibliothèque de blocs issue du fichier YAML."""
from PyQt5.QtCore import Qt, QMimeData, QSize, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QFont
from PyQt5.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QPushButton, QToolButton,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

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
    editRequested = pyqtSignal()

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
        b_reload = QPushButton("Recharger")
        b_reload.setToolTip("Relire le fichier de configuration (F5)")
        b_edit = QPushButton("Éditer la config")
        b_edit.setToolTip("Ouvrir blocks.yaml dans l'éditeur par défaut")
        row.addWidget(b_reload)
        row.addWidget(b_edit)
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
        b_reload.clicked.connect(self.reloadRequested)
        b_edit.clicked.connect(self.editRequested)
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
            self._leaf(parent, bd.id, bd.name, bd.description,
                       make_icon(bd.shape, bd.color, bd.symbol))
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
