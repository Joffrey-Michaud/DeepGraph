"""Panneau « Ordre » : ordre d'affichage (avant / arrière) des éléments du calque sélectionné.

Haut de la liste = premier plan. Glisser-déposer pour réordonner ; un élément reste
dans son groupe (les groupes se déplient pour ordonner leur contenu)."""
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from .items import BlockItem, GroupItem, JunctionItem, NodeItem, OperatorItem, SYMBOLS, TextItem

ROLE = Qt.UserRole
SYM = dict(SYMBOLS)


def item_label(it):
    if isinstance(it, BlockItem):
        t = " ".join(it.text.split()) or "(bloc sans texte)"
        if it.stack:
            t += f"  ×{it.stack + 1}"
        return t
    if isinstance(it, OperatorItem):
        return f"Opérateur : {SYM.get(it.symbol, it.symbol)}"
    if isinstance(it, TextItem):
        return f"Texte « {' '.join(it.text.split())[:40]} »"
    if isinstance(it, JunctionItem):
        return "Jonction"
    if isinstance(it, GroupItem):
        return f"Groupe « {it.label} »" if it.label else "Groupe"
    return getattr(it, "label_fr", "Élément")


def item_icon(it):
    pm = QPixmap(16, 16)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#555555"), 1))
    if isinstance(it, OperatorItem):
        p.setBrush(QColor(it.color))
        p.drawEllipse(2, 2, 12, 12)
    elif isinstance(it, GroupItem):
        p.setBrush(QColor(it.fill))
        p.drawRoundedRect(1, 3, 14, 10, 3, 3)
    elif isinstance(it, BlockItem):
        p.setBrush(QColor(it.color))
        p.drawRoundedRect(1, 4, 14, 8, 2, 2)
    elif isinstance(it, TextItem):
        p.setPen(QColor(it.color))
        f = p.font()
        f.setPixelSize(12)
        f.setBold(True)
        p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignCenter, "T")
    else:
        p.setBrush(QColor("#555555"))
        p.drawEllipse(6, 6, 4, 4)
    p.end()
    return QIcon(pm)


class OrderTree(QTreeWidget):
    reordered = pyqtSignal(bool, list)    # (dépôt valide, [parents à relire ; None = premier niveau])

    def dropEvent(self, e):
        moving = self.selectedItems()
        before = {id(i): i.parent() for i in moving}
        super().dropEvent(e)
        # un élément ne change pas de groupe ici (l'appartenance se règle en le déplaçant sur le schéma)
        ok = all(i.parent() is before[id(i)] for i in moving)
        self.reordered.emit(ok, list({id(p): p for p in before.values()}.values()))


class OrderPanel(QWidget):
    def __init__(self, scene, view=None, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.view = view
        self._syncing = False
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:gray; font-size:11px;")
        v.addWidget(self.info)
        self.tree = OrderTree()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setMouseTracking(True)
        self.tree.setToolTip("Haut de la liste = premier plan.\nGlisser-déposer pour changer l'ordre ;\n"
                             "double-clic pour centrer la vue sur l'élément.")
        v.addWidget(self.tree, 1)
        row = QHBoxLayout()
        for txt, how, tip in (("⤒", "front", "Mettre au premier plan (Ctrl+Maj+Page préc.)"),
                              ("▲", "up", "Avancer d'un cran (Ctrl+Page préc.)"),
                              ("▼", "down", "Reculer d'un cran (Ctrl+Page suiv.)"),
                              ("⤓", "back", "Mettre à l'arrière-plan (Ctrl+Maj+Page suiv.)")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(lambda checked=False, h=how: self.scene.restack_selected(h))
            row.addWidget(b)
        v.addLayout(row)
        note = QLabel("Les flèches restent toujours au-dessus des blocs de leur calque.")
        note.setWordWrap(True)
        note.setStyleSheet("color:gray; font-size:11px;")
        v.addWidget(note)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.rebuild)
        for sig in (scene.modified, scene.itemsEdited, scene.layersChanged, scene.currentLayerChanged):
            sig.connect(self.schedule)
        scene.selectionChanged.connect(self._from_scene_selection)
        self.tree.itemSelectionChanged.connect(self._to_scene_selection)
        self.tree.reordered.connect(self._on_reordered)
        self.tree.itemDoubleClicked.connect(self._center_on)
        self.tree.itemEntered.connect(self._hover)
        self.tree.viewport().installEventFilter(self)
        self._collapsed = set()

    # ------------------------------------------------------------- contenu
    def schedule(self):
        if self.isVisible():
            self._timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self.rebuild()

    def _layer_ids(self):
        return list(self.scene.active_layers())

    def rebuild(self):
        try:
            ids = self._layer_ids()
        except RuntimeError:
            return
        sc = self.scene
        names = {l["id"]: l["name"] for l in sc.layers}
        for i in range(self.tree.topLevelItemCount()):        # mémorise les groupes repliés
            self._remember(self.tree.topLevelItem(i))
        self._syncing = True
        self.tree.clear()
        multi = len(ids) > 1
        if multi:
            self.info.setText("Groupe de calques : un calque à la fois peut être réordonné "
                              "(chaque calque est une section).")
        else:
            self.info.setText(f"Calque « {names.get(ids[0], '?')} » — haut de la liste = premier plan."
                              if ids else "Aucun calque sélectionné.")
        for lid in reversed([x for x in [l["id"] for l in sc.layers] if x in ids]):   # calque du dessus d'abord
            tops = [n for n in sc.nodes() if n.parentItem() is None and sc.layer_of(n) == lid]
            parent = None
            if multi:
                parent = QTreeWidgetItem([names.get(lid, "?")])
                parent.setFlags(Qt.ItemIsEnabled)
                f = parent.font(0)
                f.setBold(True)
                parent.setFont(0, f)
                self.tree.addTopLevelItem(parent)
                parent.setExpanded(True)
            self._add_level(parent, tops)
        self._syncing = False
        self._from_scene_selection()

    def _add_level(self, parent, nodes):
        nodes = sorted(nodes, key=lambda n: (n.order if n.order is not None else -1, n.uid or 0), reverse=True)
        for n in nodes:
            ti = QTreeWidgetItem([item_label(n)])
            ti.setIcon(0, item_icon(n))
            ti.setData(0, ROLE, n.uid)
            fl = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled
            if isinstance(n, GroupItem):
                fl |= Qt.ItemIsDropEnabled
            ti.setFlags(fl)
            if parent is None:
                self.tree.addTopLevelItem(ti)
            else:
                parent.addChild(ti)
            if isinstance(n, GroupItem):
                kids = [c for c in n.childItems() if isinstance(c, NodeItem)]
                self._add_level(ti, kids)
                ti.setExpanded(n.uid not in self._collapsed)

    def _remember(self, ti):
        uid = ti.data(0, ROLE)
        if uid is not None and ti.childCount():
            (self._collapsed.discard if ti.isExpanded() else self._collapsed.add)(uid)
        for i in range(ti.childCount()):
            self._remember(ti.child(i))

    def _node(self, ti):
        uid = ti.data(0, ROLE) if ti is not None else None
        if uid is None:
            return None
        return next((n for n in self.scene.nodes() if n.uid == uid), None)

    # --------------------------------------------------------- réordonner
    def _on_reordered(self, ok, parents):
        if not ok:                            # dépôt refusé : on revient à l'état du schéma
            self.scene.itemsEdited.emit()
            self.schedule()
            return
        for p in parents:
            if p is None or p.data(0, ROLE) is None:      # premier niveau (ou section de calque)
                rows = ([self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
                        if p is None else [p.child(i) for i in range(p.childCount())])
            else:
                rows = [p.child(i) for i in range(p.childCount())]
            nodes = [n for n in (self._node(r) for r in rows) if n is not None]
            if nodes:
                self.scene.set_order(list(reversed(nodes)))     # liste = avant -> arrière
        self.scene.checkpoint()
        self.scene.itemsEdited.emit()

    # ----------------------------------------------------------- sélection
    def _all_rows(self):
        out, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            t = stack.pop()
            out.append(t)
            stack.extend(t.child(i) for i in range(t.childCount()))
        return out

    def _from_scene_selection(self):
        if self._syncing or not self.isVisible():
            return
        try:
            sel = {i.uid for i in self.scene.selectedItems() if isinstance(i, NodeItem)}
        except RuntimeError:
            return
        self._syncing = True
        first = None
        for t in self._all_rows():
            on = t.data(0, ROLE) in sel
            t.setSelected(on)
            if on and first is None:
                first = t
        if first is not None:
            self.tree.scrollToItem(first)
        self._syncing = False

    def _to_scene_selection(self):
        if self._syncing:
            return
        self._syncing = True
        sc = self.scene
        sc.clearSelection()
        for t in self.tree.selectedItems():
            n = self._node(t)
            if n is not None and n.flags() & n.ItemIsSelectable:
                n.setSelected(True)
        self._syncing = False

    def _center_on(self, ti, col=0):
        n = self._node(ti)
        if n is not None and self.view is not None:
            self.view.centerOn(n.sceneBoundingRect().center())

    # ------------------------------------------------- survol = surbrillance
    def _hover(self, ti, col=0):
        n = self._node(ti)
        if hasattr(self.scene, "highlight_items"):
            self.scene.highlight_items([n] if n is not None else [])

    def eventFilter(self, obj, e):
        if e.type() == e.Leave and hasattr(self.scene, "highlight_items"):
            self.scene.highlight_items([])
        return False
