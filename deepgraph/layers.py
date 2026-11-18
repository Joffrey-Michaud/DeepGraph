"""Panneau des calques : arbre (premier plan en haut), groupes, visibilité, verrou, opacité, menu clic droit."""
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QHeaderView, QLabel, QMenu,
                             QSlider, QSpinBox, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from .layer_model import CLIP_MIME

COL_VIS, COL_LOCK, COL_NAME = 0, 1, 2


class LayerTree(QTreeWidget):
    structureChanged = pyqtSignal()
    hovered = pyqtSignal(object)        # id du calque survolé, ou None

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.itemEntered.connect(lambda it, col: self.hovered.emit(it.data(COL_NAME, Qt.UserRole)))

    def dropEvent(self, e):
        super().dropEvent(e)
        QTimer.singleShot(0, self.structureChanged.emit)

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.itemAt(e.pos()) is None:
            self.hovered.emit(None)

    def leaveEvent(self, e):
        self.hovered.emit(None)
        super().leaveEvent(e)


class LayersPanel(QWidget):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._updating = False
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        self.tree = LayerTree()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Vis.", "Verr.", "Calque (haut = premier plan)"])
        hi = self.tree.headerItem()
        hi.setToolTip(COL_VIS, "Afficher / masquer (un calque masqué n'est pas exporté)")
        hi.setToolTip(COL_LOCK, "Verrouiller : plus aucune modification possible")
        hi.setToolTip(COL_NAME, "Seuls les éléments du calque sélectionné sont modifiables.\n"
                                "Survoler un calque met ses éléments en surbrillance.\n"
                                "Glisser-déposer pour réordonner ou ranger dans un groupe ; clic droit : menu.")
        h = self.tree.header()
        h.setSectionResizeMode(COL_VIS, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_LOCK, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setIndentation(14)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.setMinimumHeight(110)
        v.addWidget(self.tree, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Opacité"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.spin = QSpinBox()
        self.spin.setRange(0, 100)
        self.spin.setSuffix(" %")
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        v.addLayout(row)

        btns = QHBoxLayout()
        btns.setSpacing(2)
        self.b_add = self._btn("+", "Nouveau calque", lambda: self.scene.add_layer())
        self.b_grp = self._btn("Grouper", "Grouper les calques sélectionnés (ou créer un groupe vide)",
                               self._group)
        self.b_del = self._btn("−", "Supprimer le(s) calque(s) sélectionné(s) et leur contenu", self._delete)
        self.b_up = self._btn("▲", "Monter (vers le premier plan)", lambda: self._move(+1))
        self.b_down = self._btn("▼", "Descendre (vers l'arrière-plan)", lambda: self._move(-1))
        self.b_out = self._btn("Sortir", "Sortir le(s) calque(s) sélectionné(s) de leur groupe",
                               lambda: self.scene.move_out_of_group(self._selected_ids()))
        self.b_send = self._btn("Sélection →", "Envoyer les éléments sélectionnés sur ce calque",
                                lambda: self.scene.move_selection_to_layer(self.scene.current_layer))
        for b in (self.b_add, self.b_grp, self.b_out, self.b_del, self.b_up, self.b_down):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.b_send)
        v.addLayout(btns)

        self._ck = QTimer(self)             # une seule annulation par glissement d'opacité
        self._ck.setSingleShot(True)
        self._ck.setInterval(400)
        self._ck.timeout.connect(self.scene.checkpoint)

        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.itemSelectionChanged.connect(self._sync_controls)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.structureChanged.connect(self._on_structure)
        self.tree.hovered.connect(self.scene.highlight_layer)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.slider.valueChanged.connect(self._on_opacity)
        self.spin.valueChanged.connect(self._on_opacity)
        scene.layersChanged.connect(self.rebuild)
        scene.currentLayerChanged.connect(self._on_scene_current)
        self.rebuild()

    def _btn(self, text, tip, slot):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    # ------------------------------------------------------------------ vue
    def rebuild(self):
        self._updating = True
        try:
            collapsed = {self._lid(it) for it in self._all_items() if it.childCount() and not it.isExpanded()}
            selected = {self._lid(it) for it in self.tree.selectedItems()}
            if self.scene.current_layer not in selected:     # sélection périmée : on repart du calque courant
                selected = {self.scene.current_layer}
            self.tree.clear()
            cur = self._fill(self.tree.invisibleRootItem(), self.scene.ltree, collapsed)
            if cur is not None:
                self.tree.setCurrentItem(cur)
            for it in self._all_items():              # garde la sélection multiple
                if self._lid(it) in selected:
                    it.setSelected(True)
            self._sync_controls()
        finally:
            self._updating = False

    def _fill(self, parent, nodes, collapsed):
        cur = None
        dis = self.palette().brush(self.palette().Disabled, self.palette().Text)
        for n in reversed(nodes):                  # premier plan en haut
            it = QTreeWidgetItem(parent)
            it.setData(COL_NAME, Qt.UserRole, n["id"])
            group = bool(n.get("group"))
            it.setText(COL_NAME, n["name"])
            flags = (Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
                     | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            if group:
                flags |= Qt.ItemIsDropEnabled
                f = it.font(COL_NAME)
                f.setBold(True)
                it.setFont(COL_NAME, f)
            it.setFlags(flags)
            it.setCheckState(COL_VIS, Qt.Checked if n["visible"] else Qt.Unchecked)
            it.setCheckState(COL_LOCK, Qt.Checked if n["locked"] else Qt.Unchecked)
            vis, locked, op = self.scene.effective(n["id"])
            tip = [f"Opacité {round(n['opacity'] * 100)} %"] if n["opacity"] < 1 else []
            if not vis:
                it.setForeground(COL_NAME, dis)
                tip.append("masqué")
            if locked:
                tip.append("verrouillé")
            if tip:
                it.setToolTip(COL_NAME, " · ".join(tip))
            if n["id"] == self.scene.current_layer:
                cur = it
            if group:
                c = self._fill(it, n["children"], collapsed)
                cur = cur or c
                it.setExpanded(n["id"] not in collapsed)
        return cur

    def _all_items(self):
        out, stack = [], [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            out.append(it)
            stack.extend(it.child(i) for i in range(it.childCount()))
        return out

    def _sync_controls(self):
        sc = self.scene
        n, lst, _ = sc.find_layer(sc.current_layer)
        val = round((n["opacity"] if n else 1.0) * 100)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.b_del.setEnabled(n is not None)
        if n is not None:
            i = lst.index(n)
            self.b_up.setEnabled(i < len(lst) - 1)
            self.b_down.setEnabled(i > 0)
        self.b_out.setEnabled(any(sc.in_group(i) for i in self._selected_ids()))

    @staticmethod
    def _lid(item):
        return item.data(COL_NAME, Qt.UserRole) if item is not None else None

    def _selected_ids(self):
        ids = [self._lid(i) for i in self.tree.selectedItems()]
        cur = self.scene.current_layer
        return ids or ([cur] if cur is not None else [])

    # ------------------------------------------------------------- actions
    def _on_scene_current(self):
        """Le calque courant a changé : on suit, sans reconstruire la liste (garde la sélection)."""
        cid = self.scene.current_layer
        cur = self.tree.currentItem()
        if self._lid(cur) != cid:
            for it in self._all_items():
                if self._lid(it) == cid:
                    self._updating = True
                    self.tree.clearSelection()
                    self.tree.setCurrentItem(it)
                    it.setSelected(True)
                    self._updating = False
                    break
        self._sync_controls()

    def _on_current(self, cur, prev):
        if self._updating or cur is None:
            return
        self.scene.set_current_layer(self._lid(cur))
        self._sync_controls()

    def _on_item_changed(self, item, col):
        if self._updating:
            return
        lid = self._lid(item)
        if col == COL_VIS:
            val = item.checkState(COL_VIS) == Qt.Checked
            QTimer.singleShot(0, lambda: self.scene.set_layer_prop(lid, "visible", val))
        elif col == COL_LOCK:
            val = item.checkState(COL_LOCK) == Qt.Checked
            QTimer.singleShot(0, lambda: self.scene.set_layer_prop(lid, "locked", val))
        elif col == COL_NAME:
            n = self.scene.lnode(lid)
            name = item.text(COL_NAME).strip() or (n["name"] if n else "Calque")
            QTimer.singleShot(0, lambda: self.scene.set_layer_prop(lid, "name", name))

    def _on_structure(self):
        """Après un glisser-déposer : on relit l'arbre du panneau."""
        nodes = {n["id"]: n for n, _, _ in self.scene._walk()}

        def read(parent_item):
            out = []
            for i in reversed(range(parent_item.childCount())):
                it = parent_item.child(i)
                n = nodes.get(self._lid(it))
                if n is None:
                    continue
                c = {k: v for k, v in n.items() if k != "children"}
                if n.get("group"):
                    c["children"] = read(it)
                out.append(c)
            return out

        tree = read(self.tree.invisibleRootItem())
        if sum(1 for _ in self._count(tree)) == len(nodes):     # rien de perdu en route
            self.scene.set_layer_tree(tree)
        else:
            self.rebuild()

    def _count(self, lst):
        for n in lst:
            yield n
            if n.get("group"):
                yield from self._count(n["children"])

    def _on_opacity(self, v):
        if self._updating:
            return
        for w in (self.slider, self.spin):
            if w.value() != v:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self.scene.set_layer_prop(self.scene.current_layer, "opacity", v / 100.0, checkpoint=False)
        self._ck.start()

    def _group(self):
        ids = [self._lid(i) for i in self.tree.selectedItems()]
        if ids:
            self.scene.group_layers(ids)
        else:
            self.scene.add_layer(group=True)

    def _delete(self):
        self.scene.delete_layers(self._selected_ids())

    def _move(self, d):
        self.scene.move_layer(self.scene.current_layer, d)

    # ------------------------------------------------------------ clic droit
    def _menu(self, pos):
        sc = self.scene
        item = self.tree.itemAt(pos)
        if item is not None and not item.isSelected():
            self.tree.setCurrentItem(item)
        ids = self._selected_ids()
        nid = self._lid(item) if item is not None else sc.current_layer
        node = sc.lnode(nid)
        m = QMenu(self)

        def act(text, slot, enabled=True):
            a = m.addAction(text)
            a.setEnabled(enabled)
            a.triggered.connect(lambda checked=False: slot())
            return a

        act("Nouveau calque", lambda: sc.add_layer())
        act("Nouveau groupe", lambda: sc.add_layer(group=True))
        m.addSeparator()
        if node is not None:
            has = bool(sc.items_in(nid))
            clip = QApplication.clipboard().mimeData()
            can_paste = clip is not None and clip.hasFormat(CLIP_MIME)
            act("Dupliquer", lambda: [sc.duplicate_layer(i) for i in ids])
            act("Copier le contenu", lambda: sc.copy_layer_content(nid), has)
            act("Couper le contenu", lambda: sc.cut_layer_content(nid), has)
            act("Coller dans ce calque", lambda: sc.paste_into_layer(nid), can_paste)
            act("Sélectionner le contenu", lambda: sc.select_layer_content(nid), has)
            act("Envoyer la sélection ici", lambda: sc.move_selection_to_layer(nid), bool(sc.selectedItems()))
            m.addSeparator()
            act("Grouper" if len(ids) > 1 else "Mettre dans un groupe", lambda: sc.group_layers(ids))
            if any(sc.in_group(i) for i in ids):
                act("Sortir du groupe", lambda: sc.move_out_of_group(ids))
            if node.get("group"):
                act("Dégrouper", lambda: sc.ungroup_layers(nid))
            else:
                act("Fusionner avec le calque du dessous", lambda: sc.merge_down(nid),
                    sc.layer_index(nid) > 0)
            m.addSeparator()
            act("Afficher" if not node["visible"] else "Masquer",
                lambda: sc.set_layer_prop(nid, "visible", not node["visible"]))
            act("Déverrouiller" if node["locked"] else "Verrouiller",
                lambda: sc.set_layer_prop(nid, "locked", not node["locked"]))
            act("Afficher uniquement celui-ci", lambda: sc.show_only(nid))
            act("Tout afficher", sc.show_all_layers)
            m.addSeparator()
            if item is not None:
                act("Renommer", lambda: self.tree.editItem(item, COL_NAME))
            act("Supprimer" + (" (avec le contenu)" if has or node.get("group") else ""),
                lambda: sc.delete_layers(ids))
        m.exec_(self.tree.viewport().mapToGlobal(pos))
