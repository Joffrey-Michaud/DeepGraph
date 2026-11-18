"""Scène : modes d'édition, flèches, groupes, alignement, copier/coller, annuler, sérialisation."""
import json

from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF, QMimeData, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPainter
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsScene, QInputDialog, QApplication

from .config import CFG
from .items import (NodeItem, BlockItem, OperatorItem, TextItem, JunctionItem, GroupItem,
                    NODE_TYPES, SEL_COLOR, ancestors, descendants)
from .layer_model import CLIP_MIME, LayersMixin
from .arrow import (ArrowItem, Endpoint, style_from_polyline, project_on_segment,
                    heads_split, heads_join, simplify, STYLES)

FORMAT = "deepgraph"


def _depth(item):
    return sum(1 for _ in ancestors(item))


class DiagramScene(LayersMixin, QGraphicsScene):
    modified = pyqtSignal()
    itemsEdited = pyqtSignal()
    modeChanged = pyqtSignal(str)
    settingsChanged = pyqtSignal()
    layersChanged = pyqtSignal()
    currentLayerChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.mode = "select"
        self.arrow_style = "straight"
        self.snap_enabled = True
        self.grid_size = float(CFG.style["grid_size"])
        self.exporting = False
        self.export_only = None
        self.suspend_snap = False
        self.guides_enabled = True
        self._guides = []
        self._guide_cache = None
        # épaisseurs globales du diagramme (enregistrées dans le .dgraph)
        self.border_width = float(CFG.style["border_width"])
        self.arrow_width = float(CFG.style["arrow_width"])
        self.dark = False  # mode sombre (affichage seulement, jamais à l'export)
        self._init_layers()
        self._hl_layer = None      # calque survolé dans le panneau (surbrillance)
        self._uid = 0
        self._temp = None
        self._hover = None
        self._paste_n = 0
        self._undo, self._undo_i = [], -1
        self.checkpoint()

    # ------------------------------------------------------------- utilitaires
    def new_uid(self):
        self._uid += 1
        return self._uid

    def nodes(self):
        return [i for i in self.items() if isinstance(i, NodeItem)]

    def arrows(self):
        return [i for i in self.items() if isinstance(i, ArrowItem) and i is not self._temp]

    def snap_point(self, p):
        g = self.grid_size
        return QPointF(round(p.x() / g) * g, round(p.y() / g) * g)

    def top_selected_nodes(self):
        sel = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        s = set(sel)
        return [i for i in sel if not any(a in s for a in ancestors(i))]

    def set_mode(self, mode):
        if mode == self.mode:
            return
        self._cancel_temp()
        self.mode = mode
        self._set_hover(None)
        self.modeChanged.emit(mode)

    # -------------------------------------------------------------- création
    def add_node(self, item, center=None):
        if item.uid is None:
            item.uid = self.new_uid()
        self.addItem(item)
        if center is not None:
            item.setPos(center - QPointF(item.w / 2, item.h / 2))
        return item

    def create_from_def(self, bid, center):
        if bid == "__text__":
            it = TextItem("Texte")
        elif bid == "__junction__":
            it = JunctionItem()
        elif bid == "__group__":
            it = GroupItem()
        else:
            d = CFG.blocks.get(bid)
            if d is None:
                return None
            if d.shape == "circle":
                it = OperatorItem(d.symbol, d.color, d.width)
            else:
                it = BlockItem(d.label, d.color, d.width, d.height)
                if d.font_size:
                    it.font_size = int(d.font_size)
                if d.text_color:
                    it.text_color = d.text_color
                it.bold, it.stack = d.bold, d.stack
            it.def_id = bid
        self.add_node(it, center)
        self.update_membership([it])
        if isinstance(it, GroupItem):
            self.adopt_contained(it)
        self.clearSelection()
        it.setSelected(True)
        self.checkpoint()
        return it

    # ---------------------------------------------------------------- groupes
    def reparent(self, item, parent):
        old, self.suspend_snap = self.suspend_snap, True
        sp = item.scenePos()
        item.setParentItem(parent)
        item.setPos(parent.mapFromScene(sp) if parent is not None else sp)
        self.apply_item_layer(item)
        self.suspend_snap = old

    def update_membership(self, items):
        """Un élément dont le centre est dans un groupe (plus grand que lui) devient son enfant."""
        groups = [g for g in self.nodes() if isinstance(g, GroupItem)]
        for it in items:
            if not isinstance(it, NodeItem) or it.scene() is not self:
                continue
            fr = it.scene_frame()
            c, area = fr.center(), fr.width() * fr.height()
            best, best_area = None, None
            lay = self.layer_of(it)
            for g in groups:
                if g is it or it.isAncestorOf(g) or self.layer_of(g) != lay or not g.isVisible():
                    continue
                gf = g.scene_frame()
                ga = gf.width() * gf.height()
                if not gf.contains(c) or ga <= area:
                    continue
                if best is None or ga < best_area:
                    best, best_area = g, ga
            if best is not it.parentItem():
                self.reparent(it, best)

    def adopt_contained(self, group):
        gf = group.scene_frame()
        par = group.parentItem()
        for n in self.nodes():
            if n is group or n.parentItem() is not par or n.isAncestorOf(group) \
                    or self.layer_of(n) != self.layer_of(group):
                continue
            if gf.contains(n.scene_frame()):
                self.reparent(n, group)

    def refit_groups(self):
        """Les groupes « ajustés au contenu » reprennent la taille de leurs enfants + marges."""
        groups = [g for g in self.nodes() if isinstance(g, GroupItem) and g.fit]
        groups.sort(key=_depth, reverse=True)       # les plus imbriqués d'abord
        for g in groups:
            g.refit()

    def after_move(self):
        self.update_membership(self.top_selected_nodes())
        self.refit_groups()

    def after_resize(self, item):
        if isinstance(item, GroupItem):
            self.adopt_contained(item)
            item.margins_from_rect()
            self.itemsEdited.emit()
        else:
            self.update_membership([item])
        self.checkpoint()

    # ------------------------------------------------------ ordre d'affichage
    def order_siblings(self, it):
        """Éléments avec lesquels `it` s'empile : mêmes parent et calque, du fond vers l'avant."""
        par = it.parentItem()
        if par is not None:
            sib = [c for c in par.childItems() if isinstance(c, NodeItem)]
        else:
            lay = self.layer_of(it)
            sib = [n for n in self.nodes() if n.parentItem() is None and self.layer_of(n) == lay]
        return sorted(sib, key=lambda n: (n.order if n.order is not None else -1, n.uid or 0))

    def set_order(self, back_to_front):
        """Fixe l'ordre d'un ensemble de voisins (liste du fond vers l'avant)."""
        for r, n in enumerate(back_to_front):
            n.order = r
        for n in back_to_front:
            self.apply_item_layer(self.top_of(n))
        self.update()

    def restack_selected(self, how):
        """how : 'front', 'up', 'down', 'back'."""
        sel = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        done = set()
        for it in sel:
            sib = self.order_siblings(it)
            key = tuple(id(n) for n in sib)
            if key in done:
                continue
            done.add(key)
            moving = [n for n in sib if n in sel]
            rest = [n for n in sib if n not in sel]
            if how == "front":
                new = rest + moving
            elif how == "back":
                new = moving + rest
            else:
                new = list(sib)
                idxs = range(len(new) - 2, -1, -1) if how == "up" else range(1, len(new))
                for i in idxs:
                    j = i + 1 if how == "up" else i - 1
                    if new[i] in sel and new[j] not in sel:
                        new[i], new[j] = new[j], new[i]
            if new != sib:
                self.set_order(new)
        self.checkpoint()
        self.itemsEdited.emit()

    def group_selected(self):
        top = self.top_selected_nodes()
        if not top:
            return None
        parents = {i.parentItem() for i in top}
        parent = parents.pop() if len(parents) == 1 else None
        rect = QRectF(top[0].scene_frame())
        for i in top[1:]:
            rect = rect.united(i.scene_frame())
        pad = float(CFG.style["group_padding"])
        rect.adjust(-pad, -pad, pad, pad)
        g = GroupItem(rect.width(), rect.height())
        g.uid = self.new_uid()
        lays = {self.layer_of(i) for i in top}
        g.layer = lays.pop() if len(lays) == 1 else self.target_layer()
        old, self.suspend_snap = self.suspend_snap, True
        if parent is not None:
            g.setParentItem(parent)
            g.setPos(parent.mapFromScene(rect.topLeft()))
            g.setZValue(-1)
        else:
            self.addItem(g)
            g.setPos(rect.topLeft())
        self.suspend_snap = old
        for i in top:
            self.reparent(i, g)
        self.clearSelection()
        g.setSelected(True)
        self.checkpoint()
        return g

    def ungroup_selected(self):
        groups = [i for i in self.selectedItems() if isinstance(i, GroupItem)]
        if not groups:
            return
        for g in groups:
            par = g.parentItem()
            for c in [c for c in g.childItems() if isinstance(c, NodeItem)]:
                self.reparent(c, par)
                c.setSelected(True)
            self.delete_items([g])
        self.checkpoint()

    # ----------------------------------------------------------- suppression
    def delete_items(self, items):
        nodes, arrows = set(), set()
        for it in items:
            if isinstance(it, NodeItem):
                nodes.add(it)
                nodes.update(descendants(it))
            elif isinstance(it, ArrowItem):
                arrows.add(it)
        for n in nodes:
            arrows.update(n.arrows)
        for a in arrows:
            a.detach()
            if a.scene() is self:
                self.removeItem(a)
        for n in nodes:
            if n.parentItem() not in nodes and n.scene() is self:
                self.removeItem(n)

    def delete_selected(self):
        sel = self.selectedItems()
        if sel:
            self.delete_items(sel)
            self.checkpoint()

    # ------------------------------------------------------------ alignement
    def align(self, how):
        nodes = self.top_selected_nodes()
        if len(nodes) < 2:
            return
        frs = [n.scene_frame() for n in nodes]
        g = self.grid_size

        def snap(v):
            return round(v / g) * g if self.snap_enabled else v

        deltas = []
        if how == "left":
            t = min(f.left() for f in frs)
            deltas = [QPointF(t - f.left(), 0) for f in frs]
        elif how == "right":
            t = max(f.right() for f in frs)
            deltas = [QPointF(t - f.right(), 0) for f in frs]
        elif how == "top":
            t = min(f.top() for f in frs)
            deltas = [QPointF(0, t - f.top()) for f in frs]
        elif how == "bottom":
            t = max(f.bottom() for f in frs)
            deltas = [QPointF(0, t - f.bottom()) for f in frs]
        elif how == "hcenter":   # même axe vertical
            u = frs[0]
            for f in frs[1:]:
                u = u.united(f)
            t = snap(u.center().x())
            deltas = [QPointF(t - f.center().x(), 0) for f in frs]
        elif how == "vcenter":   # même axe horizontal
            u = frs[0]
            for f in frs[1:]:
                u = u.united(f)
            t = snap(u.center().y())
            deltas = [QPointF(0, t - f.center().y()) for f in frs]
        elif how in ("hdist", "vdist"):
            if len(nodes) < 3:
                return
            horiz = how == "hdist"
            order = sorted(range(len(nodes)), key=lambda i: frs[i].center().x() if horiz else frs[i].center().y())
            first, last = frs[order[0]], frs[order[-1]]
            span = (last.right() - first.left()) if horiz else (last.bottom() - first.top())
            total = sum((frs[i].width() if horiz else frs[i].height()) for i in order)
            gap = (span - total) / (len(nodes) - 1)
            deltas = [QPointF(0, 0)] * len(nodes)
            cur = first.left() if horiz else first.top()
            for i in order:
                f = frs[i]
                deltas[i] = QPointF(cur - f.left(), 0) if horiz else QPointF(0, cur - f.top())
                cur += (f.width() if horiz else f.height()) + gap
        old, self.suspend_snap = self.suspend_snap, True
        for n, d in zip(nodes, deltas):
            n.setPos(n.pos() + d)
        self.suspend_snap = old
        self.checkpoint()

    def same_size(self, dim):
        nodes = [n for n in self.top_selected_nodes() if n.resizable and not n.keep_square]
        if len(nodes) < 2:
            return
        if dim == "w":
            t = max(n.w for n in nodes)
            for n in nodes:
                n.set_size(t, n.h)
        else:
            t = max(n.h for n in nodes)
            for n in nodes:
                n.set_size(n.w, t)
        self.checkpoint()
        self.itemsEdited.emit()

    # ---------------------------------------------------------------- édition
    def edit_item(self, item):
        view = self.views()[0] if self.views() else None
        ok = False
        if isinstance(item, (BlockItem, TextItem)):
            text, ok = QInputDialog.getMultiLineText(view, "Modifier le texte", "Texte :", item.text)
            if ok:
                item.set_prop("text", text)
        elif isinstance(item, GroupItem):
            text, ok = QInputDialog.getText(view, "Étiquette du groupe", "Étiquette :", text=item.label)
            if ok:
                item.set_prop("label", text)
        elif isinstance(item, OperatorItem):
            text, ok = QInputDialog.getText(
                view, "Symbole", "plus, times, dot, minus — ou n'importe quel texte court :", text=item.symbol)
            if ok:
                item.set_prop("symbol", text.strip() or "plus")
        elif isinstance(item, ArrowItem):
            keys = [k for k, _ in STYLES]
            item.set_prop("style", keys[(keys.index(item.style) + 1) % len(keys)])
            ok = True
        if ok:
            self.checkpoint()
            self.itemsEdited.emit()

    def edit_selected(self):
        sel = self.selectedItems()
        if len(sel) == 1:
            self.edit_item(sel[0])

    def set_style_of_selected_arrows(self, style):
        arrows = [a for a in self.selectedItems() if isinstance(a, ArrowItem)]
        for a in arrows:
            a.set_prop("style", style)
        if arrows:
            self.checkpoint()
            self.itemsEdited.emit()

    def reapply_config(self):
        for n in self.nodes():
            d = CFG.blocks.get(n.def_id) if n.def_id else None
            if d is None:
                continue
            if isinstance(n, (BlockItem, OperatorItem)):
                n.set_prop("color", d.color)
            if isinstance(n, BlockItem) and d.text_color:
                n.set_prop("text_color", d.text_color)
        self.checkpoint()
        self.itemsEdited.emit()

    def refresh_style(self):
        """À appeler après rechargement de la config (polices, épaisseurs…)."""
        self.grid_size = float(CFG.style["grid_size"])
        for n in self.nodes():
            n.prepareGeometryChange()
            if isinstance(n, TextItem):
                n.on_props_changed()
            n.update_arrows()
        for a in self.arrows():
            a.update_path()
        self.update()

    # ------------------------------------------------------- réglages globaux
    def settings(self):
        d = {"border_width": self.border_width, "arrow_width": self.arrow_width}
        if CFG.family_override:
            d["font"] = CFG.family_override
        return d

    def _refresh_widths(self):
        for n in self.nodes():
            n.prepareGeometryChange()
            n.update()
        for a in self.arrows():
            a.update_path()
        self.update()

    def set_font_family(self, name):
        """Police de tout le diagramme (None = police de la configuration)."""
        name = name or None
        if name == CFG.family_override:
            return
        CFG.set_family_override(name)
        self.refresh_style()
        self.checkpoint()
        self.settingsChanged.emit()

    def set_global(self, key, value):
        if key not in ("border_width", "arrow_width") or getattr(self, key) == float(value):
            return
        setattr(self, key, float(value))
        self._refresh_widths()
        self.checkpoint()
        self.settingsChanged.emit()

    def _apply_settings(self, data):
        st = data.get("settings") or {}
        self.border_width = float(st.get("border_width", CFG.style["border_width"]))
        self.arrow_width = float(st.get("arrow_width", CFG.style["arrow_width"]))
        if (st.get("font") or None) != CFG.family_override:
            CFG.set_family_override(st.get("font") or None)
            self.refresh_style()
        self.settingsChanged.emit()

    # ----------------------------------------------------- guides d'alignement
    @staticmethod
    def _refs(frame, is_point):
        c = frame.center()
        if is_point:
            return [c.x()], [c.y()]
        return [frame.left(), c.x(), frame.right()], [frame.top(), c.y(), frame.bottom()]

    def _guide_targets(self, item):
        if self._guide_cache is not None and self._guide_cache[0] is item:
            return self._guide_cache[1]
        moving = set(self.top_selected_nodes()) | {item}
        excl = set(moving)
        for m in moving:
            excl.update(descendants(m))
        targets = []
        for n in self.nodes():
            if n in excl or not n.isVisible():
                continue
            f = n.scene_frame()
            xs, ys = self._refs(f, isinstance(n, JunctionItem))
            targets.append((f, xs, ys))
        self._guide_cache = (item, targets)
        return targets

    def _guide_tol(self):
        v = self.views()
        return 6.0 / (v[0].transform().m11() if v else 1.0)

    def guide_snap(self, item, x, y):
        """Position (x, y) du coin haut-gauche corrigée pour s'aligner sur les autres éléments."""
        tol = self._guide_tol()
        pt = isinstance(item, JunctionItem)
        offx = [item.w / 2] if pt else [0.0, item.w / 2, item.w]
        offy = [item.h / 2] if pt else [0.0, item.h / 2, item.h]
        bx = by = None
        for f, xs, ys in self._guide_targets(item):
            for t in xs:
                for o in offx:
                    d = t - (x + o)
                    if abs(d) <= tol and (bx is None or abs(d) < abs(bx)):
                        bx = d
            for t in ys:
                for o in offy:
                    d = t - (y + o)
                    if abs(d) <= tol and (by is None or abs(d) < abs(by)):
                        by = d
        return (x + bx if bx is not None else None), (y + by if by is not None else None)

    def update_guides(self, item, x, y):
        pt = isinstance(item, JunctionItem)
        mf = QRectF(x, y, item.w, item.h)
        mxs, mys = self._refs(mf, pt)
        vert, horiz = {}, {}
        for f, xs, ys in self._guide_targets(item):
            for t in xs:
                if any(abs(t - m) < 0.5 for m in mxs):
                    lo, hi = min(f.top(), mf.top()), max(f.bottom(), mf.bottom())
                    k = round(t, 1)
                    a, b = vert.get(k, (lo, hi))
                    vert[k] = (min(a, lo), max(b, hi))
            for t in ys:
                if any(abs(t - m) < 0.5 for m in mys):
                    lo, hi = min(f.left(), mf.left()), max(f.right(), mf.right())
                    k = round(t, 1)
                    a, b = horiz.get(k, (lo, hi))
                    horiz[k] = (min(a, lo), max(b, hi))
        self._guides = ([QLineF(k, a - 8, k, b + 8) for k, (a, b) in vert.items()]
                        + [QLineF(a - 8, k, b + 8, k) for k, (a, b) in horiz.items()])
        self.update()

    def clear_guides(self):
        self._guide_cache = None
        if self._guides:
            self._guides = []
            self.update()

    # ------------------------------------------------- recherche sous le curseur
    def node_at(self, pos):
        r = QRectF(pos.x() - 8, pos.y() - 8, 16, 16)
        for it in self.items(r):
            if isinstance(it, JunctionItem) and it.isVisible() and (it.scene_frame().center() - pos).manhattanLength() <= 12:
                return it
        for it in self.items(pos):
            if isinstance(it, NodeItem) and it.isVisible() and it.accepts_connection_at(pos):
                return it
        return None

    def arrow_at(self, pos):
        for it in self.items(pos):
            if isinstance(it, ArrowItem) and it is not self._temp and it.isVisible() \
                    and it.flags() & QGraphicsItem.ItemIsSelectable:
                return it
        return None

    @staticmethod
    def _project_on_arrow(a, pos):
        pts = a._pts
        best = None
        for k in range(len(pts) - 1):
            q, d = project_on_segment(pos, pts[k], pts[k + 1])
            if best is None or d < best[2]:
                best = (k, q, d)
        return best

    def split_arrow(self, a, pos, snap=True):
        """Insère une jonction sur la flèche `a` au point le plus proche de `pos`."""
        best = self._project_on_arrow(a, pos)
        if best is None:
            return None
        k, q, _ = best
        pts = a._pts
        p0, p1 = pts[k], pts[k + 1]
        if snap and self.snap_enabled:
            g = self.grid_size
            if abs(p0.x() - p1.x()) < 1e-6:
                y = min(max(round(q.y() / g) * g, min(p0.y(), p1.y())), max(p0.y(), p1.y()))
                q = QPointF(p0.x(), y)
            elif abs(p0.y() - p1.y()) < 1e-6:
                x = min(max(round(q.x() / g) * g, min(p0.x(), p1.x())), max(p0.x(), p1.x()))
                q = QPointF(x, p0.y())
        j = JunctionItem()
        old, self.suspend_snap = self.suspend_snap, True
        self.add_node(j, q)
        self.suspend_snap = old
        pts1 = simplify(pts[:k + 1] + [q])
        pts2 = simplify([q] + pts[k + 1:])
        hs, he = heads_split(a.head)
        b = ArrowItem(Endpoint(j, 0.5, 0.5), a.dst.copy())
        b.uid = self.new_uid()
        b.style, b.first, b.mid = style_from_polyline(pts2)
        b.head, b.width, b.color, b.dashed = heads_join(False, he), a.width, a.color, a.dashed
        self.addItem(b)
        b.update_path()
        a.style, a.first, a.mid = style_from_polyline(pts1)
        a.head = heads_join(hs, False)
        a.set_endpoint("dst", Endpoint(j, 0.5, 0.5))
        self.update_membership([j])
        return j

    def _endpoint_at(self, pos, snap, allow_split):
        node = self.node_at(pos)
        if node is not None:
            return Endpoint(node, *node.nearest_anchor(pos, snap))
        if allow_split:
            a = self.arrow_at(pos)
            if a is not None:
                j = self.split_arrow(a, pos, snap)
                if j is not None:
                    return Endpoint(j, 0.5, 0.5)
        return Endpoint(point=self.snap_point(pos) if (snap and self.snap_enabled) else QPointF(pos))

    # ------------------------------------------------------------ souris / clavier
    def _set_hover(self, p):
        old, self._hover = self._hover, p
        for q in (old, p):
            if q is not None:
                self.update(QRectF(q.x() - 8, q.y() - 8, 16, 16))

    def _cancel_temp(self):
        if self._temp is not None:
            self._temp.detach()
            if self._temp.scene() is self:
                self.removeItem(self._temp)
            self._temp = None

    def mousePressEvent(self, e):
        if self.mode == "arrow":
            if e.button() == Qt.LeftButton:
                snap = not (e.modifiers() & Qt.ShiftModifier)
                src = self._endpoint_at(e.scenePos(), snap, allow_split=True)
                self._temp = ArrowItem(src, Endpoint(point=e.scenePos()), self.arrow_style)
                self.addItem(self._temp)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.mode == "arrow":
            pos = e.scenePos()
            snap = not (e.modifiers() & Qt.ShiftModifier)
            node = self.node_at(pos)
            if self._temp is not None:
                if node is not None:
                    ep = Endpoint(node, *node.nearest_anchor(pos, snap))
                else:
                    ep = Endpoint(point=self.snap_point(pos) if (snap and self.snap_enabled) else pos)
                self._temp.set_endpoint("dst", ep)
            hover = None
            if node is not None:
                hover = node.anchor_scene(*node.nearest_anchor(pos, snap))
            else:
                a = self.arrow_at(pos)
                if a is not None:
                    best = self._project_on_arrow(a, pos)
                    hover = best[1] if best else None
            self._set_hover(hover)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.mode == "arrow":
            if e.button() == Qt.LeftButton and self._temp is not None:
                a = self._temp
                snap = not (e.modifiers() & Qt.ShiftModifier)
                end = self._endpoint_at(e.scenePos(), snap, allow_split=True)
                self._temp = None
                a.set_endpoint("dst", end)
                p1, p2 = a.src.pos(), a.dst.pos()
                if (p1 - p2).manhattanLength() < 3:
                    a.detach()
                    self.removeItem(a)
                else:
                    a.uid = self.new_uid()
                    if isinstance(a.dst.node, JunctionItem) and a.head == "end":
                        a.head = "none"
                        a.update_path()
                    self.clearSelection()
                    a.setSelected(True)
                self.checkpoint()
            e.accept()
            return
        super().mouseReleaseEvent(e)
        self.clear_guides()

    def mouseDoubleClickEvent(self, e):
        if self.mode == "arrow":
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            if self.mode == "arrow":
                self._cancel_temp()
                self.set_mode("select")
            else:
                self.clearSelection()
            e.accept()
            return
        moves = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
        if k in moves:
            nodes = self.top_selected_nodes()
            if nodes:
                step = 1.0 if e.modifiers() & Qt.ShiftModifier else self.grid_size
                dx, dy = moves[k]
                old, self.suspend_snap = self.suspend_snap, True
                for n in nodes:
                    n.setPos(n.pos() + QPointF(dx * step, dy * step))
                self.suspend_snap = old
                self.checkpoint()
                e.accept()
                return
        super().keyPressEvent(e)

    def highlight_layer(self, nid):
        if nid != self._hl_layer:
            self._hl_layer = nid
            self.update()

    def highlight_items(self, items):
        """Surbrillance d'éléments précis (survol dans le panneau Ordre)."""
        items = [i for i in items if i is not None]
        if items != getattr(self, "_hl_items", []):
            self._hl_items = items
            self.update()

    def drawForeground(self, painter, rect):
        if self.exporting:
            return
        if self._hl_layer is not None:
            self._draw_layer_highlight(painter)
        if getattr(self, "_hl_items", None):
            self._draw_layer_highlight(painter, self._hl_items)
        if self._guides:
            pen = QPen(QColor(230, 40, 120), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for ln in self._guides:
                painter.drawLine(ln)
        if self.mode != "arrow" or self._hover is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(SEL_COLOR, 1.5))
        painter.setBrush(QColor(30, 120, 230, 90))
        painter.drawEllipse(self._hover, 4.5, 4.5)

    def _draw_layer_highlight(self, painter, items=None):
        """Surbrillance des éléments du calque survolé dans le panneau Calques (ou de `items`)."""
        if items is None:
            items = self.items_in(self._hl_layer)
        try:
            items = [i for i in items if i.scene() is self and i.isVisible()]
        except RuntimeError:
            return
        if not items:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        col = QColor(255, 140, 0)
        fill = QColor(255, 140, 0, 45)
        pen = QPen(col, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(fill)
        for it in items:
            if isinstance(it, ArrowItem):
                p = QPen(QColor(255, 140, 0, 150), it.eff_width() + 6)
                painter.setPen(p)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(it.mapToScene(it._path))
                painter.setPen(pen)
                painter.setBrush(fill)
            else:
                r = it.mapRectToScene(it.content_rect()).adjusted(-3, -3, 3, 3)
                painter.drawRoundedRect(r, 4, 4)
        painter.restore()

    # --------------------------------------------------------- sérialisation
    def to_dict(self, subset=None, external=False):
        if subset is None:
            nodes, arrows = self.nodes(), self.arrows()
        else:
            ns = set()
            for it in subset:
                if isinstance(it, NodeItem):
                    ns.add(it)
                    ns.update(descendants(it))
            arrows = [a for a in self.arrows()
                      if a in subset or (a.src.node in ns and a.dst.node in ns)]
            nodes = list(ns)
        nodeset = set(nodes)
        nodes = sorted(nodes, key=lambda n: (_depth(n), n.uid or 0))
        out = []
        for n in nodes:
            d = n.to_dict()
            d["order"] = n.order
            par = n.parentItem()
            if par is not None and par in nodeset:
                d["parent"] = par.uid
            else:
                d["parent"] = None
                sp = n.scenePos()
                d["x"], d["y"] = round(sp.x(), 2), round(sp.y(), 2)
                d["layer"] = self.layer_of(n)
            out.append(d)
        arr = []
        for a in sorted(arrows, key=lambda a: a.uid or 0):
            ad = a.to_dict(None if external else nodeset)
            ad["layer"] = a.layer
            arr.append(ad)
        d = {"format": FORMAT, "version": 1, "nodes": out, "arrows": arr}
        if subset is None:
            d["settings"] = self.settings()
            d["layers"] = self.layers_data()
        return d

    def _ep_from(self, d, mapping, offset, ext=None):
        if "node" in d:
            n = mapping.get(d["node"]) or (ext or {}).get(d["node"])
            if n is None:
                return None
            return Endpoint(n, d.get("fx", 0.5), d.get("fy", 0.5))
        return Endpoint(point=QPointF(d["x"], d["y"]) + offset)

    def from_dict(self, data, offset=None, new_ids=False, external=False):
        """external : les flèches peuvent viser des éléments déjà présents (par uid)."""
        offset = offset or QPointF(0, 0)
        ext = {n.uid: n for n in self.nodes()} if external else None
        old, self.suspend_snap = self.suspend_snap, True
        mapping, created = {}, []
        if new_ids:
            nds = data.get("nodes", [])
            order_base = self._next_order()
            ranks = {id(d): r for r, d in enumerate(sorted(
                nds, key=lambda d: (d.get("order") if d.get("order") is not None else 1e9, d.get("uid") or 0)))}
        for d in data.get("nodes", []):
            cls = NODE_TYPES.get(d.get("type"))
            if cls is None:
                continue
            it = cls.from_dict(d)
            it.layer = d.get("layer")
            if not new_ids and d.get("order") is not None:
                it.order = int(d["order"])
            it.uid = self.new_uid() if new_ids else int(d["uid"])
            if new_ids:                       # copie : au-dessus de l'existant, ordre relatif conservé
                it.order = order_base + ranks[id(d)]
            mapping[d["uid"]] = it
            parent = mapping.get(d.get("parent"))
            if parent is not None:
                it.setParentItem(parent)
                it.setPos(d["x"], d["y"])
                self.apply_item_layer(it)
            else:
                self.addItem(it)
                it.setPos(d["x"] + offset.x(), d["y"] + offset.y())
            created.append(it)
        for d in data.get("arrows", []):
            src = self._ep_from(d["src"], mapping, offset, ext)
            dst = self._ep_from(d["dst"], mapping, offset, ext)
            if src is None or dst is None:
                continue
            a = ArrowItem(src, dst, d.get("style", "straight"))
            for k in ArrowItem.SAVE:
                if k in d:
                    setattr(a, k, d[k])
            a.layer = d.get("layer")
            a.uid = self.new_uid() if new_ids else int(d["uid"])
            self.addItem(a)
            a.update_path()
            created.append(a)
        if not new_ids and created:
            self._uid = max([self._uid] + [c.uid for c in created])
        self.suspend_snap = old
        for c in created:
            if c.parentItem() is None:
                self.apply_item_layer(c)
        return created

    # ------------------------------------------------------- annuler / rétablir
    def state(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    def current_state(self):
        return self._undo[self._undo_i]

    def checkpoint(self):
        if self._temp is not None:
            return
        if not (QApplication.mouseButtons() & Qt.LeftButton and self.mouseGrabberItem() is not None):
            self.refit_groups()
        s = self.state()
        if self._undo_i >= 0 and self._undo[self._undo_i] == s:
            return
        del self._undo[self._undo_i + 1:]
        self._undo.append(s)
        if len(self._undo) > 300:
            del self._undo[0]
        self._undo_i = len(self._undo) - 1
        self.modified.emit()

    def can_undo(self):
        return self._undo_i > 0

    def can_redo(self):
        return self._undo_i < len(self._undo) - 1

    def undo(self):
        self.checkpoint()
        if self.can_undo():
            self._undo_i -= 1
            self._restore(self._undo[self._undo_i])

    def redo(self):
        if self.can_redo():
            self._undo_i += 1
            self._restore(self._undo[self._undo_i])

    def _restore(self, s):
        self.clearSelection()
        self._temp = None
        self._hover = None
        self.clear()
        self._guides, self._guide_cache = [], None
        data = json.loads(s)
        self._apply_settings(data)
        self._apply_layers_data(data)
        self._loading = True
        try:
            self.from_dict(data)
        finally:
            self._loading = False
        self.layersChanged.emit()
        self.modified.emit()

    def load(self, data):
        from .simple_format import expand      # format simplifié (écrit à la main / par un LLM)
        data = expand(data)
        self.clearSelection()
        self._temp = None
        self.clear()
        self._guides, self._guide_cache = [], None
        self._uid = 0
        self._apply_settings(data)
        self.current_layer = None
        self._apply_layers_data(data)
        self._loading = True
        try:
            self.from_dict(data)
        finally:
            self._loading = False
        self.layersChanged.emit()
        self._undo, self._undo_i = [], -1
        self.checkpoint()

    # ------------------------------------------------------- copier / coller
    def copy_selection(self):
        sel = self.selectedItems()
        if not sel:
            return False
        md = QMimeData()
        md.setData(CLIP_MIME, json.dumps(self.to_dict(sel)).encode("utf-8"))
        QApplication.clipboard().setMimeData(md)
        self._paste_n = 0
        return True

    def cut_selection(self):
        if self.copy_selection():
            self.delete_selected()

    def _insert(self, data, offset):
        tgt = self.target_layer()
        for d in data.get("nodes", []) + data.get("arrows", []):
            d["layer"] = tgt          # on colle toujours dans le calque courant
        created = self.from_dict(data, offset, new_ids=True)
        self.clearSelection()
        for c in created:
            if isinstance(c, ArrowItem) or c.parentItem() is None or c.parentItem() not in created:
                c.setSelected(True)
        tops = [c for c in created if isinstance(c, NodeItem) and c.parentItem() is None]
        self.update_membership(tops)
        self.checkpoint()

    def paste(self):
        md = QApplication.clipboard().mimeData()
        if md is None or not md.hasFormat(CLIP_MIME):
            return
        data = json.loads(bytes(md.data(CLIP_MIME)).decode("utf-8"))
        self._paste_n += 1
        self._insert(data, QPointF(20, 20) * self._paste_n)

    def duplicate(self):
        sel = self.selectedItems()
        if sel:
            self._insert(self.to_dict(sel), QPointF(20, 20))
