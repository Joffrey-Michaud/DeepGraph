"""Flèches ancrées sur les éléments : droites, 1 angle (L) ou 2 angles (Z / U)."""
import math

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPen, QColor, QPainter, QPainterPath, QPainterPathStroker, QPolygonF
from PyQt5.QtWidgets import QGraphicsItem

from .config import CFG
from .items import SEL_COLOR, global_width, ink, is_exporting, skip_paint

STYLES = [("straight", "Droite"), ("ortho1", "1 angle"), ("ortho2", "2 angles")]
FIRSTS = [("auto", "Auto"), ("v", "Vertical d'abord"), ("h", "Horizontal d'abord")]
HEADS = [("end", "À la fin"), ("start", "Au début"), ("both", "Aux deux bouts"), ("none", "Aucune")]


# ----------------------------------------------------------------- géométrie
def _len(a, b):
    return math.hypot(b.x() - a.x(), b.y() - a.y())


def _unit(a, b):
    n = _len(a, b)
    if n < 1e-9:
        return QPointF(0, 0), 0.0
    return QPointF((b.x() - a.x()) / n, (b.y() - a.y()) / n), n


def simplify(pts):
    out = []
    for p in pts:
        if out and _len(out[-1], p) < 0.01:
            continue
        out.append(QPointF(p))
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1], out[i], out[i + 1]
        cross = (b.x() - a.x()) * (c.y() - b.y()) - (b.y() - a.y()) * (c.x() - b.x())
        dot = (b.x() - a.x()) * (c.x() - b.x()) + (b.y() - a.y()) * (c.y() - b.y())
        if abs(cross) <= 1e-3 * max(1.0, _len(a, b) * _len(b, c)) and dot >= 0:
            out.pop(i)
        else:
            i += 1
    return out


def rounded_path(pts, radius):
    path = QPainterPath(pts[0])
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        u1, n1 = _unit(b, a)
        u2, n2 = _unit(b, c)
        r = min(radius, n1 / 2, n2 / 2)
        if r < 0.5:
            path.lineTo(b)
            continue
        path.lineTo(b + u1 * r)
        path.quadTo(b, b + u2 * r)
    path.lineTo(pts[-1])
    return path


def axis_of(a, b):
    return "v" if abs(b.x() - a.x()) < abs(b.y() - a.y()) else "h"


def style_from_polyline(pts):
    """Retrouve (style, premier axe, mid) qui reproduit une polyligne de ≤ 2 coudes."""
    pts = simplify(pts)
    n = len(pts) - 2
    if n <= 0:
        return "straight", "auto", None
    first = axis_of(pts[0], pts[1])
    if n == 1:
        return "ortho1", first, None
    mid = (pts[1].y() - pts[0].y()) if first == "v" else (pts[1].x() - pts[0].x())
    return "ortho2", first, mid


def project_on_segment(p, a, b):
    dx, dy = b.x() - a.x(), b.y() - a.y()
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / l2))
    q = QPointF(a.x() + t * dx, a.y() + t * dy)
    return q, _len(p, q)


def heads_split(head):
    return head in ("start", "both"), head in ("end", "both")


def heads_join(start, end):
    return "both" if start and end else "start" if start else "end" if end else "none"


# ----------------------------------------------------------------- endpoints
class Endpoint:
    """Extrémité de flèche : ancrée sur un élément (fx, fy relatifs) ou point libre."""
    __slots__ = ("node", "fx", "fy", "point")

    def __init__(self, node=None, fx=0.5, fy=0.5, point=None):
        self.node = node
        self.fx, self.fy = float(fx), float(fy)
        self.point = QPointF(point) if point is not None else QPointF(0, 0)

    def pos(self):
        if self.node is not None:
            return self.node.anchor_scene(self.fx, self.fy)
        return QPointF(self.point)

    def direction(self):
        return self.node.anchor_dir(self.fx, self.fy) if self.node is not None else None

    def copy(self):
        return Endpoint(self.node, self.fx, self.fy, self.point)


# --------------------------------------------------------------------- arrow
class ArrowItem(QGraphicsItem):
    kind = "arrow"
    label_fr = "Flèche"
    SAVE = ["style", "first", "mid", "head", "width", "color", "dashed"]
    PROPS = [
        ("style", "Tracé", "choice", STYLES),
        ("first", "Premier segment", "choice", FIRSTS),
        ("head", "Pointe", "choice", HEADS),
        ("width", "Épaisseur", "width_opt", "arrow_width"),
        ("color", "Couleur", "color"),
        ("dashed", "Pointillés", "bool"),
    ]
    MARGIN = 10.0

    def __init__(self, src, dst, style="straight"):
        super().__init__()
        st = CFG.style
        self.uid = None
        self.src, self.dst = src, dst
        self.style = style
        self.first = "auto"
        self.mid = None
        self.head = "end"
        self.width = None          # None = épaisseur globale du diagramme
        self.color = st["arrow_color"]
        self.dashed = False
        self.layer = None
        self._raw, self._pts, self._heads = [], [], []
        self._path = QPainterPath()
        self._drag = None
        self._last = None
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self.attach()
        self.update_path()

    def eff_width(self):
        return float(self.width) if self.width is not None else global_width(self, "arrow_width")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSceneHasChanged and self.scene() is not None:
            self.update_path()
        return super().itemChange(change, value)

    # connexions --------------------------------------------------------------
    def attach(self):
        for ep in (self.src, self.dst):
            if ep.node is not None and self not in ep.node.arrows:
                ep.node.arrows.append(self)

    def detach(self):
        for ep in (self.src, self.dst):
            if ep.node is not None and self in ep.node.arrows:
                ep.node.arrows.remove(self)

    def set_endpoint(self, which, ep):
        self.detach()
        setattr(self, which, ep)
        self.attach()
        self.update_path()

    # routage -------------------------------------------------------------------
    def resolved_first(self):
        if self.first in ("h", "v"):
            return self.first
        d1, d2 = self.src.direction(), self.dst.direction()

        def axis(d):
            return "v" if d in ("up", "down") else "h"

        if self.style == "ortho1":
            if d1:
                return axis(d1)
            if d2:
                return "h" if axis(d2) == "v" else "v"
        else:
            if d1:
                return axis(d1)
            if d2:
                return axis(d2)
        p1, p2 = self.src.pos(), self.dst.pos()
        return "v" if abs(p2.y() - p1.y()) >= abs(p2.x() - p1.x()) else "h"

    def mid_coord(self, p1, p2, first):
        a1, a2 = (p1.y(), p2.y()) if first == "v" else (p1.x(), p2.x())
        if self.mid is not None:
            return a1 + self.mid
        m, clear = (a1 + a2) / 2, 20.0
        pos_dir, neg_dir = ("down", "up") if first == "v" else ("right", "left")
        d1, d2 = self.src.direction(), self.dst.direction()
        if d1 == pos_dir:
            m = max(m, a1 + clear)
        elif d1 == neg_dir:
            m = min(m, a1 - clear)
        if d2 == neg_dir:
            m = min(m, a2 - clear)
        elif d2 == pos_dir:
            m = max(m, a2 + clear)
        return m

    def compute_points(self):
        p1, p2 = self.src.pos(), self.dst.pos()
        if self.style == "straight":
            return [p1, p2]
        first = self.resolved_first()
        if self.style == "ortho1":
            corner = QPointF(p1.x(), p2.y()) if first == "v" else QPointF(p2.x(), p1.y())
            return [p1, corner, p2]
        m = self.mid_coord(p1, p2, first)
        if first == "v":
            return [p1, QPointF(p1.x(), m), QPointF(p2.x(), m), p2]
        return [p1, QPointF(m, p1.y()), QPointF(m, p2.y()), p2]

    @staticmethod
    def _head_poly(tip, u, length, width):
        perp = QPointF(-u.y(), u.x())
        base = tip - u * length
        return QPolygonF([tip, base + perp * (width / 2), base - perp * (width / 2)])

    def update_path(self):
        self.prepareGeometryChange()
        st = CFG.style
        raw = self.compute_points()
        pts = simplify(raw)
        self._raw, self._pts = raw, pts
        base_w = float(st["arrow_width"]) or 1.0
        k = math.sqrt(max(self.eff_width(), 0.1) / base_w)
        hl, hw = float(st["arrow_head_length"]) * k, float(st["arrow_head_width"]) * k
        hs, he = heads_split(self.head)
        line = [QPointF(p) for p in pts]
        heads = []
        if len(pts) >= 2:
            if he:
                u, n = _unit(pts[-2], pts[-1])
                if n > 0:
                    heads.append(self._head_poly(pts[-1], u, hl, hw))
                    line[-1] = pts[-1] - u * min(hl * 0.75, n)
            if hs:
                u, n = _unit(pts[1], pts[0])
                if n > 0:
                    heads.append(self._head_poly(pts[0], u, hl, hw))
                    line[0] = pts[0] - u * min(hl * 0.75, n)
            self._path = rounded_path(line, float(st["arrow_corner_radius"]))
        else:
            self._path = QPainterPath()
        self._heads = heads
        self.update()

    # géométrie Qt ----------------------------------------------------------------
    def path_rect(self):
        return self._path.boundingRect()

    def content_rect(self):
        r = self._path.boundingRect()
        for h in self._heads:
            r = r.united(h.boundingRect())
        m = self.eff_width() / 2
        return r.adjusted(-m, -m, m, m)

    def boundingRect(self):
        r = self.content_rect()
        for p in self._raw:
            r = r.united(QRectF(p.x() - 1, p.y() - 1, 2, 2))
        m = self.MARGIN + self.eff_width()
        return r.adjusted(-m, -m, m, m)

    def handle_points(self):
        if len(self._raw) < 2:
            return {}
        d = {"src": self._raw[0], "dst": self._raw[-1]}
        if self.style == "ortho1" and len(self._raw) == 3:
            d["corner"] = self._raw[1]
        elif self.style == "ortho2" and len(self._raw) == 4:
            a, b = self._raw[1], self._raw[2]
            d["mid"] = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        return d

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(max(8.0, self.eff_width() + 6))
        p = s.createStroke(self._path)
        for h in self._heads:
            p.addPolygon(h)
        if self.isSelected():
            for q in self.handle_points().values():
                p.addEllipse(q, 7, 7)
        return p

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        painter.setRenderHint(QPainter.Antialiasing)
        col = ink(self, self.color)
        sel = self.isSelected() and not is_exporting(self)
        painter.setBrush(Qt.NoBrush)
        if sel:
            painter.setPen(QPen(QColor(30, 120, 230, 70), self.eff_width() + 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(self._path)
        painter.setPen(QPen(col, self.eff_width(), Qt.DashLine if self.dashed else Qt.SolidLine,
                            Qt.FlatCap, Qt.RoundJoin))
        painter.drawPath(self._path)
        if self._heads:
            painter.setPen(QPen(col, 0.5, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
            painter.setBrush(col)
            for h in self._heads:
                painter.drawPolygon(h)
        if sel:
            pen = QPen(SEL_COLOR, 1.2)
            painter.setPen(pen)
            painter.setBrush(QColor("white"))
            for name, q in self.handle_points().items():
                if name in ("mid", "corner"):
                    painter.setBrush(SEL_COLOR)
                    painter.drawRect(QRectF(q.x() - 4.5, q.y() - 4.5, 9, 9))
                    painter.setBrush(QColor("white"))
                else:
                    painter.drawEllipse(q, 4, 4)

    # propriétés ------------------------------------------------------------------
    def set_prop(self, name, value):
        if name in ("style", "first"):
            self.mid = None
        setattr(self, name, value)
        self.update_path()

    def to_dict(self, nodeset=None):
        def ep(e):
            if e.node is not None and (nodeset is None or e.node in nodeset):
                return {"node": e.node.uid, "fx": round(e.fx, 4), "fy": round(e.fy, 4)}
            p = e.pos()
            return {"x": round(p.x(), 2), "y": round(p.y(), 2)}
        d = {"type": "arrow", "uid": self.uid, "src": ep(self.src), "dst": ep(self.dst)}
        for k in self.SAVE:
            d[k] = getattr(self, k)
        return d

    # interaction -----------------------------------------------------------------
    def _handle_at(self, p):
        if not self.isSelected():
            return None
        for name, q in self.handle_points().items():
            if _len(p, q) <= 7:
                return name
        return None

    def hoverMoveEvent(self, e):
        h = self._handle_at(e.scenePos())
        if h == "mid":
            self.setCursor(Qt.SizeAllCursor)
        elif h:
            self.setCursor(Qt.SizeAllCursor if h == "corner" else Qt.CrossCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            h = self._handle_at(e.scenePos())
            if h:
                self._drag = h
                self._press = e.scenePos()
                e.accept()
                return
        super().mousePressEvent(e)
        if e.button() == Qt.LeftButton and self.src.node is None and self.dst.node is None:
            self._drag, self._last = "move", e.scenePos()

    def mouseMoveEvent(self, e):
        sc = self.scene()
        p = e.scenePos()
        snap = sc.snap_enabled and not (e.modifiers() & Qt.ShiftModifier)
        g = float(sc.grid_size)
        if self._drag in ("src", "dst"):
            node = sc.node_at(p)
            if node is not None:
                ep = Endpoint(node, *node.nearest_anchor(p, snap))
            else:
                ep = Endpoint(point=sc.snap_point(p) if snap else p)
            self.set_endpoint(self._drag, ep)
            return
        if self._drag == "corner":
            # 1 angle : le coude va du côté (parmi les deux possibles) le plus proche de la souris
            p1, p2 = self.src.pos(), self.dst.pos()
            cv, ch = QPointF(p1.x(), p2.y()), QPointF(p2.x(), p1.y())
            first = "v" if _len(p, cv) <= _len(p, ch) else "h"
            if first != self.resolved_first() or self.first == "auto":
                self.first = first
                self.update_path()
            return
        if self._drag == "mid":
            self.drag_mid(p, snap, g)
            return
        if self._drag == "move":
            d = p - self._last
            if snap:
                d = QPointF(round(d.x() / g) * g, round(d.y() / g) * g)
            if d.isNull():
                return
            self._last += d
            self.src.point += d
            self.dst.point += d
            self.update_path()
            return
        super().mouseMoveEvent(e)

    def drag_mid(self, p, snap, g):
        """2 angles : le segment du milieu passe par la souris.
        Son orientation suit la position (et, entre les deux extrémités, le sens du geste) :
        tirer horizontalement = segment du milieu vertical, et inversement."""
        p1, p2 = self.src.pos(), self.dst.pos()
        in_x = min(p1.x(), p2.x()) <= p.x() <= max(p1.x(), p2.x())   # possible avec un segment horizontal
        in_y = min(p1.y(), p2.y()) <= p.y() <= max(p1.y(), p2.y())   # possible avec un segment vertical
        cur = self.resolved_first()
        if in_x and not in_y:
            first = "v"
        elif in_y and not in_x:
            first = "h"
        elif in_x and in_y:
            d = p - getattr(self, "_press", p)
            if max(abs(d.x()), abs(d.y())) < 6:
                first = cur
            else:
                first = "h" if abs(d.x()) > abs(d.y()) else "v"
        else:  # en dehors des deux plages : on garde le segment le plus proche
            dx = min(abs(p.x() - p1.x()), abs(p.x() - p2.x()))
            dy = min(abs(p.y() - p1.y()), abs(p.y() - p2.y()))
            first = cur if abs(dx - dy) < 10 else ("h" if dx > dy else "v")
        c = p.y() if first == "v" else p.x()
        if snap:
            c = round(c / g) * g
        self.first = first
        self.mid = c - (p1.y() if first == "v" else p1.x())
        self.update_path()

    def mouseReleaseEvent(self, e):
        was, self._drag = self._drag, None
        if was in ("src", "dst", "mid", "corner"):
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.scene() is not None:
            self.scene().edit_item(self)
