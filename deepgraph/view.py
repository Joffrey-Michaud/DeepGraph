"""Vue : zoom, déplacement, grille, glisser-déposer depuis la bibliothèque."""
import math

from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF, QTransform
from PyQt5.QtWidgets import QGraphicsView, QMenu

LIB_MIME = "application/x-deepgraph-block"


class DiagramView(QGraphicsView):
    zoomChanged = pyqtSignal(float)
    SUBGRID_ZOOM = 1.5     # la sous-grille apparaît à partir de 150 %

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ContainsItemShape)
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)
        self.show_grid = True
        self.context_actions = []
        self._pan = None
        scene.modeChanged.connect(self._on_mode)
        self.centerOn(0, 0)

    def _on_mode(self, mode):
        if mode == "arrow":
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.viewport().unsetCursor()

    # ------------------------------------------------------------------ grille
    def drawBackground(self, painter, rect):
        dark = getattr(self.scene(), "dark", False)
        painter.fillRect(rect, QColor("#1E1F22" if dark else "#FFFFFF"))
        if not self.show_grid:
            return
        s = self.transform().m11()
        step = 20.0                              # grille principale
        while step * s < 10:
            step *= 2
        # sous-grille : la plus fine (moitié, puis quart = pas du magnétisme) qui reste lisible
        sub = None
        if s >= self.SUBGRID_ZOOM:               # seulement à partir d'un certain zoom
            for k in (4, 2):
                if step / k * s >= 9:
                    sub = step / k
                    break
        if sub is not None:
            fade = min(1.0, (s - self.SUBGRID_ZOOM) / 0.75)   # apparition progressive en zoomant
            base = QColor(88, 91, 98) if dark else QColor(178, 178, 178)
            base.setAlphaF(0.45 + 0.55 * fade)
            self._dots(painter, rect, sub, base, 1.5, skip=step)
            main = QColor(150, 153, 160) if dark else QColor(95, 95, 95)
            self._dots(painter, rect, step, main, 2.8)
        else:
            self._dots(painter, rect, step, QColor(80, 82, 88) if dark else QColor(185, 185, 185), 1.6)

    @staticmethod
    def _dots(painter, rect, step, color, size, skip=None):
        def coords(start, end):           # mêmes additions successives que le tracé d'origine
            out, v = [], math.floor(start / step) * step
            while v <= end:
                out.append(v)
                v += step
            return out

        def on_skip(v):                    # sur la grille principale (dessinée à part)
            return abs(v / skip - round(v / skip)) < 1e-6

        xs, ys = coords(rect.left(), rect.right()), coords(rect.top(), rect.bottom())
        if skip is None:
            pts = [QPointF(x, y) for x in xs for y in ys]
        else:
            ys_off = [y for y in ys if not on_skip(y)]
            pts = []
            for x in xs:
                pts += [QPointF(x, y) for y in (ys_off if on_skip(x) else ys)]
        pen = QPen(color, size)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(QPolygonF(pts))

    # -------------------------------------------------------------------- zoom
    def zoom_by(self, f):
        s = self.transform().m11()
        ns = min(8.0, max(0.1, s * f))
        self.scale(ns / s, ns / s)
        self.zoomChanged.emit(ns)

    def zoom_reset(self):
        self.setTransform(QTransform())
        self.zoomChanged.emit(1.0)

    def fit(self):
        r = self.scene().itemsBoundingRect()
        if r.isNull():
            self.zoom_reset()
            return
        self.fitInView(r.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
        if self.transform().m11() > 2.5:
            self.setTransform(QTransform.fromScale(2.5, 2.5))
            self.centerOn(r.center())
        self.zoomChanged.emit(self.transform().m11())

    def recenter(self):
        """Vue de départ : zoom 100 %, centrée sur le diagramme (ou sur l'origine s'il est vide)."""
        r = self.scene().itemsBoundingRect()
        self.setTransform(QTransform())
        self.centerOn(QPointF(0, 0) if r.isNull() else r.center())
        self.zoomChanged.emit(1.0)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            self.zoom_by(1.0015 ** e.angleDelta().y())
            e.accept()
            return
        super().wheelEvent(e)

    # ------------------------------------------------------------ déplacement
    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = e.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.pos() - self._pan
            self._pan = e.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._pan is not None:
            self._pan = None
            self._on_mode(self.scene().mode)
            e.accept()
            return
        super().mouseReleaseEvent(e)
        if e.button() == Qt.LeftButton:
            self.scene().checkpoint()

    def contextMenuEvent(self, e):
        sc = self.scene()
        if not self.context_actions:
            return
        if sc.mode != "select":
            sc.set_mode("select")          # clic droit en mode flèche : menu de l'élément cliqué
        sc.context_pos = self.mapToScene(e.pos())     # pour « ajouter un coude ici »
        it = self.itemAt(e.pos())
        if it is not None and not it.isSelected():
            sc.clearSelection()
            it.setSelected(True)
        menu = QMenu(self)
        for a in self.context_actions:
            if a is None:
                menu.addSeparator()
            elif callable(a):
                sub = a(menu)            # sous-menu ou action construits à la volée
                if isinstance(sub, QMenu):
                    menu.addMenu(sub)
                elif isinstance(sub, list):      # plusieurs actions au premier niveau
                    for x in sub:
                        menu.addAction(x)
                elif sub is not None:
                    menu.addAction(sub)
            else:
                menu.addAction(a)
        menu.exec_(e.globalPos())

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(LIB_MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(LIB_MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasFormat(LIB_MIME):
            bid = bytes(e.mimeData().data(LIB_MIME)).decode("utf-8")
            self.scene().set_mode("select")
            self.scene().create_from_def(bid, self.mapToScene(e.pos()))
            e.acceptProposedAction()
            self.setFocus()
        else:
            super().dropEvent(e)
