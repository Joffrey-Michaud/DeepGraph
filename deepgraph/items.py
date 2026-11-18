"""Éléments du diagramme : blocs, opérateurs, texte libre, jonctions, groupes."""
import math

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPen, QColor, QPainter, QPainterPath, QFontMetricsF
from PyQt5.QtWidgets import QApplication, QGraphicsItem

from . import richtext
from .config import CFG

SEL_COLOR = QColor(30, 120, 230)
HANDLE = 8.0


def global_width(item, key):
    """Épaisseur globale du diagramme ('border_width' ou 'arrow_width')."""
    s = item.scene()
    v = getattr(s, key, None) if s is not None else None
    return float(v if v is not None else CFG.style[key])


# taille globale utilisée hors scène (ex. mesure des textes pendant simple_format.expand)
_FONT_FALLBACK = [None]


def global_font_size(item):
    """Taille de texte globale du diagramme (px)."""
    s = item.scene() if item is not None else None
    v = getattr(s, "font_size", None) if s is not None else None
    if v is None:
        v = _FONT_FALLBACK[0]
    return int(v if v is not None else CFG.style["font_size"])


def group_label_delta():
    """Écart entre l'étiquette de groupe par défaut et le texte par défaut (15 - 13 = 2 px)."""
    return int(CFG.style["group_label_size"]) - int(CFG.style["font_size"])


def free_move():
    """Maj enfoncée = pas d'aimantation (grille ni guides)."""
    return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)


# Opérateurs : (clé, libellé). Les traits de l'addition, de la différence et de la
# multiplication ne touchent pas le cercle ; ceux de la concaténation, si.
SYMBOLS = [("concat", "Concaténation"), ("plus", "Addition"), ("minus", "Différence"),
           ("times", "Multiplication"), ("dot", "Produit terme à terme")]


def draw_symbol(painter, c, rad, symbol, color, width, text_color=None, font=None):
    """Dessine le symbole intérieur d'un opérateur centré en c, de rayon rad."""
    pen = QPen(color, width)
    pen.setCapStyle(Qt.FlatCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    if symbol in ("concat", "plus", "minus"):
        L = rad if symbol == "concat" else rad * 0.76
        painter.drawLine(QPointF(c.x() - L, c.y()), QPointF(c.x() + L, c.y()))
        if symbol != "minus":
            painter.drawLine(QPointF(c.x(), c.y() - L), QPointF(c.x(), c.y() + L))
    elif symbol == "times":
        d = rad * 0.54
        painter.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
        painter.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))
    elif symbol == "dot":
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(c, rad * 0.22, rad * 0.22)
    elif symbol:
        if font is not None:
            painter.setFont(font)
        painter.setPen(text_color or color)
        painter.drawText(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), int(Qt.AlignCenter), symbol)


# ---------------------------------------------------------------- utilitaires
def snap_fraction(f, length):
    """Aimante une position relative le long d'un côté (milieu, quarts, tiers…)."""
    f = min(max(f, 0.0), 1.0)
    levels = (
        ((0.5,), 6.0),
        ((0.25, 0.75, 1 / 3, 2 / 3), 4.0),
        (tuple(k / 10 for k in range(1, 10)) + tuple(k / 8 for k in range(1, 8)), 2.5),
    )
    for cands, tol in levels:
        best = min(cands, key=lambda c: abs(c - f))
        if abs(best - f) * length <= tol:
            return best
    return f


def is_exporting(item):
    s = item.scene()
    return bool(s is not None and getattr(s, "exporting", False))


def _dark(item):
    s = item.scene()
    return bool(s is not None and getattr(s, "dark", False) and not getattr(s, "exporting", False))


def ink(item, color):
    """Couleur d'un trait ou d'un texte posé sur le fond. En mode sombre (écran seulement,
    jamais à l'export) les encres foncées deviennent claires pour rester lisibles."""
    c = QColor(color)
    if _dark(item):
        h, s_, l, a = c.getHsl()
        if l < 100:
            c = QColor.fromHsl(max(h, 0), s_, 230 - l // 2, a)
    return c


def surface(item, color):
    """Remplissage clair d'un fond de groupe : assombri à l'écran en mode sombre."""
    c = QColor(color)
    if _dark(item):
        h, s_, l, a = c.getHsl()
        if l > 150:
            c = QColor.fromHsl(max(h, 0), s_ // 3, min(110, 294 - l), a)
    return c


def skip_paint(item):
    """Pendant un export de la sélection, les éléments non retenus ne sont pas dessinés."""
    s = item.scene()
    return bool(s is not None and getattr(s, "exporting", False)
                and s.export_only is not None and item not in s.export_only)


def ancestors(item):
    p = item.parentItem()
    while p is not None:
        yield p
        p = p.parentItem()


def descendants(item):
    out = []
    for c in item.childItems():
        if isinstance(c, NodeItem):
            out.append(c)
            out.extend(descendants(c))
    return out


# ------------------------------------------------------------------ base node
class NodeItem(QGraphicsItem):
    """Élément rectangulaire (origine en haut à gauche) sur lequel on peut ancrer des flèches."""
    kind = "node"
    label_fr = "Élément"
    resizable = True
    keep_square = False
    MIN = 6.0
    SAVE = []
    PROPS = []

    def __init__(self, w, h):
        super().__init__()
        self.uid = None
        self.def_id = None
        self.w, self.h = float(w), float(h)
        self.arrows = []
        self.border_width = None   # None = épaisseur globale du diagramme
        self.layer = None          # identifiant du calque (géré par la scène)
        self.order = None          # rang d'affichage parmi ses voisins (plus grand = devant)
        self._resizing = False
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges
                      | QGraphicsItem.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

    # géométrie ---------------------------------------------------------------
    def rect(self):
        return QRectF(0, 0, self.w, self.h)

    def boundingRect(self):
        m = HANDLE + 4
        return self.rect().adjusted(-m, -m, m, m)

    def shape(self):
        p = QPainterPath()
        p.addRect(self.rect())
        if self.isSelected() and self.resizable:
            p.addRect(self.handle_rect())
        return p

    def content_rect(self):
        """Zone réellement dessinée (sert à cadrer les exports)."""
        bw = self.eff_border()
        return self.rect().adjusted(-bw, -bw, bw, bw)

    def layout_rect(self):
        """Encombrement pris en compte par un groupe qui s'ajuste à son contenu."""
        return self.rect()

    def eff_border(self):
        return float(self.border_width) if self.border_width is not None else global_width(self, "border_width")

    def scene_frame(self):
        return self.mapRectToScene(self.rect())

    def handle_rect(self):
        return QRectF(self.w + 3 - HANDLE / 2, self.h + 3 - HANDLE / 2, HANDLE, HANDLE)

    def set_size(self, w, h):
        self.prepareGeometryChange()
        self.w, self.h = float(w), float(h)
        self.update()
        self.update_arrows()

    # ancres ------------------------------------------------------------------
    def anchor_local(self, fx, fy):
        return QPointF(fx * self.w, fy * self.h)

    def anchor_scene(self, fx, fy):
        return self.mapToScene(self.anchor_local(fx, fy))

    def anchor_dir(self, fx, fy):
        """Direction sortante du côté où se trouve l'ancre."""
        if fy <= 1e-6:
            return "up"
        if fy >= 1 - 1e-6:
            return "down"
        if fx <= 1e-6:
            return "left"
        if fx >= 1 - 1e-6:
            return "right"
        return None

    def nearest_anchor(self, scene_pt, snap=True):
        p = self.mapFromScene(scene_pt)
        w, h = self.w, self.h
        x, y = min(max(p.x(), 0.0), w), min(max(p.y(), 0.0), h)
        d = {"up": abs(p.y()), "down": abs(p.y() - h), "left": abs(p.x()), "right": abs(p.x() - w)}
        side = min(d, key=d.get)
        if side in ("up", "down"):
            f = x / w if w else 0.5
            if snap:
                f = snap_fraction(f, w)
            return f, (0.0 if side == "up" else 1.0)
        f = y / h if h else 0.5
        if snap:
            f = snap_fraction(f, h)
        return (0.0 if side == "left" else 1.0), f

    def accepts_connection_at(self, scene_pt):
        return True

    def update_arrows(self):
        for a in list(self.arrows):
            a.update_path()

    # propriétés --------------------------------------------------------------
    def on_props_changed(self):
        pass

    def set_prop(self, name, value):
        if name in ("w", "h"):
            w = value if name == "w" else self.w
            h = value if name == "h" else self.h
            if self.keep_square:
                w = h = value
            self.set_size(w, h)
            return
        self.prepareGeometryChange()
        setattr(self, name, value)
        self.on_props_changed()
        self.update()
        self.update_arrows()

    def to_dict(self):
        d = {"type": self.kind, "uid": self.uid, "x": round(self.pos().x(), 2),
             "y": round(self.pos().y(), 2), "w": round(self.w, 2), "h": round(self.h, 2)}
        if self.def_id:
            d["def_id"] = self.def_id
        for k in self.SAVE:
            d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d):
        it = cls()
        it.w = float(d.get("w", it.w))
        it.h = float(d.get("h", it.h))
        it.def_id = d.get("def_id")
        for k in cls.SAVE:
            if k in d:
                setattr(it, k, d[k])
        it.on_props_changed()
        return it

    # interaction ---------------------------------------------------------------
    def itemChange(self, change, value):
        sc = self.scene()
        if change == QGraphicsItem.ItemPositionChange and sc is not None and not sc.suspend_snap:
            par = self.parentItem()
            sp = par.mapToScene(value) if par is not None else QPointF(value)
            x, y = sp.x(), sp.y()
            free = free_move()
            lead = sc.guides_enabled and sc.mouseGrabberItem() is self and not self._resizing
            gx = gy = None
            if lead and not free and len(sc.top_selected_nodes()) <= 1:
                gx, gy = sc.guide_snap(self, x, y)       # guides d'alignement prioritaires
            if sc.snap_enabled and not free:
                # aimantation du *centre* sur la grille => blocs de tailles différentes alignés
                # taille multiple de la grille : bords sur la grille ; sinon centre sur la grille
                g = float(sc.grid_size)

                def snap_axis(v, size):
                    if abs(size / g - round(size / g)) < 1e-6:
                        return round(v / g) * g
                    return round((v + size / 2) / g) * g - size / 2

                if gx is None:
                    x = snap_axis(x, self.w)
                if gy is None:
                    y = snap_axis(y, self.h)
            if gx is not None:
                x = gx
            if gy is not None:
                y = gy
            if lead:
                sc.update_guides(self, x, y)
            snapped = QPointF(x, y)
            return par.mapFromScene(snapped) if par is not None else snapped
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.update_arrows()
        return super().itemChange(change, value)

    def paint_selection(self, painter):
        if self.isSelected() and not is_exporting(self):
            pen = QPen(SEL_COLOR, 1, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(-3, -3, 3, 3))
            if self.resizable:
                pen = QPen(SEL_COLOR, 1)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(QColor("white"))
                painter.drawRect(self.handle_rect())

    def _on_handle(self, pos):
        return (self.resizable and self.isSelected()
                and self.handle_rect().adjusted(-2, -2, 2, 2).contains(pos))

    def hoverMoveEvent(self, e):
        if self._on_handle(e.pos()):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._on_handle(e.pos()):
            self._resizing = True
            sc = self.scene()
            if sc is not None and sc.snap_enabled and not (e.modifiers() & Qt.ShiftModifier):
                # coin haut-gauche sur la grille : les bords tombent alors sur les points
                g = float(sc.grid_size)
                tl = self.scenePos()
                snapped = QPointF(round(tl.x() / g) * g, round(tl.y() / g) * g)
                if snapped != tl:
                    par = self.parentItem()
                    old, sc.suspend_snap = sc.suspend_snap, True
                    self.setPos(par.mapFromScene(snapped) if par is not None else snapped)
                    sc.suspend_snap = old
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._resizing:
            p = e.pos()
            w, h = max(self.MIN, p.x() - 3), max(self.MIN, p.y() - 3)
            sc = self.scene()
            if sc is not None and sc.snap_enabled and not (e.modifiers() & Qt.ShiftModifier):
                # pas = grille d'aimantation (= sous-grille la plus fine)
                g = float(sc.grid_size)
                w, h = max(g, round(w / g) * g), max(g, round(h / g) * g)
            if self.keep_square:
                w = h = max(w, h)
            self.set_size(w, h)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        sc = self.scene()
        if self._resizing:
            self._resizing = False
            e.accept()
            if sc is not None:
                sc.after_resize(self)
            return
        super().mouseReleaseEvent(e)
        if sc is not None:
            sc.after_move()

    def mouseDoubleClickEvent(self, e):
        if self.scene() is not None:
            self.scene().edit_item(self)


# ----------------------------------------------------------------------- bloc
class BlockItem(NodeItem):
    kind = "block"
    label_fr = "Bloc"
    STACK_OFF = 4.0
    SAVE = ["text", "color", "text_color", "font_size", "bold", "italic", "stack", "border_width",
            "opacity_pct", "stack_dx", "stack_dy", "stack_fade", "base_w", "base_h"]
    PROPS = [
        ("text", "Texte", "text"),
        ("color", "Couleur", "color"),
        ("opacity_pct", "Opacité (%)", "int", (0, 100)),
        ("border_width", "Épaisseur du bord", "width_opt", "border_width"),
        ("text_color", "Couleur du texte", "color"),
        ("font_size", "Taille du texte (px)", "size_opt", "font_size"),
        ("bold", "Gras", "bool"),
        ("italic", "Italique", "bool"),
        ("stack", "Copies empilées derrière", "int", (0, 50)),
        ("stack_dx", "Empilement : décalage x", "float", (-200, 200, 1)),
        ("stack_dy", "Empilement : décalage y", "float", (-200, 200, 1)),
        ("stack_fade", "Empilement : opacité dégressive", "bool"),
        ("w", "Largeur", "float", (6, 4000, 5)),
        ("h", "Hauteur", "float", (6, 4000, 5)),
    ]

    def __init__(self, text="Bloc", color="#FFFFFF", w=None, h=None):
        st = CFG.style
        super().__init__(w or st["block_width"], h or st["block_height"])
        self.text = text
        self.color = color
        self.text_color = st["text_color"]
        self.font_size = None        # None = taille globale du diagramme
        self.bold = False
        self.italic = False
        self.stack = 0                 # nombre de copies (sans texte) dessinées derrière le bloc
        self.stack_dx = self.STACK_OFF  # décalage d'une copie à la suivante (px, signé)
        self.stack_dy = -self.STACK_OFF
        self.stack_fade = True          # copies de plus en plus transparentes
        self.opacity_pct = 100
        # taille voulue (bibliothèque ou redimensionnement) ; None = la taille actuelle.
        # Ajusté au texte, le bloc fait au moins cette taille et y revient quand le texte rétrécit.
        self.base_w = self.base_h = None

    def text_size(self):
        """Taille effective du texte (px) : la sienne, ou la taille globale du diagramme."""
        return int(self.font_size) if self.font_size is not None else global_font_size(self)

    # taille ajustée au texte ------------------------------------------------
    TEXT_PAD_X, TEXT_PAD_Y = 6.0, 1.5

    def fit_active(self):
        sc = self.scene()
        return bool(sc is not None and getattr(sc, "fit_text", False))

    def base_size(self):
        return (float(self.base_w) if self.base_w is not None else self.w,
                float(self.base_h) if self.base_h is not None else self.h)

    def text_need(self, width=None):
        """Taille minimale pour que le texte ne déborde pas. Comme à l'affichage, un texte sans
        formule revient à la ligne : la largeur doit contenir le mot le plus long, la hauteur
        les lignes obtenues à cette largeur."""
        size = self.text_size()
        text = self.text or " "
        fm = QFontMetricsF(CFG.font(size, self.bold, self.italic))
        slack = max(0.0, fm.ascent() - 0.8 * size)   # vide au-dessus des lettres les plus hautes
        if richtext.has_markup(text):
            tw, th = richtext.measure(text, size, self.bold, self.italic)
            nw = tw + 2 * self.TEXT_PAD_X
            nh = th - slack + 2 * self.TEXT_PAD_Y
        else:
            words = text.split() or [" "]
            nw = max(fm.horizontalAdvance(wd) for wd in words) + 2 * self.TEXT_PAD_X
            wrap_w = max(self.w if width is None else width, nw) - 8      # même rectangle que paint()
            r = fm.boundingRect(QRectF(0, 0, wrap_w, 1e5), int(Qt.AlignHCenter | Qt.TextWordWrap), text)
            nh = r.height() - slack + 2 * self.TEXT_PAD_Y
        return max(self.MIN, nw), max(self.MIN, nh)

    def set_size(self, w, h):
        self.base_w, self.base_h = float(w), float(h)       # taille demandée = taille voulue
        if self.fit_active():
            nw, nh = self.text_need()
            sc = self.scene()
            g = float(sc.grid_size) if sc is not None and sc.snap_enabled else 0.0
            if g > 0:           # taille minimale sur la grille pendant un redimensionnement
                nw, nh = math.ceil(nw / g - 1e-6) * g, math.ceil(nh / g - 1e-6) * g
            w, h = max(w, nw), max(h, nh)
        super().set_size(w, h)

    def fit_to_text(self, exact=False):
        """Agrandit le bloc (ou, exact=True, le met juste à la taille du texte) en gardant son centre.
        Renvoie True si la taille a changé."""
        sc = self.scene()
        shrink = exact or bool(sc is not None and getattr(sc, "fit_shrink", True))
        if exact:
            nw, nh = self.text_need()
            tw, th = nw, nh
        elif shrink:                     # retour vers la taille voulue, jamais sous le texte
            bw, bh = self.base_size()
            nw, nh = self.text_need(bw)
            tw, th = max(bw, nw), max(bh, nh)
        else:                            # agrandir seulement
            nw, nh = self.text_need()
            tw, th = max(self.w, nw), max(self.h, nh)
        g = float(sc.grid_size) if sc is not None and sc.snap_enabled else 0.0
        dw, dh = tw - self.w, th - self.h
        if g > 0:   # variation multiple de 2 × grille : les bords restent sur la grille
            dw = math.ceil(dw / (2 * g) - 1e-6) * 2 * g
            dh = math.ceil(dh / (2 * g) - 1e-6) * 2 * g
        if abs(dw) < 1e-6 and abs(dh) < 1e-6:
            return False
        old = sc.suspend_snap if sc is not None else None
        if sc is not None:
            sc.suspend_snap = True
        self.setPos(self.pos() - QPointF(dw / 2, dh / 2))
        if sc is not None:
            sc.suspend_snap = old
        if self.base_w is None:          # mémorise la taille voulue avant le premier ajustement
            self.base_w, self.base_h = self.w, self.h
        NodeItem.set_size(self, self.w + dw, self.h + dh)
        if exact:
            self.base_w, self.base_h = self.w, self.h
        return True

    def on_props_changed(self):
        if self.fit_active():
            self.fit_to_text()

    def _stack_count(self):
        return max(0, int(self.stack or 0))

    def _stack_grow(self):
        """Marges (gauche, haut, droite, bas) ajoutées par l'empilement."""
        k = self._stack_count()
        tx, ty = k * float(self.stack_dx), k * float(self.stack_dy)
        return min(0.0, tx), min(0.0, ty), max(0.0, tx), max(0.0, ty)

    def _with_stack(self, r):
        l, t, rr, b = self._stack_grow()
        return r.adjusted(l, t, rr, b)

    def boundingRect(self):
        return self._with_stack(super().boundingRect())

    def content_rect(self):
        return self._with_stack(super().content_rect())

    def layout_rect(self):
        return self._with_stack(self.rect())

    def stack_alpha(self, i):
        """Opacité relative de la i-ème copie (1 = la plus proche du bloc)."""
        if not self.stack_fade:
            return 1.0
        return max(0.08, 1.0 - 0.8 * i / (self._stack_count() + 1))

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        st = CFG.style
        painter.setRenderHint(QPainter.Antialiasing)
        r = float(st["corner_radius"])
        border = QColor(st["border_color"])
        fill = QColor(self.color)
        bw = self.eff_border()
        op0 = painter.opacity()                  # inclut l'opacité du calque
        painter.setOpacity(op0 * max(0, min(100, int(self.opacity_pct))) / 100.0)
        base_op = painter.opacity()
        k = self._stack_count()
        if k:
            # chaque copie n'est dessinée que là où elle n'est pas cachée par celles de devant :
            # avec la transparence, on ne voit donc pas les bords à travers
            def rounded(i):
                p = QPainterPath()
                p.addRoundedRect(self.rect().translated(i * self.stack_dx, i * self.stack_dy), r, r)
                return p
            front = rounded(0)
            visible = []
            for i in range(1, k + 1):
                pi = rounded(i)
                visible.append((i, pi.subtracted(front)))
                front = front.united(pi)
            pen = QPen(border, bw)
            pen.setJoinStyle(Qt.RoundJoin)
            for i, path in reversed(visible):
                painter.setOpacity(base_op * self.stack_alpha(i))
                painter.setPen(pen)
                painter.setBrush(fill)
                painter.drawPath(path)
            painter.setOpacity(base_op)
        pen = QPen(border, bw)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRoundedRect(self.rect(), r, r)
        painter.setPen(QColor(self.text_color))
        richtext.draw(painter, self.rect().adjusted(4, 1, -4, -1), self.text, self.text_size(),
                      self.bold, self.italic, Qt.AlignHCenter, wrap=True)
        painter.setOpacity(op0)
        self.paint_selection(painter)


# ------------------------------------------------------------------ opérateur
class OperatorItem(NodeItem):
    kind = "operator"
    label_fr = "Opérateur"
    keep_square = True
    SAVE = ["symbol", "color", "border_width"]
    PROPS = [
        ("symbol", "Symbole", "choice_edit", SYMBOLS),
        ("color", "Couleur", "color"),
        ("border_width", "Épaisseur du trait", "width_opt", "border_width"),
        ("w", "Diamètre", "float", (6, 400, 2)),
    ]

    def __init__(self, symbol="plus", color="#FFFFFF", size=None):
        s = size or CFG.style["operator_size"]
        super().__init__(s, s)
        self.symbol = symbol
        self.color = color

    def shape(self):
        p = QPainterPath()
        p.addEllipse(self.rect())
        if self.isSelected():
            p.addRect(self.handle_rect())
        return p

    def nearest_anchor(self, scene_pt, snap=True):
        p = self.mapFromScene(scene_pt)
        ang = math.atan2(p.y() - self.h / 2, p.x() - self.w / 2)
        if snap:
            for step, tol in ((math.pi / 2, math.radians(15)), (math.pi / 4, math.radians(8))):
                a2 = round(ang / step) * step
                if abs(a2 - ang) <= tol:
                    ang = a2
                    break
        return 0.5 + 0.5 * math.cos(ang), 0.5 + 0.5 * math.sin(ang)

    def anchor_dir(self, fx, fy):
        dx, dy = fx - 0.5, fy - 0.5
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        return "right" if dx > 0 else "left"

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        st = CFG.style
        painter.setRenderHint(QPainter.Antialiasing)
        border = QColor(st["border_color"])
        bw = self.eff_border()
        painter.setPen(QPen(border, bw))
        painter.setBrush(QColor(self.color))
        r = self.rect()
        painter.drawEllipse(r)
        draw_symbol(painter, r.center(), self.w / 2, self.symbol, border, bw,
                    QColor(st["text_color"]), CFG.font(max(6, self.w * 0.6)))
        self.paint_selection(painter)


# ---------------------------------------------------------------- texte libre
class TextItem(NodeItem):
    kind = "text"
    label_fr = "Texte"
    resizable = False
    PAD = 3.0
    ALIGNS = [("left", "Gauche"), ("center", "Centré"), ("right", "Droite")]
    SAVE = ["text", "font_size", "bold", "italic", "color", "align"]
    PROPS = [
        ("text", "Texte", "text"),
        ("font_size", "Taille (px)", "size_opt", "font_size"),
        ("bold", "Gras", "bool"),
        ("italic", "Italique", "bool"),
        ("color", "Couleur", "color"),
        ("align", "Alignement", "choice", ALIGNS),
    ]

    def __init__(self, text="Texte"):
        super().__init__(10, 10)
        st = CFG.style
        self.text = text
        self.font_size = None        # None = taille globale du diagramme
        self.bold = False
        self.italic = False
        self.color = st["text_color"]
        self.align = "center"
        self.on_props_changed()

    def text_size(self):
        """Taille effective du texte (px) : la sienne, ou la taille globale du diagramme."""
        return int(self.font_size) if self.font_size is not None else global_font_size(self)

    def _flags(self):
        h = {"left": Qt.AlignLeft, "right": Qt.AlignRight}.get(self.align, Qt.AlignHCenter)
        return int(h | Qt.AlignVCenter)

    def on_props_changed(self):
        tw, th = richtext.measure(self.text or " ", self.text_size(), self.bold, self.italic)
        self.w = tw + 2 * self.PAD
        self.h = th + 2 * self.PAD

    def set_prop(self, name, value):
        sc = self.scene()
        c = self.scene_frame().center()
        super().set_prop(name, value)
        if sc is not None:  # garder le texte centré au même endroit
            old, sc.suspend_snap = sc.suspend_snap, True
            par = self.parentItem()
            target = c - QPointF(self.w / 2, self.h / 2)
            self.setPos(par.mapFromScene(target) if par is not None else target)
            sc.suspend_snap = old
        self.update_arrows()

    def content_rect(self):
        return self.rect()

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(ink(self, self.color))
        h = {"left": Qt.AlignLeft, "right": Qt.AlignRight}.get(self.align, Qt.AlignHCenter)
        richtext.draw(painter, self.rect().adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD),
                      self.text, self.text_size(), self.bold, self.italic, h)
        self.paint_selection(painter)


# ------------------------------------------------------------------- jonction
class JunctionItem(NodeItem):
    """Point d'embranchement : plusieurs flèches peuvent partir d'ici. Invisible à l'export."""
    kind = "junction"
    label_fr = "Jonction"
    resizable = False
    SAVE = ["show_dot"]
    PROPS = [("show_dot", "Point visible à l'export", "bool")]

    def __init__(self):
        super().__init__(10, 10)
        self.show_dot = False

    def boundingRect(self):
        return self.rect().adjusted(-3, -3, 3, 3)

    def shape(self):
        p = QPainterPath()
        p.addEllipse(self.rect().adjusted(-2, -2, 2, 2))
        return p

    def content_rect(self):
        c = self.rect().center()
        r = max(2.2, 1.8 * global_width(self, "arrow_width"))
        return QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)

    def nearest_anchor(self, scene_pt, snap=True):
        return 0.5, 0.5

    def anchor_dir(self, fx, fy):
        return None

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        painter.setRenderHint(QPainter.Antialiasing)
        c = self.rect().center()
        if self.show_dot:
            painter.setPen(Qt.NoPen)
            painter.setBrush(ink(self, CFG.style["arrow_color"]))
            r = max(2.2, 1.8 * global_width(self, "arrow_width"))
            painter.drawEllipse(c, r, r)
        if not is_exporting(self):
            sel = self.isSelected()
            painter.setPen(QPen(SEL_COLOR if sel else QColor(110, 110, 110), 1.6 if sel else 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(c, 4, 4)


# ---------------------------------------------------------------------- groupe
class GroupItem(NodeItem):
    """Cadre gris arrondi ; les éléments posés dedans deviennent ses enfants.
    On le sélectionne/déplace par son bord ou son étiquette."""
    kind = "group"
    label_fr = "Groupe"
    LABEL_POS = [("left", "À gauche"), ("right", "À droite"), ("top", "Au-dessus"),
                 ("bottom", "En dessous"), ("inside", "Dedans (haut-gauche)"), ("none", "Aucune")]
    BORDERS = [("solid", "Continu"), ("dashed", "Tirets"), ("none", "Aucun")]
    SIDES = ("margin_top", "margin_bottom", "margin_left", "margin_right")
    SAVE = ["label", "label_pos", "label_size", "fill", "border_style", "border_width",
            "fit", "margin", *SIDES]
    PROPS = [
        ("label", "Étiquette", "line"),
        ("label_pos", "Position étiquette", "choice", LABEL_POS),
        ("label_size", "Taille étiquette (px)", "size_opt", "group_label_size"),
        ("fill", "Fond", "color"),
        ("border_style", "Bordure", "choice", BORDERS),
        ("border_width", "Épaisseur du bord", "width_opt", "border_width"),
        ("fit", "Ajuster au contenu", "bool"),
        ("margin", "Marges", "float", (0, 1000, 1)),
        ("margin_top", "Marge en haut", "margin_opt"),
        ("margin_bottom", "Marge en bas", "margin_opt"),
        ("margin_left", "Marge à gauche", "margin_opt"),
        ("margin_right", "Marge à droite", "margin_opt"),
        ("w", "Largeur", "float", (20, 6000, 5)),
        ("h", "Hauteur", "float", (20, 6000, 5)),
    ]
    BAND = 9.0

    def __init__(self, w=220, h=160):
        super().__init__(w, h)
        st = CFG.style
        # le cadre suit son contenu + marges ; None = « Marges » (identique sur les 4 côtés)
        self.fit = True
        self.margin = float(st["group_padding"])
        self.margin_top = self.margin_bottom = self.margin_left = self.margin_right = None
        self.label = "N×"
        self.label_pos = "left"
        self.label_size = None       # None = taille globale + 2 px
        self.fill = st["group_fill"]
        self.border_style = "solid"
        self.setZValue(-10)

    def text_size(self):
        """Taille effective de l'étiquette (px) : la sienne, ou la taille globale + 2 px."""
        if self.label_size is not None:
            return int(self.label_size)
        return global_font_size(self) + group_label_delta()

    # marges / ajustement au contenu ------------------------------------------
    def side(self, name):
        v = getattr(self, name)
        return float(self.margin if v is None else v)

    def layout_rect(self):
        if self.parentItem() is not None:
            return self.rect().united(self.label_rect())
        return self.rect()

    def content_bbox(self):
        """Encombrement des enfants (et des flèches internes), en coordonnées du groupe."""
        kids = [c for c in self.childItems() if isinstance(c, NodeItem)]
        if not kids:
            return None
        r = QRectF()
        for c in kids:
            cr = c.mapRectToParent(c.layout_rect())
            r = cr if r.isNull() else r.united(cr)
        inside = set(descendants(self))
        seen = set()
        for n in inside:
            for a in n.arrows:
                if a in seen or a.src.node not in inside or a.dst.node not in inside:
                    continue
                seen.add(a)
                ar = self.mapRectFromScene(a.mapRectToScene(a.path_rect()))
                r = r.united(ar)
        return r

    def refit(self):
        if not self.fit:
            return False
        bb = self.content_bbox()
        if bb is None:
            return False
        new = bb.adjusted(-self.side("margin_left"), -self.side("margin_top"),
                          self.side("margin_right"), self.side("margin_bottom"))
        d = new.topLeft()
        if abs(d.x()) < 1e-6 and abs(d.y()) < 1e-6 and abs(new.width() - self.w) < 1e-6 \
                and abs(new.height() - self.h) < 1e-6:
            return False
        sc = self.scene()
        old = sc.suspend_snap if sc is not None else None
        if sc is not None:
            sc.suspend_snap = True
        self.setPos(self.pos() + d)
        for c in self.childItems():
            if isinstance(c, NodeItem):
                c.setPos(c.pos() - d)
        if sc is not None:
            sc.suspend_snap = old
        self.set_size(new.width(), new.height())
        return True

    def margins_from_rect(self):
        """Après un redimensionnement à la main : les marges droite/bas prennent la nouvelle taille."""
        bb = self.content_bbox()
        if not self.fit or bb is None:
            return
        for name, v in (("margin_right", self.w - bb.right()), ("margin_bottom", self.h - bb.bottom())):
            v = round(max(0.0, v), 1)
            setattr(self, name, None if abs(v - self.margin) < 0.05 else v)

    def set_prop(self, name, value):
        super().set_prop(name, value)
        if name in ("w", "h"):
            self.margins_from_rect()
        if name == "margin" or name in self.SIDES or name in ("fit", "w", "h"):
            self.refit()
            for g in ancestors(self):
                if isinstance(g, GroupItem):
                    g.refit()

    @classmethod
    def from_dict(cls, d):
        it = super().from_dict(d)
        if "fit" not in d:      # ancien fichier : on garde exactement le cadre enregistré
            it.fit = False
        return it

    def label_rect(self):
        if not self.label or self.label_pos == "none":
            return QRectF()
        tw, th = richtext.measure(self.label, self.text_size())
        gap = 8.0
        pos = self.label_pos
        if pos == "left":
            return QRectF(-gap - tw, (self.h - th) / 2, tw, th)
        if pos == "right":
            return QRectF(self.w + gap, (self.h - th) / 2, tw, th)
        if pos == "top":
            return QRectF((self.w - tw) / 2, -gap - th, tw, th)
        if pos == "bottom":
            return QRectF((self.w - tw) / 2, self.h + gap, tw, th)
        return QRectF(10, 6, tw, th)

    def boundingRect(self):
        return super().boundingRect().united(self.label_rect().adjusted(-2, -2, 2, 2))

    def content_rect(self):
        return super().content_rect().united(self.label_rect())

    def shape(self):
        outer, inner = QPainterPath(), QPainterPath()
        outer.addRect(self.rect().adjusted(-4, -4, 4, 4))
        inner.addRect(self.rect().adjusted(self.BAND, self.BAND, -self.BAND, -self.BAND))
        p = outer.subtracted(inner)
        extra = QPainterPath()
        lr = self.label_rect()
        if not lr.isNull():
            extra.addRect(lr)
        if self.isSelected():
            extra.addRect(self.handle_rect())
        return p.united(extra)

    def accepts_connection_at(self, scene_pt):
        return self.shape().contains(self.mapFromScene(scene_pt))

    def paint(self, painter, option, widget=None):
        if skip_paint(self):
            return
        st = CFG.style
        painter.setRenderHint(QPainter.Antialiasing)
        if self.border_style == "none":
            painter.setPen(Qt.NoPen)
        else:
            pen = QPen(ink(self, st["border_color"]), self.eff_border(),
                       Qt.DashLine if self.border_style == "dashed" else Qt.SolidLine)
            painter.setPen(pen)
        painter.setBrush(surface(self, self.fill))
        r = float(st["group_corner_radius"])
        painter.drawRoundedRect(self.rect(), r, r)
        lr = self.label_rect()
        if not lr.isNull():
            painter.setPen(ink(self, st["text_color"]))
            richtext.draw(painter, lr, self.label, self.text_size(), halign=Qt.AlignLeft)
        self.paint_selection(painter)


NODE_TYPES = {c.kind: c for c in (BlockItem, OperatorItem, TextItem, JunctionItem, GroupItem)}
