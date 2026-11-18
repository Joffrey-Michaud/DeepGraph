"""Formules dans les textes : mini-syntaxe LaTeX entre $…$ (sans installer LaTeX).

Hors des $…$ le texte est affiché tel quel (« conv2_x » reste « conv2_x »). Dans une formule :
  x_t  x_{t-1}  W^Q  x^{2}  x_i^2       indices / exposants (un caractère ou {groupe})
  \\alpha … \\omega, \\Gamma … \\Omega    lettres grecques
  \\frac{a}{b}  \\sqrt{d_k}               fraction, racine
  \\hat{x} \\tilde{x} \\bar{x} \\vec{x} \\dot{x} \\overline{xy}   accents
  \\text{…} \\mathrm{…} \\mathbf{…} \\mathbb{R} \\operatorname{…}  styles
  \\times \\cdot \\oplus \\otimes \\odot \\in \\to \\infty \\sum \\leq …   symboles
  \\exp \\log \\max \\softmax …             fonctions (droites)
Les lettres latines sont en italique, chiffres et symboles droits (comme en LaTeX) ;
contrairement à LaTeX les espaces sont conservées. « \\$ » affiche un « $ ».
"""
from functools import lru_cache

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QFontMetricsF, QPainterPath, QPen

from .config import CFG

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "times": "×", "cdot": "·", "oplus": "⊕", "otimes": "⊗", "odot": "⊙", "circ": "∘", "ast": "∗",
    "star": "⋆", "pm": "±", "mp": "∓", "div": "÷", "in": "∈", "notin": "∉", "subset": "⊂",
    "subseteq": "⊆", "cup": "∪", "cap": "∩", "to": "→", "rightarrow": "→", "leftarrow": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐", "mapsto": "↦", "uparrow": "↑",
    "downarrow": "↓", "infty": "∞", "sum": "∑", "prod": "∏", "int": "∫", "partial": "∂", "nabla": "∇",
    "approx": "≈", "sim": "∼", "simeq": "≃", "equiv": "≡", "propto": "∝", "leq": "≤", "le": "≤",
    "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "ll": "≪", "gg": "≫", "ldots": "…", "dots": "…",
    "cdots": "⋯", "vdots": "⋮", "prime": "′", "top": "⊤", "intercal": "⊺", "perp": "⊥", "forall": "∀",
    "exists": "∃", "emptyset": "∅", "ell": "ℓ", "langle": "⟨", "rangle": "⟩", "lfloor": "⌊",
    "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "|": "‖", "{": "{", "}": "}", "$": "$", "%": "%",
    "#": "#", "&": "&", "_": "_", "backslash": "\\", "dagger": "†", "star_": "★",
}
SPACES = {",": 0.17, ":": 0.22, ";": 0.28, "!": -0.17, " ": 0.33, "quad": 1.0, "qquad": 2.0}
FUNCTIONS = {"exp", "log", "ln", "sin", "cos", "tan", "tanh", "sinh", "cosh", "max", "min", "arg",
             "argmax", "argmin", "det", "dim", "lim", "sup", "inf", "softmax", "sigmoid", "Pr", "mod",
             "ReLU", "GELU", "LayerNorm", "Var", "diag", "tr", "rank", "KL"}
BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ", "1": "𝟙"}
ACCENTS = {"hat": "hat", "widehat": "hat", "tilde": "tilde", "widetilde": "tilde", "bar": "bar",
           "dot": "dot", "ddot": "ddot", "vec": "vec", "check": "check", "breve": "breve"}
SCRIPT = 0.7          # taille des indices / exposants
FRAC = 0.85           # taille du numérateur / dénominateur


def has_markup(text):
    return bool(text) and "$" in text


# ----------------------------------------------------------------------- boîtes
class Box:
    w = asc = desc = 0.0

    def ink_top(self):
        return self.asc

    def draw(self, p, x, y):
        pass


class Glyphs(Box):
    def __init__(self, s, size, bold, italic):
        self.s, self.font = s, CFG.font(size, bold, italic)
        fm = QFontMetricsF(self.font)
        self.w = fm.horizontalAdvance(s)
        if italic and s:
            self.w += size * 0.06              # correction d'italique (avant un exposant…)
        self.asc, self.desc = fm.ascent(), fm.descent()
        t = fm.tightBoundingRect(s) if s.strip() else QRectF()
        self._ink = -t.top() if not t.isNull() else fm.xHeight()

    def ink_top(self):
        return self._ink

    def draw(self, p, x, y):
        p.setFont(self.font)
        p.drawText(QPointF(x, y), self.s)


class Space(Box):
    def __init__(self, w):
        self.w = w


class HBox(Box):
    def __init__(self, kids):
        self.kids = kids
        self.w = sum(k.w for k in kids)
        self.asc = max([k.asc for k in kids] or [0.0])
        self.desc = max([k.desc for k in kids] or [0.0])

    def ink_top(self):
        return max([k.ink_top() for k in self.kids] or [0.0])

    def draw(self, p, x, y):
        for k in self.kids:
            k.draw(p, x, y)
            x += k.w


class Script(Box):
    def __init__(self, base, sub, sup, size, a0, d0):
        self.base, self.sub, self.sup = base, sub, sup
        self.up = 0.36 * size + max(0.0, base.ink_top() - 0.75 * size)
        self.down = 0.18 * size + max(0.0, base.desc - d0)
        if sub is not None and sup is not None:
            self.down = max(self.down, 0.25 * size)
        self.w = base.w + max(sub.w if sub else 0, sup.w if sup else 0) + 0.04 * size
        self.asc = max(base.asc, (self.up + sup.asc) if sup else 0)
        self.desc = max(base.desc, (self.down + sub.desc) if sub else 0)

    def ink_top(self):
        return max(self.base.ink_top(), self.up + self.sup.ink_top() if self.sup else 0)

    def draw(self, p, x, y):
        self.base.draw(p, x, y)
        xs = x + self.base.w
        if self.sup:
            self.sup.draw(p, xs, y - self.up)
        if self.sub:
            self.sub.draw(p, xs, y + self.down)


def _pen_color(p):
    return p.pen().color()


def _line_pen(p, width):
    pen = QPen(p.pen().color(), width)
    pen.setCapStyle(Qt.FlatCap)
    pen.setJoinStyle(Qt.MiterJoin)
    return pen


class Frac(Box):
    def __init__(self, num, den, size):
        self.num, self.den, self.size = num, den, size
        self.axis, self.gap, self.rule = 0.28 * size, 0.1 * size, max(0.6, 0.065 * size)
        self.pad = 0.12 * size
        self.w = max(num.w, den.w) + 2 * self.pad
        self.asc = self.axis + self.gap + num.desc + num.asc
        self.desc = -self.axis + self.gap + den.asc + den.desc

    def draw(self, p, x, y):
        self.num.draw(p, x + (self.w - self.num.w) / 2, y - self.axis - self.gap - self.num.desc)
        self.den.draw(p, x + (self.w - self.den.w) / 2, y - self.axis + self.gap + self.den.asc)
        p.save()
        p.setPen(_line_pen(p, self.rule))
        ry = y - self.axis
        p.drawLine(QPointF(x + self.pad * 0.5, ry), QPointF(x + self.w - self.pad * 0.5, ry))
        p.restore()


class Sqrt(Box):
    def __init__(self, body, size):
        self.body, self.size = body, size
        self.rule = max(0.6, 0.065 * size)
        self.lead = 0.55 * size
        self.w = body.w + self.lead + 0.1 * size
        self.asc = max(body.asc, body.ink_top() + 0.2 * size)
        self.desc = max(body.desc, 0.12 * size)

    def draw(self, p, x, y):
        s, b = self.size, self.body
        top = y - b.ink_top() - 0.14 * s
        bot = y + 0.08 * s
        path = QPainterPath(QPointF(x + 0.04 * s, y - 0.28 * s))
        path.lineTo(QPointF(x + 0.15 * s, y - 0.34 * s))
        path.lineTo(QPointF(x + 0.3 * s, bot))
        path.lineTo(QPointF(x + self.lead - 0.04 * s, top))
        path.lineTo(QPointF(x + self.w - 0.02 * s, top))
        p.save()
        p.setPen(_line_pen(p, self.rule))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.restore()
        b.draw(p, x + self.lead + 0.04 * s, y)


class Over(Box):
    """Accent (dessiné, indépendant de la police) ou trait au-dessus d'un contenu."""

    def __init__(self, body, size, kind="bar"):
        self.body, self.size, self.kind = body, size, kind
        self.rule = max(0.6, 0.06 * size)
        self.base = body.ink_top() + 0.1 * size              # bas de la zone de l'accent
        self.top = self.base + (0.06 * size if kind == "bar" else 0.18 * size)
        self.w = body.w
        self.asc = max(body.asc, self.top + 0.05 * size)
        self.desc = body.desc

    def ink_top(self):
        return self.top

    def draw(self, p, x, y):
        s, b = self.size, self.body
        b.draw(p, x, y)
        italic = isinstance(b, Glyphs) and b.font.italic()
        shift = 0.07 * s if italic else 0.0                 # contenu penché : accent décalé à droite
        cx = x + b.w / 2 + shift
        hw = max(0.13 * s, min(0.38 * b.w, 0.3 * s))
        by = y - self.base
        h = 0.15 * s
        path = QPainterPath()
        p.save()
        p.setPen(_line_pen(p, self.rule))
        p.setBrush(Qt.NoBrush)
        k = self.kind
        if k == "bar":
            ly = y - self.top + self.rule / 2
            p.drawLine(QPointF(x + 0.04 * s + shift, ly), QPointF(x + b.w - 0.02 * s + shift, ly))
        elif k == "hat":
            path.moveTo(cx - hw, by)
            path.lineTo(cx, by - h)
            path.lineTo(cx + hw, by)
        elif k == "check":
            path.moveTo(cx - hw, by - h)
            path.lineTo(cx, by)
            path.lineTo(cx + hw, by - h)
        elif k == "tilde":
            path.moveTo(cx - hw, by - 0.3 * h)
            path.cubicTo(QPointF(cx - hw / 2, by - 1.3 * h), QPointF(cx + hw / 2, by + 0.3 * h),
                         QPointF(cx + hw, by - 0.7 * h))
        elif k == "breve":
            path.moveTo(cx - hw, by - h)
            path.quadTo(QPointF(cx, by + 0.6 * h), QPointF(cx + hw, by - h))
        elif k == "vec":
            x0, x1 = cx - max(hw, b.w / 2 - 0.02 * s), cx + max(hw, b.w / 2 - 0.02 * s)
            ay = by - 0.5 * h
            path.moveTo(x0, ay)
            path.lineTo(x1, ay)
            path.moveTo(x1 - 0.13 * s, ay - 0.09 * s)
            path.lineTo(x1, ay)
            path.lineTo(x1 - 0.13 * s, ay + 0.09 * s)
        elif k in ("dot", "ddot"):
            p.setBrush(_pen_color(p))
            p.setPen(Qt.NoPen)
            r = 0.055 * s
            for dx in ((0.0,) if k == "dot" else (-0.13 * s, 0.13 * s)):
                p.drawEllipse(QPointF(cx + dx, by - 0.45 * h), r, r)
        if not path.isEmpty():
            p.drawPath(path)
        p.restore()


# ------------------------------------------------------------------- analyse
class _Parser:
    def __init__(self, src, size, bold):
        self.s, self.i, self.size0, self.bold = src, 0, size, bold

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def command(self):
        """Lit le nom après « \\ » (lettres, ou un seul caractère)."""
        j = self.i
        while j < len(self.s) and self.s[j].isalpha():
            j += 1
        if j == self.i:
            j = min(len(self.s), self.i + 1)
        name, self.i = self.s[self.i:j], j
        return name

    def raw_group(self):
        """Contenu brut d'un argument {…} (ou d'un seul caractère)."""
        while self.peek() == " ":
            self.i += 1
        if self.peek() == "\\":               # x^\top, \epsilon_\theta
            j = self.i
            self.i += 1
            self.command()
            return self.s[j:self.i]
        if self.peek() != "{":
            c = self.peek()
            self.i += 1 if c else 0
            return c
        depth, j = 0, self.i
        while j < len(self.s):
            if self.s[j] == "\\":
                j += 2
                continue
            if self.s[j] == "{":
                depth += 1
            elif self.s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body, self.i = self.s[self.i + 1:j], j + 1
        return body

    def sub(self, src, size, style="math"):
        return _Parser(src, size, self.bold).parse(style)

    def parse(self, style="math"):
        kids = []
        size = self.size0
        while self.i < len(self.s):
            c = self.peek()
            if c == "}":                       # accolade orpheline : ignorée
                self.i += 1
                continue
            if c in "_^" and kids:
                base = kids.pop()
                sub = sup = None
                for _ in range(2):
                    c = self.peek()
                    if not c or c not in "_^":
                        break
                    self.i += 1
                    box = self.sub(self.raw_group(), max(5.0, size * SCRIPT), style)
                    if c == "_":
                        sub = box
                    else:
                        sup = box
                fm = QFontMetricsF(CFG.font(size, self.bold))
                kids.append(Script(base, sub, sup, size, fm.ascent(), fm.descent()))
                continue
            kids.append(self.atom(style))
        return HBox(kids)

    def text(self, s, style):
        """Morceau de texte : en mode math, lettres latines en italique."""
        size, bold = self.size0, self.bold
        if style == "text":
            return Glyphs(s, size, bold, False)
        if style == "bf":
            return Glyphs(s, size, True, False)
        if style == "bb":
            return Glyphs("".join(BLACKBOARD.get(ch, ch) for ch in s), size, bold, False)
        boxes, run, run_it = [], "", None
        for ch in s:
            it = ch.isascii() and ch.isalpha() or (ch in GREEK.values() and ch.islower())
            if run and it != run_it:
                boxes.append(Glyphs(run, size, bold, run_it))
                run = ""
            run, run_it = run + ch, it
        if run:
            boxes.append(Glyphs(run, size, bold, run_it))
        return boxes[0] if len(boxes) == 1 else HBox(boxes)

    def atom(self, style):
        c = self.peek()
        size = self.size0
        if c == "{":
            return self.sub(self.raw_group(), size, style)
        if c != "\\":
            self.i += 1
            if style == "math":
                if c == "-":
                    c = "−"
                elif c == "*":
                    c = "∗"
                elif c == "'":
                    c = "′"
            return self.text(c, style)
        self.i += 1
        name = self.command()
        if name in SPACES:
            return Space(SPACES[name] * size)
        if name in GREEK:
            return self.text(GREEK[name], style if style != "math" else "math")
        if name in SYMBOLS:
            return Glyphs(SYMBOLS[name], size, self.bold if style != "bf" else True, False)
        if name in FUNCTIONS:
            return Glyphs(name, size, self.bold, False)
        if name in ("text", "mathrm", "operatorname", "textrm", "mbox", "mathsf"):
            return self.sub(self.raw_group(), size, "text")
        if name in ("mathbf", "textbf", "boldsymbol", "bm"):
            return self.sub(self.raw_group(), size, "bf")
        if name in ("mathit", "textit"):
            return Glyphs(self.raw_group(), size, self.bold, True)
        if name in ("mathbb", "mathds"):
            return self.sub(self.raw_group(), size, "bb")
        if name == "frac" or name == "dfrac" or name == "tfrac":
            num = self.sub(self.raw_group(), size * FRAC, style)
            den = self.sub(self.raw_group(), size * FRAC, style)
            return Frac(num, den, size)
        if name == "sqrt":
            return Sqrt(self.sub(self.raw_group(), size, style), size)
        if name in ("overline",):
            return Over(self.sub(self.raw_group(), size, style), size, "bar")
        if name in ACCENTS:
            body = self.sub(self.raw_group(), size, style)
            inner = body.kids[0] if isinstance(body, HBox) and len(body.kids) == 1 else body
            return Over(inner, size, ACCENTS[name])
        if name in ("left", "right", "big", "Big", "bigl", "bigr", "displaystyle", "mathcal", "limits"):
            return Space(0)                     # commandes de mise en forme ignorées
        return Glyphs("\\" + name, size, self.bold, False)     # inconnue : affichée telle quelle


def _split(line):
    """Découpe une ligne en morceaux (texte, est_formule) ; « \\$ » = dollar littéral."""
    parts, buf, math, i = [], "", False, 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "$":
            buf += "\\$" if math else "$"
            i += 2
            continue
        if ch == "$":
            parts.append((buf, math))
            buf, math = "", not math
            i += 1
            continue
        buf += ch
        i += 1
    if math:                                    # $ non refermé : affiché tel quel
        parts.append(("$" + buf, False))
    else:
        parts.append((buf, False))
    return [(t, m) for t, m in parts if t]


class Layout:
    def __init__(self, lines, line_h, ascs):
        self.lines, self.line_h, self.ascs = lines, line_h, ascs
        self.w = max([ln.w for ln in lines] or [0.0])
        self.h = sum(line_h)


@lru_cache(maxsize=512)
def _layout(text, size, bold, italic, family):
    fm = QFontMetricsF(CFG.font(size, bold, italic))
    lines, hs, ascs = [], [], []
    for raw in text.split("\n"):
        kids = []
        for chunk, math in _split(raw):
            if math:
                kids.append(_Parser(chunk, float(size), bold).parse())
            else:
                kids.append(Glyphs(chunk, size, bold, italic))
        box = HBox(kids)
        asc = max(fm.ascent(), box.asc)
        desc = max(fm.descent(), box.desc)
        lines.append(box)
        ascs.append(asc)
        hs.append(asc + desc + max(0.0, fm.lineSpacing() - fm.height()))
    return Layout(lines, hs, ascs)


def layout(text, size, bold=False, italic=False):
    return _layout(text or "", int(size), bool(bold), bool(italic), CFG.font_family())


def measure(text, size, bold=False, italic=False, wrap_width=None):
    """(largeur, hauteur) du texte, formules comprises."""
    if not has_markup(text):
        fm = QFontMetricsF(CFG.font(size, bold, italic))
        flags = int(Qt.AlignLeft | Qt.AlignTop)
        r = fm.boundingRect(QRectF(0, 0, 1e5, 1e5), flags, text or " ")
        return r.width(), r.height()
    lay = layout(text, size, bold, italic)
    return lay.w, lay.h


def draw(p, rect, text, size, bold=False, italic=False, halign=Qt.AlignHCenter, wrap=False):
    """Dessine le texte centré verticalement dans rect (pinceau courant = couleur du texte)."""
    if not has_markup(text):
        p.setFont(CFG.font(size, bold, italic))
        flags = int(halign | Qt.AlignVCenter | (Qt.TextWordWrap if wrap else 0))
        p.drawText(rect, flags, text)
        return
    lay = layout(text, size, bold, italic)
    y = rect.center().y() - lay.h / 2
    for box, h, asc in zip(lay.lines, lay.line_h, lay.ascs):
        if halign & Qt.AlignLeft:
            x = rect.left()
        elif halign & Qt.AlignRight:
            x = rect.right() - box.w
        else:
            x = rect.center().x() - box.w / 2
        box.draw(p, x, y + asc)
        y += h
