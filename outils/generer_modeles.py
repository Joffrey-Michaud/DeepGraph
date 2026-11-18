"""Génère les schémas de modèles types (menu Fichier > Modèles types).

    python outils/generer_modeles.py            # écrit modeles/*.dgraph + modeles/index.json
    python outils/generer_modeles.py --png DIR  # + un aperçu PNG de chaque modèle dans DIR

Chaque modèle est une fonction `build_xxx(d)` qui dessine avec les petites aides de `Draw`.
Les blocs viennent de blocks.yaml quand c'est possible (leur description s'affiche alors
dans Propriétés) ; les blocs propres à un article sont des blocs libres.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF as P  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

from deepgraph.arrow import ArrowItem, Endpoint  # noqa: E402
from deepgraph.config import CFG  # noqa: E402
from deepgraph.export import export_scene  # noqa: E402
from deepgraph.items import BlockItem, JunctionItem, TextItem  # noqa: E402
from deepgraph.scene import DiagramScene  # noqa: E402

CFG.load()
OUT = os.path.join(ROOT, "modeles")

# ancres usuelles (fraction x, fraction y)
T, B, L, R, C = (.5, 0), (.5, 1), (0, .5), (1, .5), (.5, .5)

# couleurs (cohérentes avec blocks.yaml)
CONV, POOL, FC, NORM, ACT = "#CFE8CC", "#E4E4E4", "#DCDDF2", "#F2F5C4", "#FFE3C4"
IO, ATT, FF, REC = "#F0F0F0", "#FCDDB8", "#C6E6F5", "#CDEBE6"
STAGE, FILL = "#D6E6F7", "#EEF2F3"
GREY = "#555555"


class Draw:
    def __init__(self):
        self.sc = DiagramScene()

    # ------------------------------------------------------------- éléments
    def _place(self, it, x, y):
        old, self.sc.suspend_snap = self.sc.suspend_snap, True
        it.setPos(P(x - it.w / 2, y - it.h / 2))
        self.sc.suspend_snap = old
        return it

    def lib(self, bid, x, y, text=None, w=None, h=None, color=None, stack=None, size=None):
        """Bloc de la bibliothèque (garde sa description)."""
        assert bid in CFG.blocks, bid
        it = self.sc.create_from_def(bid, P(x, y))
        self._tune(it, text, w, h, color, stack, size)
        return self._place(it, x, y)

    def blk(self, text, x, y, color="#FFFFFF", w=None, h=None, stack=None, size=None):
        """Bloc libre."""
        it = BlockItem(text, color, w, h)
        self.sc.add_node(it, P(x, y))
        self._tune(it, None, None, None, None, stack, size)
        return self._place(it, x, y)

    @staticmethod
    def _tune(it, text, w, h, color, stack, size):
        if text is not None:
            it.set_prop("text", text)
        if w or h:
            it.set_size(w or it.w, h or it.h)
        if color:
            it.set_prop("color", color)
        if stack is not None:
            it.set_prop("stack", stack)
        if size:
            it.set_prop("font_size", size)

    def op(self, sym, x, y):
        it = self.sc.create_from_def("op_" + sym, P(x, y))
        return self._place(it, x, y)

    def junc(self, x, y):
        it = JunctionItem()
        self.sc.add_node(it, P(x, y))
        return self._place(it, x, y)

    def txt(self, t, x, y, size=13, color="#000000", italic=False, bold=False, align="center"):
        it = TextItem(t)
        it.font_size, it.color, it.italic, it.bold, it.align = size, color, italic, bold, align
        it.on_props_changed()
        self.sc.add_node(it, P(x, y))
        self._place(it, x, y)
        if align == "left":                      # x = bord gauche
            self._place(it, x + it.w / 2, y)
        elif align == "right":                   # x = bord droit
            self._place(it, x - it.w / 2, y)
        return it

    def title(self, name, ref, x, y):
        self.txt(name, x, y, 16, bold=True)
        self.txt(ref, x, y + 22, 10, GREY, italic=True)

    # --------------------------------------------------------------- flèches
    def ar(self, a, b, sa=B, sb=T, style="straight", first="auto", mid=None, head="end",
           dashed=False, color=None):
        """Flèche de a vers b ; a / b = élément (ancre sa / sb) ou point libre (QPointF)."""
        e1 = Endpoint(point=a) if isinstance(a, P) else Endpoint(a, *sa)
        e2 = Endpoint(point=b) if isinstance(b, P) else Endpoint(b, *sb)
        x = ArrowItem(e1, e2, style)
        x.first, x.mid, x.head, x.dashed = first, mid, head, dashed
        if color:
            x.color = color
        x.uid = self.sc.new_uid()
        self.sc.addItem(x)
        x.update_path()
        return x

    def chain(self, nodes, up=False, **kw):
        sa, sb = (T, B) if up else (B, T)
        for a, b in zip(nodes, nodes[1:]):
            self.ar(a, b, sa, sb, **kw)

    def col(self, specs, x, y0, dy, up=False, link=True):
        """Colonne de blocs ; spec = (bid ou None, texte, options). Renvoie les éléments."""
        out, y = [], y0
        for spec in specs:
            bid, text, kw = (list(spec) + [{}])[:3]
            if bid is None:
                out.append(self.blk(text, x, y, **kw))
            else:
                out.append(self.lib(bid, x, y, text, **kw))
            y += -dy if up else dy
        if link:
            self.chain(out, up=up)
        return out

    def group(self, items, label="", pos="none", **props):
        sc = self.sc
        sc.clearSelection()
        for i in items:
            i.setSelected(True)
        g = sc.group_selected()
        g.set_prop("label", label)
        g.set_prop("label_pos", pos)
        for k, v in props.items():
            g.set_prop(k, v)
        sc.clearSelection()
        return g

    def side_note(self, item, text, dx=14, size=10):
        r = item.scene_frame()
        return self.txt(text, r.right() + dx, r.center().y(), size, GREY, align="left")

    def residual(self, src, dst, side=-1, off=70, sa=C, sb=None):
        """Connexion résiduelle : part horizontalement d'un côté puis rejoint dst par le côté."""
        sb = sb or (L if side < 0 else R)
        return self.ar(src, dst, sa, sb, "ortho2", first="h", mid=side * off)

    def legend_arrow(self, x, y, color, text, dashed=False):
        self.ar(P(x, y), P(x + 34, y), color=color, dashed=dashed)
        self.txt(text, x + 44, y, 11, align="left")


# ===================================================================== CNN
def build_lenet5(d):
    specs = [
        (None, "Image 32×32", dict(color=IO)),
        ("Conv2D", "Conv 5×5, 6", dict(w=130)),
        ("Avg Pooling", "Sous-échant. 2×2", dict(w=130)),
        ("Conv2D", "Conv 5×5, 16", dict(w=130)),
        ("Avg Pooling", "Sous-échant. 2×2", dict(w=130)),
        ("Conv2D", "Conv 5×5, 120", dict(w=130)),
        ("Linear", "FC 84", dict(w=130, h=28)),
        ("Linear", "FC 10", dict(w=130, h=28)),
        ("Softmax", "Softmax", dict(w=130, h=28)),
    ]
    n = d.col(specs, 0, 0, 50)
    for it, t in zip(n, ["1@32×32", "C1 : 6@28×28", "S2 : 6@14×14", "C3 : 16@10×10",
                         "S4 : 16@5×5", "C5 : 120", "F6 : 84 (tanh)", "10 classes", ""]):
        if t:
            d.side_note(it, t)
    d.txt("activations : tanh", -80, 225, 10, GREY, italic=True, align="right")
    d.title("LeNet-5", "LeCun et al., Gradient-based learning applied to document recognition, Proc. IEEE 1998",
            0, 490)


def build_alexnet(d):
    W = 150
    specs = [
        (None, "Image 227×227×3", dict(color=IO, w=W)),
        ("Conv2D", "Conv 11×11, 96, /4", dict(w=W)),
        ("Local Response Norm", "LRN", dict(w=W)),
        ("Max Pooling", "Max Pool 3×3, /2", dict(w=W)),
        ("Conv2D", "Conv 5×5, 256", dict(w=W)),
        ("Local Response Norm", "LRN", dict(w=W)),
        ("Max Pooling", "Max Pool 3×3, /2", dict(w=W)),
        ("Conv2D", "Conv 3×3, 384", dict(w=W)),
        ("Conv2D", "Conv 3×3, 384", dict(w=W)),
        ("Conv2D", "Conv 3×3, 256", dict(w=W)),
        ("Max Pooling", "Max Pool 3×3, /2", dict(w=W)),
        ("Linear", "FC 4096 + Dropout", dict(w=W, h=28)),
        ("Linear", "FC 4096 + Dropout", dict(w=W, h=28)),
        ("Linear", "FC 1000", dict(w=W, h=28)),
        ("Softmax", "Softmax", dict(w=W, h=28)),
    ]
    n = d.col(specs, 0, 0, 46)
    for i, t in {1: "55×55×96", 3: "27×27×96", 4: "27×27×256", 6: "13×13×256", 7: "13×13×384",
                 8: "13×13×384", 9: "13×13×256", 10: "6×6×256", 14: "1000 classes"}.items():
        d.side_note(n[i], t)
    d.txt("ReLU après chaque\nconvolution et FC", -95, 180, 10, GREY, italic=True, align="right")
    d.title("AlexNet", "Krizhevsky, Sutskever & Hinton, ImageNet classification with deep CNNs, NeurIPS 2012",
            0, 14 * 46 + 60)


def build_vgg16(d):
    W, DY = 140, 40
    y = 0
    inp = d.blk("Image 224×224×3", 0, y, IO, w=W)
    prev, last = inp, inp
    y += DY + 6
    stages = [(64, 2, "224×224"), (128, 2, "112×112"), (256, 3, "56×56"), (512, 3, "28×28"), (512, 3, "14×14")]
    for k, (ch, reps, size) in enumerate(stages, 1):
        convs = []
        for _ in range(reps):
            convs.append(d.lib("Conv2D", 0, y, f"Conv 3×3, {ch}", w=W))
            y += DY
        pool = d.lib("Max Pooling", 0, y, "Max Pool 2×2", w=W)
        y += DY + 14
        items = convs + [pool]
        d.chain([last] + items)
        last = pool
        g = d.group(items, f"Bloc {k}", "left", fill=FILL, margin=7, label_size=11)
        d.txt(size, g.scene_frame().right() + 12, g.scene_frame().center().y(), 10, GREY, align="left")
    fcs = d.col([("Linear", "FC 4096", dict(w=W, h=28)), ("Linear", "FC 4096", dict(w=W, h=28)),
                 ("Linear", "FC 1000", dict(w=W, h=28)), ("Softmax", "Softmax", dict(w=W, h=28))], 0, y, DY)
    d.ar(last, fcs[0])
    d.txt("Conv : pas 1, padding 1, ReLU\nFC 4096 : ReLU + Dropout", 0, y + 4 * DY + 10, 10, GREY, italic=True)
    d.title("VGG-16", "Simonyan & Zisserman, Very deep convolutional networks for large-scale image "
            "recognition, ICLR 2015", 0, y + 4 * DY + 60)


def build_resnet18(d):
    W = 160
    specs = [
        (None, "Image 224×224×3", dict(color=IO, w=W)),
        ("Conv2D", "Conv 7×7, 64, /2", dict(w=W)),
        (None, "BN + ReLU", dict(color=NORM, w=W)),
        ("Max Pooling", "Max Pool 3×3, /2", dict(w=W)),
    ]
    top = d.col(specs, 0, 0, 46)
    y = 3 * 46 + 66
    stages = []
    for name, ch, down in (("conv2_x", 64, ""), ("conv3_x", 128, ", /2"), ("conv4_x", 256, ", /2"),
                           ("conv5_x", 512, ", /2")):
        b = d.blk(f"Basic Block\n{ch}{down}", 0, y, STAGE, w=W, h=40, stack=1)
        g = d.group([b], "2×", "left", fill=FILL, margin=8)
        d.txt(name, g.scene_frame().right() + 12, y, 10, GREY, align="left")
        stages.append(b)
        y += 74
    tail = d.col([("Avg Pooling", "Avg Pool global", dict(w=W)), ("Linear", "FC 1000", dict(w=W, h=28)),
                  ("Softmax", "Softmax", dict(w=W, h=28))], 0, y - 8, 44)
    d.chain([top[-1]] + stages + [tail[0]])

    # détail du bloc résiduel
    X, Y = 330, 120
    x_in = d.junc(X, Y)
    det = d.col([("Conv2D", "Conv 3×3, C", dict(w=130)), ("Batch Norm", "Batch Norm", dict(w=130)),
                 ("ReLU", "ReLU", dict(w=130)), ("Conv2D", "Conv 3×3, C", dict(w=130)),
                 ("Batch Norm", "Batch Norm", dict(w=130))], X, Y + 40, 44)
    plus = d.op("plus", X, Y + 40 + 4 * 44 + 40)
    relu = d.lib("ReLU", X, Y + 40 + 4 * 44 + 80, w=130)
    d.ar(P(X, Y - 40), x_in, sb=C, head="none")
    d.txt("x", X - 10, Y - 34, 13, italic=True)
    d.ar(x_in, det[0], sa=C)
    d.ar(det[-1], plus)
    d.ar(plus, relu)
    d.ar(relu, P(X, Y + 40 + 4 * 44 + 120))
    d.residual(x_in, plus, side=+1, off=105, sb=R)
    d.txt("identité x\n(Conv 1×1, /2\nsi sous-échant.)", X + 115, Y + 150, 10, GREY, italic=True, align="left")
    d.group([x_in] + det + [plus, relu], "Basic Block", "top", fill="#FFFFFF", border_style="dashed",
            margin=16, margin_right=110, label_size=13)
    d.title("ResNet-18", "He, Zhang, Ren & Sun, Deep residual learning for image recognition, CVPR 2016",
            150, y + 3 * 44 + 30)


def build_inception(d):
    xs = [-210, -70, 70, 210]
    prev = d.blk("Couche précédente", 0, 0, IO, w=560)
    r1 = [d.lib("Conv 1×1", xs[0], 150, "Conv 1×1"), d.lib("Conv 1×1", xs[1], 80, "Conv 1×1"),
          d.lib("Conv 1×1", xs[2], 80, "Conv 1×1"), d.lib("Max Pooling", xs[3], 80, "Max Pool 3×3")]
    r2 = [None, d.lib("Conv 3×3", xs[1], 150, "Conv 3×3"), d.lib("Conv2D", xs[2], 150, "Conv 5×5"),
          d.lib("Conv 1×1", xs[3], 150, "Conv 1×1")]
    cat = d.lib("Concat", 0, 230, "Concaténation des filtres", w=560, h=28)
    for i, x in enumerate(xs):
        f = (x + 280) / 560
        d.ar(prev, r1[i], (f, 1), T)
        last = r1[i]
        if r2[i] is not None:
            d.ar(r1[i], r2[i])
            last = r2[i]
        d.ar(last, cat, B, (f, 0))
    d.ar(cat, P(0, 290))
    d.txt("(les Conv 1×1 avant 3×3 / 5×5 réduisent la dimension ; ReLU après chaque conv.)",
          0, 315, 10, GREY, italic=True)
    d.title("Module Inception (GoogLeNet)", "Szegedy et al., Going deeper with convolutions, CVPR 2015", 0, 355)


def build_unet(d):
    chans = [64, 128, 256, 512]
    DX, DY, W = 75, 95, 150
    enc, dec = [], []
    for l, c in enumerate(chans):
        enc.append(d.lib("Conv2D", -330 + DX * l, DY * l, f"2× (Conv 3×3 + ReLU)\n{c} canaux", w=W, h=40))
        dec.append(d.lib("Conv2D", 330 - DX * l, DY * l, f"2× (Conv 3×3 + ReLU)\n{c} canaux", w=W, h=40))
    bott = d.lib("Conv2D", 0, DY * 4, "2× (Conv 3×3 + ReLU)\n1024 canaux", w=W, h=40)
    RED, GREEN, GRAY, TEAL = "#C0392B", "#2E8B57", "#8A8A8A", "#2A7F9E"
    inp = d.blk("Image d'entrée", -330, -80, IO, w=W)
    d.ar(inp, enc[0])
    downs = enc[1:] + [bott]
    ups = dec[1:] + [bott]
    for l in range(4):
        if l < 3:
            d.ar(enc[l], downs[l], B, T, "ortho2", first="v", color=RED)
            d.ar(ups[l], dec[l], T, B, "ortho2", first="v", color=GREEN)
        else:                                    # bas du U : entrée / sortie par les côtés
            d.ar(enc[l], bott, B, L, "ortho1", first="v", color=RED)
            d.ar(bott, dec[l], R, B, "ortho1", first="h", color=GREEN)
        d.ar(enc[l], dec[l], R, L, color=GRAY, dashed=True)
    head = d.lib("Conv 1×1", 330, -70, "Conv 1×1", w=W)
    out = d.blk("Carte de segmentation", 330, -140, IO, w=W)
    d.ar(dec[0], head, T, B, color=TEAL)
    d.ar(head, out, T, B)
    lx, ly = -330, DY * 4 + 80
    d.legend_arrow(lx, ly, RED, "Max Pool 2×2 (sous-échantillonnage)")
    d.legend_arrow(lx, ly + 22, GREEN, "Up-conv 2×2 (sur-échantillonnage)")
    d.legend_arrow(lx, ly + 44, GRAY, "copie + concaténation (skip)", dashed=True)
    d.legend_arrow(lx, ly + 66, TEAL, "Conv 1×1 vers les classes")
    d.title("U-Net", "Ronneberger, Fischer & Brox, U-Net: convolutional networks for biomedical image "
            "segmentation, MICCAI 2015", 0, ly + 115)


# ============================================================== Transformers
def _attention_inputs(d, j, att, fracs=(.2, .5, .8)):
    for f in fracs:
        if abs(f - .5) < 1e-6:
            d.ar(j, att, C, (f, 1))
        else:
            d.ar(j, att, C, (f, 1), "ortho1", first="h")


def build_transformer(d):
    XE, XD = -150, 150
    W = 150
    # ---------------------------------------------------------------- encodeur
    d.txt("Inputs", XE, 0, 13)
    emb = d.lib("Input Embedding", XE, -50, w=W)
    pe = d.op("plus", XE, -98)
    pos = d.lib("Positional Encoding", XE - 140, -98, "Positional\nEncoding", w=90)
    d.ar(P(XE, -12), emb, sb=B)
    d.ar(emb, pe, T, B)
    d.ar(pos, pe, R, L)
    j1 = d.junc(XE, -135)
    mha = d.lib("Multi-Head Attention", XE, -180, "Multi-Head\nAttention", w=W)
    an1 = d.lib("Add & Norm", XE, -232, w=W)
    j2 = d.junc(XE, -262)
    ffn = d.lib("Feed Forward", XE, -300, w=W)
    an2 = d.lib("Add & Norm", XE, -350, w=W)
    d.ar(pe, j1, T, C, head="none")
    _attention_inputs(d, j1, mha)
    d.ar(mha, an1, T, B)
    d.ar(an1, j2, T, C, head="none")
    d.ar(j2, ffn, C, B)
    d.ar(ffn, an2, T, B)
    d.residual(j1, an1, -1, 95)
    d.residual(j2, an2, -1, 95)
    d.group([j1, mha, an1, j2, ffn, an2], "N×", "left", margin=12, margin_left=24, margin_top=14,
            margin_bottom=4)
    # ---------------------------------------------------------------- décodeur
    d.txt("Outputs\n(shifted right)", XD, -5, 13)
    emb2 = d.lib("Output Embedding", XD, -50, w=W)
    pe2 = d.op("plus", XD, -98)
    pos2 = d.lib("Positional Encoding", XD + 140, -98, "Positional\nEncoding", w=90)
    d.ar(P(XD, -24), emb2, sb=B)
    d.ar(emb2, pe2, T, B)
    d.ar(pos2, pe2, L, R)
    k1 = d.junc(XD, -135)
    mmha = d.lib("Masked Multi-Head Attention", XD, -185, w=W)
    dan1 = d.lib("Add & Norm", XD, -240, w=W)
    k2 = d.junc(XD, -275)
    mha2 = d.lib("Multi-Head Attention", XD, -425, "Multi-Head\nAttention", w=W)
    dan2 = d.lib("Add & Norm", XD, -477, w=W)
    k3 = d.junc(XD, -507)
    ffn2 = d.lib("Feed Forward", XD, -545, w=W)
    dan3 = d.lib("Add & Norm", XD, -595, w=W)
    d.ar(pe2, k1, T, C, head="none")
    _attention_inputs(d, k1, mmha)
    d.ar(mmha, dan1, T, B)
    d.ar(dan1, k2, T, C, head="none")
    d.ar(k2, mha2, C, (.8, 1), "ortho1", first="h")
    d.ar(mha2, dan2, T, B)
    d.ar(dan2, k3, T, C, head="none")
    d.ar(k3, ffn2, C, B)
    d.ar(ffn2, dan3, T, B)
    d.residual(k1, dan1, +1, 95)
    d.residual(k2, dan2, +1, 95)
    d.residual(k3, dan3, +1, 95)
    # sortie de l'encodeur -> K, V de l'attention croisée
    je = d.junc(XE, -385)
    d.ar(an2, je, T, C, head="none")
    d.ar(je, mha2, C, (.2, 1), "ortho1", first="h")
    d.ar(je, mha2, C, (.45, 1), "ortho1", first="h")
    d.group([k1, mmha, dan1, k2, mha2, dan2, k3, ffn2, dan3], "N×", "right",
            margin=12, margin_right=24, margin_top=14, margin_bottom=4)
    # ------------------------------------------------------------------ sortie
    lin = d.lib("Linear", XD, -660, w=W)
    sm = d.lib("Softmax", XD, -705, w=W)
    d.ar(dan3, lin, T, B)
    d.ar(lin, sm, T, B)
    d.ar(sm, P(XD, -740), T)
    d.txt("Output\nProbabilities", XD, -765, 13)
    d.title("Transformer", "Vaswani et al., Attention is all you need, NeurIPS 2017", 0, 60)


def build_attention(d):
    # ------------------------------------------- Scaled Dot-Product Attention
    X = -170
    for t, x in (("Q", X - 30), ("K", X + 10), ("V", X + 60)):
        d.txt(t, x, 0, 13)
    mm1 = d.lib("MatMul", X - 10, -50)
    sc_ = d.lib("Scale", X - 10, -95)
    mk = d.lib("Mask (opt.)", X - 10, -140)
    sm = d.lib("softmax_attention", X - 10, -185)
    mm2 = d.lib("MatMul", X + 20, -240)
    d.ar(P(X - 30, -12), mm1, sb=(.25, 1))
    d.ar(P(X + 10, -12), mm1, sb=(.75, 1))
    d.chain([mm1, sc_, mk, sm], up=True)
    d.ar(sm, mm2, T, (.25, 1), "ortho1", first="v")
    d.ar(P(X + 60, -12), mm2, sb=(.75, 1), style="ortho1", first="v")
    d.ar(mm2, P(X + 20, -285), T)
    d.txt("Scaled Dot-Product Attention", X + 10, -310, 14, bold=True)
    d.txt("Attention(Q, K, V) = softmax(QKᵀ / √d_k) V", X + 10, 40, 11, GREY, italic=True)
    # ----------------------------------------------------- Multi-Head Attention
    X = 170
    for t, x in (("V", X - 70), ("K", X), ("Q", X + 70)):
        d.txt(t, x, 0, 13)
    lins = [d.lib("linear_proj", x, -55) for x in (X - 70, X, X + 70)]
    for x, li in zip((X - 70, X, X + 70), lins):
        d.ar(P(x, -12), li, sb=B)
    sdpa = d.lib("Scaled Dot-Product Attention", X, -125, "Scaled Dot-Product\nAttention")
    for li, f in zip(lins, (.2, .5, .8)):
        d.ar(li, sdpa, T, (f, 1), "ortho1", first="v")
    cat = d.lib("Concat", X, -190)
    lin = d.lib("Linear", X, -235, w=80)
    d.ar(sdpa, cat, T, B)
    d.ar(cat, lin, T, B)
    d.ar(lin, P(X, -275), T)
    d.txt("h", sdpa.scene_frame().right() + 14, -125, 13, italic=True)
    d.txt("Multi-Head Attention", X, -310, 14, bold=True)
    d.txt("MultiHead(Q, K, V) = Concat(head₁, …, head_h) Wᴼ", X, 40, 11, GREY, italic=True)
    d.title("Attention", "Vaswani et al., Attention is all you need, NeurIPS 2017 (fig. 2)", 0, 90)


def build_vit(d):
    X, W = 0, 160
    d.txt("Image → patchs 16×16", X, 0, 13)
    pe = d.lib("Patch Embedding", X, -50, "Linear Projection of\nFlattened Patches", w=W)
    plus = d.op("plus", X, -105)
    pos = d.blk("[class] token +\nPosition Embedding", X - 160, -105, "#E9E3F3", w=130, h=40)
    d.ar(P(X, -12), pe, sb=B)
    d.ar(pe, plus, T, B)
    d.ar(pos, plus, R, L)
    j1 = d.junc(X, -135)
    ln1 = d.lib("Layer Norm", X, -170, "Norm", w=W)
    mha = d.lib("Multi-Head Attention", X, -220, "Multi-Head\nAttention", w=W)
    a1 = d.op("plus", X, -270)
    j2 = d.junc(X, -295)
    ln2 = d.lib("Layer Norm", X, -330, "Norm", w=W)
    mlp = d.lib("MLP", X, -380, w=W)
    a2 = d.op("plus", X, -430)
    d.ar(plus, j1, T, C, head="none")
    d.ar(j1, ln1, C, B)
    d.ar(ln1, mha, T, B)
    d.ar(mha, a1, T, B)
    d.ar(a1, j2, T, C, head="none")
    d.ar(j2, ln2, C, B)
    d.ar(ln2, mlp, T, B)
    d.ar(mlp, a2, T, B)
    d.residual(j1, a1, +1, 110, sb=R)
    d.residual(j2, a2, +1, 110, sb=R)
    d.group([j1, ln1, mha, a1, j2, ln2, mlp, a2], "L×", "right", margin=12, margin_right=30,
            margin_top=14)
    head = d.lib("MLP", X, -500, "MLP Head", w=W)
    d.ar(a2, head, T, B)
    d.ar(head, P(X, -540), T)
    d.txt("Classe", X, -555, 13)
    d.txt("Transformer Encoder\n(pre-norm)", X - 175, -300, 11, GREY, italic=True)
    d.title("Vision Transformer (ViT)", "Dosovitskiy et al., An image is worth 16×16 words: Transformers "
            "for image recognition at scale, ICLR 2021", 0, 50)


# ================================================================ autres
def build_mlp(d):
    W = 140
    specs = [(None, "Entrée x ∈ ℝᵈ", dict(color=IO, w=W)), ("Linear", "Linear d → 256", dict(w=W, h=28)),
             ("ReLU", "ReLU", dict(w=W)), ("Dropout", "Dropout", dict(w=W)),
             ("Linear", "Linear 256 → 256", dict(w=W, h=28)), ("ReLU", "ReLU", dict(w=W)),
             ("Linear", "Linear 256 → K", dict(w=W, h=28)), ("Softmax", "Softmax", dict(w=W, h=28)),
             (None, "Probabilités (K classes)", dict(color=IO, w=W + 30))]
    n = d.col(specs, 0, 0, 44)
    d.group(n[1:4], "couche cachée", "left", fill=FILL, margin=7, label_size=11)
    d.group(n[4:6], "couche cachée", "left", fill=FILL, margin=7, label_size=11)
    d.title("Perceptron multicouche (MLP)", "Rumelhart, Hinton & Williams, Learning representations by "
            "back-propagating errors, Nature 1986", 0, 8 * 44 + 50)


def build_seq2seq(d):
    DX = 105
    enc = [d.lib("LSTM", -330 + DX * i, 0, w=80) for i in range(3)]
    dec = [d.lib("LSTM", 30 + DX * i + 30, 0, w=80) for i in range(3)]
    for i, c in enumerate(enc):
        x = c.scene_frame().center().x()
        d.txt(f"x{'₁₂₃'[i]}", x, 75, 13, italic=True)
        emb = d.lib("Embedding", x, 45, "Emb.", w=60, h=22)
        d.ar(P(x, 63), emb, sb=B)
        d.ar(emb, c, T, B)
    toks = ["<s>", "y₁", "y₂"]
    for i, c in enumerate(dec):
        x = c.scene_frame().center().x()
        d.txt(toks[i], x, 75, 13, italic=True)
        emb = d.lib("Embedding", x, 45, "Emb.", w=60, h=22)
        d.ar(P(x, 63), emb, sb=B)
        d.ar(emb, c, T, B)
        sm = d.lib("Linear", x, -50, "Linear +\nSoftmax", w=70, h=34)
        d.ar(c, sm, T, B)
        d.ar(sm, P(x, -90), T)
        d.txt(f"y{'₁₂₃'[i]}", x, -102, 13, italic=True)
    for a, b in zip(enc + dec, (enc + dec)[1:]):
        d.ar(a, b, R, L)
    d.txt("h, c", (enc[-1].scene_frame().right() + dec[0].scene_frame().left()) / 2, -12, 11, GREY,
          italic=True)
    d.group(enc, "Encodeur", "top", fill=FILL, margin=10, label_size=12)
    d.group(dec, "Décodeur", "right", fill=FILL, margin=10, label_size=12)
    d.txt("en génération, la sortie y_t est réinjectée en entrée au pas t+1", 0, 110, 10, GREY,
          italic=True)
    d.title("Seq2seq (LSTM encodeur–décodeur)", "Sutskever, Vinyals & Le, Sequence to sequence learning "
            "with neural networks, NeurIPS 2014", 0, 150)


def build_vae(d):
    x = d.blk("x", -300, 0, IO, w=60)
    enc = d.lib("Encoder", -190, 0, "Encodeur\nq_φ(z | x)", w=110)
    mu = d.lib("Linear", -50, -35, "μ", w=60, h=26)
    lv = d.lib("Linear", -50, 35, "log σ²", w=60, h=26)
    samp = d.lib("Sampling", 85, 0, "z = μ + σ ⊙ ε", w=120)
    eps = d.lib("Noise", 85, 75, "ε ~ N(0, I)", w=100)
    dec = d.lib("Decoder", 230, 0, "Décodeur\np_θ(x | z)", w=110)
    out = d.blk("x'", 340, 0, IO, w=60)
    d.ar(x, enc, R, L)
    d.ar(enc, mu, R, L, "ortho2", first="h")
    d.ar(enc, lv, R, L, "ortho2", first="h")
    d.ar(mu, samp, R, (0, .3), "ortho2", first="h")
    d.ar(lv, samp, R, (0, .7), "ortho2", first="h")
    d.ar(eps, samp, T, B)
    d.ar(samp, dec, R, L)
    d.ar(dec, out, R, L)
    d.txt("astuce de reparamétrisation", 85, -38, 10, GREY, italic=True)
    d.txt("Perte = ‖x − x'‖² (reconstruction) + KL(q_φ(z | x) ‖ N(0, I))", 20, 130, 11, GREY, italic=True)
    d.title("Autoencodeur variationnel (VAE)", "Kingma & Welling, Auto-encoding variational Bayes, ICLR 2014",
            20, 175)


def build_gan(d):
    z = d.lib("Noise", -300, -50, "z ~ N(0, I)", w=100)
    gen = d.lib("Generator", -160, -50, "Générateur G", w=120)
    fake = d.blk("G(z)\néchantillon généré", -10, -50, IO, w=130, h=40)
    real = d.blk("x\ndonnée réelle", -10, 50, IO, w=130, h=40)
    disc = d.lib("Discriminator", 150, 0, "Discriminateur D", w=130)
    out = d.lib("Sigmoid", 280, 0, w=80)
    d.ar(z, gen, R, L)
    d.ar(gen, fake, R, L)
    d.ar(fake, disc, R, (0, .3), "ortho2", first="h")
    d.ar(real, disc, R, (0, .7), "ortho2", first="h")
    d.ar(disc, out, R, L)
    d.ar(out, P(370, 0), R)
    d.txt("réel / généré", 380, 0, 12, align="left")
    d.txt("min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]", 30, 115, 11, GREY, italic=True)
    d.title("GAN", "Goodfellow et al., Generative adversarial nets, NeurIPS 2014", 30, 160)


MODELS = [
    # (fichier, titre du menu, catégorie, référence, fonction)
    ("lenet5", "LeNet-5", "Réseaux convolutifs", "LeCun et al., 1998", build_lenet5),
    ("alexnet", "AlexNet", "Réseaux convolutifs", "Krizhevsky et al., 2012", build_alexnet),
    ("vgg16", "VGG-16", "Réseaux convolutifs", "Simonyan & Zisserman, 2015", build_vgg16),
    ("inception", "Module Inception (GoogLeNet)", "Réseaux convolutifs", "Szegedy et al., 2015",
     build_inception),
    ("resnet18", "ResNet-18", "Réseaux convolutifs", "He et al., 2016", build_resnet18),
    ("unet", "U-Net", "Réseaux convolutifs", "Ronneberger et al., 2015", build_unet),
    ("transformer", "Transformer (encodeur–décodeur)", "Attention / Transformers", "Vaswani et al., 2017",
     build_transformer),
    ("attention", "Scaled Dot-Product & Multi-Head Attention", "Attention / Transformers",
     "Vaswani et al., 2017", build_attention),
    ("vit", "Vision Transformer (ViT)", "Attention / Transformers", "Dosovitskiy et al., 2021", build_vit),
    ("vae", "Autoencodeur variationnel (VAE)", "Génératifs", "Kingma & Welling, 2014", build_vae),
    ("gan", "GAN", "Génératifs", "Goodfellow et al., 2014", build_gan),
    ("mlp", "Perceptron multicouche (MLP)", "Autres", "Rumelhart et al., 1986", build_mlp),
    ("seq2seq", "Seq2seq LSTM", "Autres", "Sutskever et al., 2014", build_seq2seq),
]


def main():
    png_dir = None
    if "--png" in sys.argv:
        png_dir = sys.argv[sys.argv.index("--png") + 1]
        os.makedirs(png_dir, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    index = []
    for fname, title, cat, ref, fn in MODELS:
        d = Draw()
        fn(d)
        d.sc.clearSelection()
        d.sc.refit_groups()
        data = d.sc.to_dict()
        path = os.path.join(OUT, fname + ".dgraph")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        index.append({"file": fname + ".dgraph", "title": title, "category": cat, "ref": ref})
        if png_dir:
            export_scene(d.sc, os.path.join(png_dir, fname + ".png"), "png", scale=1.5)
        print(f"{fname:12s} {len(d.sc.nodes()):3d} éléments, {len(d.sc.arrows()):3d} flèches")
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
