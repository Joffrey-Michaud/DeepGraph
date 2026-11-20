#!/usr/bin/env python3
"""Brouillon de figure DeepGraph à partir d'un modèle PyTorch (PROTOTYPE, en évaluation).

Le modèle est exécuté une fois sur des entrées factices ; on observe quels modules s'appellent
et quelles opérations du forward (cat, +, ×…) relient leurs sorties. Le niveau de détail suit la
hiérarchie des modules : --profondeur 1 montre les sous-modules directs du modèle (les
ModuleList / Sequential ne comptent pas comme un niveau), 2 un niveau de plus, etc.

    python outils/depuis_pytorch.py mon_modele.py:MonModele --entree x:1,3,224,224 -o brouillon.dgraph
    python outils/depuis_pytorch.py paquet.module:Classe --args '{"dim": 256}' --entree a:1,16,64 --entree b:1,16,64
    python outils/depuis_pytorch.py torchvision.models:vit_b_16 -e image:1,3,224,224 -p 2 -o vit.dgraph
    python outils/depuis_pytorch.py modele.py:Unet -e x:1,3,256,256 --detail EncoderBlock --detail DecoderBlock -o unet.dgraph

Vue d'ensemble + encarts : --detail Classe dessine l'intérieur de cette classe à droite (préférable à
une grande profondeur, qui mélange tout). Les blocs gardent le nom qualifié du module dans « ref ».

Le résultat est un .dgraph au format simplifié, à ouvrir et retoucher dans l'éditeur.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    from torch.overrides import TorchFunctionMode
except ImportError:  # pragma: no cover
    sys.exit("PyTorch est nécessaire pour cet outil (pip install torch).")

CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.Sequential)
TRANSPARENT_MODULES = (nn.Identity,)
# remises en forme sans calcul : invisibles (la forme suivante le montre)
TRANSPARENT_NAMES = {"Permute", "Transpose", "Reshape", "View", "Rearrange", "Unflatten"}
# régularisation : leur taux change souvent d'un bloc à l'autre (stochastic depth croissante) ; on
# l'ignore pour reconnaître les blocs identiques
REGULARIZERS = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "StochasticDepth",
                "DropPath", "DropBlock2d", "DropBlock3d"}
CONTAINER_NAMES = {"IntermediateLayerGetter"}   # enveloppes qui ne font qu'appeler leurs enfants
NOISE = {"randn_like", "rand_like", "normal", "randn", "rand", "normal_"}


def is_container(m):
    """Liste de modules sans calcul propre : ModuleList / ModuleDict / Sequential exactement (une
    sous-classe avec son propre forward, comme un bloc de DenseNet, est un vrai module)."""
    return type(m) in CONTAINERS or type(m).__name__ in CONTAINER_NAMES


def struct_sig(m):
    """Signature d'un module pour repérer les blocs identiques : classe, réglages et sous-modules,
    sans les taux de régularisation."""
    name = type(m).__name__
    if name in REGULARIZERS:
        return name
    kids = ",".join(f"{k}={struct_sig(c)}" for k, c in m.named_children())
    return f"{name}({m.extra_repr()};{kids})"


# ======================================================================= traçage
class Node:
    def __init__(self, nid, kind, order, **kw):
        self.id, self.kind, self.order = nid, kind, order
        self.label = kw.get("label", "")
        self.cls = kw.get("cls", "")
        self.module = kw.get("module")
        self.parent = kw.get("parent")          # module « déplié » qui contient ce nœud (cadre)
        self.sig = kw.get("sig")                # pour repérer les blocs identiques répétés
        self.op = kw.get("op")
        self.in_shapes, self.out_shape = [], None
        self.repeat = 1


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _tensors(x)


def _named_tensors(obj, key=""):
    """Comme _tensors, avec le nom de chaque tenseur dans un dictionnaire (clé), sinon ""."""
    if isinstance(obj, torch.Tensor):
        yield key, obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _named_tensors(x, key)
    elif isinstance(obj, dict):
        for k, x in obj.items():
            yield from _named_tensors(x, str(k))


OPS = {
    "add": ("__add__", "__radd__", "__iadd__", "add", "add_"),
    "sub": ("__sub__", "__rsub__", "__isub__", "sub", "subtract"),
    "mul": ("__mul__", "__rmul__", "__imul__", "mul", "multiply"),
    "cat": ("cat", "concat", "concatenate", "stack", "hstack", "vstack", "dstack"),
    "matmul": ("matmul", "__matmul__", "bmm", "mm", "einsum", "baddbmm"),
    "act": ("relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "log_softmax", "leaky_relu", "elu",
            "mish", "softplus", "hardswish", "selu"),
    "norm": ("layer_norm", "group_norm", "batch_norm", "instance_norm", "normalize", "rms_norm"),
    "resize": ("interpolate", "upsample", "pixel_shuffle", "avg_pool1d", "avg_pool2d", "max_pool1d",
               "max_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d"),
    "attn": ("scaled_dot_product_attention",),
}
OP_OF = {name: cat for cat, names in OPS.items() for name in names}


class Tracer:
    def __init__(self, model, depth=1, unfold=(), fold=()):
        self.model, self.depth = model, depth
        self.unfold, self.fold = set(unfold), set(fold)
        self.info = {}                          # module -> (nom qualifié, niveau)
        self._walk(model, "", 0)
        self.params = {id(p): n for n, p in model.named_parameters()}
        self.nodes, self.edges = {}, []
        self.producer = {}                      # id(tenseur) -> id du nœud qui l'a produit
        self.keep = []                          # garde les tenseurs en vie (ids stables)
        self.inside = 0                         # > 0 : dans un module affiché comme un bloc
        self.stack = []                         # modules dépliés en cours d'exécution
        self.calls = defaultdict(int)
        self.frames = []                        # piles des hooks (une entrée par appel de module)
        self.groups = {}                        # module déplié -> libellé du cadre
        self.group_sig, self.group_parent = {}, {}

    # ---------------------------------------------------------------- hiérarchie
    def _walk(self, m, qual, level):
        for name, child in m.named_children():
            if child in self.info:
                continue
            q = f"{qual}.{name}" if qual else name
            lv = level if is_container(child) else level + 1
            self.info[child] = (q, lv)
            self._walk(child, q, lv)

    def _role(self, m):
        """« leaf » (un bloc), « open » (déplié : on montre son contenu), « skip » (conteneur)."""
        if m is self.model:
            return "open"
        if is_container(m):
            return "skip"
        q, lv = self.info.get(m, ("?", 99))
        cls = type(m).__name__
        builtin = type(m).__module__.startswith("torch.")
        has_children = any(True for _ in m.children())
        if cls in self.fold or not has_children:
            return "leaf"
        if cls in self.unfold and lv <= self.depth:
            return "open"                          # --deplier : un niveau de plus pour cette classe
        if lv >= self.depth:
            return "leaf"
        if builtin and cls not in self.unfold:
            return "leaf"
        return "open"

    # --------------------------------------------------------------------- nœuds
    def _new(self, nid, kind, **kw):
        n = Node(nid, kind, len(self.nodes), parent=self.stack[-1] if self.stack else None, **kw)
        self.nodes[nid] = n
        return n

    def _src(self, t):
        """Nœud producteur d'un tenseur (paramètre du modèle : nœud créé à la demande)."""
        nid = self.producer.get(id(t))
        if nid is None and id(t) in self.params:
            q = self.params[id(t)]
            nid = "param:" + q
            if nid not in self.nodes:
                self._new(nid, "param", label=q.split(".")[-1]).out_shape = tuple(t.shape)
        return nid

    def _inputs(self, objs):
        out = []
        for t in _tensors(objs):
            nid = self._src(t)
            if nid is not None and nid not in [o[0] for o in out]:
                out.append((nid, tuple(t.shape)))
        return out

    def _bind(self, outputs, nid):
        for t in _tensors(outputs):
            self.producer[id(t)] = nid
            self.keep.append(t)

    def _link(self, ins, node):
        for src, shape in ins:
            self.edges.append((src, node.id))
            node.in_shapes.append(shape)

    # --------------------------------------------------------------------- hooks
    def pre_hook(self, mod, args, kwargs):
        if self.inside:
            self.inside += 1
            self.frames.append(("in", None))
            return
        role = self._role(mod)
        if role == "leaf":
            if isinstance(mod, TRANSPARENT_MODULES) or type(mod).__name__ in TRANSPARENT_NAMES:
                self.frames.append(("transparent", self._inputs((args, kwargs))))
                self.inside = 1
                return
            q = self.info[mod][0]
            self.calls[q] += 1
            nid = q if self.calls[q] == 1 else f"{q}#{self.calls[q]}"
            node = self._new(nid, "module", cls=type(mod).__name__, module=mod, sig=struct_sig(mod))
            self._link(self._inputs((args, kwargs)), node)
            self.inside = 1
            self.frames.append(("leaf", node))
        elif role == "open" and mod is not self.model:
            q = self.info[mod][0]
            if q not in self.groups:
                self.groups[q] = type(mod).__name__
                self.group_sig[q] = struct_sig(mod)
                self.group_parent[q] = self.stack[-1] if self.stack else None
            self.stack.append(q)
            self.frames.append(("open", None))
        else:
            self.frames.append(("skip", None))

    def post_hook(self, mod, args, kwargs, output):
        kind, data = self.frames.pop()
        if kind == "in":
            self.inside -= 1
        elif kind == "transparent":
            self.inside = 0
            if data:
                for t in _tensors(output):
                    self.producer[id(t)] = data[0][0]
                    self.keep.append(t)
        elif kind == "leaf":
            self.inside = 0
            outs = list(_tensors(output))
            if outs:
                data.out_shape = tuple(outs[0].shape)
            self._bind(output, data.id)
        elif kind == "open":
            self.stack.pop()

    # ------------------------------------------------------------- opérations
    def on_op(self, func, args, kwargs, out):
        outs = list(_tensors(out))
        if not outs:
            return
        name = getattr(func, "__name__", "") or ""
        if name in NOISE:
            k = sum(1 for n in self.nodes.values() if n.kind == "noise") + 1
            node = self._new(f"noise{k}", "noise", label="Noise")
            node.out_shape = tuple(outs[0].shape)
            self._bind(out, node.id)
            return
        ins = self._inputs((args, kwargs))
        if not ins:
            return
        cat = OP_OF.get(name)
        if cat in ("add", "sub", "mul", "cat", "matmul") and len(ins) >= 2 \
                or cat in ("act", "norm", "resize", "attn") \
                or (cat is None and len(ins) >= 2 and not name.startswith("__")):
            k = sum(1 for n in self.nodes.values() if n.kind == "op") + 1
            node = self._new(f"op{k}:{name}", "op", op=cat or name, label=name)
            self._link(ins, node)
            node.out_shape = tuple(outs[0].shape)
            self._bind(out, node.id)
        else:                                   # view, transpose, x * 0.5… : transparent
            for t in outs:
                self.producer[id(t)] = ins[0][0]
                self.keep.append(t)

    # ---------------------------------------------------------------- exécution
    def run(self, inputs, call=None):
        """inputs : [(nom, tenseur)] -> nœuds d'entrée. call : (args, kwargs) de l'appel, si ce ne sont
        pas simplement les tenseurs dans l'ordre (détail d'un sous-module)."""
        tracer = self

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                if tracer.inside == 0:
                    tracer.on_op(func, args, kwargs, out)
                return out

        handles = []
        for m in self.model.modules():
            handles.append(m.register_forward_pre_hook(self.pre_hook, with_kwargs=True))
            handles.append(m.register_forward_hook(self.post_hook, with_kwargs=True))
        for name, t in inputs:
            n = self._new("in:" + name, "input", label=name)
            n.out_shape = tuple(t.shape)
            self._bind(t, n.id)
        try:
            with torch.no_grad(), Mode():
                if call is None:
                    result = self.model(*[t for _, t in inputs])
                else:
                    result = self.model(*call[0], **call[1])
        finally:
            for h in handles:
                h.remove()
        named = list(_named_tensors(result))
        outs = [t for _, t in named]
        for i, (key, t) in enumerate(named):
            n = self._new(f"out:{i}", "output",
                          label=key or ("Output" if len(outs) == 1 else f"Output {i + 1}"))
            n.parent = None
            src = self._src(t)
            if src is not None:
                self.edges.append((src, n.id))
                n.in_shapes.append(tuple(t.shape))
                n.out_shape = tuple(t.shape)
        return self.graph()

    def graph(self):
        g = Graph(self.nodes, self.edges, self.groups)
        g.group_sig, g.group_parent = self.group_sig, self.group_parent
        g.containers = {q: type(m).__name__ for m, (q, lv) in self.info.items() if is_container(m)}
        g.prune()
        return g


# ===================================================================== graphe
class Graph:
    def __init__(self, nodes, edges, groups):
        self.nodes = dict(nodes)
        seen, self.edges = set(), []
        for e in edges:
            if e not in seen and e[0] != e[1]:
                seen.add(e)
                self.edges.append(e)
        self.groups = groups
        self.group_sig, self.group_parent, self.group_repeat = {}, {}, {}
        self.containers = {}

    def preds(self, nid):
        return [a for a, b in self.edges if b == nid]

    def succs(self, nid):
        return [b for a, b in self.edges if a == nid]

    def prune(self):
        """Ne garde que ce qui contribue à une sortie (entrées toujours gardées)."""
        alive = set()
        todo = [n for n in self.nodes if self.nodes[n].kind == "output"]
        while todo:
            n = todo.pop()
            if n in alive:
                continue
            alive.add(n)
            todo.extend(self.preds(n))
        alive |= {n for n in self.nodes if self.nodes[n].kind == "input"}
        self.nodes = {k: v for k, v in self.nodes.items() if k in alive}
        self.edges = [e for e in self.edges if e[0] in alive and e[1] in alive]

    def fold_repeats(self):
        """Chaîne de blocs identiques (même classe et mêmes réglages) -> un bloc dans un cadre « N× »."""
        order = sorted(self.nodes.values(), key=lambda n: n.order)
        gone = set()
        for n in order:
            if n.id in gone or n.kind != "module":
                continue
            run = [n]
            while True:
                last = run[-1]
                nxt = self.succs(last.id)
                if len(nxt) != 1:
                    break
                v = self.nodes[nxt[0]]
                if v.kind != "module" or v.sig != n.sig or v.parent != n.parent or self.preds(v.id) != [last.id]:
                    break
                run.append(v)
            if len(run) < 2:
                continue
            first, last = run[0], run[-1]
            first.repeat = len(run)
            first.out_shape = last.out_shape
            for v in run[1:]:
                gone.add(v.id)
            self.edges = [(first.id if a == last.id else a, b) for a, b in self.edges
                          if not (a in gone or b in gone) or a == last.id]
            self.edges = [(a, b) for a, b in self.edges if a not in gone and b not in gone]
            for v in run[1:]:
                del self.nodes[v.id]

    def in_group(self, n, q):
        """Le nœud n est-il (à n'importe quelle profondeur) dans le module déplié q ?"""
        p = self.nodes[n].parent
        while p is not None:
            if p == q:
                return True
            p = self.group_parent.get(p)
        return False

    def members(self, q):
        return {n for n in self.nodes if self.in_group(n, q)}

    def fold_open_repeats(self):
        """Modules dépliés identiques qui se suivent (ex. 6 blocs de Transformer) : on garde le contenu
        du premier, dans un cadre « N× ». Une entrée externe commune à tous (mémoire de l'encodeur,
        encodage du pas de temps…) est permise."""
        changed = True
        while changed:
            changed = False
            quals = [q for q in self.groups if self.members(q)]
            for i, q0 in enumerate(quals):
                sibs = [q for q in quals[i + 1:] if self.group_parent.get(q) == self.group_parent.get(q0)]
                run = [q0]
                for q in sibs:
                    if self.group_sig.get(q) != self.group_sig.get(q0):
                        break
                    S, T = self.members(run[-1]), self.members(q)
                    ext_S = {a for a, b in self.edges if b in S and a not in S}
                    into_T = {a for a, b in self.edges if b in T and a not in T}
                    out_S = {b for a, b in self.edges if a in S and b not in S}
                    if not (into_T - S <= ext_S and out_S <= T | {b for b in out_S if b not in T and
                                                                   not any(self.in_group(b, r) for r in sibs)}):
                        break
                    if not any(b in T for b in out_S):
                        break                      # ne s'enchaînent pas
                    run.append(q)
                if len(run) >= 2 and self._fold_run(run):
                    changed = True
                    break

    def _fold_run(self, run):
        members = [sorted((n for n in self.nodes if self.in_group(n, q)), key=lambda k: self.nodes[k].order)
                   for q in run]
        if len({len(m) for m in members}) != 1:
            return False
        first, last = members[0], members[-1]
        to_first = dict(zip(last, first))
        gone = {n for m in members[1:] for n in m}
        edges = []
        for a, b in self.edges:
            if a in gone and b not in gone:
                if a in to_first:
                    edges.append((to_first[a], b))
            elif a not in gone and b not in gone:
                edges.append((a, b))
        seen = set()
        self.edges = [e for e in edges if not (e in seen or seen.add(e))]
        for n in gone:
            del self.nodes[n]
        self.group_repeat[run[0]] = len(run)
        return True

    def summary(self):
        return {"nodes": [{"id": n.id, "kind": n.kind, "class": n.cls or n.op, "label": n.label,
                           "in_shapes": n.in_shapes, "out_shape": n.out_shape, "repeat": n.repeat,
                           "group": n.parent} for n in sorted(self.nodes.values(), key=lambda n: n.order)],
                "edges": [list(e) for e in self.edges]}


# ============================================================ libellés / style
def _dims(shape):
    s = list(shape or [])[1:]                   # sans la dimension du batch
    return "×".join(str(x) for x in s)


def _k(v):
    return v[0] if isinstance(v, (tuple, list)) and len(set(v)) == 1 else v


def builtin_label(m):
    """(texte, identifiant de bibliothèque) d'un module de torch.nn."""
    c = type(m).__name__
    if c.startswith("ConvTranspose"):
        return f"{c} k{_k(m.kernel_size)}" + (f" s{_k(m.stride)}" if _k(m.stride) != 1 else ""), "Transposed Conv"
    if c in ("Conv1d", "Conv2d", "Conv3d"):
        lib = {"Conv1d": "Conv1D", "Conv2d": "Conv2D", "Conv3d": "Conv3D"}[c]
        txt = f"{c} k{_k(m.kernel_size)}" + (f" s{_k(m.stride)}" if _k(m.stride) != 1 else "")
        if getattr(m, "groups", 1) > 1 and m.groups == m.in_channels:
            return txt + " (depthwise)", "Depthwise Conv"
        return txt, lib
    if c == "Linear":
        return f"Linear {m.in_features} → {m.out_features}", "Linear"
    if c == "MultiheadAttention":
        return f"Multi-Head Attention ({m.num_heads} têtes)", "Multi-Head Attention"
    simple = {"LayerNorm": ("LayerNorm", "Layer Norm"), "BatchNorm1d": ("BatchNorm", "Batch Norm"),
              "BatchNorm2d": ("BatchNorm", "Batch Norm"), "BatchNorm3d": ("BatchNorm", "Batch Norm"),
              "InstanceNorm1d": ("InstanceNorm", "Instance Norm"), "InstanceNorm2d": ("InstanceNorm", "Instance Norm"),
              "RMSNorm": ("RMSNorm", "RMS Norm"), "ReLU": ("ReLU", "ReLU"), "GELU": ("GELU", "GELU"),
              "ReLU6": ("ReLU6", "ReLU"), "PReLU": ("PReLU", "Leaky ReLU"), "Hardswish": ("Hardswish", "SiLU / Swish"),
              "Hardsigmoid": ("Hardsigmoid", "Sigmoid"), "Softplus": ("Softplus", "Activation"),
              "GLU": ("GLU", "Activation"), "LogSoftmax": ("LogSoftmax", "Softmax"),
              "MaxPool3d": ("Max Pooling", "Max Pooling"), "AvgPool3d": ("Avg Pooling", "Avg Pooling"),
              "AdaptiveMaxPool1d": ("Global Max Pooling", "Max Pooling"),
              "AdaptiveMaxPool2d": ("Global Max Pooling", "Max Pooling"),
              "AdaptiveAvgPool3d": ("Global Avg Pooling", "Avg Pooling"), "Unfold": ("Unfold", None),
              "InstanceNorm3d": ("InstanceNorm", "Instance Norm"), "PixelShuffle": ("PixelShuffle", "Upsample"),
              "SiLU": ("SiLU", "SiLU / Swish"), "Sigmoid": ("Sigmoid", "Sigmoid"), "Tanh": ("Tanh", "Tanh"),
              "LeakyReLU": ("Leaky ReLU", "Leaky ReLU"), "ELU": ("ELU", "ELU"), "Mish": ("Mish", "Mish"),
              "Softmax": ("Softmax", "Softmax"), "Dropout": ("Dropout", "Dropout"), "Dropout1d": ("Dropout", "Dropout"),
              "Dropout2d": ("Dropout", "Dropout"), "Flatten": ("Flatten", "Flatten"), "Embedding": ("Embedding", "Embedding"),
              "LSTM": ("LSTM", "LSTM"), "GRU": ("GRU", "GRU"), "RNN": ("RNN", "RNN"), "Upsample": ("Upsample", "Upsample"),
              "MaxPool1d": ("Max Pooling", "Max Pooling"), "MaxPool2d": ("Max Pooling", "Max Pooling"),
              "AvgPool1d": ("Avg Pooling", "Avg Pooling"), "AvgPool2d": ("Avg Pooling", "Avg Pooling"),
              "AdaptiveAvgPool1d": ("Global Avg Pooling", "Avg Pooling"),
              "AdaptiveAvgPool2d": ("Global Avg Pooling", "Avg Pooling"),
              "TransformerEncoderLayer": ("Transformer Encoder Layer", "Encoder"),
              "TransformerDecoderLayer": ("Transformer Decoder Layer", "Decoder")}
    if c == "GroupNorm":
        return f"GroupNorm ({m.num_groups})", "Group Norm"
    if c in simple:
        return simple[c]
    return c, None


COLOR_HINTS = [  # (mots dans le nom de la classe, bloc de bibliothèque dont on prend la couleur)
    (("crossattention", "cross_attention", "crossattn"), "Cross-Attention"),
    (("attention", "attn"), "Multi-Head Attention"),
    (("decoder", "upsample", "upblock", "decod"), "Decoder"),
    (("encoder", "downsample", "downblock", "encod"), "Encoder"),
    (("bottleneck",), "Bottleneck"),
    (("basicblock", "residual", "resblock", "resnet"), "Conv 3×3"),
    (("transformer", "swin"), "Encoder"),
    (("conv", "cnn"), "Conv1D"),
    (("norm",), "Layer Norm"),
    (("mlp", "feedforward", "ffn", "linear", "dense"), "MLP"),
    (("embed", "tokeniz", "patch"), "Embedding"),
    (("pool",), "Max Pooling"),
    (("lstm", "gru", "rnn", "recurrent"), "LSTM"),
    (("generator",), "Generator"), (("discriminator",), "Discriminator"),
]
RESIZE_TEXT = {"max_pool1d": ("Max Pooling", "Max Pooling"), "max_pool2d": ("Max Pooling", "Max Pooling"),
               "avg_pool1d": ("Avg Pooling", "Avg Pooling"), "avg_pool2d": ("Avg Pooling", "Avg Pooling"),
               "adaptive_avg_pool1d": ("Avg Pooling", "Avg Pooling"),
               "adaptive_avg_pool2d": ("Avg Pooling", "Avg Pooling"),
               "interpolate": ("Interpolate", "Upsample"), "upsample": ("Upsample", "Upsample"),
               "pixel_shuffle": ("PixelShuffle", "Upsample")}
OP_LIB = {"add": "op_plus", "sub": "op_minus", "mul": "op_times", "cat": "op_concat"}
OP_TEXT = {"matmul": ("MatMul", "MatMul"), "attn": ("Scaled Dot-Product Attention", "Scaled Dot-Product Attention")}


def builtin_chain(m):
    """Sous-classe de Sequential faite de 2 à 4 couches de torch.nn : leurs libellés, ou None."""
    if not isinstance(m, nn.Sequential):
        return None
    kids = [c for c in m.children() if type(c).__name__ not in REGULARIZERS]
    if not 2 <= len(kids) <= 4 or not all(type(c).__module__.startswith("torch.nn") for c in kids):
        return None
    return [builtin_label(c) for c in kids]


def user_color(cls, blocks):
    low = cls.lower()
    for words, lib in COLOR_HINTS:
        if any(w in low for w in words) and lib in blocks:
            return blocks[lib].color
    return "#EDEDED"


def shape_line(n, mode):
    if mode == "aucune" or not n.out_shape:
        return ""
    out = _dims(n.out_shape)
    if mode == "sortie" or not n.in_shapes:
        return out
    inn = _dims(n.in_shapes[0])
    return out if inn == out else f"{inn} → {out}"


def style_node(n, blocks, shapes):
    """Dictionnaire du format simplifié (sans position) pour un nœud."""
    if n.kind == "op" and n.op in OP_LIB:
        return {"lib": OP_LIB[n.op]}
    lines, d = [], {}
    if n.kind == "input":
        lines.append(n.label)
        d["lib"] = "Input"
    elif n.kind == "output":
        lines.append(n.label)
        d["lib"] = "Output"
    elif n.kind == "noise":
        lines.append("Noise ε ~ 𝒩(0, I)")
        if "Noise" in blocks:
            d["lib"] = "Noise"
    elif n.kind == "param":
        lines.append(n.label)
        d["color"] = blocks["Positional Encoding"].color if "pos" in n.label.lower() and "Positional Encoding" in blocks \
            else "#F3F3F3"
    elif n.kind == "op":
        txt, lib = OP_TEXT.get(n.op, (n.label.replace("_", " ").capitalize(), None))
        if n.op == "act":
            lib = next((k for k in blocks if k.lower() == n.label.lower()), "Activation")
            txt = blocks[lib].label if lib in blocks else n.label
        elif n.op == "norm":
            txt, lib = n.label.replace("_", " ").title().replace(" ", ""), "Layer Norm"
        elif n.op == "resize":
            txt, lib = RESIZE_TEXT.get(n.label, (txt, "Upsample"))
            if n.label.startswith("adaptive") and n.out_shape and all(v == 1 for v in n.out_shape[2:]):
                txt = "Global " + txt
        lines.append(txt)
        if lib in blocks:
            d["lib"] = lib
    else:
        m = n.module
        if type(m).__module__.startswith("torch."):
            txt, lib = builtin_label(m)
            lines.append(txt)
            if lib in blocks:
                d["lib"] = lib
            else:
                d["color"] = user_color(n.cls, blocks)
        else:
            chain = builtin_chain(m)
            if chain:                            # Conv2dNormActivation… : « Conv2d k3 s2 + BatchNorm + ReLU6 »
                lines.append(" + ".join(t for t, _ in chain))
                lib = chain[0][1]
                if lib in blocks:
                    d["lib"] = lib
            else:
                lines.append(n.cls)
                d["color"] = user_color(n.cls, blocks)
    sl = shape_line(n, shapes) if n.kind not in ("op",) or n.op not in OP_LIB else ""
    if n.kind == "module" and n.cls == "Linear":
        sl = ""                                  # « Linear 512 → 1000 » dit déjà tout
    if sl and n.kind not in ("param", "noise"):
        lines.append(sl)
    d["text"] = "\n".join(lines)
    return d


# ===================================================================== mise en page
class Layout:
    """Mise en page en colonnes.

    1. rang = position dans le flux (plus long chemin depuis les entrées) ;
    2. colonne principale = plus long chemin d'une entrée à une sortie ; si de longues liaisons
       relient sa première et sa seconde moitié (U-Net), elle est repliée en U : la seconde moitié
       remonte dans une autre colonne et ces liaisons deviennent horizontales ;
    3. les autres chaînes de blocs forment des colonnes (une chaîne courte rejoint une colonne
       où sa ligne est libre) ;
    4. l'ordre des colonnes est choisi parmi toutes les permutations : liaisons les plus courtes,
       et le moins possible de colonnes traversées.
    """
    GAP_Y, GAP_X = 46.0, 60.0
    LONG = 3                                   # une liaison « longue » saute au moins 3 rangs
    FRAME_GAP = 26.0                           # place en plus par cadre qui s'ouvre ou se ferme

    def frame_keys(self, n):
        """Cadres susceptibles d'entourer n : modules dépliés qui le contiennent, listes nommées."""
        g, keys = self.g, set()
        p = g.nodes[n].parent
        while p is not None:
            keys.add(p)
            p = g.group_parent.get(p)
        base = n.split("#")[0]
        for q in getattr(g, "containers", {}):
            if base.startswith(q + "."):
                keys.add("list:" + q)
        if g.nodes[n].repeat > 1:
            keys.add("rep:" + n)
        return keys

    def __init__(self, g, sizes):
        self.g, self.sizes = g, sizes
        self.ids = sorted(g.nodes, key=lambda k: g.nodes[k].order)

    # ------------------------------------------------------------------ rangs
    def ranks(self):
        g, r = self.g, {}
        for n in self.ids:                       # ordre d'exécution = ordre topologique
            r[n] = max((r[p] + 1 for p in g.preds(n) if p in r), default=0)
        for n in self.ids:                       # paramètres : sur la ligne de leur usage (flèche horizontale)
            if not g.preds(n) and g.succs(n) and g.nodes[n].kind != "input":
                r[n] = min(r[s] for s in g.succs(n))
        return r

    def main_path(self, r):
        g, best = self.g, {}
        for n in sorted(self.ids, key=lambda k: r[k], reverse=True):
            ss = [s for s in g.succs(n) if r[s] > r[n]]
            if not ss:
                best[n] = (0, [n])
            else:
                s = max(ss, key=lambda s: (best[s][0], -g.nodes[s].order))
                best[n] = (best[s][0] + 1, [n] + best[s][1])
        starts = [n for n in self.ids if g.nodes[n].kind == "input"] or self.ids[:1]
        return max((best[s] for s in starts), key=lambda b: b[0])[1]

    def fold(self, r):
        """Pivot du repli en U (None si inutile)."""
        long_edges = [(a, b) for a, b in self.g.edges if abs(r[b] - r[a]) >= self.LONG]
        if len(long_edges) < 2:
            return None
        flat = sum(abs(r[b] - r[a]) for a, b in long_edges)
        R = max(r.values())
        best = None
        for twice in range(2, 2 * R - 1):            # pivots entiers et demi-entiers
            P = twice / 2
            # le U suppose des liaisons emboîtées qui enjambent toutes le pivot (U-Net) ;
            # des résiduelles qui se suivent (x + f(x), puis + g(...)) ne doivent pas replier
            span = [(r[a], r[b]) for a, b in long_edges if min(r[a], r[b]) <= P < max(r[a], r[b])]
            if len(span) < 2 or len(span) < 0.75 * len(long_edges):
                continue
            iv = sorted((min(e), max(e)) for e in span)
            if any(not (i0 <= j0 and j1 <= i1) and not (j0 <= i0 and i1 <= j1)
                   for k, (i0, i1) in enumerate(iv) for (j0, j1) in iv[k + 1:]):
                continue

            def row(x):
                return x if x <= P else 2 * P - x
            cost = sum(abs(row(r[b]) - row(r[a])) for a, b in long_edges)
            if best is None or cost < best[0]:
                best = (cost, P)
        return best[1] if best and best[0] <= 0.4 * flat else None

    # --------------------------------------------------------------- colonnes
    def lift(self, r, spine):
        """Un bloc hors du chemin principal, alimenté par plusieurs blocs de la ligne au-dessus et dont
        la sortie part loin (ex. attention entre deux encodeurs, reliée au décodeur), remonte sur la ligne
        de ses entrées : ses liaisons deviennent horizontales."""
        g, on = self.g, set(spine)
        for n in self.ids:
            ps, ss = g.preds(n), g.succs(n)
            if n in on or len(ps) < 2 or not ss:
                continue
            if all(r[p] == r[n] - 1 for p in ps) and all(r[s] - r[n] >= 2 for s in ss):
                r[n] -= 1
        return r

    def run(self):
        g = self.g
        r = self.ranks()
        spine = self.main_path(r)
        r = self.lift(r, spine)
        P = self.fold(r)
        if P is None:
            row = dict(r)
        else:
            row = {n: (r[n] if r[n] <= P else 2 * P - r[n]) for n in self.ids}
        lanes = []                                    # listes de nœuds
        lane_of = {}

        def add_lane(nodes):
            lanes.append(list(nodes))
            for n in nodes:
                lane_of[n] = len(lanes) - 1
        if P is None:
            add_lane(spine)
        else:
            add_lane([n for n in spine if r[n] <= P])
            add_lane([n for n in spine if r[n] > P])
        fixed = len(lanes)
        # chaînes restantes, des plus longues aux plus courtes
        rest = [n for n in self.ids if n not in lane_of]
        while rest:
            chains = {}
            for n in sorted(rest, key=lambda k: r[k], reverse=True):
                ss = [s for s in g.succs(n) if s not in lane_of and s in rest and row[s] != row[n]
                      and s in chains]
                if ss:                           # même classe d'abord (un encodeur reste un encodeur)
                    s = max(ss, key=lambda s: (g.nodes[s].cls == g.nodes[n].cls or g.nodes[n].kind == "input",
                                               len(chains[s]), -g.nodes[s].order))
                    chains[n] = [n] + chains[s]
                else:
                    chains[n] = [n]
            chain = max(chains.values(), key=len)
            rows = {row[n] for n in chain}
            # une chaîne courte rejoint la colonne (non principale) la plus liée où ses lignes sont libres
            target = None
            if len(chain) <= 2:
                cands = []
                for i in range(fixed, len(lanes)):
                    if rows & {row[n] for n in lanes[i]}:
                        continue
                    links = sum(1 for a, b in g.edges if (a in chain and lane_of.get(b) == i)
                                or (b in chain and lane_of.get(a) == i))
                    sib = sum(1 for m in lanes[i] if g.nodes[m].sig and g.nodes[m].sig.split("(")[0]
                              == (g.nodes[chain[0]].sig or "").split("(")[0])
                    cands.append((sib, links, -i, i))
                if cands:
                    target = max(cands)[3]
            if target is None:
                add_lane(chain)
            else:
                lanes[target] += chain
                for n in chain:
                    lane_of[n] = target
            rest = [n for n in rest if n not in lane_of]
        order = self.order_lanes(lanes, lane_of, row, fixed)
        # abscisses : largeur de chaque colonne = son bloc le plus large
        widths = [max(self.sizes[n][0] for n in lane) for lane in lanes]
        x, cur = {}, 0.0
        lane_x = {}
        for i in order:
            lane_x[i] = cur + widths[i] / 2
            cur += widths[i] + self.GAP_X
        shift = lane_x[0]
        for n in self.ids:
            x[n] = lane_x[lane_of[n]] - shift
        # ordonnées : une ligne par valeur de row ; entre deux lignes, de la place en plus pour
        # chaque cadre (module déplié, liste de modules) qui se ferme ou s'ouvre entre elles
        y, top = {}, 0.0
        rows = sorted(set(row.values()))
        prev_keys = None
        for rv in rows:
            members = [n for n in self.ids if row[n] == rv]
            keys = set().union(*(self.frame_keys(n) for n in members))
            if prev_keys is not None:
                top += self.FRAME_GAP * len(keys ^ prev_keys)
            prev_keys = keys
            h = max(self.sizes[n][1] for n in members)
            for n in members:
                y[n] = top + h / 2
            top += h + self.GAP_Y
        self.x, self.y, self.row, self.lane_of, self.lanes, self.r = x, y, row, lane_of, lanes, r
        return x, y

    def order_lanes(self, lanes, lane_of, row, fixed):
        """Ordre gauche -> droite des colonnes minimisant longueur des liaisons et traversées."""
        import itertools
        g = self.g
        idx = list(range(len(lanes)))
        rows_of = [sorted({row[n] for n in lane}) for lane in lanes]

        def cost(order):
            pos = {l: i for i, l in enumerate(order)}
            c = 0.0
            for a, b in g.edges:
                la, lb = lane_of[a], lane_of[b]
                if la == lb:
                    continue
                lo, hi = sorted((pos[la], pos[lb]))
                c += hi - lo
                r0, r1 = sorted((row[a], row[b]))
                for l in order[lo + 1:hi]:
                    if any(r0 <= rv <= r1 for rv in rows_of[l]):
                        c += 1.5
            return c
        if len(lanes) <= 7:
            perms = (p for p in itertools.permutations(idx) if fixed < 2 or p.index(0) < p.index(1))
            return list(min(perms, key=cost))
        order = list(range(fixed))                    # trop de colonnes : insertion gloutonne
        for l in idx[fixed:]:
            order = min((order[:i] + [l] + order[i:] for i in range(len(order) + 1)), key=cost)
        return order


# ===================================================================== export
def measure_sizes(g, styled):
    """Taille de chaque nœud telle que l'éditeur l'affichera (texte mesuré, blocs ajustés)."""
    from deepgraph.config import CFG
    from deepgraph.items import BlockItem
    st = CFG.style
    sizes = {}
    for nid, d in styled.items():
        lib = CFG.blocks.get(d.get("lib")) if d.get("lib") else None
        if lib is not None and lib.shape == "circle":
            s = float(lib.width or st["operator_size"])
            sizes[nid] = (s, s)
            continue
        w0 = float(lib.width or st["block_width"]) if lib else float(st["block_width"])
        h0 = float(lib.height or st["block_height"]) if lib else float(st["block_height"])
        b = BlockItem(d.get("text", ""))
        b.w, b.h = w0, h0
        nw, nh = b.text_need()
        g10 = lambda v: -(-v // 10) * 10              # noqa: E731  arrondi à la dizaine supérieure
        sizes[nid] = (max(w0, g10(nw)), max(h0, g10(nh)))
    return sizes


GENERIC_NAMES = {"n", "l", "ln", "bn", "gn", "norm", "conv", "fc", "lin", "linear", "act", "relu", "gelu",
                 "drop", "dropout", "pool", "layer", "block", "proj", "mlp", "ff", "ffn", "attn", "attention",
                 "dense", "up", "down", "out", "net", "model", "module", "head", "stem", "body"}


def name_twins(g, styled):
    """Blocs au texte identique (ex. deux « Linear 400 → 20 ») : le nom d'attribut (mu, logvar,
    q_proj…) en première ligne, s'il dit quelque chose de plus que la classe."""
    by_text = defaultdict(list)
    for nid, n in g.nodes.items():
        if n.kind == "module":
            by_text[styled[nid]["text"]].append(nid)
    for text, ids in by_text.items():
        names = {nid: nid.split("#")[0].split(".")[-1] for nid in ids}
        if len(set(names.values())) < 2:
            continue
        for nid, name in names.items():
            core = name.strip("_").rstrip("0123456789").rstrip("_").lower()
            if name.isdigit() or not core or core in GENERIC_NAMES or core == g.nodes[nid].cls.lower():
                continue
            styled[nid]["text"] = name + "\n" + text


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication(["depuis_pytorch"])


def build(g, title, shapes):
    _app = _qt()                                 # noqa: F841  (polices : application Qt nécessaire)
    from deepgraph.config import CFG
    CFG.load()
    blocks = CFG.blocks
    styled = {nid: style_node(n, blocks, shapes) for nid, n in g.nodes.items()}
    name_twins(g, styled)
    sizes = measure_sizes(g, styled)
    lay = Layout(g, sizes)
    x, y = lay.run()
    nodes = []
    for nid in lay.ids:
        d = {"id": nid, "ref": nid, **styled[nid], "cx": round(x[nid], 1), "cy": round(y[nid], 1)}
        if "lib" not in d or not (blocks.get(d["lib"]) and blocks[d["lib"]].shape == "circle"):
            d["w"], d["h"] = sizes[nid]
        nodes.append(d)
    # cadres « N× » autour des blocs répétés
    for nid, n in g.nodes.items():
        if n.repeat > 1:
            nodes.append({"id": f"rep:{nid}", "type": "group", "label": f"{n.repeat}×", "children": [nid]})
    # cadre autour du contenu d'un module déplié (s'il n'englobe aucun bloc étranger)
    from deepgraph import richtext

    def box_ok(ids, m=14.0, top_m=30.0, extra=0.0):
        x0 = min(x[i] - sizes[i][0] / 2 for i in ids) - m - extra
        x1 = max(x[i] + sizes[i][0] / 2 for i in ids) + m + extra
        y0 = min(y[i] - sizes[i][1] / 2 for i in ids) - top_m
        y1 = max(y[i] + sizes[i][1] / 2 for i in ids) + m
        return not [i for i in lay.ids if i not in ids and x0 < x[i] < x1 and y0 < y[i] < y1]

    # cadres des listes de modules nommées (self.encoder = nn.ModuleList(...)) : leur nom
    # d'attribut est souvent celui qu'on écrirait sur la figure
    frame_of = {}                                  # nœud ou cadre -> cadre de liste qui le contient

    def wrap(i):
        while i in frame_of:
            i = frame_of[i]
        return i

    def opened_parent(q):
        parts = q.split(".")
        for k in range(len(parts) - 1, 0, -1):
            if ".".join(parts[:k]) in g.groups:
                return ".".join(parts[:k])
        return None
    for q in sorted(g.containers, key=lambda q: -q.count(".")):
        par = opened_parent(q)
        ids = [n for n, nd in g.nodes.items() if nd.kind == "module" and nd.parent == par
               and n.split("#")[0].startswith(q + ".")]
        kids = list(dict.fromkeys(wrap(f"rep:{i}" if g.nodes[i].repeat > 1 else i) for i in ids))
        if len(ids) < 2 or len(kids) < 2 or len({lay.lane_of[i] for i in ids}) != 1 or not box_ok(ids):
            continue
        fid = f"list:{q}"
        label = q.split(".")[-1].replace("_", " ")
        if label.isdigit():
            label = ""                             # features.3 : un numéro n'apprend rien
        # l'étiquette (en haut à gauche) ne doit pas passer sous la flèche qui entre au centre
        half = max(sizes[i][0] for i in ids) / 2 + 14.0
        if 10 + richtext.measure(label, 11)[0] > half - 6 and " " in label:
            label = label.replace(" ", "\n")
        nodes.append({"id": fid, "type": "group", "label": label,
                      "label_pos": "inside", "label_size": 11, "margin": 14.0,
                      "margin_top": (30.0 + 13.0 * label.count("\n")) if label else 14.0,
                      "fill": "#F7F7F7", "children": kids})
        for k in kids:
            frame_of[k] = fid
    framed = {}                                   # module déplié -> id du cadre (s'il a été dessiné)
    quals = [q for q in g.groups if any(g.in_group(n, q) for n in g.nodes)]
    for q in sorted(quals, key=lambda q: -q.count(".")):    # les plus profonds d'abord
        all_ids = [n for n in g.nodes if g.in_group(n, q)]
        direct = [n for n in all_ids if g.nodes[n].parent == q or
                  (g.nodes[n].parent not in framed and g.nodes[n].parent != q)]
        inner = [framed[s] for s in framed if g.group_parent.get(s) == q]
        m, top_m = 14.0, 30.0
        x0 = min(x[i] - sizes[i][0] / 2 for i in all_ids) - m - 12 * len(inner)
        x1 = max(x[i] + sizes[i][0] / 2 for i in all_ids) + m + 12 * len(inner)
        y0 = min(y[i] - sizes[i][1] / 2 for i in all_ids) - top_m
        y1 = max(y[i] + sizes[i][1] / 2 for i in all_ids) + m
        if [i for i in lay.ids if i not in all_ids and x0 < x[i] < x1 and y0 < y[i] < y1]:
            continue                               # le cadre engloberait un bloc étranger
        kids = [f"rep:{i}" if g.nodes[i].repeat > 1 else i for i in direct
                if not any(g.in_group(i, s) for s in framed if g.group_parent.get(s) == q)] + inner
        kids = list(dict.fromkeys(wrap(k) for k in kids))
        n_rep = g.group_repeat.get(q, 1)
        label = (f"{n_rep}× " if n_rep > 1 else "") + g.groups.get(q, q)
        if n_rep == 1:
            label += f"  ({q})"
        framed[q] = f"mod:{q}"
        nodes.append({"id": framed[q], "type": "group", "label": label,
                      "label_pos": "inside", "label_size": 12 if n_rep > 1 else 11, "margin": m,
                      "margin_top": top_m, "fill": "#FAFAFA", "border_style": "dashed", "children": kids})
    # flèches
    arrows = []
    row, lane = lay.row, lay.lane_of
    plan = []
    for a, b in g.edges:
        if lane[a] == lane[b]:
            down = row[b] > row[a]
            fs, ts = ("bottom", "top") if down else ("top", "bottom")
            lo, hi = sorted((row[a], row[b]))
            blocked = any(lo < row[n] < hi for n in lay.lanes[lane[a]])
            if blocked:                           # saute des blocs de sa colonne (résiduelle) : par le côté,
                plan.append((a, b, None, None, "auto"))   # choisi une fois les autres flèches placées
            else:
                plan.append((a, b, fs, ts, None))
        else:
            right = x[b] > x[a]
            fs, ts = ("right", "left") if right else ("left", "right")
            if row[a] == row[b]:
                lo, hi = sorted((x[a], x[b]))
                blocked = any(row[n] == row[a] and lo < x[n] < hi for n in lay.ids)
                plan.append((a, b, fs, ts, "auto" if blocked else None))
            else:
                plan.append((a, b, fs, ts, "auto"))
    used = defaultdict(int)
    for a, b, fs, ts, st in plan:
        if fs:
            used[(a, fs)] += 1
            used[(b, ts)] += 1
    for k, (a, b, fs, ts, st) in enumerate(plan):
        if fs is None:
            side = min(("right", "left"), key=lambda sd: (used[(a, sd)] + used[(b, sd)], sd != "right"))
            used[(a, side)] += 1
            used[(b, side)] += 1
            plan[k] = (a, b, side, side, st)
    # toutes les attaches d'un même côté (entrées ET sorties) se répartissent le long de ce côté,
    # dans l'ordre de position de l'autre extrémité : pas de flèches superposées
    side_all = defaultdict(list)
    for k, (a, b, fs, ts, st) in enumerate(plan):
        side_all[(a, fs)].append((b, k))
        side_all[(b, ts)].append((a, k))
    circle = {nid for nid, d in styled.items() if d.get("lib") in blocks and blocks[d["lib"]].shape == "circle"}

    def at(node, side, k):
        lst = side_all[(node, side)]
        if len(lst) < 2 or node in circle:
            return {}
        key = (lambda e: (x[e[0]], e[1])) if side in ("top", "bottom") else (lambda e: (y[e[0]], e[1]))
        i = [e[1] for e in sorted(lst, key=key)].index(k)
        return {"at": round((i + 1) / (len(lst) + 1), 2)}
    for k, (a, b, fs, ts, st) in enumerate(plan):
        frm = {"node": a, "side": fs, **at(a, fs, k)}
        to = {"node": b, "side": ts, **at(b, ts, k)}
        if fs in ("left", "right") and row[a] == row[b] and abs(sizes[a][1] - sizes[b][1]) < 1:
            # horizontale entre deux blocs de même hauteur : un côté seul suit l'autre (reste droite)
            if "at" in to and len(side_all[(a, fs)]) == 1 and a not in circle:
                frm["at"] = to["at"]
            elif "at" in frm and len(side_all[(b, ts)]) == 1 and b not in circle:
                to["at"] = frm["at"]
        if st is None and (frm.get("at", .5) != to.get("at", .5)):
            st = "auto"                           # attaches décalées : la flèche ne serait plus droite
        arrows.append({"from": frm, "to": to, **({"style": st} if st else {})})
    top = min(y[n] - sizes[n][1] / 2 for n in lay.ids)
    from deepgraph.config import CFG as _C
    from PyQt5.QtGui import QFontMetricsF
    tw = QFontMetricsF(_C.font(16, True)).horizontalAdvance(title)
    cx = (min(x.values()) + max(x.values())) / 2
    bbox = (min(min(x[n] - sizes[n][0] / 2 for n in lay.ids) - 30, cx - tw / 2 - 10), top - 60,
            max(max(x[n] + sizes[n][0] / 2 for n in lay.ids) + 30, cx + tw / 2 + 10),
            max(y[n] + sizes[n][1] / 2 for n in lay.ids))
    nodes.append({"id": "titre", "type": "text", "text": title, "bold": True, "font_size": 16,
                  "cx": round((min(x.values()) + max(x.values())) / 2, 1), "cy": round(top - 40, 1)})
    return {"format": "deepgraph", "settings": {"fit_text": True}, "nodes": nodes, "arrows": arrows}, bbox


# ===================================================================== encarts
def capture_calls(model, inputs, classes):
    """Premier appel de chaque classe demandée : (module, args, kwargs) avec les vraies entrées."""
    found, handles = {}, []

    def make(cls):
        def hook(mod, args, kwargs):
            if cls not in found:
                found[cls] = (mod, args, kwargs)
        return hook
    for m in model.modules():
        c = type(m).__name__
        if c in classes:
            handles.append(m.register_forward_pre_hook(make(c), with_kwargs=True))
    try:
        with torch.no_grad():
            model(*[t for _, t in inputs])
    finally:
        for h in handles:
            h.remove()
    return found


def detail_graph(mod, args, kwargs, depth, fold=True):
    import inspect
    try:
        bound = inspect.signature(mod.forward).bind(*args, **kwargs)
        named = list(bound.arguments.items())
    except (TypeError, ValueError):
        named = [(f"x{i}", a) for i, a in enumerate(args)] + list(kwargs.items())
    inputs = [(name, v) for name, v in named if isinstance(v, torch.Tensor)]
    g = Tracer(mod, depth).run(inputs, call=(args, kwargs))
    if fold:
        g.fold_open_repeats()
        g.fold_repeats()
    return g


def merge_inset(doc, inset, dx, dy, prefix):
    """Ajoute les éléments d'un encart à doc, décalés de (dx, dy), identifiants préfixés."""
    def pid(v):
        return prefix + str(v)
    for d in inset["nodes"]:
        d = dict(d)
        d["id"] = pid(d["id"])
        if "ref" in d:
            d["ref"] = pid(d["ref"])
        if "children" in d:
            d["children"] = [pid(c) for c in d["children"]]
        if "cx" in d:
            d["cx"] = round(d["cx"] + dx, 1)
            d["cy"] = round(d["cy"] + dy, 1)
        doc["nodes"].append(d)
    for a in inset["arrows"]:
        a = dict(a)
        for k in ("from", "to"):
            e = dict(a[k])
            e["node"] = pid(e["node"])
            a[k] = e
        doc["arrows"].append(a)


# ===================================================================== CLI
def load_class(spec):
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise SystemExit("modèle attendu sous la forme fichier.py:Classe ou paquet.module:Classe")
    if path.endswith(".py") or os.path.sep in path:
        path = os.path.abspath(path)
        sys.path.insert(0, os.path.dirname(path))
        # le dossier courant (racine du projet) d'abord : « from models.x import … » doit marcher
        # comme quand on lance son propre code depuis la racine du projet
        sys.path.insert(0, os.getcwd())
        modname = "_deepgraph_modele_" + os.path.splitext(os.path.basename(path))[0]
        sp = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[modname] = mod
        sp.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(path)
    return getattr(mod, name)


def parse_input(s):
    """nom:forme[:type] -> (nom, tenseur). type : float (défaut), long / int (indices < 10), bool."""
    parts = s.split(":")
    kind = parts.pop() if parts[-1] in ("float", "long", "int", "bool") else "float"
    shape = parts.pop()
    name = ":".join(parts)
    dims = [int(v) for v in shape.replace("x", ",").split(",") if v.strip()]
    if kind in ("long", "int"):
        t = torch.randint(0, 10, dims, dtype=torch.long if kind == "long" else torch.int)
    elif kind == "bool":
        t = torch.rand(*dims) > 0.5
    else:
        t = torch.randn(*dims)
    return name or f"x{len(dims)}", t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("modele", help="fichier.py:Classe ou paquet.module:Classe")
    ap.add_argument("--entree", "-e", action="append", default=[],
                    help="nom:forme[:type] d'une entrée, ex. x:1,3,224,224 ou tokens:1,32:long "
                         "(dans l'ordre des arguments du forward)")
    ap.add_argument("--args", default="{}", help="arguments du constructeur, en JSON")
    ap.add_argument("--profondeur", "-p", type=int, default=1, help="niveau de détail (défaut : 1)")
    ap.add_argument("--deplier", action="append", default=[], help="classe à montrer en détail (répétable)")
    ap.add_argument("--replier", action="append", default=[], help="classe à montrer comme un seul bloc")
    ap.add_argument("--formes", choices=["entree-sortie", "sortie", "aucune"], default="entree-sortie",
                    help="formes des tenseurs dans les blocs")
    ap.add_argument("--sans-repetitions", action="store_true", help="ne pas regrouper les blocs identiques en « N× »")
    ap.add_argument("--detail", action="append", default=[],
                    help="classe dont l'intérieur est dessiné en encart, à droite (répétable)")
    ap.add_argument("--resume", help="écrit aussi un résumé JSON du graphe (pour un LLM ou pour vérifier)")
    ap.add_argument("-o", "--out", required=True, help="fichier .dgraph à écrire")
    a = ap.parse_args(argv)
    sys.path.insert(0, ROOT)

    cls = load_class(a.modele)
    model = cls(**json.loads(a.args)).eval()
    if not a.entree:
        raise SystemExit("donner au moins une entrée : --entree nom:forme")
    inputs = []
    for s in a.entree:
        inputs.append(parse_input(s))
    g = Tracer(model, a.profondeur, a.deplier, a.replier).run(inputs)
    if not a.sans_repetitions:
        g.fold_open_repeats()
        g.fold_repeats()
    if a.resume:
        with open(a.resume, "w", encoding="utf-8") as f:
            json.dump(g.summary(), f, ensure_ascii=False, indent=1)
    doc, bbox = build(g, getattr(cls, "__name__", "Modèle"), a.formes)
    if a.detail:
        calls = capture_calls(model, inputs, set(a.detail))
        x_cur = bbox[2] + 60
        for c in a.detail:
            if c not in calls:
                print(f"[détail] classe « {c} » jamais appelée : ignorée", file=sys.stderr)
                continue
            mod, args_, kwargs_ = calls[c]
            sub = detail_graph(mod, args_, kwargs_, 1, not a.sans_repetitions)
            inset, ib = build(sub, c, a.formes)
            dx, dy = x_cur - ib[0], bbox[1] - ib[1]
            merge_inset(doc, inset, dx, dy, f"{c}/")
            x_cur += (ib[2] - ib[0]) + 40
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    n_blocks = sum(1 for n in g.nodes.values() if n.kind in ("module", "op", "param"))
    print(f"{a.out} : {n_blocks} blocs, {len(g.edges)} liaisons", file=sys.stderr)


if __name__ == "__main__":
    main()
