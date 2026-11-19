"""Schéma .dgraph -> squelette de code PyTorch (l'inverse de outils/depuis_pytorch.py).

Le schéma est lu comme un graphe de flux de données :
  - flèches = tenseurs (sens donné par la pointe) ; jonction = tenseur réutilisé ;
  - bloc = couche, reconnue par son bloc de bibliothèque et par son texte (« Conv 7×7, 64, /2 »,
    « BN + ReLU », « 2× (Conv 3×3 + ReLU) 64 canaux ») ; opérateur ⊕ ⊗ ⊙ − concat = opération ;
  - point libre ou texte au départ d'une flèche = entrée du forward (nommée d'après le texte voisin),
    à l'arrivée = sortie ;
  - cadre « N× » = bloc répété N fois ; cadre nommé (« Bloc 1 ») = nn.Sequential ;
    cadre en pointillés = définition d'une classe, utilisée par les blocs qui portent son titre ;
  - flèches de couleur expliquées par une légende (flèche + texte) = opération sur la liaison
    (« Max Pool 2×2 », « Up-conv 2×2 », « copie + concaténation »).

Le code produit utilise les modules « paresseux » (nn.LazyConv2d, nn.LazyLinear…) : le nombre de
canaux d'entrée, que les schémas ne donnent presque jamais, est déduit au premier passage. Ce qui
manque au schéma reçoit une valeur par défaut marquée « TODO ».

Aucune dépendance à Qt ni à torch : le module lit le format complet (celui qu'écrit l'éditeur).
"""
import keyword
import re
import unicodedata
from dataclasses import dataclass, field

# ============================================================================ textes

GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "eps", "ϵ": "eps", "ζ": "zeta",
         "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lam", "μ": "mu", "ν": "nu", "ξ": "xi",
         "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi", "ϕ": "phi", "χ": "chi",
         "ψ": "psi", "ω": "omega", "Σ": "sigma", "Φ": "phi", "Ω": "omega", "Δ": "delta"}
SUBSUP = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
LATEX = re.compile(r"\\(mathrm|mathbf|text|operatorname|hat|tilde|bar|vec|dot|overline)\s*")
STOP = {"de", "des", "du", "la", "le", "les", "d", "l", "of", "the", "et", "and", "vers", "a", "an",
        "un", "une", "en", "to", "par", "pour"}
RESERVED = set(keyword.kwlist) | {"self", "torch", "nn", "F", "math", "super", "print", "len", "range",
                                  "input", "type", "list", "dict", "set", "min", "max", "sum", "id"}


TEX = {"times": "×", "cdot": "·", "to": "→", "rightarrow": "→", "leftarrow": "←", "in": "∈", "leq": "≤",
       "geq": "≥", "approx": "≈", "infty": "∞", "sum": "Σ", "prod": "Π", "top": "ᵀ", "odot": "⊙",
       "oplus": "⊕", "otimes": "⊗", "sqrt": "√", "partial": "∂", "nabla": "∇", "ldots": "…", "dots": "…",
       "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
       "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
       "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
       "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ",
       "Psi": "Ψ", "Omega": "Ω", "quad": " ", "qquad": " "}


def tex(s):
    """Formules $…$ -> texte lisible : « Conv $3\\times3$ » -> « Conv 3×3 », « $\\mathcal{F}(x_l)$ » -> « F(x_l) »."""
    s = s or ""
    if "$" not in s and "\\" not in s:
        return s
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\(?:mathcal|mathbf|mathrm|mathbb|text|operatorname|hat|tilde|bar|vec|dot|overline)\s*"
               r"\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: TEX.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\[,;! ]", " ", s)
    return s.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")


def plain(s):
    """Texte affiché -> texte ASCII approché (formules, grec translittéré, indices, sans accents)."""
    s = tex(s).translate(SUBSUP)
    s = LATEX.sub("", s).replace("\\", "")
    s = "".join(GREEK.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def first_line(s):
    return (s or "").strip().split("\n")[0].strip()


def norm_key(s):
    """Clé de comparaison de titres : « Scaled Dot-Product\\nAttention » == « scaled dot product attention »."""
    return " ".join(re.findall(r"[a-z0-9]+", plain(s).lower()))


def ident(text, default="x"):
    """Nom de variable lisible tiré d'un texte du schéma."""
    t = plain(first_line(text))
    t = re.sub(r"\([^)]*\)$", "", t).strip() or t
    low = t.lower()
    m = re.match(r"^\s*([a-z]\w*)\s*(\(([a-z]\w*)\))?\s*(?:[∈~=:]|$)", low)
    if m:
        name = m.group(1) + (("_" + m.group(3)) if m.group(3) else "")
    else:
        m = re.search(r"(?<![\w])([a-z]\w?)\s*[∈~]", low)
        if m:
            name = m.group(1)
        else:
            low = re.split(r"[→:(=∈~,]|->", low)[0]
            words = [w for w in re.findall(r"[a-z][a-z0-9]*", low) if w not in STOP]
            name = "_".join(words[:2]) if words else ""
    name = re.sub(r"\W+", "_", name).strip("_")
    if not name:
        return default
    if name[0].isdigit():
        name = "x" + name
    return name + "_" if name in RESERVED else name


def camel(text, default="Modele"):
    """Titre -> nom de classe : « ResNet-18 » -> ResNet18, « Autoencodeur variationnel (VAE) » -> VAE."""
    t = plain(first_line(text))
    m = re.search(r"\(\s*([A-Za-z][A-Za-z0-9]*)\s*\)", t)
    if m and len(m.group(1)) >= 2 and re.search(r"[A-Z]", m.group(1)):     # sigle : (VAE), (ViT)
        return m.group(1)
    t = re.sub(r"\([^)]*\)", " ", t)
    words = re.findall(r"[A-Za-z0-9]+", t)
    name = "".join(w[0].upper() + w[1:] for w in words)
    if not name:
        return default
    return "M" + name if name[0].isdigit() else name


def snake(name):
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"\W+", "_", s).strip("_").lower()
    return s or "bloc"


def note_text(s):
    return " ".join(tex(s).split())


def group_name(label):
    """Nom tiré de l'étiquette d'un cadre (« Bloc 1 » -> bloc_1) ; « 0 », « 12 » -> ''."""
    t = first_line(plain(label))
    t = re.sub(r"\(.*?\)", "", t).strip()
    if not re.search(r"[A-Za-z]", t):
        return ""
    return re.sub(r"\W+", "_", t).strip("_").lower()


def repeat_label(label):
    """Étiquette de cadre répété -> (N, nom) : « N× » -> ('N', ''), « 2× InvertedResidual » -> ('2', 'InvertedResidual'),
    « ×4 » -> ('4', '') ; sinon None."""
    t = plain(label).strip()
    m = re.match(r"^(\d+|[A-Za-z])\s*[×x](?:\s+(\S.*))?$", t) or re.match(r"^[×x]\s*(\d+|[A-Za-z]\w*)()$", t)
    if not m:
        return None
    return m.group(1), (m.group(2) or "").strip()


def label_line(text):
    """Texte d'un bloc sur une ligne : « Multi-Head\nAttention » -> « Multi-Head Attention »,
    « Conv 3×3\n64 canaux » -> « Conv 3×3 / 64 canaux » (ligne de réglages)."""
    lines = [x.strip() for x in tex(text).strip().split("\n") if x.strip()]
    out = lines[:1]
    for ln in lines[1:]:
        wrapped = not re.search(r"\d", out[-1] + ln) and not re.search(r"[=(|]", ln)
        out.append((" " if wrapped else " / ") + ln)
    return "".join(out)


# ============================================================================ couches

@dataclass
class Spec:
    """Couche reconnue dans un bloc (kind + réglages lus dans le texte)."""
    kind: str
    p: dict = field(default_factory=dict)


SIMPLE = {"conv", "pool", "gpool", "upsample", "norm", "act", "softmax", "dropout", "flatten", "linear",
          "embedding", "ff", "mlp", "head", "patch", "posenc", "identity", "todo"}

DEF_SPECS = {
    "Embedding": Spec("embedding"), "Input Embedding": Spec("embedding"),
    "Output Embedding": Spec("embedding"), "Positional Encoding": Spec("posenc", {"learned": False}),
    "Patch Embedding": Spec("patch"), "Linear": Spec("linear"), "linear_proj": Spec("linear", {"proj": True}),
    "MLP": Spec("mlp"), "Feed Forward": Spec("ff"), "Dropout": Spec("dropout"), "Flatten": Spec("flatten"),
    "Conv1D": Spec("conv", {"dim": 1}), "Conv2D": Spec("conv", {"dim": 2}), "Conv3D": Spec("conv", {"dim": 3}),
    "Conv 3×3": Spec("conv", {"k": 3}), "Conv 1×1": Spec("conv", {"k": 1}),
    "Transposed Conv": Spec("conv", {"transposed": True}), "Depthwise Conv": Spec("conv", {"depthwise": True}),
    "Max Pooling": Spec("pool", {"op": "max"}), "Avg Pooling": Spec("pool", {"op": "avg"}),
    "Upsample": Spec("upsample"),
    "Batch Norm": Spec("norm", {"op": "batch"}), "Layer Norm": Spec("norm", {"op": "layer"}),
    "Group Norm": Spec("norm", {"op": "group"}), "Instance Norm": Spec("norm", {"op": "instance"}),
    "RMS Norm": Spec("norm", {"op": "rms"}), "Local Response Norm": Spec("norm", {"op": "lrn"}),
    "L2 Norm": Spec("todo", {"what": "normalisation L2 : F.normalize(x, dim=1)"}),
    "AdaIN": Spec("todo", {"what": "AdaIN (normalisation modulée par un style)"}),
    "AdaLN": Spec("todo", {"what": "AdaLN (LayerNorm modulée par un conditionnement)"}),
    "Weight Norm": Spec("todo", {"what": "nn.utils.parametrizations.weight_norm sur la couche voisine"}),
    "Spectral Norm": Spec("todo", {"what": "nn.utils.parametrizations.spectral_norm sur la couche voisine"}),
    "Activation": Spec("act", {"op": "relu", "guess": True}),
    "ReLU": Spec("act", {"op": "relu"}), "Leaky ReLU": Spec("act", {"op": "leaky"}),
    "PReLU": Spec("act", {"op": "prelu"}), "ELU": Spec("act", {"op": "elu"}), "SELU": Spec("act", {"op": "selu"}),
    "GELU": Spec("act", {"op": "gelu"}), "SiLU / Swish": Spec("act", {"op": "silu"}),
    "Mish": Spec("act", {"op": "mish"}), "Sigmoid": Spec("act", {"op": "sigmoid"}),
    "Tanh": Spec("act", {"op": "tanh"}), "Hard Sigmoid": Spec("act", {"op": "hardsigmoid"}),
    "Hard Swish": Spec("act", {"op": "hardswish"}), "Softplus": Spec("act", {"op": "softplus"}),
    "GLU": Spec("act", {"op": "glu"}), "SwiGLU": Spec("todo", {"what": "SwiGLU : Linear → SiLU ⊙ Linear"}),
    "Softmax": Spec("softmax"), "softmax_attention": Spec("softmax", {"dim": -1}),
    "Self-Attention": Spec("mha", {"self": True}), "Cross-Attention": Spec("mha"),
    "Multi-Head Attention": Spec("mha"), "Masked Multi-Head Attention": Spec("mha", {"masked": True}),
    "Scaled Dot-Product Attention": Spec("sdpa"),
    "RNN": Spec("rnn", {"cell": "RNN"}), "LSTM": Spec("rnn", {"cell": "LSTM"}), "GRU": Spec("rnn", {"cell": "GRU"}),
    "BiLSTM": Spec("rnn", {"cell": "LSTM", "bi": True}),
    "Encoder": Spec("stub"), "Decoder": Spec("stub"), "Bottleneck": Spec("stub"), "Denoising": Spec("stub"),
    "Generator": Spec("stub"), "Discriminator": Spec("stub"),
    "Noise": Spec("noise"), "Sampling": Spec("sampling"), "Add & Norm": Spec("addnorm"),
    "MatMul": Spec("matmul"), "Scale": Spec("scale"), "Mask (opt.)": Spec("mask"), "Concat": Spec("concat"),
    "Input": Spec("input"), "Output": Spec("output"),
}
# blocs dont le texte peut préciser ou remplacer la couche (« BN + ReLU » posé avec le bloc Batch Norm…)
TEXT_FIRST = {"conv", "pool", "upsample", "norm", "act", "softmax", "dropout", "flatten", "linear", "todo"}

ACTS = [("relu6", r"\brelu\s*6\b"), ("leaky", r"leaky\s*-?\s*relu"), ("prelu", r"\bprelu\b"), ("selu", r"\bselu\b"), ("gelu", r"\bgelu\b"),
        ("elu", r"\belu\b"), ("relu", r"\brelu\b"), ("silu", r"\b(silu|swish)\b"), ("mish", r"\bmish\b"),
        ("hardsigmoid", r"hard\s*-?\s*sigmoid"), ("hardswish", r"hard\s*-?\s*swish"),
        ("sigmoid", r"\b(sigmoid|sigmoide)\b"), ("tanh", r"\btanh\b"), ("softplus", r"\bsoftplus\b"),
        ("glu", r"\bglu\b")]
ACT_CTOR = {"relu6": "nn.ReLU6()", "relu": "nn.ReLU()", "leaky": "nn.LeakyReLU(0.2)", "prelu": "nn.PReLU()", "elu": "nn.ELU()",
            "selu": "nn.SELU()", "gelu": "nn.GELU()", "silu": "nn.SiLU()", "mish": "nn.Mish()",
            "sigmoid": "nn.Sigmoid()", "tanh": "nn.Tanh()", "hardsigmoid": "nn.Hardsigmoid()",
            "hardswish": "nn.Hardswish()", "softplus": "nn.Softplus()", "glu": "nn.GLU(dim=-1)"}

KERNEL = re.compile(r"(\d+)\s*[×x*]\s*(\d+)(?:\s*[×x*]\s*(\d+))?")
ARROW = re.compile(r"([A-Za-z0-9_]+)\s*(?:→|->)\s*([A-Za-z0-9_]+)")
SYMBOL = re.compile(r"(?<![\w/])([A-Za-z](?:\d+|_\w+)?)(?![\w])")


def _size(tok):
    """« 64 » -> 64, « K » / « d_model » -> 'K' / 'd_model', sinon None."""
    if tok is None:
        return None
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    if re.fullmatch(r"[A-Za-z](?:\d+|_\w+)?", tok) and tok.lower() not in ("x", "d"):
        return tok
    return None


def _numbers(s):
    """Réglages courants d'un morceau de texte : taille de noyau, pas, taille de sortie."""
    out = {}
    m = ARROW.search(s)
    if m:
        out["out"] = _size(m.group(2))
        s = s[:m.start()] + " " + s[m.end():]
    m = KERNEL.search(s)
    if m:
        out["k"] = int(m.group(1))
        s = s[:m.start()] + " " + s[m.end():]
    m = re.search(r"\bk\s*=?\s*(\d+)\b", s)
    if m and "k" not in out:
        out["k"] = int(m.group(1))
        s = s[:m.start()] + " " + s[m.end():]
    m = re.search(r"/\s*(\d+)|\bs(?:tride)?\s*=?\s*(\d+)\b|\bpas\s*(\d+)", s)
    if m:
        out["stride"] = int(next(g for g in m.groups() if g))
        s = s[:m.start()] + " " + s[m.end():]
    if "out" not in out:
        nums = re.findall(r"(?<![\w.])(\d+)(?![\w.%])", s)
        if nums:
            out["out"] = int(nums[-1])
        else:
            syms = [t for t in SYMBOL.findall(s) if len(t) <= 3 and t[0].isupper()]
            if syms:
                out["out"] = syms[-1]
    return out


def parse_part(part):
    """Un morceau de texte (« Conv 3×3, 64 », « BN », « FC 4096 ») -> Spec, ou None."""
    s = plain(part).strip()
    low = s.lower()
    if not low:
        return None
    if re.search(r"attention", low):
        return Spec("mha", {"masked": "mask" in low, "self": "self" in low})
    if re.search(r"feed\s*-?\s*forward|\bffn\b", low):
        return Spec("ff")
    if re.search(r"\bmlp\b", low):
        return Spec("head") if "head" in low else Spec("mlp")
    if re.search(r"patch", low) and re.search(r"emb|proj", low):
        return Spec("patch")
    if re.search(r"position", low):
        return Spec("posenc", {"learned": "embedding" in low or "appris" in low, "cls": "token" in low})
    if re.search(r"\b(bi\s*-?\s*lstm|lstm|gru|rnn)\b", low):
        cell = "GRU" if "gru" in low else "LSTM" if "lstm" in low else "RNN"
        return Spec("rnn", {"cell": cell, "bi": bool(re.search(r"\bbi", low)), **_numbers(s)})
    if re.search(r"\bemb(edding)?\b|\bemb\.", low):
        return Spec("embedding", _numbers(s))
    if re.search(r"conv|d[ée]conv", low):
        p = _numbers(s)
        if re.search(r"transpos|deconv|up\s*-?\s*conv", low):
            p["transposed"] = True
        m = re.search(r"\b([123])\s*d\b", low)
        if m:
            p["dim"] = int(m.group(1))
        if "depthwise" in low:
            p["depthwise"] = True
        return Spec("conv", p)
    if re.search(r"pool|sous\s*-?\s*[ée]chant|subsampl|downsampl", low):
        p = _numbers(s)
        p.pop("out", None)
        p["op"] = "max" if "max" in low else "avg"
        if re.search(r"global|adaptive|adaptative", low):
            return Spec("gpool", p)
        return Spec("pool", p)
    if re.search(r"upsampl|up\s*-?\s*sampl|sur\s*-?\s*[ée]chant|interpol", low):
        return Spec("upsample", _numbers(s))
    for op, pat in (("batch", r"\bbn\b|batch\s*-?\s*norm"), ("layer", r"layer\s*-?\s*norm|\bln\b"),
                    ("group", r"group\s*-?\s*norm|\bgn\b"), ("instance", r"instance\s*-?\s*norm"),
                    ("rms", r"rms\s*-?\s*norm"), ("lrn", r"\blrn\b|local\s+response")):
        if re.search(pat, low):
            return Spec("norm", {"op": op})
    if re.fullmatch(r"norm(ali[sz]ation)?\.?", low):
        return Spec("norm", {})
    if re.search(r"\bdropout\b", low):
        m = re.search(r"(\d*[.,]\d+)|(\d+)\s*%", low)
        p = {}
        if m:
            p["p"] = float(m.group(1).replace(",", ".")) if m.group(1) else int(m.group(2)) / 100
        return Spec("dropout", p)
    if re.search(r"\bflatten\b|aplati", low):
        return Spec("flatten")
    if re.search(r"log\s*-?\s*softmax", low):
        return Spec("softmax", {"log": True})
    if re.search(r"softmax", low):
        return Spec("softmax")
    for op, pat in ACTS:
        if re.search(pat, low):
            return Spec("act", {"op": op})
    if re.search(r"\b(fc|linear|lin[ée]aire|dense|fully\s*-?\s*connected|projection|proj)\b", low):
        return Spec("linear", _numbers(s))
    if re.search(r"concat", low):
        return Spec("concat")
    if re.search(r"identit", low):
        return Spec("identity")
    return None


SHAPE_LINE = re.compile(r"^[\d\s×x*,()\[\]]+(?:(?:→|->)[\d\s×x*,()\[\]]+)?$")


def shape_line(line):
    """« 3×224×224 → 64×112×112 » -> ([3, 224, 224], [64, 112, 112]) ; « 64×56×56 » -> (même, même)."""
    t = plain(line).strip()
    if not t or not re.search(r"\d", t) or not SHAPE_LINE.match(t):
        return None
    sides = re.split(r"→|->", t)
    dims = [[int(x) for x in re.findall(r"\d+", s)] for s in sides]
    if not all(dims):
        return None
    return dims[0], dims[-1]


def split_shapes(text):
    """Texte d'un bloc -> (texte sans les lignes de formes, (forme d'entrée, forme de sortie) ou None)."""
    keep, shape = [], None
    for ln in (text or "").split("\n"):
        sh = shape_line(ln)
        if sh and shape is None:
            shape = sh
        elif not sh:
            keep.append(ln)
    return "\n".join(keep), shape


def parse_text(text):
    """Texte d'un bloc -> (liste de Spec, répétition, canaux par défaut) ou None si non reconnu.

    « BN + ReLU » -> [norm, act] ; « 2× (Conv 3×3 + ReLU)\\n64 canaux » -> [conv, act], 2, 64.
    Une ligne de formes (« 3×224×224 → 64×112×112 ») donne les canaux de sortie."""
    text, shape = split_shapes(text)
    t = " ".join(plain(text).split())
    if not t:
        return None
    reps, rest = 1, ""
    m = re.match(r"^(\d+)\s*[×x]\s*\((.*)\)\s*(.*)$", t)
    if m:
        reps, t, rest = int(m.group(1)), m.group(2), m.group(3)
    parts = [p for p in re.split(r"\s\+\s|\+\s|\s\+", t) if p.strip()]
    specs = [parse_part(p) for p in parts]
    if not specs or any(s is None for s in specs):
        whole = parse_part(t)
        if whole is None:
            return None
        specs = [whole]
    chans = None
    m = re.search(r"(\d+)\s*(canaux|channels|filtres|filters|ch\b)", rest + " " + t)
    if m:
        chans = int(m.group(1))
    elif shape and specs[0].kind in ("conv", "linear", "embedding", "rnn") and specs[0].p.get("out") is None:
        chans = shape[1][0] if specs[0].kind == "conv" else shape[1][-1]
    if shape:
        specs[0].p["_shape"] = shape
    return specs, reps, chans


def hidden_layers(tree, parent_hidden=False):
    """Identifiants des calques masqués (un groupe de calques masqué masque son contenu)."""
    out = set()
    for n in tree:
        if not isinstance(n, dict):
            continue
        h = parent_hidden or not n.get("visible", True)
        if h:
            out.add(n.get("id"))
        out |= hidden_layers(n.get("children") or [], h)
    return out


# ============================================================================ format simplifié

OPS = {"op_plus": "plus", "op_concat": "concat", "op_times": "times", "op_dot": "dot", "op_minus": "minus"}
SIDES = {"top": (.5, 0.0), "bottom": (.5, 1.0), "left": (0.0, .5), "right": (1.0, .5), "center": (.5, .5),
         "top-left": (0.0, 0.0), "top-right": (1.0, 0.0), "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
         "haut": (.5, 0.0), "bas": (.5, 1.0), "gauche": (0.0, .5), "droite": (1.0, .5), "centre": (.5, .5)}


def from_simple(nodes, arrows):
    """Format simplifié (écrit par un LLM ou par depuis_pytorch) -> nœuds et flèches du format complet,
    en coordonnées absolues. Les tailles absentes sont estimées (l'éditeur les mesure au chargement)."""
    out, by_id = [], {}
    for i, d in enumerate(nodes):
        d = dict(d)
        uid = d.pop("id", d.get("uid", f"n{i}"))
        lib = d.pop("lib", None)
        t = d.get("type") or ("operator" if lib in OPS else "block")
        if lib in OPS:
            d.setdefault("symbol", OPS[lib])
        elif lib:
            d["def_id"] = lib
            if t == "block" and not d.get("text"):
                d["text"] = lib
        lines = max(1, len(str(d.get("text") or "").split("\n")))
        size = {"operator": (18, 18), "junction": (10, 10), "group": (0, 0),
                "text": (7 * max(len(x) for x in str(d.get("text") or " ").split("\n")), 16 * lines)}
        w, h = size.get(t, (110, max(28, 16 * lines + 8)))
        w, h = float(d.get("w") or w), float(d.get("h") or h)
        if "cx" in d or "cy" in d:
            x, y = float(d.get("cx", 0)) - w / 2, float(d.get("cy", 0)) - h / 2
        else:
            x, y = float(d.get("x", 0)), float(d.get("y", 0))
        d.update(uid=uid, type=t, x=x, y=y, w=w, h=h, parent=None)
        out.append(d)
        by_id[uid] = d
    for d in out:                                    # groupes : parenté et cadre autour des enfants
        for c in d.get("children") or []:
            if c in by_id:
                by_id[c]["parent"] = d["uid"]

    def fit(g, depth=0):
        kids = [k for k in out if k["parent"] == g["uid"]]
        for k in kids:
            if k["type"] == "group" and depth < 20:
                fit(k, depth + 1)
        if kids:
            m = float(g.get("margin", 15) or 15)
            x0, y0 = min(k["x"] for k in kids), min(k["y"] for k in kids)
            x1, y1 = max(k["x"] + k["w"] for k in kids), max(k["y"] + k["h"] for k in kids)
            g.update(x=x0 - m, y=y0 - m, w=x1 - x0 + 2 * m, h=y1 - y0 + 2 * m)
    for d in out:
        if d["type"] == "group" and d["parent"] is None:
            fit(d)

    def end(e, other):
        if isinstance(e, (str, int)):
            e = {"node": e}
        if not isinstance(e, dict):
            return {"x": 0, "y": 0}
        if "node" not in e:
            return {"x": e.get("x", 0), "y": e.get("y", 0)}
        n, o = by_id.get(e["node"]), by_id.get(other.get("node") if isinstance(other, dict) else other)
        if n is None:
            return {"x": 0, "y": 0}
        side = str(e.get("side") or "").lower()
        if side in SIDES:
            fx, fy = SIDES[side]
        elif o is not None:                          # côté face à l'autre extrémité
            dx = (o["x"] + o["w"] / 2) - (n["x"] + n["w"] / 2)
            dy = (o["y"] + o["h"] / 2) - (n["y"] + n["h"] / 2)
            fx, fy = ((1.0 if dx > 0 else 0.0), .5) if abs(dx) > abs(dy) else (.5, (1.0 if dy > 0 else 0.0))
        else:
            fx, fy = .5, .5
        if e.get("at") is not None:
            if fy in (0.0, 1.0) and fx == .5:
                fx = float(e["at"])
            elif fx in (0.0, 1.0) and fy == .5:
                fy = float(e["at"])
        return {"node": e["node"], "fx": fx, "fy": fy}

    arr = []
    for a in arrows:
        s, d = a.get("from", a.get("src")), a.get("to", a.get("dst"))
        sd = s if isinstance(s, dict) else {"node": s}
        dd = d if isinstance(d, dict) else {"node": d}
        arr.append({"src": end(s, dd), "dst": end(d, sd), "head": a.get("head", "end"),
                    "color": a.get("color", "#000000"), "dashed": a.get("dashed", False)})
    return out, arr


# ============================================================================ graphe

@dataclass
class Node:
    uid: object
    type: str                 # block operator text junction group point edgeop sub
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    text: str = ""
    def_id: str = ""
    symbol: str = ""
    label: str = ""
    border: str = "solid"
    parent: object = None
    stack: int = 0
    bold: bool = False
    fs: float = 13
    hint: str = ""            # nom lu dans un texte voisin (points libres)
    ref: str = ""             # référence d'origine (depuis_pytorch : « param:… » = paramètre du modèle)
    data: dict = field(default_factory=dict)

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


@dataclass
class Edge:
    src: object
    dst: object
    sp: tuple                 # point de départ (absolu)
    dp: tuple                 # point d'arrivée (absolu)
    color: str = "#000000"
    dashed: bool = False
    skip: bool = False        # liaison « copie + concaténation » (légende)
    port: int = 0             # sortie utilisée (1 = état d'un RNN)


class DiagramGraph:
    def __init__(self, data):
        self.nodes, self.edges, self.notes = {}, [], []
        self.legend = {}                         # (couleur, pointillés) -> texte de légende
        hidden = hidden_layers(data.get("layers") or [])     # calques masqués : ignorés
        raw = [d for d in (data.get("nodes") or []) if isinstance(d, dict) and d.get("layer") not in hidden]
        arrows = [a for a in (data.get("arrows") or []) if isinstance(a, dict) and a.get("layer") not in hidden]
        kept = {d.get("uid", d.get("id")) for d in raw}
        arrows = [a for a in arrows if all(not isinstance(e, dict) or "node" not in e or e["node"] in kept
                                           for e in (a.get("src"), a.get("dst")))]
        if any("uid" not in d for d in raw) or any("from" in a or "to" in a for a in arrows):
            raw, arrows = from_simple(raw, arrows)
            for d in raw:
                d["_abs"] = True
        by_uid = {d.get("uid"): d for d in raw}

        def absolute(d, depth=0):
            x, y = float(d.get("x", 0)), float(d.get("y", 0))
            par = by_uid.get(d.get("parent"))
            if par is not None and depth < 50 and not d.get("_abs"):
                px, py = absolute(par, depth + 1)
                x, y = x + px, y + py
            return x, y

        for d in raw:
            x, y = absolute(d)
            t = d.get("type", "block")
            n = Node(d.get("uid"), t, x, y, float(d.get("w", 0) or 0), float(d.get("h", 0) or 0),
                     text=d.get("text") or "", def_id=d.get("def_id") or "", symbol=d.get("symbol") or "",
                     label=d.get("label") or "", border=d.get("border_style") or "solid",
                     parent=d.get("parent"), stack=int(d.get("stack") or 0), bold=bool(d.get("bold")),
                     fs=float(d.get("font_size") or 13), ref=str(d.get("ref") or ""))
            self.nodes[n.uid] = n
        self._next = 10 ** 6
        self._arrows(arrows)

    # ---------------------------------------------------------------- outils
    def new_uid(self):
        self._next += 1
        return f"v{self._next}"

    def ancestors(self, uid):
        out, n = [], self.nodes.get(uid)
        while n is not None and n.parent in self.nodes and len(out) < 50:
            out.append(n.parent)
            n = self.nodes[n.parent]
        return out

    def ins(self, uid):
        return [e for e in self.edges if e.dst == uid]

    def outs(self, uid):
        return [e for e in self.edges if e.src == uid]

    def texts(self):
        return [n for n in self.nodes.values() if n.type == "text"]

    # ---------------------------------------------------------------- flèches
    def _arrows(self, arrows):
        used_texts = set()
        free = []
        for a in arrows:
            if not isinstance(a, dict):
                continue
            ends = []
            for key in ("src", "dst"):
                e = a.get(key) or {}
                if "node" in e and e["node"] in self.nodes:
                    n = self.nodes[e["node"]]
                    fx, fy = float(e.get("fx", .5)), float(e.get("fy", .5))
                    ends.append((n.uid, (n.x + fx * n.w, n.y + fy * n.h)))
                else:
                    p = (float(e.get("x", 0)), float(e.get("y", 0)))
                    ends.append((None, p))
            (s, sp), (d, dp) = ends
            if a.get("head") == "start":
                (s, sp), (d, dp) = (d, dp), (s, sp)
            color, dashed = str(a.get("color") or "#000000").lower(), bool(a.get("dashed"))
            if s is None and d is None:
                free.append((sp, dp, color, dashed))
                continue
            if s is None:
                s = self._point(sp)
            if d is None:
                d = self._point(dp)
            self.edges.append(Edge(s, d, sp, dp, color, dashed))

        # légende : flèche entre deux points libres + texte juste à côté
        for sp, dp, color, dashed in free:
            best = None
            for t in self.texts():
                if t.uid in used_texts:
                    continue
                dx = t.x - max(sp[0], dp[0])
                dy = abs(t.cy - (sp[1] + dp[1]) / 2)
                if -5 <= dx <= 40 and dy <= 12 and (best is None or dx < best[0]):
                    best = (dx, t)
            if best and color != "#000000" or best and dashed:
                self.legend[(color, dashed)] = best[1].text
                used_texts.add(best[1].uid)
                best[1].data["legend"] = True

        # nom des points libres : texte le plus proche
        for n in list(self.nodes.values()):
            if n.type != "point":
                continue
            best = None
            for t in self.texts():
                if t.data.get("legend") or self.ins(t.uid) or self.outs(t.uid):
                    continue
                dx = max(t.x - n.x, 0, n.x - (t.x + t.w))
                dy = max(t.y - n.y, 0, n.y - (t.y + t.h))
                d = (dx * dx + dy * dy) ** .5
                if d <= 30 and len(first_line(t.text)) <= 24 and not t.bold and t.fs < 15 \
                        and (best is None or d < best[0]):
                    best = (d, t)
            if best:
                n.hint = best[1].text
                best[1].data["label"] = True

    def _point(self, p):
        uid = self.new_uid()
        self.nodes[uid] = Node(uid, "point", p[0], p[1], 0, 0)
        return uid

    # ------------------------------------------------------------ simplification
    def simplify(self):
        """Jonctions et textes de passage -> liaisons directes ; flèches vers un cadre -> ses blocs ;
        flèches de couleur légendée -> opération sur la liaison."""
        changed = True
        while changed:
            changed = False
            for n in list(self.nodes.values()):
                ins, outs = self.ins(n.uid), self.outs(n.uid)
                passing = n.type == "junction" or (n.type == "text" and ins and outs)
                if not passing or not outs:
                    continue
                if not ins:
                    if n.type == "junction":
                        continue
                if len(ins) > 1:
                    self.notes.append("jonction alimentée par plusieurs flèches : seule la première est suivie")
                src = ins[0] if ins else None
                for e in outs:
                    if src is not None:
                        e.src, e.sp, e.port = src.src, src.sp, src.port
                        if n.type == "text" and not self.nodes[src.src].hint:
                            self.nodes[src.src].hint = n.text
                for e in ins:
                    self.edges.remove(e)
                if src is not None:
                    changed = True
        # jonctions sans entrée : point de départ
        for n in self.nodes.values():
            if n.type == "junction" and self.outs(n.uid) and not self.ins(n.uid):
                n.type = "point"
        # flèches attachées à un cadre
        for e in list(self.edges):
            for end in ("src", "dst"):
                uid = getattr(e, end)
                if self.nodes[uid].type != "group":
                    continue
                inner = [m.uid for m in self.nodes.values() if uid in self.ancestors(m.uid)
                         and m.type not in ("group", "text")]
                if end == "dst":
                    cands = [m for m in inner if not any(x.src not in inner for x in self.ins(m))
                             and not [x for x in self.ins(m) if x.src in inner]]
                else:
                    cands = [m for m in inner if not [x for x in self.outs(m) if x.dst in inner]]
                if not cands:
                    self.edges.remove(e)
                    break
                setattr(e, end, cands[0])
        # opérations des flèches légendées
        for e in list(self.edges):
            text = self.legend.get((e.color, e.dashed))
            if not text:
                continue
            low = plain(text).lower()
            if re.search(r"concat|copie|copy|skip", low):
                e.skip = True
                continue
            parsed = parse_text(re.sub(r"\(.*?\)", "", text))
            if not parsed:
                continue
            specs = parsed[0]
            dst = self.nodes[e.dst]
            dparsed = parse_text(dst.text) if dst.type == "block" else None
            if dparsed and [s.kind for s in dparsed[0]][:1] == [specs[0].kind] and \
                    dparsed[0][0].p.get("k") == specs[0].p.get("k"):
                continue                     # la légende nomme le bloc d'arrivée, rien à insérer
            uid = self.new_uid()
            par = self._common_parent(e.src, e.dst)
            mx, my = (e.sp[0] + e.dp[0]) / 2, (e.sp[1] + e.dp[1]) / 2
            self.nodes[uid] = Node(uid, "edgeop", mx, my, 0, 0, text=re.sub(r"\(.*?\)", "", text).strip(),
                                   parent=par, data={"specs": specs, "to": e.dst})
            self.edges.append(Edge(uid, e.dst, (mx, my), e.dp, e.color, e.dashed, e.skip))
            e.dst, e.dp = uid, (mx, my)

    def _common_parent(self, a, b):
        A = self.ancestors(a)
        for g in self.ancestors(b):
            if g in A:
                return g
        return None


# ============================================================================ génération

HP_DEFAULTS = {"d_model": 512, "num_heads": 8, "d_ff": 2048, "vocab_size": 10000, "hidden_size": 256,
               "patch_size": 16, "num_classes": 10, "d_k": 64, "N": 6, "L": 12, "K": 10, "C": 64,
               "latent_dim": 16}
HP_NOTES = {"d_model": "taille des vecteurs (embeddings, attention)", "num_heads": "têtes d'attention",
            "d_ff": "couche cachée des Feed Forward / MLP", "vocab_size": "taille du vocabulaire",
            "hidden_size": "taille de l'état des RNN", "patch_size": "côté des patchs",
            "num_classes": "nombre de classes (sortie)", "d_k": "dimension des clés (Scale)"}


@dataclass
class Value:
    name: str
    meta: tuple = (None, None)       # (genre, taille) genre ∈ img1d img2d img3d seq vec tokens
    hint: str = ""                   # nom d'origine (q, k, v…) suivi le long des couches
    is_input: bool = False


@dataclass
class ClassCode:
    name: str
    doc: str
    params: dict                     # nom -> défaut (None = obligatoire)
    init: list
    forward_args: list
    forward: list
    inputs: list = field(default_factory=list)    # (nom, genre, forme) pour l'essai
    top: bool = False


class Generator:
    def __init__(self, data, user_defs=None, title=""):
        self.g = DiagramGraph(data)
        self.g.simplify()
        self.user_defs = user_defs or {}
        self.file_title = title
        self.classes = []            # ClassCode dans l'ordre de dépendance
        self.class_names = set()
        self.helpers = []
        self.stubs = {}
        self.defs = {}               # clé de titre -> scope de définition
        self.def_classes = {}        # clé -> ClassCode construite
        self.notes = list(self.g.notes)

    # ---------------------------------------------------------------- portées
    def components(self):
        g = self.g
        graph_nodes = [u for u, n in g.nodes.items() if n.type not in ("group",)
                       and (g.ins(u) or g.outs(u))]
        adj = {u: set() for u in graph_nodes}
        for e in g.edges:
            if e.src in adj and e.dst in adj:
                adj[e.src].add(e.dst)
                adj[e.dst].add(e.src)
        seen, comps = set(), []
        for u in graph_nodes:
            if u in seen:
                continue
            stack, comp = [u], []
            seen.add(u)
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            if any(g.nodes[v].type in ("block", "operator", "edgeop") for v in comp):
                comps.append(comp)
        return comps

    def bbox(self, uids):
        ns = [self.g.nodes[u] for u in uids]
        return (min(n.x for n in ns), min(n.y for n in ns), max(n.x + n.w for n in ns), max(n.y + n.h for n in ns))

    def assign_titles(self, comps):
        """Titre de chaque composante : cadre pointillé qui la contient, sinon texte gras le plus proche."""
        g = self.g
        titles = {}
        free_titles = [t for t in g.texts() if not g.ins(t.uid) and not g.outs(t.uid) and not t.data
                       and (t.bold or t.fs >= 15)]
        pairs = []
        for i, comp in enumerate(comps):
            dashed = None
            core = [u for u in comp if g.nodes[u].type in ("block", "operator", "edgeop")]
            for u in core:
                for a in g.ancestors(u):
                    if g.nodes[a].border == "dashed":
                        dashed = a
            if dashed is not None and all(dashed in g.ancestors(u) for u in core):
                titles[i] = (g.nodes[dashed].label or "Bloc détaillé", dashed)
                continue
            x0, y0, x1, y1 = self.bbox(comp)
            for t in free_titles:
                dx = max(x0 - (t.x + t.w), 0, t.x - x1)
                dy = max(y0 - (t.y + t.h), 0, t.y - y1)
                pairs.append(((dx * dx + dy * dy) ** .5, i, t))
        used = set()
        for d, i, t in sorted(pairs, key=lambda p: p[0]):
            if i in titles or t.uid in used or (d > 150 and len(comps) > 1):
                continue
            titles[i] = (t.text, None)
            used.add(t.uid)
            t.data["title"] = True
        return titles

    # ---------------------------------------------------------------- entrée
    def run(self):
        g = self.g
        comps = self.components()
        titles = self.assign_titles(comps)
        # noms près des cadres (« conv2_x » à droite d'un cadre 2×)
        for n in g.nodes.values():
            if n.type != "group":
                continue
            for t in g.texts():
                if t.data or g.ins(t.uid) or g.outs(t.uid) or t.uid in self._descendants(n.uid):
                    continue
                if not re.search(r"[A-Za-z]", t.text) or len(t.text) > 24 or "\n" in t.text:
                    continue
                gap = t.x - (n.x + n.w)
                if -5 <= gap <= 40 and n.y <= t.cy <= n.y + n.h:
                    n.data["hint"] = t.text
                    t.data["hint"] = True
        # définitions : cadres pointillés, et composantes dont le titre est le texte d'un bloc
        block_keys = {}
        for u, n in g.nodes.items():
            if n.type == "block":
                for k in {norm_key(n.text), norm_key(first_line(n.text))}:
                    block_keys.setdefault(k, set()).add(u)
        mains = []
        for i, comp in enumerate(comps):
            title, dashed = titles.get(i, ("", None))
            key = norm_key(title)
            users = {u for u in block_keys.get(key, set()) if u not in comp} if key else set()
            if dashed is not None or users:
                self.defs[key] = {"comp": comp, "title": title, "dashed": dashed, "users": users}
            else:
                mains.append((comp, title))
        if not mains and comps:                     # tout est référencé : on garde la première
            k = next(iter(self.defs))
            d = self.defs.pop(k)
            mains.append((d["comp"], d["title"]))
        for comp, title in mains:
            name = camel(title, camel(self.file_title or "Modele"))
            cc = self.build(comp, name, kind="main",
                            doc=f"{first_line(tex(title)) or name} (d'après le schéma).")
            cc.top = True
        for key, d in self.defs.items():           # définitions jamais utilisées
            if key not in self.def_classes:
                self.def_class(key)
        for t in g.texts():
            if not t.data and not g.ins(t.uid) and not g.outs(t.uid) and t.text.strip():
                self.notes.append(note_text(t.text))
        return self.render()

    def _descendants(self, gid):
        return {u for u in self.g.nodes if gid in self.g.ancestors(u)}

    # ---------------------------------------------------------------- classes
    def unique_class(self, name):
        base, k = name, 2
        while name in self.class_names:
            name, k = f"{base}{k}", k + 1
        self.class_names.add(name)
        return name

    def def_class(self, key):
        if key in self.def_classes:
            return self.def_classes[key]
        d = self.defs[key]
        self.def_classes[key] = None              # garde contre les références circulaires
        strided = False
        for u in d["users"]:
            text = self.g.nodes[u].text
            rest = " ".join(text.split("\n")[1:])
            sh = split_shapes(text)[1]
            if re.search(r"/\s*\d", rest) or (sh and len(sh[0]) > 1 and sh[0][-1] > sh[1][-1]):
                strided = True
        name = camel(d["title"], "Bloc")
        cc = self.build(d["comp"], name, kind="def", doc=f"« {first_line(d['title'])} » (cadre du schéma).",
                        root_group=d["dashed"], strided=strided)
        self.def_classes[key] = cc
        return cc

    def build(self, comp, name, kind, doc, root_group=None, strided=False, ext_inputs=None):
        b = ClassBuilder(self, comp, self.unique_class(name), kind, doc, root_group, strided, ext_inputs)
        cc = b.build()
        self.classes.append(cc)
        return cc

    def stub_class(self, text, n_in):
        name = camel(first_line(text), "Bloc")
        key = (name, n_in)
        if key in self.stubs:
            return self.stubs[key]
        cname = self.unique_class(name)
        args = ["x"] if n_in <= 1 else [f"x{i + 1}" for i in range(n_in)]
        label = label_line(text)
        body = [f"return {args[0]}  # TODO : calcul de « {label} »" +
                (" (combiner les entrées)" if n_in > 1 else "")]
        cc = ClassCode(cname, f"« {label} » : le schéma ne détaille pas ce bloc.", {},
                       [f"# TODO : couches de « {label} »"], args, body)
        self.classes.append(cc)
        self.stubs[key] = cname
        return cname

    def helper(self, name):
        if name not in self.helpers:
            self.helpers.append(name)

    # ---------------------------------------------------------------- rendu
    def render(self):
        out = []
        head = ["Squelette PyTorch généré par DeepGraph à partir du schéma"
                + (f" « {self.file_title} »" if self.file_title else "") + ".",
                "",
                "Brouillon : la structure suit le schéma ; les réglages absents du schéma ont une valeur",
                "par défaut marquée TODO. Les modules « Lazy » déduisent le nombre de canaux d'entrée au",
                "premier passage : faire un passage d'essai avant de créer l'optimiseur.",
                "",
                "Conventions : convolutions avec padding « same » (k // 2) ; pooling sans pas indiqué :",
                "pas 2 pour un noyau 2×2, sinon pas 1 ; entrées d'une attention, de gauche à droite : V, K, Q."]
        notes = [n for n in dict.fromkeys(self.notes) if n]
        if notes:
            head += ["", "Textes du schéma :"] + [f"  - {n}" for n in notes]
        out.append('"""' + "\n".join(head).replace('"""', "'''") + '\n"""')
        out.append("import math\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F")
        for h in self.helpers:
            out.append(HELPERS[h].strip("\n"))
        for cc in self.classes:
            out.append(render_class(cc))
        out.append(self.render_main())
        code = "\n\n\n".join(out) + "\n"
        body = code.split("import torch.nn.functional as F", 1)[1]
        if not re.search(r"\bmath\.", body):
            code = code.replace("import math\n\n", "", 1)
        if not re.search(r"(?<![\w.])F\.", body):
            code = code.replace("\nimport torch.nn.functional as F", "", 1)
        return code

    def render_main(self):
        tops = [c for c in self.classes if c.top]
        lines = ['if __name__ == "__main__":',
                 "    # essai : entrées aléatoires de forme plausible (TODO : formes réelles)"]
        for cc in tops:
            v = snake(cc.name)
            lines.append(f"    {v} = {cc.name}()")
            args = []
            for nm, kind, shape in cc.inputs:
                if kind == "tokens":
                    lines.append(f"    {nm} = torch.randint(0, 100, {shape})")
                else:
                    lines.append(f"    {nm} = torch.randn{shape}")
                args.append(nm)
            lines.append("    with torch.no_grad():")
            lines.append(f"        out = {v}({', '.join(args)})")
            lines.append("    outs = out if isinstance(out, tuple) else (out,)")
            lines.append(f"    print(\"{cc.name} :\", \", \".join(str(tuple(o.shape)) for o in outs),")
            lines.append(f"          f\"— {{sum(p.numel() for p in {v}.parameters()):,}} paramètres\")")
        if not tops:
            lines.append("    pass")
        return "\n".join(lines)


def render_class(cc):
    params = ", ".join(["self"] + [p if d is None else f"{p}={d!r}" for p, d in cc.params.items()])
    lines = [f"class {cc.name}(nn.Module):", f'    """{cc.doc}"""', "",
             f"    def __init__({params}):", "        super().__init__()"]
    lines += ["        " + ln if ln else "" for ln in cc.init]
    lines += ["", f"    def forward({', '.join(['self'] + cc.forward_args)}):"]
    lines += ["        " + ln if ln else "" for ln in cc.forward]
    return "\n".join(lines)


HELPERS = {
    "PositionalEncoding": '''
class PositionalEncoding(nn.Module):
    """Encodage de position sinusoïdal (Vaswani et al., 2017), ajouté à l'entrée (B, T, d_model)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
''',
    "PositionEmbedding": '''
class PositionEmbedding(nn.Module):
    """Position apprise ajoutée à l'entrée (B, T, d_model), précédée d'un jeton de classe si class_token."""

    def __init__(self, d_model, max_len=1024, class_token=False):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if class_token else None
        self.pos = nn.Parameter(torch.randn(1, max_len + int(class_token), d_model) * 0.02)

    def forward(self, x):
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        return x + self.pos[:, :x.size(1)]
''',
    "PatchEmbedding": '''
class PatchEmbedding(nn.Module):
    """Découpe l'image en patchs p×p et projette chacun : (B, C, H, W) -> (B, N, d_model)."""

    def __init__(self, d_model, patch_size):
        super().__init__()
        self.proj = nn.LazyConv2d(d_model, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
''',
}


# ============================================================================ une classe

@dataclass
class Op:
    node: object
    inputs: list                      # [(Value, Edge)]
    lines: list = field(default_factory=list)
    outs: list = field(default_factory=list)


class ClassBuilder:
    def __init__(self, gen, comp, name, kind, doc, root_group, strided, ext_inputs):
        self.gen, self.g = gen, gen.g
        self.comp = list(comp)
        self.name, self.kind, self.doc = name, kind, doc
        self.root_group = root_group
        self.strided = strided
        self.ext_inputs = ext_inputs or {}        # uid de point -> (nom, meta, indice)
        self.params = {}
        self.init, self.fwd = [], []
        self.attr_count = {}
        self.attrs_used = set()
        self.var_used = set()
        self.values = {}                          # (uid, port) -> Value
        self.consumers = {}
        self.stride_done = False
        self.inputs_spec = []
        self.forward_args = []
        self.shared = {}                          # uid -> ensemble de nœuds qui partagent un module
        self.shared_attr = {}                     # id(ensemble) -> nom de l'attribut

    # ---------------------------------------------------------------- noms
    def hp(self, name, default=None):
        if name not in self.params:
            self.params[name] = HP_DEFAULTS.get(name, 64) if default is None else default
        return name

    def fwd_hp(self, name):
        """Réglage utilisé dans forward : gardé dans self au début de __init__."""
        self.hp(name)
        line = f"self.{name} = {name}"
        if line not in self.init:
            self.init.insert(0, line)
        return name

    def size_expr(self, v, fallback_hp=None, todo=None):
        """Taille lue dans le schéma (int ou symbole) -> expression ; symbole -> paramètre."""
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str) and v:
            if self.kind == "def" and v in self.def_symbols:
                return v
            return self.hp(v)
        if fallback_hp:
            return self.hp(fallback_hp)
        return None

    def attr(self, base):
        base = re.sub(r"\W+", "_", base).strip("_").lower() or "bloc"
        if base[0].isdigit():
            base = "b" + base
        k = self.attr_count.get(base, 0) + 1
        self.attr_count[base] = k
        name = base if k == 1 else f"{base}{k}"
        while name in self.attrs_used or name in self.params:
            k += 1
            name = f"{base}{k}"
        self.attrs_used.add(name)
        return name

    def var(self, base="h"):
        base = base if base and base not in RESERVED else "h"
        if base not in self.var_used and base not in self.params:
            self.var_used.add(base)
            return base
        k = 1 if base in ("x", "h") else 2
        while f"{base}{k}" in self.var_used or f"{base}{k}" in self.params:
            k += 1
        self.var_used.add(f"{base}{k}")
        return f"{base}{k}"

    # ---------------------------------------------------------------- construction
    def build(self):
        g = self.g
        self.def_symbols = set()
        if self.kind == "def":
            self.collect_def_symbols()
        self.collapse_groups()
        nodes = [u for u in self.comp if u in g.nodes]
        order = self.topo(nodes)
        self.compute_sharing(order)
        self.pre_absorb(order)
        # entrées
        inputs = [u for u in order if self.is_source(u)]
        inputs.sort(key=lambda u: self.input_sort_key(u))
        names = {}
        for u in inputs:
            n = g.nodes[u]
            if u in self.ext_inputs:
                nm, meta, hint = self.ext_inputs[u][:3]
                nm = self.var(nm)
                for tok in re.findall(r"[A-Za-z_]\w*", str(meta[1] or "")):
                    self.hp(tok)
            else:
                nm = self.var(ident(n.hint or n.text, "x"))
                meta, hint = self.input_meta(u), ident(n.hint or n.text, "")
            names[u] = nm
            self.values[(u, 0)] = Value(nm, meta, hint, is_input=True)
            self.forward_args.append(nm)
            if u not in self.ext_inputs:
                self.inputs_spec.append((nm, meta[0], self.example_shape(u, n, meta)))
        returns = []
        for u in order:
            if u in names or self.g.nodes[u].data.get("absorbed"):
                continue
            n = g.nodes[u]
            ins = self.collect_inputs(u)
            if self.is_sink(u):
                for v, e in ins:
                    returns.append((v, n))
                continue
            self.emit_node(u, ins)
        # sorties : blocs sans successeur qui produisent une valeur
        for u in order:
            n = g.nodes[u]
            if (u, 0) in self.values and not g.outs(u) and not self.values[(u, 0)].is_input \
                    and not self.is_sink(u):
                returns.append((self.values[(u, 0)], n))
        if not returns:
            returns = [(v, None) for k, v in self.values.items() if v.is_input][:1]
        self.finish_returns(returns)
        params = dict(self.params)
        if self.kind == "def":
            ordered = {s: None for s in self.def_symbol_order if s in self.def_symbols}
            if self.strided:
                ordered["stride"] = 1
            ordered.update({k: v for k, v in params.items() if k not in ordered})
            params = ordered
        cc = ClassCode(self.name, self.doc, params, self.init, self.forward_args, self.fwd,
                       self.inputs_spec)
        return cc

    def finish_returns(self, returns):
        seen, vals, labels = set(), [], []
        for v, n in returns:
            if id(v) in seen:
                continue
            seen.add(id(v))
            vals.append(v.name)
            lab = first_line(tex(n.hint or n.text)) if n is not None and self.role(n.uid) in ("point", "data") \
                else ""
            labels.append(lab)
        if not vals:
            self.fwd.append("return None  # TODO : aucune sortie trouvée dans le schéma")
            return
        com = ", ".join(lab for lab in labels if lab)
        self.fwd.append(f"return {', '.join(vals)}" + (f"  # {com}" if com else ""))

    # ---------------------------------------------------------------- tri
    def flow(self, nodes):
        sx = sy = 0.0
        S = set(nodes)
        for e in self.g.edges:
            if e.src in S and e.dst in S:
                sx += e.dp[0] - e.sp[0]
                sy += e.dp[1] - e.sp[1]
        L = (sx * sx + sy * sy) ** .5 or 1
        return sx / L, sy / L

    def topo(self, nodes):
        g, S = self.g, set(nodes)
        fx, fy = self.flow(nodes)
        key = {u: (g.nodes[u].cx * fx + g.nodes[u].cy * fy, str(u)) for u in nodes}
        indeg = {u: 0 for u in nodes}
        for e in g.edges:
            if e.src in S and e.dst in S:
                indeg[e.dst] += 1
        order, ready = [], sorted([u for u in nodes if indeg[u] == 0], key=key.get)
        left = set(nodes)
        while left:
            if not ready:                                    # boucle : on la coupe
                u = min(left, key=key.get)
                for e in list(g.edges):
                    if e.dst == u and e.src in left:
                        g.edges.remove(e)
                        self.gen.notes.append("boucle dans le schéma : une flèche de retour a été ignorée")
                ready = [u]
            u = ready.pop(0)
            if u not in left:
                continue
            left.discard(u)
            order.append(u)
            succ = set()
            for e in g.edges:
                if e.src == u and e.dst in left:
                    succ.add(e.dst)
                    indeg[e.dst] -= 1
                    if indeg[e.dst] == 0:
                        ready.append(e.dst)
            ready.sort(key=lambda v: (v not in succ, key[v]))
        return order

    # ---------------------------------------------------------------- classification
    def node_specs(self, u):
        """Spec(s) d'un nœud : (liste, répétition, canaux) ; None si non reconnu."""
        n = self.g.nodes[u]
        if n.type == "edgeop":
            return n.data["specs"], 1, None
        if n.type == "operator":
            return [Spec("op", {"symbol": n.symbol or "plus"})], 1, None
        if n.type == "sub":
            return [Spec("sub")], 1, None
        if n.type != "block":
            return None
        if n.def_id and n.def_id in self.gen.user_defs:
            return [Spec("user", {"expr": self.gen.user_defs[n.def_id]})], 1, None
        key_full, key_first = norm_key(n.text), norm_key(first_line(n.text))
        for k in (key_full, key_first):
            if k and k in self.gen.defs and u not in self.gen.defs[k]["comp"]:
                return [Spec("defref", {"key": k})], 1, None
        base = DEF_SPECS.get(n.def_id)
        parsed = parse_text(n.text)
        if base is not None and base.kind == "mlp" and parsed and [s.kind for s in parsed[0]] == ["head"]:
            return parsed
        if base is not None and base.kind not in TEXT_FIRST:
            p = dict(base.p)
            if parsed and len(parsed[0]) == 1 and parsed[0][0].kind == base.kind:
                p.update({k: v for k, v in parsed[0][0].p.items() if v is not None})
            elif base.kind in ("embedding", "rnn"):
                p.update({k: v for k, v in _numbers(plain(n.text)).items() if k == "out"})
            return [Spec(base.kind, p)], 1, None
        if parsed:
            specs, reps, chans = parsed
            if base is not None and len(specs) == 1 and specs[0].kind == base.kind:
                merged = dict(base.p)
                merged.update({k: v for k, v in specs[0].p.items() if v is not None})
                specs = [Spec(base.kind, merged)]
            return specs, reps, chans
        if base is not None:
            p = dict(base.p)
            nums = _numbers(plain(n.text))
            if base.kind in ("conv", "linear"):
                p.update(nums)
            return [Spec(base.kind, p)], 1, None
        return None

    def is_data_text(self, n):
        t = first_line(n.text)
        if not t:
            return True
        return bool(re.search(r"[=∈~]|^\S{1,4}$|^[^\s(]{1,3}\(.*\)", t))

    def role(self, u):
        """input / output / pass / layer / stub / absorbed"""
        n = self.g.nodes[u]
        if n.type in ("point", "text", "junction"):
            return "point"
        if n.type == "block":
            if n.def_id in ("Input", "Output"):
                return "data"
            sp = self.node_specs(u)
            if sp is None:
                return "data" if (self.is_data_text(n) or not self.g.ins(u) or not self.g.outs(u)) else "stub"
            if sp[0][0].kind == "stub":
                return "stub" if (self.g.ins(u) and self.g.outs(u)) or self.g.ins(u) else "data"
            if sp[0][0].kind in ("input", "output"):
                return "data"
        return "layer"

    def is_source(self, u):
        g = self.g
        if g.ins(u):
            return False
        n = g.nodes[u]
        if n.data.get("absorbed"):
            return False
        if u in self.ext_inputs:
            return True
        r = self.role(u)
        if r in ("point", "data"):
            return True
        if r == "layer" and n.type == "block":
            sp = self.node_specs(u)
            if sp and sp[0][0].kind == "noise":
                return True
            return False
        return False

    def is_sink(self, u):
        g = self.g
        if g.outs(u):
            return False
        return self.role(u) in ("point", "data")

    def input_sort_key(self, u):
        n = self.g.nodes[u]
        nm = ident(n.hint or n.text, "")
        qkv = {"q": 0, "k": 1, "v": 2}
        if u in self.ext_inputs:
            return (-1, self.ext_inputs[u][3] if len(self.ext_inputs[u]) > 3 else 0, 0)
        return (0, qkv.get(nm, 3), n.cx + n.cy * 0.001)

    # ---------------------------------------------------------------- formes
    def input_meta(self, u):
        """Genre de l'entrée u, d'après la première couche qui le révèle en aval."""
        g = self.g
        seen, frontier = {u}, [u]
        while frontier:
            nxt = []
            for v in frontier:
                for e in g.outs(v):
                    w = e.dst
                    if w in seen:
                        continue
                    seen.add(w)
                    sp = self.node_specs(w) if g.nodes[w].type in ("block", "edgeop", "operator", "sub") else None
                    kind = None
                    if g.nodes[w].type == "sub":
                        sd = g.nodes[w].data
                        if sd["mode"] == "class":
                            kind = self.input_meta(sd["points"][sd["srcs"][0]])[0] if sd.get("srcs") else None
                        else:
                            sp = self.node_specs(sd["chain"][0])
                            kind = self.kind_of(sp[0][0]) if sp else None
                    elif sp:
                        kind = self.kind_of(sp[0][0])
                    if kind:
                        return (kind, None)
                    nxt.append(w)
            frontier = nxt
        return ("vec", None)

    def kind_of(self, s):
        """Genre de tenseur qu'attend la couche s (None si elle ne le dit pas)."""
        if s.kind == "conv":
            return f"img{s.p.get('dim', 2)}d"
        if s.kind in ("pool", "gpool", "patch", "upsample"):
            return "img2d"
        if s.kind == "embedding":
            return "tokens"
        if s.kind in ("mha", "sdpa", "rnn", "posenc", "addnorm", "matmul", "ff") or \
                (s.kind == "linear" and s.p.get("proj")) or (s.kind == "norm" and s.p.get("op") == "layer"):
            return "seq"
        if s.kind in ("linear", "mlp", "head"):
            return "vec"
        if s.kind == "defref":
            return self.gen.def_class(s.p["key"]) and self.def_in_kind(s.p["key"])
        return None

    def def_in_kind(self, key):
        cc = self.gen.def_classes.get(key)
        if cc and cc.inputs:
            return cc.inputs[0][1]
        return None

    def residual_channels(self, u):
        """Entrée ajoutée telle quelle à une sortie (bloc résiduel dessiné seul) : ses canaux doivent être
        ceux de la première convolution qui suit."""
        g = self.g
        for e in g.outs(u):                           # à travers un cadre répété : son entrée principale
            sd = g.nodes[e.dst].data
            if g.nodes[e.dst].type == "sub" and sd.get("mode") == "class" and sd.get("srcs"):
                return self.residual_channels(sd["points"][sd["srcs"][0]])
        if not any(g.nodes[e.dst].type == "operator" and (g.nodes[e.dst].symbol or "plus") == "plus"
                   for e in g.outs(u)):
            return None
        seen, todo = set(), [e.dst for e in g.outs(u)]
        while todo:
            w = todo.pop(0)
            if w in seen:
                continue
            seen.add(w)
            sp = self.node_specs(w) if g.nodes[w].type in ("block", "edgeop") else None
            if sp and sp[0][0].kind == "conv":
                out = sp[0][0].p.get("out") or sp[2]
                return out if isinstance(out, int) else 64
            todo += [e.dst for e in g.outs(w)]
        return None

    def example_shape(self, u, n, meta):
        kind = meta[0]
        t = plain(n.hint + " " + n.text)
        res = self.residual_channels(u) if kind and kind.startswith("img") else None
        dims = [int(x) for x in re.findall(r"\d+", KERNEL.search(t).group(0))] if KERNEL.search(t) and \
            "patch" not in t.lower() else []
        if kind == "tokens":
            return "(1, 16)"
        if kind and kind.startswith("img"):
            d = int(kind[3])
            sp = 64
            if len(dims) == 3:                         # H×W×C (ou C×H×W)
                c = dims[2] if dims[2] <= 4 else dims[0]
                hw = [x for x in dims if x != c][:2] or dims[:2]
                return f"(1, {c}, {hw[0]}, {hw[-1]})" if d == 2 else f"(1, {c}" + f", {hw[0]}" * d + ")"
            if len(dims) == 2:
                return f"(1, 3, {dims[0]}, {dims[1]})" if d == 2 else "(1, 3" + f", {dims[0]}" * d + ")"
            c = res or 3
            if d == 1:
                return f"(1, {c}, 128)"
            return f"(1, {c}" + f", {sp}" * d + ")"
        if kind == "seq":
            dm = self.params.get("d_model", 64)
            return f"(1, 16, {dm})"
        nums = [x for x in re.findall(r"\b\d+\b", re.sub(r"\(.*?\)", "", t)) if int(x) >= 2]
        return f"(1, {nums[0] if nums else 64})"

    # ---------------------------------------------------------------- groupes
    def collapse_groups(self):
        g = self.g
        comp = set(self.comp)
        groups = set()
        for u in comp:
            for a in g.ancestors(u):
                if a != self.root_group and (self.root_group is None or self.root_group in g.ancestors(a)):
                    groups.add(a)
        groups = [x for x in groups if g.nodes[x].type == "group"]
        groups.sort(key=lambda x: -len(g.ancestors(x)))
        for gid in groups:
            gn = g.nodes[gid]
            inner = [u for u in self.comp if u in g.nodes and gid in g.ancestors(u)]
            if not inner:
                continue
            rep = repeat_label(gn.label)
            if rep:
                gn.data.setdefault("name", rep[1])
                self.collapse(gid, inner, repeat=rep[0])
            elif gn.label.strip() and self.is_chain(inner):
                self.collapse(gid, inner, repeat=None)

    def is_chain(self, inner):
        g, S = self.g, set(inner)
        if len(inner) < 1:
            return False
        ext_in = {e.src for u in inner for e in g.ins(u) if e.src not in S}
        ext_out = [e for u in inner for e in g.outs(u) if e.dst not in S]
        if len(ext_in) != 1 or len({e.src for e in ext_out}) > 1:
            return False
        for u in inner:
            if g.nodes[u].type not in ("block", "edgeop"):
                return False
            if len([e for e in g.ins(u) if e.src in S]) > 1 or len([e for e in g.outs(u) if e.dst in S]) > 1:
                return False
            if self.role(u) != "layer":
                return False
            sp = self.node_specs(u)
            if not sp or not all(s.kind in SIMPLE and s.kind != "posenc" for s in sp[0]):
                return False
        return True

    def collapse(self, gid, inner, repeat):
        """Remplace les nœuds du cadre par un nœud « sub » (classe répétée, ou nn.Sequential)."""
        g, S = self.g, set(inner)
        gn = g.nodes[gid]
        uid = g.new_uid()
        sub = Node(uid, "sub", gn.x, gn.y, gn.w, gn.h, label=gn.label, parent=gn.parent,
                   data={"group": gid, "repeat": repeat, "inner": list(inner), "hint": gn.data.get("hint", "")})
        g.nodes[uid] = sub
        in_edges = [e for u in inner for e in g.ins(u) if e.src not in S]
        out_edges = [e for u in inner for e in g.outs(u) if e.dst not in S]
        # sous-classe (cadre répété contenant plusieurs nœuds) : entrées -> points virtuels
        single = repeat is not None and len([u for u in inner if g.nodes[u].type in ("block", "operator",
                                                                                      "edgeop", "sub")]) == 1
        if repeat is None or single:
            sub.data["mode"] = "seq" if repeat is None else "single"
            sub.data["chain"] = self.topo(list(inner))
            for e in in_edges:
                sub.data.setdefault("first_in", e)
                e.dst = uid
            for e in out_edges:
                e.src = uid
            for u in inner:
                g.nodes[u].data["inside"] = uid
            self.comp = [u for u in self.comp if u not in S] + [uid]
            sub.data["in_kind"] = None
            return
        sub.data["mode"] = "class"
        srcs = []
        for e in in_edges:
            if (e.src, e.port) not in srcs:
                srcs.append((e.src, e.port))
        # entrée principale : celle qui arrive à l'opposé de la sortie
        cx, cy = gn.cx, gn.cy
        if out_edges:
            ox, oy = out_edges[0].dp[0] - cx, out_edges[0].dp[1] - cy
        else:
            ox, oy = self.flow(self.comp)
        def score(sp):
            s = g.nodes[sp[0]]
            return (s.cx - cx) * ox + (s.cy - cy) * oy
        srcs.sort(key=score)
        points = {}
        for i, sp in enumerate(srcs):
            pu = g.new_uid()
            first = next(e for e in in_edges if (e.src, e.port) == sp)
            g.nodes[pu] = Node(pu, "point", first.sp[0], first.sp[1], 0, 0, parent=gid)
            points[sp] = pu
        new_edges = []
        for e in in_edges:
            pu = points[(e.src, e.port)]
            new_edges.append(Edge(e.src, uid, e.sp, e.dp, e.color, e.dashed, e.skip, e.port))
            e.src, e.port = pu, 0
        seen = set()
        for e in new_edges:                       # une liaison par valeur extérieure
            if (e.src, e.port) not in seen:
                seen.add((e.src, e.port))
                g.edges.append(e)
        outs = []
        for e in out_edges:
            if e.src not in outs:
                outs.append(e.src)
        out_pts = {}
        for o in outs:
            pu = g.new_uid()
            g.nodes[pu] = Node(pu, "point", g.nodes[o].cx, g.nodes[o].cy, 0, 0, parent=gid)
            out_pts[o] = pu
            g.edges.append(Edge(o, pu, (0, 0), (0, 0)))
        for e in out_edges:
            e.src = uid
        sub.data["srcs"] = srcs
        sub.data["points"] = points
        sub.data["inner_nodes"] = list(inner) + list(points.values()) + list(out_pts.values())
        self.comp = [u for u in self.comp if u not in S] + [uid]

    # ---------------------------------------------------------------- partage
    def compute_sharing(self, order):
        """Cellules récurrentes dessinées dépliées dans le temps (même texte, reliées par leur état) :
        un seul module ; idem pour les blocs identiques qui ne servent qu'à ces cellules."""
        g = self.g
        sets = []
        rnn = [u for u in order if self.spec0(u) == "rnn"]
        parent_of = {}
        for u in rnn:
            parent_of[u] = u
        def find(x):
            while parent_of[x] != x:
                x = parent_of[x]
            return x
        for e in g.edges:
            if e.src in parent_of and e.dst in parent_of:
                a, b = g.nodes[e.src], g.nodes[e.dst]
                if a.text == b.text and a.def_id == b.def_id and a.parent == b.parent:
                    parent_of[find(e.src)] = find(e.dst)
        groups = {}
        for u in rnn:
            groups.setdefault(find(u), []).append(u)
        for members in groups.values():
            if len(members) > 1:
                sets.append(set(members))
        changed = True
        while changed:
            changed = False
            owner = {u: i for i, s in enumerate(sets) for u in s}
            cands = {}
            for u in order:
                n = g.nodes[u]
                if u in owner or n.type != "block" or self.role(u) != "layer":
                    continue
                for direction, es in (("out", g.outs(u)), ("in", g.ins(u))):
                    peers = {owner.get(e.dst if direction == "out" else e.src) for e in es}
                    if len(es) == 1 and len(peers) == 1 and None not in peers:
                        cands.setdefault((n.def_id, n.text, direction, peers.pop()), []).append(u)
                        break
            for members in cands.values():
                if len(members) > 1:
                    sets.append(set(members))
                    changed = True
        for s in sets:
            for u in s:
                self.shared[u] = s

    def spec0(self, u):
        sp = self.node_specs(u) if self.g.nodes[u].type in ("block", "edgeop") else None
        return sp[0][0].kind if sp else None

    def pre_absorb(self, order):
        """Encodage de position posé à côté d'un ⊕ : devient un module appliqué à l'autre entrée.
        Bruit (ε ~ N(0, I)) qui n'alimente qu'un échantillonnage : tiré dans le forward."""
        g = self.g
        for u in order:
            if self.spec0(u) == "noise" and g.outs(u) and all(self.spec0(e.dst) == "sampling" for e in g.outs(u)):
                g.nodes[u].data["absorbed"] = True
        for u in order:
            n = g.nodes[u]
            if n.type != "operator":
                continue
            for e in g.ins(u):
                s = g.nodes[e.src]
                if s.type == "block" and not g.ins(e.src) and s.ref.startswith("param:"):
                    s.data["absorbed"] = True
                    n.data.setdefault("params", []).append(e.src)
            if (n.symbol or "plus") != "plus":
                continue
            for e in g.ins(u):
                s = e.src
                if g.ins(s) or g.nodes[s].type != "block":
                    continue
                sp = self.node_specs(s)
                if sp and sp[0][0].kind == "posenc":
                    g.nodes[s].data["absorbed"] = True
                    n.data["posenc"] = (s, sp[0][0])

    # ---------------------------------------------------------------- entrées d'un nœud
    def collect_inputs(self, u):
        g = self.g
        ins = []
        for e in g.ins(u):
            if g.nodes[e.src].data.get("absorbed"):
                continue
            v = self.values.get((e.src, e.port)) or self.values.get((e.src, 0))
            if v is None:
                continue
            ins.append((v, e))
        ins.sort(key=lambda ve: (ve[1].dp[0], ve[1].dp[1]))
        return ins

    # ---------------------------------------------------------------- émission
    def out_value(self, u, ins, meta, port=0, hint="", base=None):
        """Nom de la valeur produite : reprend celui de l'entrée unique si elle ne sert plus."""
        g = self.g
        name = None
        if len(ins) == 1 and port == 0:
            v, e = ins[0]
            n_cons = len([x for x in g.edges if x.src == e.src and x.port == e.port])
            if n_cons == 1 and re.fullmatch(r"[xh]\d*", v.name):
                name = v.name
        n = g.nodes[u]
        if name is None:
            if base:
                name = self.var(base)
            elif n.type == "block" and self.role(u) == "layer" and parse_text(n.text) is None:
                name = self.var(ident(n.text, "x"))
            else:
                name = self.var("x")
        if not hint and len(ins) == 1:
            hint = ins[0][0].hint
        val = Value(name, meta, hint)
        self.values[(u, port)] = val
        return val

    def comment(self, n):
        if n.type in ("block", "edgeop") and n.text.strip():
            return "  # " + label_line(n.text)
        if n.type == "sub" and n.label.strip():
            return "  # cadre « " + n.label.strip() + " »"
        return ""

    def emit_node(self, u, ins):
        g = self.g
        n = g.nodes[u]
        if n.type == "sub":
            return self.emit_sub(u, ins)
        if n.type == "operator":
            return self.emit_operator(u, ins)
        if not ins:
            return
        r = self.role(u)
        if r == "data":                               # bloc de données au milieu : simple nom
            v = ins[0][0]
            nm = self.var(ident(n.text, "h"))
            self.fwd.append(f"{nm} = {v.name}{self.comment(n)}")
            self.values[(u, 0)] = Value(nm, v.meta, v.hint)
            return
        if r == "stub":
            cname = self.gen.stub_class(n.text, len(ins))
            return self.emit_module(u, ins, f"{cname}()", snake(cname), ins[0][0].meta)
        specs, reps, chans = self.node_specs(u)
        s0 = specs[0]
        if s0.kind == "user":
            return self.emit_module(u, ins, s0.p["expr"], ident(n.text, "bloc"), ins[0][0].meta)
        if s0.kind == "defref":
            return self.emit_defref(u, ins, s0.p["key"])
        handler = getattr(self, "emit_" + s0.kind, None)
        if handler is not None and s0.kind not in SIMPLE:
            return handler(u, ins, s0)
        return self.emit_simple(u, ins, specs, reps, chans)

    # .................................................................. modules simples
    def align_kinds(self, ins, force_seq=False):
        """Images et séquences mêlées (ViT : patchs + jeton de classe) : l'image devient une séquence
        de positions, opération que les schémas ne dessinent pas."""
        kinds = {v.meta[0] for v, _ in ins}
        if ("seq" not in kinds and not force_seq) or not any(k and k.startswith("img") for k in kinds):
            return ins
        out = []
        for v, e in ins:
            if v.meta[0] and v.meta[0].startswith("img"):
                nm = self.var("x")
                self.fwd.append(f"{nm} = {v.name}.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H·W, C)")
                v = Value(nm, ("seq", v.meta[1]), v.hint)
            out.append((v, e))
        return out

    def param_operand(self, p_uid, sym, ins, names):
        """Paramètre appris (jeton de classe, position) : nn.Parameter de forme déduite de l'autre entrée."""
        pn = self.g.nodes[p_uid]
        nm = self.attr(ident(pn.text, "param"))
        meta = ins[0][0].meta if ins else (None, None)
        d = meta[1] or self.hp("d_model")
        x = names[0] if names else "x"
        if meta[0] == "seq" and sym == "concat":
            self.init.append(f"self.{nm} = nn.Parameter(torch.zeros(1, 1, {d}))  # {label_line(pn.text)}")
            expr = f"self.{nm}.expand({x}.size(0), -1, -1)"
            return [expr] + names if re.search(r"token|cls|class", pn.text, re.I) else names + [expr]
        if meta[0] == "seq":
            self.init.append(f"self.{nm} = nn.Parameter(torch.zeros(1, {self.hp('max_len', 1024)}, {d}))"
                             f"  # {label_line(pn.text)} (max_len positions)")
            return names + [f"self.{nm}[:, :{x}.size(1)]"]
        self.init.append(f"self.{nm} = nn.Parameter(torch.zeros(1))  # {label_line(pn.text)} — TODO : forme")
        return names + [f"self.{nm}"]

    def combine(self, u, ins):
        """Plusieurs entrées pour une couche à une entrée : concaténation (légende) ou TODO."""
        if len(ins) <= 1:
            return ins[0][0].name, ins[0][0].meta, ins[0][0].hint
        ins = self.align_kinds(ins)
        skips = [ve for ve in ins if ve[1].skip]
        others = [ve for ve in ins if not ve[1].skip]
        ordered = skips + others
        kind = ordered[0][0].meta[0]
        dim = 1 if kind and kind.startswith("img") else -1
        nm = self.var("h")
        line = f"{nm} = torch.cat([{', '.join(v.name for v, _ in ordered)}], dim={dim})"
        if not skips:
            line += "  # TODO : plusieurs entrées, concaténées par défaut"
        self.fwd.append(line)
        return nm, (kind, None), ""

    def emit_simple(self, u, ins, specs, reps, chans):
        g = self.g
        n = g.nodes[u]
        x, meta, hint = self.combine(u, ins)
        stride_param = None
        if self.kind == "def" and self.strided and not self.stride_done and \
                any(s.kind in ("conv", "pool") for s in specs):
            stride_param = "stride"
            self.stride_done = True
        last = self.is_last_layer(u)
        sized = [i for i, s in enumerate(specs) if s.kind not in ("act", "softmax", "dropout", "identity")]
        last_i = sized[-1] if sized else len(specs) - 1
        if n.type == "edgeop" and chans is None:          # up-conv d'une légende : canaux du bloc d'arrivée
            dp = parse_text(g.nodes[n.data["to"]].text) if n.data.get("to") in g.nodes else None
            chans = dp[2] if dp else None
            if chans is None and dp and dp[0][0].p.get("out") is not None:
                chans = dp[0][0].p.get("out")
        ctors, todos = [], []
        for _ in range(max(1, reps)):
            for i, s in enumerate(specs):
                c, meta, t = self.layer(s, meta, chans, stride_param, last and i == last_i)
                stride_param = None
                if c:
                    ctors.extend(c)
                todos.extend(t)
        todos = list(dict.fromkeys(todos))
        if not ctors:
            return self.passthrough(u, ins, x, meta, hint, todos)
        ctor = ctors[0] if len(ctors) == 1 else "nn.Sequential(\n    " + ",\n    ".join(ctors) + ",\n)"
        base = self.attr_base(specs, n)
        att = self.module_attr(u, base, ctor, n, todos)
        fake_ins = [(Value(x, meta, hint), ins[0][1])] if len(ins) > 1 else ins
        out = self.out_value(u, fake_ins if len(ins) == 1 else [], meta, hint=hint)
        self.fwd.append(f"{out.name} = self.{att}({x}){self.comment(n)}")

    def passthrough(self, u, ins, x, meta, hint, todos):
        n = self.g.nodes[u]
        out = self.out_value(u, ins if len(ins) == 1 else [], meta, hint=hint)
        line = f"{out.name} = {x}" + self.comment(n)
        if todos:
            line += ("  # " if "#" not in line else " — ") + "TODO : " + "; ".join(todos)
        if out.name != x or todos:
            self.fwd.append(line)

    def module_attr(self, u, base, ctor, n, todos):
        """Attribut du module de u (créé au premier usage ; partagé par les cellules dépliées)."""
        key = id(self.shared[u]) if u in self.shared else None
        if key is not None and key in self.shared_attr:
            return self.shared_attr[key]
        att = self.attr(base)
        self.add_init(att, ctor, n, todos)
        if key is not None:
            self.shared_attr[key] = att
        return att

    def add_init(self, att, ctor, n, todos):
        com = self.comment(n)
        lines = ctor.split("\n")
        first = f"self.{att} = {lines[0]}"
        tail = ""
        if todos:
            tail = "  # TODO : " + "; ".join(todos)
        if len(lines) == 1:
            self.init.append(first + (com if not todos else com + (" — " + tail[4:] if com else tail)))
        else:
            self.init.append(first + com)
            self.init.extend(lines[1:-1])
            self.init.append(lines[-1] + tail)

    def attr_base(self, specs, n):
        names = {"conv": "conv", "pool": "pool", "gpool": "pool", "upsample": "up", "norm": "norm",
                 "act": "act", "softmax": "softmax", "dropout": "dropout", "flatten": "flatten",
                 "linear": "fc", "embedding": "embedding", "ff": "feed_forward", "mlp": "mlp",
                 "head": "head", "patch": "patch_embedding", "posenc": "position", "identity": "identity",
                 "todo": "bloc"}
        s = specs[0]
        base = names.get(s.kind, s.kind)
        if s.kind == "conv" and s.p.get("transposed"):
            base = "upconv"
        if s.kind == "pool":
            base = s.p.get("op", "max") + "pool"
        if s.kind == "norm":
            base = {"batch": "bn", "layer": "norm", "group": "gn", "instance": "inorm", "rms": "norm",
                    "lrn": "lrn"}.get(s.p.get("op"), "norm")
        if s.kind == "act":
            base = s.p.get("op", "act")
        if len(specs) > 1 and specs[1].kind not in ("act", "dropout", "norm", "softmax"):
            base = base + "_" + names.get(specs[1].kind, specs[1].kind)
        elif len(specs) > 1 and s.kind in ("norm",) and specs[1].kind == "act":
            base = base + "_" + specs[1].p.get("op", "act")
        if len(specs) > 1 and all(x.kind == s.kind or x.kind == "act" for x in specs) and s.kind == "conv":
            base = "convs"
        par = self.g.nodes.get(n.parent) if n.parent is not None else None
        if par is not None and par.type == "group" and par.label.strip() and not \
                repeat_label(par.label) and par.border != "dashed":
            pre = group_name(par.label)
            base = pre + "_" + base if pre and not base.startswith(pre) else base
        return base

    def is_last_layer(self, u):
        """Vrai si u est la dernière couche « à taille » avant une sortie (seules des activations suivent)."""
        g = self.g
        seen = set()
        frontier = [u]
        while frontier:
            v = frontier.pop()
            for e in g.outs(v):
                w = e.dst
                if w in seen:
                    continue
                seen.add(w)
                if self.is_sink(w):
                    continue
                k = self.spec0(w)
                if k in ("act", "softmax", "dropout", "identity"):
                    frontier.append(w)
                    continue
                return False
        return bool(seen) or not g.outs(u)

    def layer(self, s, meta, chans, stride_param=None, last=False):
        """Spec -> ([constructeurs], meta de sortie, [todos])."""
        kind, feat = meta
        todos = []
        k = s.kind
        p = s.p
        img = kind if kind and kind.startswith("img") else None
        dim = int(img[3]) if img else 2

        def out_size(fallback=64, what="la taille de sortie"):
            v = p.get("out")
            if v is None and chans:
                v = chans
            if self.kind == "def" and v is not None and v == getattr(self, "def_cout", None):
                v = "C"
            if v is None and last:
                if "vocab_size" in self.params:
                    return self.hp("vocab_size")
                return self.hp("d_model" if kind == "seq" else "num_classes")
            e = self.size_expr(v)
            if e is None:
                todos.append(f"préciser {what}, que le schéma n'indique pas")
                return str(fallback)
            return e

        if k == "conv":
            d = p.get("dim") or dim
            ks = p.get("k") or (2 if p.get("transposed") else 3)
            out = out_size(64, "le nombre de canaux")
            st = p.get("stride") or (ks if p.get("transposed") else 1)
            stv = stride_param or (str(st) if st != 1 else None)
            if p.get("transposed"):
                c = f"nn.LazyConvTranspose{d}d({out}, kernel_size={ks}" + (f", stride={stv}" if stv else "") + ")"
            else:
                sh = p.get("_shape")
                if sh and len(sh[0]) > 1 and len(sh[1]) > 1 and isinstance(st, int):
                    pad = max(0, -(-((sh[1][-1] - 1) * st + ks - sh[0][-1]) // 2))
                else:
                    pad = max(0, (ks - st + 1) // 2) if isinstance(st, int) else ks // 2
                args = [f"kernel_size={ks}"]
                if stv:
                    args.append(f"stride={stv}")
                if pad:
                    args.append(f"padding={pad}")
                cin = str(sh[0][0]) if sh else (feat if isinstance(feat, str) and feat.isdigit() else None)
                if p.get("depthwise") and cin:
                    c = f"nn.Conv{d}d({cin}, {cin}, {', '.join(args)}, groups={cin})"
                    out = cin
                else:
                    c = f"nn.LazyConv{d}d({', '.join([out] + args)})"
                    if p.get("depthwise"):
                        todos.append("convolution depthwise : ajouter groups=nombre de canaux d'entrée")
            pre = []
            if kind == "seq":
                todos.append("convolution sur une séquence : passer en (B, C, T) avec x.transpose(1, 2)")
            return pre + [c], (f"img{d}d", out), todos
        if k == "pool":
            ks = p.get("k") or 2
            st = p.get("stride") or (2 if ks == 2 else 1)
            op = "Max" if p.get("op") == "max" else "Avg"
            args = [f"kernel_size={ks}"]
            stv = stride_param or str(st)
            args.append(f"stride={stv}")
            if st == 1 and not stride_param and ks // 2:
                args.append(f"padding={ks // 2}")
            return [f"nn.{op}Pool{dim}d({', '.join(args)})"], (img or "img2d", feat), todos
        if k == "gpool":
            op = "Max" if p.get("op") == "max" else "Avg"
            return [f"nn.Adaptive{op}Pool{dim}d(1)", "nn.Flatten()"], ("vec", feat), todos
        if k == "upsample":
            sc = p.get("out") if isinstance(p.get("out"), int) and p.get("out") <= 8 else 2
            return [f"nn.Upsample(scale_factor={sc})"], meta, todos
        if k == "norm":
            op = p.get("op") or ("batch" if img else "layer")
            if op == "batch":
                if kind == "seq":
                    todos.append("BatchNorm sur une séquence : nn.BatchNorm1d attend (B, C, T)")
                bd = dim if img else 1
                return [f"nn.LazyBatchNorm{bd}d()"], meta, todos
            if op == "instance":
                return [f"nn.LazyInstanceNorm{dim}d()"], meta, todos
            if op == "lrn":
                return ["nn.LocalResponseNorm(5)"], meta, todos
            sh = p.get("_shape")
            f = (str(sh[1][-1] if kind == "seq" or len(sh[1]) <= 2 else sh[1][0]) if sh else None) \
                or feat or self.hp("d_model")
            if feat is None and not sh:
                todos.append("taille normalisée inconnue (d_model par défaut)")
            if op == "group":
                return [f"nn.GroupNorm(32, {f})"], meta, todos
            if op == "rms":
                return [f"nn.RMSNorm({f})"], meta, todos
            if img:
                todos.append("LayerNorm sur une image : préciser les dimensions normalisées")
            return [f"nn.LayerNorm({f})"], meta, todos
        if k == "act":
            if p.get("guess"):
                todos.append("activation non précisée (ReLU par défaut)")
            return [ACT_CTOR.get(p.get("op"), "nn.ReLU()")], meta, todos
        if k == "softmax":
            d = p.get("dim", -1 if kind == "seq" else 1)
            fn = "LogSoftmax" if p.get("log") else "Softmax"
            if last and p.get("dim") is None:
                todos.append("retirer si la perte est nn.CrossEntropyLoss (elle attend des logits)")
            return [f"nn.{fn}(dim={d})"], meta, todos
        if k == "dropout":
            return [f"nn.Dropout({p.get('p', 0.5)})"], meta, todos
        if k == "flatten":
            return ["nn.Flatten()"], ("vec", None), todos
        if k == "linear":
            pre = []
            if img:
                pre = ["nn.Flatten()"]
                kind = "vec"
            if p.get("proj") and p.get("out") is None:
                out = self.hp("d_model")
            else:
                out = out_size(128, "la taille de sortie")
            return pre + [f"nn.LazyLinear({out})"], (kind or "vec", out), todos
        if k == "embedding":
            d = self.size_expr(p.get("out"), "d_model")
            return [f"nn.Embedding({self.hp('vocab_size')}, {d})"], ("seq", d), todos
        if k == "ff" or k == "mlp":
            d = feat or self.hp("d_model")
            act = "nn.ReLU()" if k == "ff" else "nn.GELU()"
            return [f"nn.Sequential(nn.LazyLinear({self.hp('d_ff')}), {act}, nn.Linear({self.hp('d_ff')}, {d}))"], \
                (kind, d), todos
        if k == "head":
            nm = "vocab_size" if "vocab_size" in self.params else "num_classes"
            if kind == "seq":
                todos.append("tête appliquée à chaque jeton : prendre x[:, 0] (jeton de classe) ou x.mean(1)")
            return [f"nn.LazyLinear({self.hp(nm)})"], (kind, nm), todos
        if k == "patch":
            self.gen.helper("PatchEmbedding")
            d = self.hp("d_model")
            return [f"PatchEmbedding({d}, {self.hp('patch_size')})"], ("seq", d), todos
        if k == "posenc":
            d = feat or self.hp("d_model")
            if p.get("learned"):
                self.gen.helper("PositionEmbedding")
                return [f"PositionEmbedding({d}" + (", class_token=True" if p.get("cls") else "") + ")"], meta, todos
            self.gen.helper("PositionalEncoding")
            return [f"PositionalEncoding({d})"], meta, todos
        if k == "identity":
            return ["nn.Identity()"], meta, todos
        if k == "todo":
            todos.append(p.get("what", "bloc à compléter"))
            return [], meta, todos
        if k == "concat":
            todos.append("concaténation avec une seule entrée (copies empilées ?)")
            return [], meta, todos
        todos.append(f"bloc « {k} » non traduit")
        return [], meta, todos

    def emit_module(self, u, ins, ctor, base, meta, call_args=None):
        n = self.g.nodes[u]
        att = self.module_attr(u, base, ctor, n, [])
        args = call_args or [v.name for v, _ in ins]
        out = self.out_value(u, ins if len(ins) == 1 else [], meta)
        self.fwd.append(f"{out.name} = self.{att}({', '.join(args)}){self.comment(n)}")
        return out

    # .................................................................. cas particuliers
    def emit_operator(self, u, ins):
        n = self.g.nodes[u]
        sym = n.symbol or "plus"
        if n.data.get("posenc"):
            s_uid, spec = n.data["posenc"]
            x, meta, hint = (ins[0][0].name, ins[0][0].meta, ins[0][0].hint) if len(ins) == 1 else \
                (" + ".join(v.name for v, _ in ins), ins[0][0].meta, "")
            c, meta2, todos = self.layer(spec, meta, None)
            att = self.attr("position")
            self.add_init(att, c[0], self.g.nodes[s_uid], todos)
            out = self.out_value(u, ins if len(ins) == 1 else [], meta, hint=hint)
            self.fwd.append(f"{out.name} = self.{att}({x}){self.comment(self.g.nodes[s_uid])}")
            return
        if not ins and not n.data.get("params"):
            return
        seq_params = any(re.search(r"token|cls|pos|embed", self.g.nodes[p].text, re.I)
                         for p in n.data.get("params", []))
        ins = self.align_kinds(ins, force_seq=seq_params)
        names = [v.name for v, _ in ins]
        meta = ins[0][0].meta if ins else (None, None)
        todo = ""
        for p_uid in n.data.get("params", []):         # paramètre du modèle posé à côté de l'opérateur
            names = self.param_operand(p_uid, sym, ins, names)
        if sym == "plus":
            expr = " + ".join(names)
            if self.kind == "def" and self.strided and len(ins) == 2:
                inp = [v for v, _ in ins if v.is_input]
                if inp:
                    att = self.attr("shortcut")
                    width = "C" if "C" in self.def_symbols else None
                    if width:
                        self.init.append(f"self.{att} = nn.Identity() if stride == 1 else nn.Sequential(")
                        self.init.append(f"    nn.LazyConv2d({width}, kernel_size=1, stride=stride, bias=False),")
                        self.init.append("    nn.LazyBatchNorm2d(),")
                        self.init.append(")  # raccourci : Conv 1×1 quand la résolution change "
                                         "(TODO : aussi si le nombre de canaux change)")
                        expr = " + ".join(f"self.{att}({v.name})" if v.is_input else v.name for v, _ in ins)
        elif sym in ("times", "dot"):
            expr = " * ".join(names)
        elif sym == "minus":
            expr = " - ".join(names)
            if len(ins) == 2:
                todo = "  # TODO : vérifier l'ordre (a − b)"
        elif sym == "concat":
            kind = meta[0]
            tokens = seq_params and kind == "seq"           # jeton ajouté : axe des positions
            dim = 1 if (kind and kind.startswith("img")) or tokens else -1
            expr = f"torch.cat([{', '.join(names)}], dim={dim})"
            meta = (kind, meta[1] if tokens else None)
        else:
            expr = names[0]
            todo = f"  # TODO : opérateur « {sym} »"
        out = self.out_value(u, ins if len(ins) == 1 else [], meta,
                             hint=ins[0][0].hint if len(ins) == 1 else "")
        self.fwd.append(f"{out.name} = {expr}{todo}")

    def order_qkv(self, ins):
        """Entrées d'une attention -> (q, k, v) : noms (q, k, v) s'ils sont connus, sinon position V K Q."""
        byhint = {v.hint.lower()[:1]: v for v, _ in ins if v.hint and v.hint.lower()[:1] in "qkv"}
        if len(byhint) == len(ins) and len(ins) >= 2:
            q = byhint.get("q", ins[-1][0])
            k = byhint.get("k", byhint.get("v", ins[0][0]))
            v = byhint.get("v", k)
            return q, k, v, False
        vals = [v for v, _ in ins]
        if len(vals) == 1:
            return vals[0], vals[0], vals[0], False
        if len(vals) == 2:
            return vals[1], vals[0], vals[0], True
        return vals[2], vals[1], vals[0], False

    def emit_mha(self, u, ins, s):
        n = self.g.nodes[u]
        meta = ins[0][0].meta
        d = meta[1] or self.hp("d_model")
        att = self.attr("attention" if not s.p.get("masked") else "masked_attention")
        self.add_init(att, f"nn.MultiheadAttention({d}, {self.hp('num_heads')}, batch_first=True)", n, [])
        q, k, v, guess = self.order_qkv(ins)
        out = self.out_value(u, [], (meta[0], d))
        extra = ""
        if s.p.get("masked"):
            mask = self.var("mask")
            self.fwd.append(f"{mask} = nn.Transformer.generate_square_subsequent_mask({q.name}.size(1), "
                            f"device={q.name}.device)")
            extra = f", attn_mask={mask}"
        line = f"{out.name} = self.{att}({q.name}, {k.name}, {v.name}{extra}, need_weights=False)[0]" + \
            self.comment(n)
        if guess:
            line += " — TODO : vérifier quelle entrée est Q"
        self.fwd.append(line)

    def emit_sdpa(self, u, ins, s):
        n = self.g.nodes[u]
        q, k, v, guess = self.order_qkv(ins)
        out = self.out_value(u, [], q.meta)
        self.fwd.append(f"{out.name} = F.scaled_dot_product_attention({q.name}, {k.name}, {v.name})"
                        + self.comment(n))

    def emit_addnorm(self, u, ins, s):
        n = self.g.nodes[u]
        meta = ins[0][0].meta
        d = next((v.meta[1] for v, _ in ins if v.meta[1]), None) or self.hp("d_model")
        att = self.attr("norm")
        self.add_init(att, f"nn.LayerNorm({d})", n, [])
        out = self.out_value(u, [], (meta[0], d))
        self.fwd.append(f"{out.name} = self.{att}({' + '.join(v.name for v, _ in ins)}){self.comment(n)}")

    def emit_rnn(self, u, ins, s):
        g = self.g
        n = g.nodes[u]
        rnn_ins = [(v, e) for v, e in ins if self.spec0(e.src) == "rnn"]
        data = [(v, e) for v, e in ins if (v, e) not in rnn_ins]
        if not data:
            data = rnn_ins[:1]
            rnn_ins = rnn_ins[1:]
        x, meta, hint = self.combine(u, data)
        feat = meta[1] or self.hp("d_model")
        hid = self.size_expr(s.p.get("out"), "hidden_size")
        cell = s.p.get("cell", "LSTM")
        bi = ", bidirectional=True" if s.p.get("bi") else ""
        ctor = f"nn.{cell}({feat}, {hid}, batch_first=True{bi})"
        base = cell.lower()
        par = g.nodes.get(n.parent) if n.parent is not None else None
        if par is not None and par.type == "group" and par.label.strip():
            base = ident(par.label, "") + "_" + base
        first = u not in self.shared or id(self.shared[u]) not in self.shared_attr
        att = self.module_attr(u, base, ctor, n, [])
        if first and u in self.shared:
            self.init[-1] += " — dépliée dans le temps : mêmes poids à chaque pas"
        out_kind = ("seq", f"2 * {hid}" if s.p.get("bi") else hid)
        has_seq_out = any(self.spec0(e.dst) != "rnn" for e in g.outs(u)) or not g.outs(u)
        has_state_out = any(self.spec0(e.dst) == "rnn" for e in g.outs(u))
        o = self.out_value(u, [], out_kind, base="out") if has_seq_out else None
        st = Value(self.var("state"), (None, None)) if has_state_out else None
        if st:
            self.values[(u, 1)] = st
            for e in g.outs(u):
                if self.spec0(e.dst) == "rnn":
                    e.port = 1
        args = x + (f", {rnn_ins[0][0].name}" if rnn_ins else "")
        lhs = f"{o.name if o else '_'}, {st.name if st else '_'}"
        self.fwd.append(f"{lhs} = self.{att}({args}){self.comment(n)}")

    def emit_matmul(self, u, ins, s):
        if len(ins) < 2:
            return self.passthrough(u, ins, ins[0][0].name, ins[0][0].meta, ins[0][0].hint,
                                    ["MatMul avec une seule entrée"])
        a, b = ins[0][0], ins[1][0]
        bt = f"{b.name}.transpose(-2, -1)" if b.hint.lower().startswith("k") else b.name
        out = self.out_value(u, [], a.meta)
        line = f"{out.name} = torch.matmul({a.name}, {bt})" + self.comment(self.g.nodes[u])
        self.fwd.append(line)

    def emit_scale(self, u, ins, s):
        v = ins[0][0]
        out = self.out_value(u, ins, v.meta)
        self.fwd.append(f"{out.name} = {v.name} / math.sqrt(self.{self.fwd_hp('d_k')})"
                        + self.comment(self.g.nodes[u]))

    def emit_mask(self, u, ins, s):
        v = ins[0][0]
        out = self.out_value(u, ins, v.meta)
        line = f"{out.name} = {v.name}" + self.comment(self.g.nodes[u]) + \
            " — facultatif : .masked_fill(mask == 0, float('-inf'))"
        if out.name != v.name:
            self.fwd.append(line)
        else:
            self.fwd.append("# " + line.split("#", 1)[1].strip())

    def emit_concat(self, u, ins, s):
        if len(ins) == 1:
            v = ins[0][0]
            out = self.out_value(u, ins, v.meta)
            self.fwd.append(f"# {self.comment(self.g.nodes[u])[4:]} — TODO : concaténation des têtes "
                            "(copies empilées du schéma)")
            if out.name != v.name:
                self.fwd.insert(len(self.fwd) - 1, f"{out.name} = {v.name}")
            return
        kind = ins[0][0].meta[0]
        dim = 1 if kind and kind.startswith("img") else -1
        out = self.out_value(u, [], (kind, None))
        self.fwd.append(f"{out.name} = torch.cat([{', '.join(v.name for v, _ in ins)}], dim={dim})"
                        + self.comment(self.g.nodes[u]))

    def emit_sampling(self, u, ins, s):
        vals = [v for v, _ in ins if not (self.spec0(_.src) == "noise")]
        mu = next((v for v in vals if not re.search(r"sigma|log|var|std", v.hint + v.name)), vals[0])
        lv = next((v for v in vals if v is not mu), None)
        out = self.out_value(u, [], mu.meta, base="z")
        if lv is not None:
            self.fwd.append(f"{out.name} = {mu.name} + torch.exp(0.5 * {lv.name}) * torch.randn_like({mu.name})"
                            + self.comment(self.g.nodes[u]))
        else:
            self.fwd.append(f"{out.name} = {mu.name} + torch.randn_like({mu.name})  # TODO : écart-type")

    def emit_noise(self, u, ins, s):
        v = ins[0][0]
        self.values[(u, 0)] = v

    def emit_stub(self, u, ins, s):
        cname = self.gen.stub_class(self.g.nodes[u].text, len(ins))
        return self.emit_module(u, ins, f"{cname}()", snake(cname), ins[0][0].meta)

    def defref_ctor(self, u, key):
        """Bloc qui porte le titre d'une définition -> (classe, constructeur(avec_pas)).

        « Basic Block\n128, /2 » : les nombres remplissent les tailles symboliques de la définition (C),
        « /2 » son paramètre stride."""
        n = self.g.nodes[u]
        cc = self.gen.def_class(key)
        if cc is None:
            return None, None
        rest = " ".join(n.text.split("\n")[1:]) if norm_key(first_line(n.text)) == key else ""
        nums = re.findall(r"(?<![/\w])(\d+)(?![\w])", plain(rest))
        m = re.search(r"/\s*(\d+)", plain(rest))
        sh = split_shapes(n.text)[1]
        if sh:                                       # « 64×56×56 → 128×28×28 » : C = 128, pas 2
            nums = [str(sh[1][0])]
            if len(sh[0]) > 1 and sh[0][-1] > sh[1][-1]:
                m = re.search(r"(\d+)", str(sh[0][-1] // sh[1][-1]))
        args = []
        for i, p in enumerate([p for p, d in cc.params.items() if d is None]):
            args.append(nums[i] if i < len(nums) else self.hp(p))
        args += [f"{p}={self.hp(p, d)}" for p, d in cc.params.items() if d is not None and p != "stride"]
        stride = m.group(1) if m and "stride" in cc.params else None

        def ctor(with_stride=True):
            return f"{cc.name}({', '.join(args + ([f'stride={stride}'] if stride and with_stride else []))})"
        return cc, ctor

    def emit_defref(self, u, ins, key):
        cc, ctor = self.defref_ctor(u, key)
        if cc is None:
            return self.emit_stub(u, ins, None)
        call = [v.name for v, _ in ins]
        hints = {v.hint.lower(): v.name for v, _ in ins if v.hint}
        if len(cc.forward_args) == len(ins) and all(a in hints for a in cc.forward_args):
            call = [f"{a}={hints[a]}" for a in cc.forward_args]       # q=…, k=…, v=…
        return self.emit_module(u, ins, ctor(), snake(cc.name), ins[0][0].meta, call_args=call)

    # .................................................................. cadres
    def emit_sub(self, u, ins, s=None):
        g = self.g
        n = g.nodes[u]
        mode = n.data["mode"]
        if mode in ("seq", "single"):
            chain = n.data["chain"]
            x, meta, hint = self.combine(u, ins) if ins else ("x", (None, None), "")
            ctors, todos = [], []
            for c_uid in chain:
                sp = self.node_specs(c_uid)
                if mode == "single" and (not sp or sp[0][0].kind in ("defref", "stub", "user")
                                         or self.role(c_uid) != "layer"):
                    return self.emit_repeat_single(u, ins, c_uid, sp[0][0] if sp and self.role(c_uid) == "layer"
                                                   else Spec("stub"))
                specs, reps, chans = sp
                stride_param = None
                last = False
                for _ in range(max(1, reps)):
                    for sp_ in specs:
                        c, meta, t = self.layer(sp_, meta, chans, stride_param, last)
                        ctors.extend(c)
                        todos.extend(t)
            if mode == "single":
                r = n.data["repeat"]
                inner = ctors[0] if len(ctors) == 1 else "nn.Sequential(" + ", ".join(ctors) + ")"
                ctor = f"nn.Sequential(*[{inner} for _ in range({self.rep_expr(r)})])"
            else:
                ctor = "nn.Sequential(\n    " + ",\n    ".join(ctors) + ",\n)" if len(ctors) > 1 else \
                    (ctors[0] if ctors else "nn.Identity()")
            base = group_name(n.data.get("hint") or n.label)
            first = self.node_specs(chain[0]) if chain else None
            if not base:                                 # cadre « 0 », « 1 » : nommé d'après son contenu
                base = self.attr_base(first[0], g.nodes[chain[0]]) if first else "bloc"
            att = self.attr(base)
            com = self.comment(n) if group_name(n.label) else \
                "  # " + " + ".join(first_line(split_shapes(g.nodes[c].text)[0]) for c in chain)
            self.add_init(att, ctor, n, list(dict.fromkeys(todos)))
            if not group_name(n.label) and "\n" not in ctor:
                self.init[-1] = self.init[-1].split("  # ")[0] + com
            elif not group_name(n.label):
                self.init[-len(ctor.split("\n"))] = f"self.{att} = nn.Sequential({com}"
            out = self.out_value(u, ins if len(ins) == 1 else [], meta, hint=hint)
            self.fwd.append(f"{out.name} = self.{att}({x}){com}")
            return
        # cadre répété contenant plusieurs nœuds : classe de couche + boucle
        srcs, points = n.data["srcs"], n.data["points"]
        ext = {}
        main_val = None
        for i, sp in enumerate(srcs):
            pu = points[sp]
            v = self.values.get(sp) or self.values.get((sp[0], 0))
            nm = "x" if i == 0 else ("context" if i == 1 else f"context{i}")
            ext[pu] = (nm, v.meta if v else (None, None), v.hint if v else "", i)
            if i == 0:
                main_val = v
        gname = g.nodes[n.data["group"]].data.get("name")
        lname = camel(gname) if gname else \
            self.name + (camel(n.data.get("hint"), "") if n.data.get("hint") else "Bloc")
        cc = self.gen.build(n.data["inner_nodes"], lname, kind="layer",
                            doc=f"Bloc répété « {n.label.strip()} » de {self.name}.",
                            root_group=n.data["group"], ext_inputs=ext)
        kw = ", ".join(f"{p}={self.hp(p, d)}" for p, d in cc.params.items())
        r = self.rep_expr(n.data["repeat"])
        gname = g.nodes[n.data["group"]].data.get("name")
        base = ident(n.data.get("hint"), "") or (snake(camel(gname)) + "s" if gname else "blocs")
        att = self.attr(base)
        self.init.append(f"self.{att} = nn.ModuleList([{cc.name}({kw}) for _ in range({r})])"
                         + self.comment(n))
        others = [self.values.get(sp) or self.values.get((sp[0], 0)) for sp in srcs[1:]]
        meta = main_val.meta if main_val else (None, None)
        x = main_val.name if main_val else "x"
        main_ins = [(main_val, next(e for e in g.ins(u) if (e.src, e.port) == srcs[0]))] if main_val else []
        out = self.out_value(u, main_ins, meta)
        if out.name != x:
            self.fwd.append(f"{out.name} = {x}")
        extra = "".join(f", {v.name}" for v in others if v)
        self.fwd.append(f"for layer in self.{att}:")
        self.fwd.append(f"    {out.name} = layer({out.name}{extra})")

    def emit_repeat_single(self, u, ins, c_uid, spec):
        """Cadre « N× » autour d'un seul bloc (« Basic Block, 128, /2 ») : nn.Sequential de N copies ;
        un pas (/2) ne s'applique qu'à la première (convention ResNet)."""
        g = self.g
        n, cn = g.nodes[u], g.nodes[c_uid]
        r = n.data["repeat"]
        x, meta, hint = self.combine(u, ins)
        name = ident(n.data.get("hint"), "")
        note = []
        if spec.kind == "defref":
            cc, ctor = self.defref_ctor(c_uid, spec.p["key"])
            if r.isdigit() and int(r) <= 4:
                parts = [ctor(True)] + [ctor(False)] * (int(r) - 1)
                inner = "nn.Sequential(\n    " + ",\n    ".join(parts) + ",\n)"
            else:
                inner = f"nn.Sequential({ctor(True)}, *[{ctor(False)} for _ in range({self.rep_expr(r)} - 1)])"
            if ctor(True) != ctor(False):
                note = ["pas : premier bloc seulement"]
            att = self.attr(name or snake(cc.name) + "s")
        else:
            one = f"{self.gen.stub_class(cn.text, 1)}()" if spec.kind == "stub" else spec.p["expr"]
            inner = f"nn.Sequential(*[{one} for _ in range({self.rep_expr(r)})])"
            att = self.attr(name or ident(cn.text, "bloc") + "s")
        self.add_init(att, inner, n, [])
        if note:
            self.init[-1] += "  # " + note[0]
        out = self.out_value(u, ins if len(ins) == 1 else [], meta, hint=hint)
        self.fwd.append(f"{out.name} = self.{att}({x})  # {label_line(cn.text)} ×{r}")

    def rep_expr(self, r):
        if r is None:
            return "1"
        return r if r.isdigit() else self.hp(r)

    # .................................................................. définitions
    def collect_def_symbols(self):
        """Symboles des tailles (« Conv 3×3, C ») d'un cadre de définition -> paramètres obligatoires."""
        order = []
        for u in self.comp:
            n = self.g.nodes[u]
            if n.type != "block":
                continue
            parsed = parse_text(n.text)
            if not parsed:
                continue
            for s in parsed[0]:
                v = s.p.get("out")
                if isinstance(v, str) and v not in order:
                    order.append(v)
        # définition dessinée pour un exemple (depuis_pytorch : formes concrètes) : les tailles égales à
        # la sortie de l'exemple deviennent le paramètre C, fourni par chaque bloc qui l'utilise
        self.def_cout = None
        if not order:
            for u in self.comp:
                n = self.g.nodes[u]
                if not self.g.outs(u) and (n.type in ("text", "point") or n.def_id == "Output"):
                    sh = split_shapes(n.text)[1]
                    if sh:
                        self.def_cout = sh[1][0]
                        order.append("C")
        self.def_symbols = set(order)
        self.def_symbol_order = order


def generate(data, user_defs=None, title=""):
    """Diagramme .dgraph (format complet ou simplifié) -> (code, infos).

    user_defs : {identifiant de bloc: constructeur PyTorch} pour les blocs de l'utilisateur (champ
    « pytorch » de mes_blocs.yaml). infos : classes, main (classes de premier niveau), todo (nombre de
    TODO dans les classes), notes."""
    gen = Generator(data, user_defs, title)
    code = gen.run()
    return code, {"classes": [c.name for c in gen.classes], "main": [c.name for c in gen.classes if c.top],
                  "todo": sum(render_class(c).count("TODO") for c in gen.classes), "notes": gen.notes}
