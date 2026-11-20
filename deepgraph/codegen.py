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

Le code produit utilise des couches classiques (nn.Conv2d(in, out, …), nn.Linear(in, out)…) : la
taille d'entrée de chaque couche est suivie le long du graphe (canaux, et taille de l'image pour un
Flatten suivi d'une Linear). Ce qui manque au schéma, ou ne peut pas en être déduit, reçoit une valeur
par défaut marquée « TODO ».

Aucune dépendance à Qt ni à torch : le module lit le format complet (celui qu'écrit l'éditeur).
"""
import keyword
import re
import unicodedata
from dataclasses import dataclass, field, replace

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
    m = re.fullmatch(r"(\d+)\s*\*?\s*([A-Za-z](?:_\w+)?)", tok)       # « 4C » -> « 4 * C »
    if m and m.group(2).lower() not in ("x", "d"):
        return f"{m.group(1)} * {m.group(2)}"
    return None


def _numbers(s):
    """Réglages courants d'un morceau de texte : taille de noyau, pas, taille de sortie."""
    out = {}
    m = ARROW.search(s)
    if m:
        out["out"] = _size(m.group(2))
        if _size(m.group(1)) is not None:
            out["in"] = _size(m.group(1))
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
    m = re.search(r"\bp(?:ad(?:ding)?)?\s*=?\s*(\d+)\b", s)
    if m:
        out["pad"] = int(m.group(1))
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
        p = {"masked": "mask" in low, "self": "self" in low}
        m = re.search(r"(\d+)\s*(?:t[êe]tes?|heads?)\b", low)
        if m:
            p["heads"] = int(m.group(1))
        return Spec("mha", p)
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
        m = re.search(r"(?<![\d.])([123])\s*d\b", low)
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
            m = re.search(r"(\d+)", low) if op == "group" else None
            return Spec("norm", {"op": op, **({"groups": int(m.group(1))} if m else {})})
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
    path: tuple = ()          # tracé (points absolus), coudes compris
    label: str = ""           # texte posé le long de la flèche (« skip », « K, V »)


def arrow_path(a, sp, dp, first_side=None):
    """Tracé d'une flèche comme le dessine l'éditeur (droite, 1 ou 2 angles, coudes libres)."""
    style = a.get("style") or "straight"
    if style == "straight":
        return (sp, dp)
    first = a.get("first") if a.get("first") in ("h", "v") else (first_side or "v")
    if style == "free":
        pts, cur, ax = [sp], sp, first
        for c in a.get("bends") or []:
            cur = (cur[0], sp[1] + float(c)) if ax == "v" else (sp[0] + float(c), cur[1])
            pts.append(cur)
            ax = "h" if ax == "v" else "v"
        pts.append((cur[0], dp[1]) if ax == "v" else (dp[0], cur[1]))
        return tuple(pts) + (dp,)
    if style == "ortho1":
        return (sp, (sp[0], dp[1]) if first == "v" else (dp[0], sp[1]), dp)
    a1, a2 = (sp[1], dp[1]) if first == "v" else (sp[0], dp[0])
    m = a1 + float(a["mid"]) if a.get("mid") is not None else (a1 + a2) / 2
    return (sp, (sp[0], m), (dp[0], m), dp) if first == "v" else (sp, (m, sp[1]), (m, dp[1]), dp)


def rect_segment_dist(r, p, q):
    """Distance entre un rectangle (x, y, w, h) et le segment pq."""
    x0, y0, x1, y1 = r[0], r[1], r[0] + r[2], r[1] + r[3]

    def pt_rect(px, py):
        dx, dy = max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1)
        return (dx * dx + dy * dy) ** .5

    def pt_seg(px, py):
        vx, vy = q[0] - p[0], q[1] - p[1]
        L2 = vx * vx + vy * vy
        t = 0 if L2 == 0 else max(0, min(1, ((px - p[0]) * vx + (py - p[1]) * vy) / L2))
        dx, dy = px - (p[0] + t * vx), py - (p[1] + t * vy)
        return (dx * dx + dy * dy) ** .5
    # segment horizontal ou vertical qui traverse le rectangle
    if (p[1] == q[1] and y0 <= p[1] <= y1 and min(p[0], q[0]) <= x1 and max(p[0], q[0]) >= x0) or \
            (p[0] == q[0] and x0 <= p[0] <= x1 and min(p[1], q[1]) <= y1 and max(p[1], q[1]) >= y0):
        return 0.0
    return min([pt_rect(*p), pt_rect(*q)] + [pt_seg(cx, cy) for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))])


def with_todo(line, todo):
    """Ajoute un TODO au bout d'une ligne de code (après son commentaire éventuel)."""
    if "TODO : " in line:
        return f"{line} ; {todo}"
    if "  # " in line:
        return f"{line} — TODO : {todo}"
    return f"{line}  # TODO : {todo}"


def io_channels(text):
    """« Bloc Conv\n64 → 128 » -> (64, 128) ; « 512 (+ skip 256) → 256 » -> (512, 256) ; sinon (None, None)."""
    rest = plain("\n".join((text or "").split("\n")[1:]))
    m = re.search(r"(?:(\d+)[^→\n]*?)?(?:→|->)\s*(\d+)\b", rest)
    if not m:
        return None, None
    return (int(m.group(1)) if m.group(1) else None), int(m.group(2))


def edge_hint(label):
    """Texte d'une flèche -> nom court : « K, V » -> k ; « skip : copie de l'entrée » -> skip."""
    words = [w for w in re.split(r"[\s,:;/·()]+", plain(label).strip()) if w]
    return ident(words[0], "") if words else ""


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
            ends, side = [], None
            for key in ("src", "dst"):
                e = a.get(key) or {}
                if "node" in e and e["node"] in self.nodes:
                    n = self.nodes[e["node"]]
                    fx, fy = float(e.get("fx", .5)), float(e.get("fy", .5))
                    ends.append((n.uid, (n.x + fx * n.w, n.y + fy * n.h)))
                    if key == "src":
                        side = "h" if fx in (0.0, 1.0) and 0 < fy < 1 else "v" if fy in (0.0, 1.0) else None
                else:
                    p = (float(e.get("x", 0)), float(e.get("y", 0)))
                    ends.append((None, p))
            (s, sp), (d, dp) = ends
            path = arrow_path(a, sp, dp, side)
            if a.get("head") == "start":
                (s, sp), (d, dp) = (d, dp), (s, sp)
                path = path[::-1]
            color, dashed = str(a.get("color") or "#000000").lower(), bool(a.get("dashed"))
            if s is None and d is None:
                free.append((sp, dp, color, dashed))
                continue
            if s is None:
                s = self._point(sp)
            if d is None:
                d = self._point(dp)
            self.edges.append(Edge(s, d, sp, dp, color, dashed, path=path))

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

        # textes posés le long d'une flèche (« skip », « K, V ») : nom de la valeur qu'elle porte
        for t in self.texts():
            if t.data or self.ins(t.uid) or self.outs(t.uid) or t.bold or t.fs >= 15 \
                    or "\n" in t.text.strip() or len(t.text.strip()) > 24 or not edge_hint(t.text):
                continue
            best = None
            for e in self.edges:
                pts = e.path or (e.sp, e.dp)
                d = min(rect_segment_dist((t.x, t.y, t.w, t.h), p, q) for p, q in zip(pts, pts[1:]))
                if d <= 12 and (best is None or d < best[0]):
                    best = (d, e)
            if best:
                best[1].label = t.text.strip()
                t.data["edge_label"] = True

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
            self.edges.append(Edge(uid, e.dst, (mx, my), e.dp, e.color, e.dashed, e.skip, label=e.label))
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
    meta: tuple = (None, None)       # (genre, taille[, Spatial]) genre ∈ img1d img2d img3d seq vec tokens
    hint: str = ""                   # nom d'origine (q, k, v…) suivi le long des couches
    is_input: bool = False


@dataclass(frozen=True)
class Spatial:
    """Taille spatiale d'une image suivie de couche en couche (pour un Flatten suivi d'une Linear)."""
    dims: tuple                      # (H, W), (T,) ou (D, H, W)
    why: frozenset = frozenset()     # raisons d'en douter : « entrée 64×64 » (supposée), « padding »


def hw_of(meta):
    return meta[2] if len(meta) > 2 else None


def paren(e):
    e = str(e)
    return f"({e})" if re.search(r"[+\-]", e) else e


def size_sum(feats):
    """Canaux d'une concaténation : « 512 », « 512 » -> « 1024 » ; « C », « C » -> « 2 * C » ;
    None si l'une des tailles est inconnue."""
    if not feats or any(f is None for f in feats):
        return None
    feats = [str(f) for f in feats]
    nums = [int(f) for f in feats if f.isdigit()]
    syms = [f for f in feats if not f.isdigit()]
    terms = [s if syms.count(s) == 1 else f"{syms.count(s)} * {paren(s)}" for s in dict.fromkeys(syms)]
    if nums:
        terms.append(str(sum(nums)))
    return " + ".join(terms)


def subst(expr, mapping):
    """Taille exprimée avec les paramètres d'une classe -> avec les valeurs d'un appel (« C » -> « 128 »)."""
    if expr is None:
        return None
    out = re.sub(r"\b[A-Za-z_]\w*\b",
                 lambda m: paren(mapping[m.group(0)]) if mapping.get(m.group(0)) is not None else m.group(0),
                 str(expr))
    if re.fullmatch(r"[\d\s+*()]+", out) and "**" not in out:     # purement numérique : calculée
        try:
            return str(eval(out, {"__builtins__": {}}))  # noqa: S307 — chiffres, +, * et parenthèses
        except (SyntaxError, TypeError, ValueError):
            return out
    return out


def conv_dims(sp, k, s, pad, transposed=False, guessed=False):
    """Taille spatiale après une convolution ou un pooling (None si elle ne peut pas être suivie)."""
    if sp is None or not all(isinstance(v, int) for v in (k, s, pad)):
        return None
    if transposed:
        dims = tuple((h - 1) * s - 2 * pad + k for h in sp.dims)
    else:
        dims = tuple((h + 2 * pad - k) // s + 1 for h in sp.dims)
    if min(dims) < 1:
        return None
    return Spatial(dims, sp.why | ({"padding"} if guessed else frozenset()))


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
    in_params: dict = field(default_factory=dict)  # paramètre de taille d'entrée -> indice de l'argument
    out_meta: tuple = (None, None)                 # meta de la sortie (tailles en noms de paramètres)


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
                        root_group=d["dashed"], strided=strided, users=d["users"])
        self.def_classes[key] = cc
        return cc

    def build(self, comp, name, kind, doc, root_group=None, strided=False, ext_inputs=None, users=None):
        b = ClassBuilder(self, comp, self.unique_class(name), kind, doc, root_group, strided, ext_inputs, users)
        cc = b.build()
        self.classes.append(cc)
        return cc

    def stub_class(self, text, n_in, proj=None):
        """Classe à compléter pour un bloc non détaillé ; proj (genre du tenseur) : le bloc change la taille
        (« 256 → 128 ») et reçoit une projection provisoire, pour que le reste du réseau s'exécute."""
        name = camel(first_line(text), "Bloc")
        key = (name, n_in, proj)
        if key in self.stubs:
            return self.stubs[key]
        cname = self.unique_class(name)
        args = ["x"] if n_in <= 1 else [f"x{i + 1}" for i in range(n_in)]
        label = label_line(text)
        if proj:
            conv = {"img1d": "nn.Conv1d", "img2d": "nn.Conv2d", "img3d": "nn.Conv3d"}.get(proj)
            a, b = ("in_channels", "out_channels") if conv else ("in_features", "out_features")
            layer = f"{conv}({a}, {b}, kernel_size=1)" if conv else f"nn.Linear({a}, {b})"
            cc = ClassCode(cname, f"« {label} » : le schéma ne détaille pas ce bloc.", {a: None, b: None},
                           [f"self.proj = {layer}  # TODO : projection provisoire, à remplacer par les "
                            f"couches de « {label} »"], args,
                           [f"return self.proj({args[0]})  # TODO : calcul de « {label} »"])
            self.classes.append(cc)
            self.stubs[key] = cname
            return cname
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
                "par défaut marquée TODO. La taille d'entrée de chaque couche est déduite de la couche",
                "précédente : changer une taille de sortie demande de changer aussi l'entrée qui suit.",
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

    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

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
    def __init__(self, gen, comp, name, kind, doc, root_group, strided, ext_inputs, users=None):
        self.def_users = set(users or ())
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
        self.in_syms = {}                         # in_channels… -> (défaut, lu dans le schéma, libellé, indice)
        self.bound = {}                           # symbole du schéma -> taille suivie (« C » de « C → 4C »)
        self.out_meta = (None, None)

    # ---------------------------------------------------------------- tailles d'entrée
    def in_size(self, feat, todos, kind=None, what="le nombre de canaux d'entrée"):
        """Taille d'entrée d'une couche, suivie depuis l'entrée du réseau ; inconnue -> défaut + TODO."""
        if feat is not None:
            return str(feat)
        if kind == "seq":
            return self.hp("d_model")
        todos.append(f"préciser {what}, que le schéma ne permet pas de déduire")
        return "64"

    def flat_size(self, feat, sp, todos):
        """Taille après Flatten d'une image (C × H × W), avec un TODO si elle repose sur une supposition."""
        if feat is None or sp is None:
            return None
        if all(d == 1 for d in sp.dims):
            return str(feat)
        expr = " * ".join([paren(feat)] + [str(d) for d in sp.dims])
        why = []
        for w in sorted(sp.why):
            if w.startswith("entrée"):
                why.append(f"une {w} supposée (celle de l'essai)")
            elif w == "padding":
                why.append("un padding supposé (« same » pour les convolutions)")
        if why:
            todos.append(f"{expr} calculé avec {' et '.join(why)} : à vérifier")
        return expr

    def shape_meta(self, meta, out):
        """Ligne de formes du schéma (« → 64×56×56 ») : taille de sortie exacte."""
        kind, feat = meta[0], meta[1]
        if kind and kind.startswith("img") and len(out) > 1:
            return (kind, feat if feat is not None else str(out[0]), Spatial(tuple(out[1:])))
        if kind in ("vec", "seq") and out:
            return (kind, feat if feat is not None else str(out[-1]))
        return meta

    def input_dims(self, u, n, kind):
        """Entrée u : (canaux ou taille, dimensions spatiales, canaux lus dans le schéma ?, dimensions lues ?)."""
        t = plain(n.hint + " " + n.text)
        if kind == "vec":
            nums = [x for x in re.findall(r"\b\d+\b", re.sub(r"\(.*?\)", "", t)) if int(x) >= 2]
            return (int(nums[0]) if nums else 64), None, bool(nums), False
        d = int(kind[3])
        m = KERNEL.search(t)
        dims = [int(x) for x in re.findall(r"\d+", m.group(0))] if m and "patch" not in t.lower() else []
        word = 3 if re.search(r"\brgb\b|\bcouleurs?\b|\bcolou?r\b", t, re.I) else \
            1 if re.search(r"\bgr[ia]y\w*|\bgris\b|\bniveaux? de gris|\bmono", t, re.I) else None
        if len(dims) == 3:                             # H×W×C (ou C×H×W)
            c = dims[2] if dims[2] <= 4 else dims[0]
            hw = [x for x in dims if x != c][:2] or dims[:2]
            return c, ((hw[0], hw[-1]) if d == 2 else (hw[0],) * d), True, True
        if len(dims) == 2 and d == 1:                  # signal : C × L (ou L × C)
            return min(dims), (max(dims),), True, True
        res = self.residual_channels(u)                # entrée additionnée à la sortie d'une convolution
        c = word or res or (64 if res == 0 else 3)     # 64 : canaux par défaut d'une convolution
        if len(dims) == 2:
            return c, ((dims[0], dims[1]) if d == 2 else (dims[0],) * d), bool(word or res), True
        return c, ((128,) if d == 1 else (64,) * d), bool(word or res), False

    def input_size(self, u, n, nm, meta, idx, mandatory):
        """Entrée image ou vecteur : sa taille devient un paramètre (in_channels, in_features)."""
        kind = meta[0]
        img = bool(kind and kind.startswith("img"))
        if not img and kind != "vec":
            return meta
        what = "channels" if img else "features"
        named = self.kind != "def" and getattr(self, "n_sized_inputs", 1) > 1 and \
            not re.fullmatch(r"(x|h|inputs?)\d*", nm)
        sym = f"in_{what}" if f"in_{what}" not in self.in_syms and not named else f"{ident(nm, 'x')}_{what}"
        if mandatory:
            self.in_syms[sym] = (None, True, nm, idx)
            return (kind, sym, hw_of(meta))
        c, hw, c_text, hw_text = self.input_dims(u, n, kind)
        self.in_syms[sym] = (c, c_text, label_line(n.hint or n.text) or nm, idx)
        sp = None
        if img:
            sp = Spatial(tuple(hw), frozenset() if hw_text else frozenset({f"entrée {'×'.join(map(str, hw))}"}))
        return (kind, sym, sp)

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
        """Taille lue dans le schéma (int, symbole ou « 4 * C ») -> expression ; symbole -> paramètre,
        sauf s'il nomme une taille reçue (« Linear C → 4C » : C = taille de l'entrée)."""
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str) and v:
            def tok(m):
                t = m.group(0)
                if t in self.bound:
                    return paren(self.bound[t])
                if self.kind == "def" and t in self.def_symbols:
                    return t
                return self.hp(t)
            return subst(re.sub(r"\b[A-Za-z_]\w*\b", tok, v), {})
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
        self.n_sized_inputs = sum(1 for u in inputs if u not in self.ext_inputs and
                                  (self.input_meta(u)[0] or "").startswith(("img", "vec")))
        for idx, u in enumerate(inputs):
            n = g.nodes[u]
            if u in self.ext_inputs:
                ext = self.ext_inputs[u]
                nm, meta, hint = ext[:3]
                nm = self.var(nm)
                if (ext[3] if len(ext) > 3 else 0) == 0 and self.kind == "layer":
                    meta = self.input_size(u, n, nm, meta, idx, mandatory=True)   # bloc répété : paramètre
                for tok in re.findall(r"[A-Za-z_]\w*", str(meta[1] or "")):
                    if tok not in self.in_syms:
                        self.hp(tok)
            else:
                nm = self.var(ident(n.hint or n.text, "x"))
                meta, hint = self.input_meta(u), ident(n.hint or n.text, "")
                meta = self.input_size(u, n, nm, meta, idx, mandatory=self.kind == "def")
            names[u] = nm
            self.values[(u, 0)] = Value(nm, meta, hint, is_input=True)
            self.forward_args.append(nm)
            if u not in self.ext_inputs:
                self.inputs_spec.append((nm, meta[0], (u, n, meta)))
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
        self.inputs_spec = [(nm, kind, self.example_shape(*a)) for nm, kind, a in self.inputs_spec]
        # tailles d'entrée (in_channels…) : paramètres en tête, s'ils servent
        used = "\n".join(self.init)
        in_params, in_idx = {}, {}
        for sym, (default, from_text, label, idx) in self.in_syms.items():
            if not re.search(rf"\b{sym}\b", used):
                continue
            in_params[sym] = default
            in_idx[sym] = idx
            if default is not None and not from_text:
                what = "les canaux" if sym.endswith("channels") else "la taille"
                i = next(i for i, ln in enumerate(self.init) if re.search(rf"\b{sym}\b", ln))
                self.init[i] = with_todo(self.init[i], f"{sym}={default} par défaut, le schéma n'indique pas "
                                                       f"{what} de « {label} »")
        params = dict(in_params)
        if self.kind == "def":
            params.update({s: None for s in self.def_symbol_order if s in self.def_symbols})
            if self.strided:
                params["stride"] = 1
        params.update({k: v for k, v in self.params.items() if k not in params})
        cc = ClassCode(self.name, self.doc, params, self.init, self.forward_args, self.fwd,
                       self.inputs_spec, in_params=in_idx, out_meta=self.out_meta)
        return cc

    def finish_returns(self, returns):
        if returns:
            self.out_meta = returns[0][0].meta
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
                return out if isinstance(out, int) else 0          # 0 : addition, canaux non indiqués
            todo += [e.dst for e in g.outs(w)]
        return None

    def example_shape(self, u, n, meta):
        """Forme de l'entrée de l'essai (appelée en fin de construction : réglages connus)."""
        kind = meta[0]
        if kind == "tokens":
            return "(1, 16)"
        if kind == "seq":
            feat = meta[1] if meta[1] is not None else "d_model"
            return f"(1, 16, {self.params.get(feat, feat if str(feat).isdigit() else 64)})"
        c, hw, _, _ = self.input_dims(u, n, kind if kind and kind.startswith("img") else "vec")
        if kind and kind.startswith("img"):
            return "(" + ", ".join(map(str, (1, c) + tuple(hw))) + ")"
        return f"(1, {c})"

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
            if e.label and edge_hint(e.label):
                v = replace(v, hint=edge_hint(e.label))
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
            return self.emit_stub(u, ins, None)
        specs, reps, chans = self.node_specs(u)
        s0 = specs[0]
        if s0.kind == "user":
            return self.emit_user(u, ins, s0.p["expr"])
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
        return nm, (kind, size_sum([v.meta[1] for v, _ in ordered]), hw_of(ordered[0][0].meta)), ""

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
        ctors, meta, todos = self.run_specs(specs, reps, chans, meta, stride_param, last, last_i)
        todos = list(dict.fromkeys(todos))
        if not ctors:
            return self.passthrough(u, ins, x, meta, hint, todos)
        ctor = ctors[0] if len(ctors) == 1 else "nn.Sequential(\n    " + ",\n    ".join(ctors) + ",\n)"
        base = self.attr_base(specs, n)
        att = self.module_attr(u, base, ctor, n, todos)
        extra, com = "", self.comment(n)
        skip = self.concat_partner(u) if len(ctors) == 1 and ctors[0].startswith("nn.ConvTranspose") else None
        if skip is not None:                        # taille de sortie calée sur la branche concaténée
            extra = f", output_size={skip.name}.shape"
            com += (" — " if com else "  # ") + f"taille calée sur {skip.name}"
            meta = (meta[0], meta[1], hw_of(skip.meta))
        fake_ins = [(Value(x, meta, hint), ins[0][1])] if len(ins) > 1 else ins
        out = self.out_value(u, fake_ins if len(ins) == 1 else [], meta, hint=hint)
        self.fwd.append(f"{out.name} = self.{att}({x}{extra}){com}")

    def concat_partner(self, u):
        """Convolution transposée dont la seule sortie va à une concaténation : l'autre entrée de celle-ci,
        si elle est déjà calculée (sa taille fixe celle de la transposée : 2L ou 2L − 1)."""
        outs = self.g.outs(u)
        if len(outs) != 1:
            return None
        w = self.g.nodes[outs[0].dst]
        if w.type != "operator" or (w.symbol or "") != "concat":
            return None
        for e in self.g.ins(w.uid):
            if e.src != u:
                v = self.values.get((e.src, e.port)) or self.values.get((e.src, 0))
                if v is not None and v.meta[0] and v.meta[0].startswith("img"):
                    return v
        return None

    def run_specs(self, specs, reps, chans, meta, stride_param=None, last=False, last_i=None):
        """Couches d'un bloc, la taille suivie de l'une à l'autre -> (constructeurs, meta, todos)."""
        ctors, todos = [], []
        for _ in range(max(1, reps)):
            for i, s in enumerate(specs):
                c, meta, t = self.layer(s, meta, chans, stride_param, last and i == last_i)
                if s.kind in ("conv", "pool"):
                    stride_param = None
                ctors.extend(c)
                todos.extend(t)
        sh = specs[0].p.get("_shape") if specs else None
        if sh:
            meta = self.shape_meta(meta, sh[1])
        return ctors, meta, todos

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
        """Spec -> ([constructeurs], meta de sortie, [todos]) ; meta = (genre, taille[, Spatial])."""
        kind, feat, sp = meta[0], meta[1], hw_of(meta)
        todos = []
        k = s.kind
        p = s.p
        pin = p.get("in")                          # « Linear C → 4C » : C désigne la taille reçue
        if isinstance(pin, str) and re.fullmatch(r"[A-Za-z]\w*", pin) and pin not in self.bound \
                and str(feat) != pin and not (self.kind == "def" and pin in self.def_symbols):
            known = feat if feat is not None else (self.hp("d_model") if kind == "seq" else None)
            if known is not None:
                self.bound[pin] = str(known)
        img = kind if kind and kind.startswith("img") else None
        dim = int(img[3]) if img else 2
        sh = p.get("_shape")

        def out_size(fallback=64, what="la taille de sortie"):
            v = p.get("out")
            if v is None and chans:
                v = chans
            if self.kind == "def" and v is not None and v == getattr(self, "def_cout", None):
                v = "C"
            if self.kind == "def" and v is None and getattr(self, "def_auto_c", False):
                v = "C"                              # « Bloc Conv / 64 → 128 » : C fourni par chaque bloc
            if v is None and last:
                if "vocab_size" in self.params:
                    return self.hp("vocab_size")
                return self.hp("d_model" if kind == "seq" else "num_classes")
            e = self.size_expr(v)
            if e is None:
                todos.append(f"préciser {what}, que le schéma n'indique pas")
                return str(fallback)
            return e

        def channels_in():
            """Canaux d'entrée : suivis le long du graphe, sinon lus dans la ligne de formes."""
            shown = sh[0][0] if sh and len(sh[0]) > 1 else None
            if feat is not None and shown is not None and str(feat).isdigit() and int(feat) != shown:
                todos.append(f"le schéma indique {shown} canaux d'entrée")
            return self.in_size(feat if feat is not None else shown, todos)

        def exact_pad(ks, st):
            if sh and len(sh[0]) > 1 and len(sh[1]) > 1 and isinstance(st, int) and isinstance(ks, int):
                return max(0, -(-((sh[1][-1] - 1) * st + ks - sh[0][-1]) // 2))
            return None

        if k == "conv":
            d = p.get("dim") or dim
            ks = p.get("k") or (2 if p.get("transposed") else 3)
            out = out_size(64, "le nombre de canaux")
            st = p.get("stride") or (ks if p.get("transposed") else 1)
            stv = stride_param or (str(st) if st != 1 else None)
            cin = channels_in()
            run_st = st if not stride_param else None          # pas en paramètre : taille non suivie
            if p.get("transposed"):
                tp = p.get("pad") or 0
                c = f"nn.ConvTranspose{d}d({cin}, {out}, kernel_size={ks}" + (f", stride={stv}" if stv else "") + \
                    (f", padding={tp}" if tp else "") + ")"
                nsp = conv_dims(sp, ks, run_st, tp, transposed=True)
            else:
                pad = p.get("pad") if p.get("pad") is not None else exact_pad(ks, st)
                guessed = pad is None and isinstance(ks, int) and ks > 1
                if pad is None:
                    pad = max(0, (ks - st + 1) // 2) if isinstance(st, int) else ks // 2
                args = [f"kernel_size={ks}"]
                if stv:
                    args.append(f"stride={stv}")
                if pad:
                    args.append(f"padding={pad}")
                if p.get("depthwise"):
                    c = f"nn.Conv{d}d({cin}, {cin}, {', '.join(args)}, groups={cin})"
                    out = cin
                else:
                    c = f"nn.Conv{d}d({', '.join([cin, out] + args)})"
                nsp = conv_dims(sp, ks, run_st, pad, guessed=guessed)
            if sh and len(sh[1]) > 1:
                nsp = Spatial(tuple(sh[1][1:]))
            if kind == "seq":
                todos.append("convolution sur une séquence : passer en (B, C, T) avec x.transpose(1, 2)")
            return [c], (f"img{d}d", out, nsp), todos
        if k == "pool":
            ks = p.get("k") or 2
            st = p.get("stride") or (2 if ks == 2 else 1)
            op = "Max" if p.get("op") == "max" else "Avg"
            args = [f"kernel_size={ks}"]
            stv = stride_param or str(st)
            args.append(f"stride={stv}")
            pad = exact_pad(ks, st)
            guessed = pad is None and isinstance(ks, int) and isinstance(st, int) and ks > st
            if pad is None:
                pad = ks // 2 if st == 1 and not stride_param else 0
            if pad:
                args.append(f"padding={pad}")
            nsp = conv_dims(sp, ks, st if not stride_param else None, pad, guessed=guessed)
            if sh and len(sh[1]) > 1:
                nsp = Spatial(tuple(sh[1][1:]))
            return [f"nn.{op}Pool{dim}d({', '.join(args)})"], (img or "img2d", feat, nsp), todos
        if k == "gpool":
            op = "Max" if p.get("op") == "max" else "Avg"
            return [f"nn.Adaptive{op}Pool{dim}d(1)", "nn.Flatten()"], ("vec", feat), todos
        if k == "upsample":
            sc = p.get("out") if isinstance(p.get("out"), int) and p.get("out") <= 8 else 2
            nsp = Spatial(tuple(h * sc for h in sp.dims), sp.why) if sp else None
            return [f"nn.Upsample(scale_factor={sc})"], (kind, feat, nsp), todos
        if k == "norm":
            op = p.get("op") or ("batch" if img else "layer")
            if op == "batch":
                if kind == "seq":
                    todos.append("BatchNorm sur une séquence : nn.BatchNorm1d attend (B, C, T)")
                bd = dim if img else 1
                return [f"nn.BatchNorm{bd}d({channels_in()})"], meta, todos
            if op == "instance":
                return [f"nn.InstanceNorm{dim}d({channels_in()})"], meta, todos
            if op == "lrn":
                return ["nn.LocalResponseNorm(5)"], meta, todos
            if img:
                f = channels_in()
            else:
                f = (str(sh[1][-1]) if sh else None) if feat is None else str(feat)
                f = f or self.hp("d_model")
                if feat is None and not sh:
                    todos.append("taille normalisée inconnue (d_model par défaut)")
            if op == "group":
                if not p.get("groups"):
                    todos.append("nombre de groupes non précisé (32 par défaut)")
                return [f"nn.GroupNorm({p.get('groups') or 32}, {f})"], meta, todos
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
            return ["nn.Flatten()"], ("vec", self.flat_size(feat, sp, todos) if img else
                                      (feat if kind == "vec" else None)), todos
        if k in ("linear", "head"):
            pre = []
            if img:
                pre = ["nn.Flatten()"]
                if sh and len(sh[0]) == 1:
                    fin = str(sh[0][0])
                elif sh and len(sh[0]) > 1:
                    fin = " * ".join(map(str, sh[0]))
                else:
                    fin = self.flat_size(feat, sp, todos)
                kind = "vec"
            else:
                fin = feat if feat is not None else (str(sh[0][-1]) if sh else None)
            fin = self.in_size(fin, todos, kind, "la taille d'entrée")
            if k == "head":
                nm = "vocab_size" if "vocab_size" in self.params else "num_classes"
                if kind == "seq":
                    todos.append("tête appliquée à chaque jeton : prendre x[:, 0] (jeton de classe) ou x.mean(1)")
                return pre + [f"nn.Linear({fin}, {self.hp(nm)})"], (kind, nm), todos
            if p.get("proj") and p.get("out") is None:
                out = self.hp("d_model")
            else:
                out = out_size(128, "la taille de sortie")
            return pre + [f"nn.Linear({fin}, {out})"], (kind or "vec", out), todos
        if k == "embedding":
            d = self.size_expr(p.get("out"), "d_model")
            return [f"nn.Embedding({self.hp('vocab_size')}, {d})"], ("seq", d), todos
        if k == "ff" or k == "mlp":
            d = feat or self.hp("d_model")
            act = "nn.ReLU()" if k == "ff" else "nn.GELU()"
            dff = self.hp("d_ff")
            return [f"nn.Sequential(nn.Linear({d}, {dff}), {act}, nn.Linear({dff}, {d}))"], (kind, d), todos
        if k == "patch":
            self.gen.helper("PatchEmbedding")
            d = self.hp("d_model")
            return [f"PatchEmbedding({channels_in()}, {d}, {self.hp('patch_size')})"], ("seq", d), todos
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

    def emit_module(self, u, ins, ctor, base, meta, call_args=None, todos=None):
        n = self.g.nodes[u]
        att = self.module_attr(u, base, ctor, n, todos or [])
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
        if sym in ("plus", "times", "dot", "minus"):          # taille : celle d'une entrée calculée
            known = [v for v, _ in ins if v.meta[1] is not None]
            pref = [v for v in known if not v.is_input] or known
            if pref:
                meta = pref[0].meta
        if sym == "plus":
            expr = " + ".join(names)
            if self.kind == "def" and len(ins) == 2:          # bloc résiduel : raccourci si la forme change
                inp = [v for v, _ in ins if v.is_input]
                oth = [v for v, _ in ins if not v.is_input]
                kd = inp[0].meta[0] if inp else None
                cin, cout = (inp[0].meta[1], oth[0].meta[1]) if inp and oth else (None, None)
                conds = (["stride == 1"] if self.strided else []) + \
                    ([f"{cin} == {paren(cout)}"] if cin and cout and str(cin) != str(cout) else [])
                if kd and kd.startswith("img") and cin and cout and conds:
                    d = kd[3]
                    att = self.attr("shortcut")
                    st = ", stride=stride" if self.strided else ""
                    self.init.append(f"self.{att} = nn.Identity() if {' and '.join(conds)} else nn.Sequential(")
                    self.init.append(f"    nn.Conv{d}d({cin}, {cout}, kernel_size=1{st}, bias=False),")
                    self.init.append(f"    nn.BatchNorm{d}d({cout}),")
                    self.init.append(")  # raccourci : Conv 1×1 quand la forme change")
                    expr = " + ".join(f"self.{att}({v.name})" if v.is_input else v.name for v, _ in ins)
                    meta = oth[0].meta
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
            meta = (kind, meta[1] if tokens else size_sum([v.meta[1] for v, _ in ins]), hw_of(meta))
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
        if byhint and len(ins) - len(byhint) == 1:       # une seule entrée sans nom : celle qui manque
            rest = next(v for v, _ in ins if not (v.hint and v.hint.lower()[:1] in "qkv"))
            missing = [x for x in "vkq" if x not in byhint]
            byhint[missing[0]] = rest
            q = byhint.get("q", rest)
            k = byhint.get("k", byhint.get("v"))
            return q, k, byhint.get("v", k), False
        vals = [v for v, _ in ins]
        if len(vals) == 1:
            return vals[0], vals[0], vals[0], False
        if len(vals) == 2:
            return vals[1], vals[0], vals[0], True
        return vals[2], vals[1], vals[0], False

    def emit_mha(self, u, ins, s):
        n = self.g.nodes[u]
        q, k, v, guess = self.order_qkv(ins)
        meta = q.meta
        chans = (self.node_specs(u) or [None, None, None])[2]       # « 256 canaux · 8 têtes »
        todos = []
        if chans is not None and meta[1] is not None and str(meta[1]).isdigit() and int(meta[1]) != chans:
            todos.append(f"l'entrée a {meta[1]} canaux, le schéma en indique {chans}")
        d = str(chans) if chans is not None else (meta[1] or self.hp("d_model"))
        heads = s.p.get("heads") or self.hp("num_heads")
        att = self.attr("attention" if not s.p.get("masked") else "masked_attention")
        self.add_init(att, f"nn.MultiheadAttention({d}, {heads}, batch_first=True)", n, todos)
        img = bool(meta[0] and meta[0].startswith("img"))
        names = {}
        for val in (q, k, v):                       # image / signal (B, C, …) -> séquence (B, positions, C)
            if id(val) in names:
                continue
            if val.meta[0] and val.meta[0].startswith("img"):
                nm = self.var(val.name + "_seq" if re.fullmatch(r"[a-z]\w*", val.name) else "seq")
                flat = "" if val.meta[0] == "img1d" else ".flatten(2)"
                self.fwd.append(f"{nm} = {val.name}{flat}.transpose(1, 2)  # (B, C, …) -> (B, positions, C)")
                names[id(val)] = nm
            else:
                names[id(val)] = val.name
        out = self.out_value(u, [], (meta[0], d, hw_of(meta)) if img else (meta[0], d))
        extra = ""
        if s.p.get("masked"):
            mask = self.var("mask")
            self.fwd.append(f"{mask} = nn.Transformer.generate_square_subsequent_mask({names[id(q)]}.size(1), "
                            f"device={q.name}.device)")
            extra = f", attn_mask={mask}"
        back = ""
        if img:                                     # retour à la forme de l'image / du signal
            back = ".transpose(1, 2)" if meta[0] == "img1d" else f".transpose(1, 2).reshape({q.name}.shape)"
        line = f"{out.name} = self.{att}({names[id(q)]}, {names[id(k)]}, {names[id(v)]}{extra}, " \
               f"need_weights=False)[0]{back}" + self.comment(n)
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
        out = self.out_value(u, [], (kind, size_sum([v.meta[1] for v, _ in ins]), hw_of(ins[0][0].meta)))
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

    def emit_user(self, u, ins, expr):
        """Bloc personnel (champ « Code PyTorch ») : « {in} » y est remplacé par la taille d'entrée."""
        v0, todos = ins[0][0], []
        meta = v0.meta
        if "{in}" in expr:
            expr = expr.replace("{in}", self.in_size(meta[1], todos, meta[0]))
        outs = re.findall(r"nn\.(?:Conv\w*|Linear)\(\s*[^,()]+,\s*([^,()=]+)[,)]", expr) or \
            re.findall(r"nn\.Lazy(?:Conv\w*|Linear)\(\s*([^,()=]+)[,)]", expr)
        if outs:                                     # dernière couche à taille : ses canaux de sortie
            meta = (meta[0], outs[-1].strip())
        return self.emit_module(u, ins, expr, ident(self.g.nodes[u].text, "bloc"), meta, todos=todos)

    def emit_stub(self, u, ins, s):
        n, v0, todos = self.g.nodes[u], ins[0][0], []
        kind = v0.meta[0]
        a, b = io_channels(n.text)
        if b is not None and len(ins) == 1 and kind and (kind.startswith("img") or kind in ("vec", "seq")):
            cin = self.in_size(v0.meta[1], todos, kind)
            if a is not None and cin.isdigit() and int(cin) != a:
                todos.append(f"le schéma indique {a} canaux d'entrée, le graphe en donne {cin}")
            if cin != str(b):
                cname = self.gen.stub_class(n.text, 1, proj=kind)
                return self.emit_module(u, ins, f"{cname}({cin}, {b})", snake(cname),
                                        (kind, str(b), hw_of(v0.meta)), todos=todos)
        cname = self.gen.stub_class(n.text, len(ins))
        return self.emit_module(u, ins, f"{cname}()", snake(cname), v0.meta, todos=todos)

    def defref_ctor(self, u, key):
        """Bloc qui porte le titre d'une définition -> (classe, ctor(avec_pas, tailles), sortie(tailles, meta)).

        « Basic Block\n128, /2 » : les nombres remplissent les tailles symboliques de la définition (C),
        « /2 » son paramètre stride ; ses tailles d'entrée (in_channels) sont celles des valeurs reçues."""
        n = self.g.nodes[u]
        cc = self.gen.def_class(key)
        if cc is None:
            return None, None, None
        rest = " ".join(n.text.split("\n")[1:]) if norm_key(first_line(n.text)) == key else ""
        nums = re.findall(r"(?<![/\w])(\d+)(?![\w])", plain(rest))
        m = re.search(r"/\s*(\d+)", plain(rest))
        sh = split_shapes(n.text)[1]
        io = io_channels(n.text) if rest else (None, None)
        if io[1] is not None and not sh:             # « 64 → 128 » : C = 128
            nums = [str(io[1])]
        if sh:                                       # « 64×56×56 → 128×28×28 » : C = 128, pas 2
            nums = [str(sh[1][0])]
            if len(sh[0]) > 1 and sh[0][-1] > sh[1][-1]:
                m = re.search(r"(\d+)", str(sh[0][-1] // sh[1][-1]))
        stride = m.group(1) if m and "stride" in cc.params else None

        def call(feats):
            args, mapping, i = [], {}, 0
            for p, d in cc.params.items():
                if d is not None:
                    continue
                if p in cc.in_params:
                    j = cc.in_params[p]
                    val = feats[j] if j < len(feats) and feats[j] is not None else "64"
                else:
                    val = nums[i] if i < len(nums) else self.hp(p)
                    i += 1
                args.append(str(val))
                mapping[p] = str(val)
            for p, d in cc.params.items():
                if d is not None and p != "stride":
                    args.append(f"{p}={self.hp(p, d)}")
                    mapping[p] = p
            return args, mapping

        def ctor(with_stride=True, feats=()):
            args = call(list(feats))[0]
            return f"{cc.name}({', '.join(args + ([f'stride={stride}'] if stride and with_stride else []))})"

        def out(feats, meta_in, with_stride=True):
            mapping = call(list(feats))[1]
            kind = cc.out_meta[0] or meta_in[0]
            feat = subst(cc.out_meta[1], mapping)
            sp = hw_of(meta_in) if kind and kind.startswith("img") else None
            if sp and sh and len(sh[1]) > 1:
                sp = Spatial(tuple(sh[1][1:]))
            elif sp and stride and with_stride:
                sp = Spatial(tuple(-(-h // int(stride)) for h in sp.dims), sp.why | {"padding"})
            return (kind, feat, sp)
        return cc, ctor, out

    def def_feats(self, cc, metas, todos):
        """Tailles d'entrée passées à une définition (in_channels…), dans l'ordre de ses arguments."""
        feats = [None] * len(metas)
        for j in cc.in_params.values():
            if j < len(metas):
                feats[j] = self.in_size(metas[j][1], todos, metas[j][0])
        return feats

    def emit_defref(self, u, ins, key):
        cc, ctor, out = self.defref_ctor(u, key)
        if cc is None:
            return self.emit_stub(u, ins, None)
        vals = [v for v, _ in ins]
        hints = {v.hint.lower(): v for v in vals if v.hint}
        if len(cc.forward_args) == len(ins) and all(a in hints for a in cc.forward_args):
            vals = [hints[a] for a in cc.forward_args]
            call = [f"{a}={v.name}" for a, v in zip(cc.forward_args, vals)]       # q=…, k=…, v=…
        else:
            byname = {}                             # flèche nommée comme un argument (« skip »)
            for a in cc.forward_args:
                m = next((v for v in vals if v.hint and v.hint.lower() == a
                          and all(v is not x for x in byname.values())), None)
                if m is not None:
                    byname[a] = m
            if byname and len(cc.forward_args) == len(vals):
                rest = [v for v in vals if all(v is not x for x in byname.values())]
                vals = [byname.get(a) or rest.pop(0) for a in cc.forward_args]
            call = [v.name for v in vals]
        metas = [v.meta for v in vals]
        todos = []
        feats = self.def_feats(cc, metas, todos)
        a = io_channels(self.g.nodes[u].text)[0]
        if a is not None and feats and feats[0] and feats[0].isdigit() and int(feats[0]) != a:
            todos.append(f"le schéma indique {a} canaux d'entrée, le graphe en donne {feats[0]}")
        return self.emit_module(u, ins, ctor(True, feats), snake(cc.name), out(feats, metas[0]),
                                call_args=call, todos=todos)

    # .................................................................. cadres
    def emit_sub(self, u, ins, s=None):
        g = self.g
        n = g.nodes[u]
        mode = n.data["mode"]
        if mode in ("seq", "single"):
            chain = n.data["chain"]
            x, meta, hint = self.combine(u, ins) if ins else ("x", (None, None), "")
            if mode == "single":
                for c_uid in chain:
                    sp = self.node_specs(c_uid)
                    if not sp or sp[0][0].kind in ("defref", "stub", "user") or self.role(c_uid) != "layer":
                        return self.emit_repeat_single(u, ins, c_uid, sp[0][0] if sp and self.role(c_uid) == "layer"
                                                       else Spec("stub"))

            def one_pass(m):
                cs, ts = [], []
                for c_uid in chain:
                    specs, reps, chans = self.node_specs(c_uid)
                    c, m, t = self.run_specs(specs, reps, chans, m)
                    cs.extend(c)
                    ts.extend(t)
                return cs, m, ts

            ctors, meta1, todos = one_pass(meta)
            if mode == "single":                          # « 3× Conv 3×3, 256 » : la 1re copie reçoit l'entrée
                r = n.data["repeat"]
                passes, m = [ctors], meta1
                for _ in range((int(r) if r.isdigit() else 2) - 1):
                    c, m, t = one_pass(m)
                    passes.append(c)
                    todos.extend(t)
                if not r.isdigit() and hw_of(m) != hw_of(meta1):
                    m = (m[0], m[1])                      # nombre de copies symbolique : taille non suivie
                meta = m

                def wrap(cs):
                    return cs[0] if len(cs) == 1 else "nn.Sequential(" + ", ".join(cs) + ")"
                if all(p == passes[0] for p in passes):
                    ctor = f"nn.Sequential(*[{wrap(passes[0])} for _ in range({self.rep_expr(r)})])"
                elif r.isdigit() and int(r) <= 4:
                    ctor = "nn.Sequential(\n    " + ",\n    ".join(wrap(p) for p in passes) + ",\n)"
                else:
                    ctor = f"nn.Sequential({wrap(passes[0])}, *[{wrap(passes[1])} for _ in range({self.rep_expr(r)} - 1)])"
            else:
                meta = meta1
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
                self.init[-1] = self.init[-1].split("  # ")[0] + com + \
                    (" — TODO : " + self.init[-1].split("TODO : ", 1)[1] if "TODO : " in self.init[-1] else "")
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
        kw = ", ".join(f"{p}={self.hp(p, d)}" for p, d in cc.params.items() if p not in cc.in_params)
        r = self.rep_expr(n.data["repeat"])
        gname = g.nodes[n.data["group"]].data.get("name")
        base = ident(n.data.get("hint"), "") or (snake(camel(gname)) + "s" if gname else "blocs")
        att = self.attr(base)
        meta = main_val.meta if main_val else (None, None)
        in_p = next((p for p, j in cc.in_params.items() if j == 0), None)
        todos = []
        if in_p:                                         # taille d'entrée : celle reçue, puis celle produite
            cin = self.in_size(meta[1], todos, meta[0])
            o = cc.out_meta[1]
            grows = o is not None and str(o) != in_p and re.search(rf"\b{in_p}\b", str(o))
            cout = subst(o, {in_p: cin}) if o is not None and not grows else cin
            step = None
            if grows:                                    # « in_channels + 32 » (DenseNet) : croissance régulière
                terms = [t.strip() for t in str(o).split(" + ")]
                if in_p in terms:
                    terms.remove(in_p)
                    if terms and not any(re.search(rf"\b{in_p}\b", t) for t in terms):
                        step = " + ".join(terms)
                if step is None:
                    todos.append("la taille change à chaque répétition : préciser l'entrée de chaque bloc")
            if step is not None:
                body = f"[{cc.name}({', '.join([f'{cin} + i * {paren(step)}'] + ([kw] if kw else []))}) " \
                       f"for i in range({r})]"
                cout, grows = subst(f"{paren(cin)} + {paren(r)} * {paren(step)}", {}), False
            elif cout == cin or grows:
                body = f"[{cc.name}({', '.join([cin] + ([kw] if kw else []))}) for _ in range({r})]"
            else:
                body = f"[{cc.name}({', '.join([f'{cin} if i == 0 else {cout}'] + ([kw] if kw else []))}) " \
                       f"for i in range({r})]"
            sp = hw_of(meta)
            if sp and (hw_of(cc.out_meta) is None or hw_of(cc.out_meta).dims != sp.dims):
                sp = None
            meta = (meta[0], None if grows else cout, sp)
        else:
            body = f"[{cc.name}({kw}) for _ in range({r})]"
        self.add_init(att, f"nn.ModuleList({body})", n, todos)
        others = [self.values.get(sp) or self.values.get((sp[0], 0)) for sp in srcs[1:]]
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
        un pas (/2) ne s'applique qu'à la première (convention ResNet), qui reçoit aussi la taille d'entrée."""
        g = self.g
        n, cn = g.nodes[u], g.nodes[c_uid]
        r = n.data["repeat"]
        x, meta, hint = self.combine(u, ins)
        name = ident(n.data.get("hint"), "")
        note, todos = [], []
        if spec.kind == "defref":
            cc, ctor, out = self.defref_ctor(c_uid, spec.p["key"])
            f1 = self.def_feats(cc, [meta], todos)
            m1 = out(f1, meta)
            f2 = self.def_feats(cc, [m1], todos)
            c1, c2 = ctor(True, f1), ctor(False, f2)
            if r.isdigit() and int(r) <= 4:
                parts = [c1] + [c2] * (int(r) - 1)
                inner = "nn.Sequential(\n    " + ",\n    ".join(parts) + ",\n)"
            else:
                inner = f"nn.Sequential({c1}, *[{c2} for _ in range({self.rep_expr(r)} - 1)])"
            if ctor(True, f2) != ctor(False, f2):
                note = ["pas : premier bloc seulement"]
            meta = out(f2, m1, with_stride=False)
            att = self.attr(name or snake(cc.name) + "s")
        else:
            one = f"{self.gen.stub_class(cn.text, 1)}()" if spec.kind == "stub" else \
                spec.p["expr"].replace("{in}", self.in_size(meta[1], todos, meta[0]))
            inner = f"nn.Sequential(*[{one} for _ in range({self.rep_expr(r)})])"
            att = self.attr(name or ident(cn.text, "bloc") + "s")
        self.add_init(att, inner, n, todos)
        if note:
            self.init[-1] += "  # " + note[0]
        out_v = self.out_value(u, ins if len(ins) == 1 else [], meta, hint=hint)
        self.fwd.append(f"{out_v.name} = self.{att}({x})  # {label_line(cn.text)} ×{r}")

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
                for t in re.findall(r"[A-Za-z_]\w*", v) if isinstance(v, str) else []:   # « 4 * C » -> C
                    if t not in order:
                        order.append(t)
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
        self.def_auto_c = False
        if not order and any(io_channels(self.g.nodes[u].text)[1] is not None for u in self.def_users):
            self.def_auto_c = True
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
