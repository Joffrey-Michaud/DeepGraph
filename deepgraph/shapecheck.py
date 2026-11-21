"""Vérification d'un schéma : formes des tenseurs calculées, incohérences, tailles écrites sur le schéma.

Le schéma est traduit en code PyTorch (deepgraph/codegen.py, avec des « sondes »), puis ce code est
exécuté une fois, dans un processus à part, sur des entrées factices : chaque bloc note la forme de
sa sortie. Les formes sont donc exactes pour le code généré, avec les mêmes règles que l'export en
code (padding « same » par défaut, tailles suivies de couche en couche…).

Incohérences signalées :
  - erreur à l'exécution (⊕ entre 64 et 128 canaux, concaténation de tailles différentes, Linear qui
    reçoit 2048 valeurs au lieu de 4096…), localisée sur le bloc ;
  - forme écrite dans un bloc (« 64×56×56 ») ou à côté (« 55×55×96 ») différente de la forme calculée ;
  - réglage du schéma contredit par le graphe (« le schéma indique 128 canaux d'entrée ») ;
et, à part, les hypothèses dont dépendent les formes (taille d'entrée non écrite, bloc non reconnu).

Ce module n'utilise pas Qt (sauf write_labels, qui place les textes dans une scène).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

from .codegen import SHAPE_LABEL_REF, Generator, annot_split, first_line, split_shapes, tex

RUNNER = r"""
import json, sys, traceback
path, out = sys.argv[1], sys.argv[2]
res = {"error": None, "calls": [], "loose": {}, "cur": None, "ins": None, "params": None, "line": None}
ns = {"__name__": "__main__"}
try:
    src = open(path, encoding="utf-8").read()
    exec(compile(src, path, "exec"), ns)
except BaseException as e:
    res["error"] = f"{type(e).__name__}: {e}"
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == path]
    res["line"] = lines[-1] if lines else None
dg = ns.get("_DG") or {}
for k in ("calls", "loose", "cur", "ins"):
    if k in dg:
        res[k] = dg[k]
try:
    import torch.nn as nn
    mods = [v for v in ns.values() if isinstance(v, nn.Module)]
    if mods:
        res["params"] = sum(p.numel() for p in mods[0].parameters())
except BaseException:
    pass
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, default=str)
"""

LEVELS = ("erreur", "attention", "hypothèse")
SIZE_WORDS = re.compile(r"canaux|taille|padding|\bpas\b|forme|côtés|longueur|features|calculé", re.I)
SYMBOLS = {1: ["d"], 2: ["T", "d"], 3: ["C", "H", "W"], 4: ["C", "D", "H", "W"]}


# ------------------------------------------------------------------ génération et exécution
def probe_code(data, user_defs=None, title=""):
    """Diagramme -> (code avec sondes, générateur)."""
    gen = Generator(data, user_defs or {}, title, probes=True)
    code = gen.run()
    return code, gen


def runner_args(code_path, out_path):
    return ["-c", RUNNER, code_path, out_path]


def write_temp(code):
    fd, code_path = tempfile.mkstemp(suffix=".py", prefix="deepgraph_verif_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="deepgraph_verif_")
    os.close(fd)
    return code_path, out_path


def read_result(out_path, stderr=""):
    try:
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        msg = (stderr or "").strip().splitlines()
        return {"error": msg[-1] if msg else "l'exécution n'a rien produit", "calls": [], "loose": {}}


def run(code, python=None, timeout=300):
    """Exécute le code sondé (sans Qt : ligne de commande, tests) -> résultat brut."""
    code_path, out_path = write_temp(code)
    try:
        p = subprocess.run([python or sys.executable] + runner_args(code_path, out_path),
                           capture_output=True, text=True, timeout=timeout)
        return read_result(out_path, p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"{type(e).__name__}: {e}", "calls": [], "loose": {}}
    finally:
        for f in (code_path, out_path):
            try:
                os.remove(f)
            except OSError:
                pass


# ------------------------------------------------------------------ formes
def _is_dims(sh):
    return isinstance(sh, list) and all(isinstance(v, int) for v in sh)


def fmt(dims):
    return "×".join(str(d) for d in dims)


def _ratio(val, ref, sym):
    if not ref:
        return None
    r = Fraction(val, ref)
    if r == 1:
        return sym
    if r.numerator == 1:
        return f"{sym}/{r.denominator}"
    if r.denominator == 1:
        return f"{r.numerator}{sym}"
    return None


def generalize(samples, kind=None):
    """Formes d'un bloc appelé plusieurs fois avec des tailles différentes (bloc défini, cadre répété) :
    chaque dimension relative à la sortie de l'appel (« C/4×H×W »), ou None.

    samples : [(dims sans batch, dims de sortie de l'appel sans batch)]. On préfère les appels qui
    gardent la forme (entrée = sortie), comme l'exemple que dessine un schéma."""
    rank = len(samples[0][0])
    if any(len(d) != rank for d, _ in samples):
        return None
    out = []
    for j in range(rank):
        vals = {d[j] for d, _ in samples}
        if len(vals) == 1:
            out.append(str(vals.pop()))
            continue
        refs = [o for _, o in samples]
        if not all(o and len(o) > j for o in refs):
            return None
        syms = SYMBOLS.get(len(refs[0]), [])
        if kind == "img1d" and len(refs[0]) == 2:
            syms = ["C", "L"]
        if j >= len(syms):
            return None
        rs = {_ratio(d[j], o[j], syms[j]) for d, o in samples}
        if len(rs) != 1 or None in rs:
            return None
        out.append(rs.pop())
    return "×".join(out)


def node_shapes(res, class_kinds=None):
    """Résultat brut -> {"uid" ou "uid|port": texte de la forme (sans le batch)}."""
    per = {}
    for c in res.get("calls") or []:
        outs = c.get("outs") or []
        ins = c.get("ins") or []
        o = outs[0][1:] if outs and _is_dims(outs[0]) else None
        keep = bool(ins and outs and ins[0] == outs[0])
        for k, sh in (c.get("nodes") or {}).items():
            per.setdefault(k, []).append((sh, o, keep, c.get("cls")))
    for k, lst in (res.get("loose") or {}).items():
        for sh in lst:
            per.setdefault(k, []).append((sh, None, False, None))
    shapes, exact = {}, {}
    for k, lst in per.items():
        dims = [sh[1:] for sh, _, _, _ in lst if _is_dims(sh) and sh]
        if len(dims) != len(lst) or not dims:
            continue
        uid, port = k.rsplit("|", 1)
        key = uid if port == "0" else k
        if all(d == dims[0] for d in dims):
            shapes[key] = fmt(dims[0])
            exact[key] = dims[0]
            continue
        samples = [(sh[1:], o) for sh, o, keep, _ in lst if keep and o is not None] or \
            [(sh[1:], o) for sh, o, _, _ in lst if o is not None]
        kind = (class_kinds or {}).get(lst[0][3])
        g = generalize(samples, kind) if samples else None
        if g:
            shapes[key] = g
    return shapes, exact


# ------------------------------------------------------------------ incohérences
def node_label(n):
    if n is None:
        return "?"
    if n.type == "operator":
        return {"plus": "⊕", "times": "⊗", "dot": "⊙", "concat": "concaténation", "minus": "⊖"}.get(
            n.symbol or "plus", n.symbol or "opérateur")
    if n.type == "group":
        return f"cadre « {first_line(tex(n.label)) or 'sans titre'} »"
    t = first_line(tex(split_shapes(n.text)[0])) or first_line(tex(n.text))
    if n.type == "edgeop":
        return f"flèche « {t} »"
    return f"« {t} »" if t else "bloc"


def explain_error(msg, ins):
    """Message de PyTorch -> phrase sur le schéma (formes sans le batch)."""
    shapes = [fmt(s[1:]) for s in (ins or []) if _is_dims(s) and s]
    m = re.search(r"size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", msg)
    if m and len(shapes) >= 2:
        return f"formes incompatibles : {' et '.join(shapes)}"
    if "Sizes of tensors must match" in msg and shapes:
        return f"formes incompatibles : {', '.join(shapes)}"
    m = re.search(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)", msg)
    if m:
        return f"reçoit {m.group(2)} valeurs, la couche en attend {m.group(3)}"
    m = re.search(r"expected input\[[^\]]*\] to have (\d+) channels, but got (\d+) channels", msg)
    if m:
        return f"reçoit {m.group(2)} canaux, la couche en attend {m.group(1)}"
    m = re.search(r"Kernel size can't be greater than actual input size|Output size is too small", msg)
    if m:
        return f"entrée trop petite pour la couche ({', '.join(shapes)})" if shapes else "entrée trop petite pour la couche"
    short = msg.split("\n")[0]
    return short + (f" (entrée {', '.join(shapes)})" if shapes else "")


def _same_shape(written, computed):
    """Forme écrite (C×H×W, ou H×W×C comme dans beaucoup de figures) = forme calculée ?"""
    if written == computed:
        return True
    if len(written) == len(computed) >= 3 and written == computed[1:] + computed[:1]:
        return True
    return False


def analyse(gen, res):
    """-> rapport : {"shapes", "exact", "issues": [{uid, level, text}], "error", "params", "missing"}."""
    g = gen.g
    kinds = {}
    for cc in gen.classes:
        kinds[cc.name] = cc.out_meta[0] if cc.out_meta else None
    shapes, exact = node_shapes(res, kinds)
    issues = []
    err = res.get("error")
    no_torch = bool(err and "No module named 'torch'" in err)
    if err and not no_torch:
        cur = res.get("cur")
        n = g.nodes.get(cur) if cur is not None else None
        if n is None and cur is not None:
            n = next((x for u, x in g.nodes.items() if str(u) == str(cur)), None)
        text = explain_error(err.split(": ", 1)[-1], res.get("ins"))
        if n is not None and n.type == "sub":
            gid = n.data.get("group")
            n, cur = g.nodes.get(gid, n), gid
        issue = {"uid": cur if n is not None else None, "level": "erreur",
                 "text": (f"{node_label(n)} : " if n is not None else "") + text}
        if n is not None and n.type == "edgeop":           # opération portée par une flèche (légende)
            issue.update(uid=None, arrow=n.data.get("arrow"))
        issues.append(issue)
    # formes écrites dans les blocs ou à côté
    written = {str(u) for u, n in g.nodes.items()      # forme complète écrite (pas « 256 » seul)
               if n.type == "block" and ((split_shapes(n.text)[1] or ((), ()))[1][1:] or n.data.get("annot"))}
    node_ins = {}
    for c in res.get("calls") or []:
        for k, v in (c.get("node_ins") or {}).items():
            node_ins.setdefault(k, []).append(v)
    for u, n in g.nodes.items():
        comp = exact.get(str(u))
        if n.type != "block" or comp is None:
            continue
        sh = split_shapes(n.text)[1]
        if sh:
            w_in, w_out = sh
            if len(w_out) == len(comp) and not _same_shape(w_out, comp):
                issues.append({"uid": u, "level": "attention",
                               "text": f"{node_label(n)} : écrit {fmt(w_out)}, calculé {fmt(comp)}"})
            seen_in = node_ins.get(str(u)) or []
            one = seen_in and seen_in[0] and len(seen_in[0]) == 1 and _is_dims(seen_in[0][0])
            ci = seen_in[0][0][1:] if one else None       # plusieurs entrées (skip, DenseNet) : pas comparé
            if w_in != w_out and ci is not None and len(w_in) == len(ci) and not _same_shape(w_in, ci) \
                    and all(s == seen_in[0] for s in seen_in):
                issues.append({"uid": u, "level": "attention",
                               "text": f"{node_label(n)} : entrée écrite {fmt(w_in)}, reçue {fmt(ci)}"})
        a = n.data.get("annot")
        if a and len(comp) >= 2:
            kind = {2: "img1d", 3: "img2d", 4: "img3d"}.get(len(comp))
            c, hw = annot_split(a, kind, comp[0])
            bad = (c is not None and c != comp[0]) or (hw and list(hw) != list(comp[1:]))
            if bad:
                issues.append({"uid": u, "level": "attention",
                               "text": f"{node_label(n)} : écrit à côté {fmt(a['dims'])}"
                                       f"{' (' + str(a['c']) + ' canaux)' if a.get('c') else ''}, "
                                       f"calculé {fmt(comp)}"})
    # TODO du générateur, hypothèses
    assumed = {}
    for u, info in gen.node_info.items():
        n = g.nodes.get(u)
        if n is None:
            continue
        if n.type == "sub":
            u, n = n.data.get("group", u), g.nodes.get(n.data.get("group"), n)
        flagged = any(i["uid"] == u and i["level"] != "hypothèse" for i in issues)
        for t in info.get("todos", []):
            if "le schéma indique" in t:
                if flagged and any(i["uid"] == u and "écrit" in i["text"] for i in issues):
                    continue                             # déjà dit : « écrit 32×32×32, calculé 32×16×16 »
                issues.append({"uid": u, "level": "attention", "text": f"{node_label(n)} : {t}"})
            elif SIZE_WORDS.search(t):
                issues.append({"uid": u, "level": "hypothèse", "text": f"{node_label(n)} : {t}"})
        if info.get("assumed"):
            assumed.setdefault(", ".join(info["assumed"]), []).append((u, n))
        if info.get("stub"):
            issues.append({"uid": u, "level": "hypothèse",
                           "text": f"{node_label(n)} : bloc non reconnu, sa sortie est supposée de même forme "
                                   "que son entrée"})
    for what, nodes in assumed.items():            # entrées dont la taille n'est pas écrite
        img = "canaux" in what or "côtés" in what
        seq = "longueur" in what
        how = "ex. « Image 3×224×224 »" if img else "ex. une ligne « 128 », le nombre de positions" if seq \
            else "ex. « x 784 »"
        names = ", ".join(node_label(n) for _, n in nodes)
        issues.append({"uid": nodes[0][0], "level": "hypothèse",
                       "text": f"{'entrées' if len(nodes) > 1 else 'entrée'} {names} : {what} par défaut — écrire "
                               f"la forme dans le bloc ({how}) pour des tailles justes"})
    # blocs sans forme
    missing = []
    for u, n in g.nodes.items():
        if n.type in ("block", "operator") and g.outs(u) and str(u) not in shapes \
                and not n.data.get("absorbed") and not n.data.get("inside"):
            missing.append(u)
    order = {lv: i for i, lv in enumerate(LEVELS)}
    issues.sort(key=lambda i: order[i["level"]])
    # flèche qui porte une opération (couleur expliquée par une légende : Max Pool, Up-conv) : la
    # taille écrite est celle qui arrive au bloc suivant
    arrow_ops = {str(n.data["arrow"]): shapes[str(u)] for u, n in g.nodes.items()
                 if n.type == "edgeop" and n.data.get("arrow") is not None and str(u) in shapes}
    return {"shapes": shapes, "exact": {k: v for k, v in exact.items()}, "issues": issues, "arrow_ops": arrow_ops,
            "written": written,
            "error": err, "no_torch": no_torch, "params": res.get("params"), "missing": missing}


def check(data, python=None, title=""):
    """Tout en un (sans Qt) : diagramme -> rapport."""
    code, gen = probe_code(data, title=title)
    return analyse(gen, run(code, python))


# ------------------------------------------------------------------ tailles écrites sur le schéma
def arrow_source_uid(arrow, seen=None):
    """Nœud qui produit la valeur d'une flèche (en remontant les jonctions)."""
    from .items import JunctionItem
    node = arrow.src.node
    seen = seen or set()
    if node is None or id(arrow) in seen:
        return None
    seen.add(id(arrow))
    if isinstance(node, JunctionItem):
        ins = [a for a in node.arrows if a.dst.node is node]
        return arrow_source_uid(ins[0], seen) if ins else None
    return node.uid


def shape_for(report, uid):
    return report["shapes"].get(str(uid))


def remove_labels(scene):
    """Retire les tailles écrites par une vérification précédente."""
    from .items import TextItem
    gone = [it for it in scene.items() if isinstance(it, TextItem) and str(it.ref or "").startswith(SHAPE_LABEL_REF)]
    for it in gone:
        for a in list(getattr(it, "arrows", [])):
            scene.removeItem(a)
        scene.removeItem(it)
    return len(gone)


def _overlap(r, avoid):
    hits = 0.0
    for o in avoid:
        i = o.intersected(r)
        if not i.isEmpty():
            hits += i.width() * i.height() + 1
    return hits


def _label_spot(lines, w, h, avoid, late=False, srcs=()):
    """Position d'un texte w×h le long d'une flèche (tracés : la flèche, et celles qui continuent après
    une jonction) : milieu des plus longs segments, à droite d'un segment vertical, au-dessus d'un
    segment horizontal ; on évite les rectangles pris. À défaut, à côté du bloc (ou du cadre) source."""
    from PyQt5.QtCore import QRectF
    segs = sorted(((pts[i], pts[i + 1]) for pts in lines for i in range(len(pts) - 1)),
                  key=lambda s: -((s[1].x() - s[0].x()) ** 2 + (s[1].y() - s[0].y()) ** 2))
    best = None
    for a, b in segs[:4]:
        vertical = abs(b.y() - a.y()) >= abs(b.x() - a.x())
        for t in ((0.75, 0.65, 0.85, 0.5) if late else (0.5, 0.35, 0.65, 0.2, 0.8)):
            m = a + (b - a) * t
            for side in (1, -1):
                if vertical:
                    r = QRectF(m.x() + 5 if side > 0 else m.x() - 5 - w, m.y() - h / 2, w, h)
                else:
                    r = QRectF(m.x() - w / 2, m.y() - h - 3 if side > 0 else m.y() + 3, w, h)
                hits = _overlap(r, avoid)
                if best is None or hits < best[0]:
                    best = (hits, r)
                if hits == 0:
                    return r
    for src in srcs:                    # blocs trop serrés : à côté du bloc qui produit la valeur
        spots = [(x, y) for x in (src.right() + 6, src.left() - 6 - w)
                 for y in (src.center().y() - h / 2, src.bottom() - h, src.bottom() + 2)]
        for x, y in spots:
            r = QRectF(x, y, w, h)
            hits = _overlap(r, avoid)
            if hits == 0:
                return r
            if best is None or hits < best[0]:
                best = (hits, r)
    return best[1] if best else None


def write_labels(scene, report, mode="arrows", font_size=None, color="#555555"):
    """Écrit les formes calculées sur le schéma. mode « arrows » : un texte le long de chaque flèche
    (repéré par sa référence, pour être remplacé à la vérification suivante) ; « blocks » : une ligne
    de formes à la fin du texte de chaque bloc. -> nombre de tailles écrites."""
    from .items import BlockItem, GroupItem, JunctionItem, NodeItem, TextItem
    remove_labels(scene)
    n = 0
    if mode == "blocks":
        for it in scene.items():
            if not isinstance(it, BlockItem):
                continue
            sh = report["exact"].get(str(it.uid))
            if sh is None or len(sh) < 2 or str(it.uid) in report.get("written", ()) \
                    or not any(a.src.node is it for a in it.arrows):
                continue                             # « 4096 » seul serait lu comme une entrée
            text, old = split_shapes(it.text)
            if old and len(old[1]) < 2:                  # « SE-Bottleneck\n256 » : on garde la ligne
                text = it.text
            nums = sorted(int(x) for x in re.findall(r"\d+", text))
            if not any(a.dst.node is it for a in it.arrows) and nums == sorted(sh):
                continue                             # entrée « Image 224×224×3 » : déjà écrite
            line = fmt(sh)
            new = (text.rstrip("\n") + "\n" + line) if text.strip() else line
            if new != it.text:
                it.set_prop("text", new)
                n += 1
        return n
    from PyQt5.QtCore import QRectF
    nodes = [i for i in scene.items() if isinstance(i, NodeItem) and not isinstance(i, (GroupItem, JunctionItem))
             and i.isVisible()]
    avoid = [i.scene_frame().adjusted(-2, -2, 2, 2) for i in nodes]
    for gi in scene.items():                     # bords des cadres (un texte peut être dans un cadre)
        if isinstance(gi, GroupItem) and gi.isVisible():
            f = gi.scene_frame()
            avoid += [QRectF(f.left() - 3, f.top() - 3, f.width() + 6, 6),
                      QRectF(f.left() - 3, f.bottom() - 3, f.width() + 6, 6),
                      QRectF(f.left() - 3, f.top() - 3, 6, f.height() + 6),
                      QRectF(f.right() - 3, f.top() - 3, 6, f.height() + 6)]
            if (gi.label or "").strip():
                avoid.append(gi.mapRectToScene(gi.label_rect()).adjusted(-2, -2, 2, 2))
    for a in scene.arrows():                     # tracés des flèches
        pts = [a.mapToScene(p) for p in a._pts]
        avoid += [QRectF(p, q).normalized().adjusted(-2, -2, 2, 2) for p, q in zip(pts, pts[1:])]
    arrows = sorted(scene.arrows(), key=lambda a: (a.path_rect().top(), a.path_rect().left()))
    for a in arrows:
        src = a.src.node
        if src is None or isinstance(src, JunctionItem):
            continue
        uid = arrow_source_uid(a)
        op = report.get("arrow_ops", {}).get(str(a.uid))
        text = op or shape_for(report, uid)
        dst = a.dst.node
        if not text or (not op and str(uid) in report.get("written", ())):
            continue                                 # taille déjà écrite par l'auteur (« 55×55×96 »)
        pts = [a.mapToScene(p) for p in a._pts]
        if len(pts) < 2:
            continue
        lines = [pts]
        if isinstance(dst, JunctionItem):            # la valeur continue après la jonction
            lines += [[b.mapToScene(p) for p in b._pts] for b in dst.arrows if b.src.node is dst]
        t = TextItem(text)
        t.color = color
        t.ref = f"{SHAPE_LABEL_REF}{a.uid}"
        scene.add_node(t)
        t.font_size = font_size or max(7, round(t.text_size() * 0.85))
        t.on_props_changed()
        srcs = []
        if sum(1 for b in src.arrows if b.src.node is src) == 1 and not op:
            srcs.append(src.scene_frame())
            end = pts[-1]
            par = src.parentItem()                   # la flèche sort d'un cadre : à côté du cadre
            while isinstance(par, GroupItem):
                if not par.scene_frame().contains(end):
                    srcs.append(par.scene_frame())
                par = par.parentItem()
        r = _label_spot(lines, t.w, t.h, avoid, late=bool(op), srcs=srcs)
        if r is None:
            scene.removeItem(t)
            continue
        t.setPos(r.topLeft())
        scene.update_membership([t])
        avoid.append(t.sceneBoundingRect())
        n += 1
    return n
