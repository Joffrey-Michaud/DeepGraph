"""Format simplifié (pensé pour être écrit à la main ou par un LLM) -> format .dgraph complet.

Voir docs/DGRAPH_POUR_LLM.md. Tout est facultatif sauf "format" et les identifiants :
  - "id" (texte ou entier) au lieu de "uid" ; les flèches peuvent omettre leur identifiant ;
  - "cx"/"cy" = centre de l'élément (coordonnées absolues) au lieu de "x"/"y" (coin haut-gauche) ;
  - "lib" = identifiant d'un bloc de blocks.yaml (texte, couleur, taille par défaut) ;
  - groupe : "children" = liste des identifiants contenus (coordonnées des enfants absolues) ;
  - flèche : "from"/"to" (ou "src"/"dst") = identifiant, ou {"node", "side", "at"}, ou {"x", "y"}.

Un fichier au format complet (celui qu'écrit l'éditeur) traverse cette fonction sans changement.
"""
import copy

from .config import CFG
from . import items as _items
from .items import NODE_TYPES

SIDES = {"top": (.5, 0.0), "bottom": (.5, 1.0), "left": (0.0, .5), "right": (1.0, .5), "center": (.5, .5),
         "top-left": (0.0, 0.0), "top-right": (1.0, 0.0), "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0)}
SIDE_ALIASES = {"haut": "top", "bas": "bottom", "gauche": "left", "droite": "right", "centre": "center",
                "n": "top", "s": "bottom", "w": "left", "o": "left", "e": "right", "c": "center"}


class FormatError(ValueError):
    pass


def migrate_font_sizes(data):
    """Fichiers d'avant la taille de texte globale (pas de settings.font_size) : les tailles
    égales au défaut de l'époque (texte 13 px, étiquette de groupe 15 px) suivent désormais
    la taille globale. Les tailles choisies à la main sont conservées."""
    st = data.get("settings")
    if isinstance(st, dict) and "font_size" in st:
        return data
    fs, gs = int(CFG.style["font_size"]), int(CFG.style["group_label_size"])
    for d in data.get("nodes") or []:
        if d.get("type") in ("block", "text") and d.get("font_size") == fs:
            d["font_size"] = None
        elif d.get("type") == "group" and d.get("label_size") == gs:
            d["label_size"] = None
    return data


def _side(name):
    key = str(name).strip().lower()
    key = SIDE_ALIASES.get(key, key)
    if key not in SIDES:
        raise FormatError(f"côté inconnu « {name} » (attendu : {', '.join(SIDES)})")
    return key


def _lib_defaults(d):
    """Complète un nœud qui référence un bloc de la bibliothèque (clé "lib" ou "def_id")."""
    bid = d.pop("lib", None) or d.get("def_id")
    if not bid:
        return
    bd = CFG.blocks.get(bid)
    if bd is None:                       # bloc inconnu : bloc libre portant ce nom
        d.pop("def_id", None)
        d.setdefault("type", "block")
        d.setdefault("text", bid)
        return
    d["def_id"] = bid
    if bd.shape == "circle":
        d.setdefault("type", "operator")
        d.setdefault("symbol", bd.symbol)
        d.setdefault("color", bd.color)
        if bd.width:
            d.setdefault("w", bd.width)
            d.setdefault("h", d["w"])
        return
    d.setdefault("type", "block")
    d.setdefault("text", bd.label)
    d.setdefault("color", bd.color)
    if bd.width:
        d.setdefault("w", bd.width)
    if bd.height:
        d.setdefault("h", bd.height)
    if bd.font_size:
        d.setdefault("font_size", bd.font_size)
    if bd.text_color:
        d.setdefault("text_color", bd.text_color)
    if bd.bold:
        d.setdefault("bold", bd.bold)
    if bd.stack:
        d.setdefault("stack", bd.stack)


def _probe_sizes(nodes):
    """Valeurs par défaut (bibliothèque, type) et taille réelle de chaque nœud."""
    for d in nodes:
        _lib_defaults(d)
        d.setdefault("type", "block")
        cls = NODE_TYPES.get(d["type"])
        if cls is None:
            raise FormatError(f"« {d['_key']} » : type inconnu « {d['type']} » "
                              f"(attendu : {', '.join(NODE_TYPES)})")
        if d["type"] == "group" and "children" in d:
            d.setdefault("label", "")
        probe = cls.from_dict(d)              # taille réelle (textes mesurés, tailles par défaut)
        d["w"], d["h"] = float(probe.w), float(probe.h)


def expand(data):
    """Renvoie une copie au format complet (lève FormatError avec un message lisible)."""
    if not isinstance(data, dict):
        raise FormatError("le fichier doit contenir un objet JSON")
    data = copy.deepcopy(data)
    nodes = data.get("nodes") or []
    arrows = data.get("arrows") or []
    if not isinstance(nodes, list) or not isinstance(arrows, list):
        raise FormatError("« nodes » et « arrows » doivent être des listes")
    data["nodes"], data["arrows"] = nodes, arrows

    # ------------------------------------------------ identifiants -> entiers
    ids, used = {}, set()
    for d in nodes:
        if "uid" in d and isinstance(d["uid"], int):
            used.add(d["uid"])
    nxt = max(used | {0}) + 1
    for i, d in enumerate(nodes):
        if not isinstance(d, dict):
            raise FormatError(f"nœud n° {i + 1} : objet JSON attendu")
        key = d.pop("id", None)
        if key is None:
            key = d.get("uid")
        if key is None:
            raise FormatError(f"nœud n° {i + 1} ({d.get('text') or d.get('lib') or d.get('type')}) : "
                              "« id » manquant")
        if key in ids:
            raise FormatError(f"identifiant en double : « {key} »")
        if isinstance(d.get("uid"), int):
            uid = d["uid"]
        elif isinstance(key, int) and key not in used:
            uid = key
            used.add(uid)
        else:
            while nxt in used:
                nxt += 1
            uid = nxt
            used.add(uid)
        d["uid"] = uid
        d["_key"] = key
        ids[key] = d
        if key != uid:
            ids.setdefault(uid, d)

    def node(key, where):
        if isinstance(key, float) and key.is_integer():
            key = int(key)
        d = ids.get(key)
        if d is None:
            raise FormatError(f"{where} : élément « {key} » introuvable")
        return d

    # --------------------------------------------- valeurs par défaut, tailles
    st = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    _items._FONT_FALLBACK[0] = st.get("font_size")          # mesure des textes à la bonne taille
    try:
        _probe_sizes(nodes)
    finally:
        _items._FONT_FALLBACK[0] = None

    # ---------------------------------------- groupes déclarés par « children »
    parent_of = {}
    for d in nodes:
        kids = d.pop("children", None)
        if kids is None:
            continue
        if d["type"] != "group":
            raise FormatError(f"« {d['_key']} » : seuls les groupes ont des « children »")
        for k in kids:
            c = node(k, f"enfants du groupe {d['uid']}")
            if c is d:
                raise FormatError(f"le groupe « {k} » se contient lui-même")
            if id(c) in parent_of:
                raise FormatError(f"« {k} » est dans deux groupes")
            parent_of[id(c)] = d
            c["parent"] = d["uid"]
            c["_abs"] = True
    by_uid = {d["uid"]: d for d in nodes}

    # ------------------------------------------------ positions absolues (x, y)
    def place(d):
        """Coin haut-gauche absolu de d (calculé une fois)."""
        if "_ax" in d:
            return d["_ax"], d["_ay"]
        if "cx" in d or "cy" in d:
            x = float(d.pop("cx", 0.0)) - d["w"] / 2
            y = float(d.pop("cy", 0.0)) - d["h"] / 2
        elif d["type"] == "group" and any(parent_of.get(id(c)) is d for c in nodes) and "x" not in d:
            x = y = None                       # cadre ajusté au contenu : calculé ci-dessous
        else:
            x, y = float(d.get("x", 0.0)), float(d.get("y", 0.0))
            par = by_uid.get(d.get("parent"))
            if par is not None and not d.get("_abs"):          # format complet : relatif au parent
                px, py = place(par)
                x, y = x + px, y + py
        if x is None:
            kids = [c for c in nodes if parent_of.get(id(c)) is d]
            pts = [place(c) for c in kids]
            m = float(d.get("margin", CFG.style["group_padding"]))
            x = min(p[0] for p in pts) - float(d.get("margin_left") if d.get("margin_left") is not None else m)
            y = min(p[1] for p in pts) - float(d.get("margin_top") if d.get("margin_top") is not None else m)
            d.setdefault("fit", True)
        d["_ax"], d["_ay"] = x, y
        return x, y

    for d in nodes:
        place(d)
    for d in nodes:
        par = by_uid.get(d.get("parent"))
        if par is not None:
            d["x"], d["y"] = d["_ax"] - par["_ax"], d["_ay"] - par["_ay"]
        else:
            d["x"], d["y"] = d["_ax"], d["_ay"]
            d["parent"] = None
    # les parents doivent précéder leurs enfants
    depth = {}

    def dep(d):
        if d["uid"] not in depth:
            par = by_uid.get(d.get("parent"))
            depth[d["uid"]] = 0 if par is None else dep(par) + 1
        return depth[d["uid"]]

    for d in nodes:
        if dep(d) > 50:
            raise FormatError("groupes imbriqués en boucle")
    order = {id(d): i for i, d in enumerate(nodes)}
    nodes.sort(key=lambda d: (dep(d), order[id(d)]))

    # ----------------------------------------------------------------- flèches
    def center(d):
        return d["_ax"] + d["w"] / 2, d["_ay"] + d["h"] / 2

    def endpoint(e, other, where, is_src):
        if isinstance(e, (str, int, float)) and not isinstance(e, bool):
            e = {"node": e}
        if not isinstance(e, dict):
            raise FormatError(f"{where} : extrémité invalide")
        if "node" not in e:
            if "x" in e and "y" in e:
                return {"x": float(e["x"]), "y": float(e["y"])}, None
            raise FormatError(f"{where} : extrémité sans « node » ni « x »/« y »")
        d = node(e["node"], where)
        if "fx" in e and "fy" in e:
            return {"node": d["uid"], "fx": float(e["fx"]), "fy": float(e["fy"])}, d
        side = e.get("side")
        if side is None:
            return {"node": d["uid"], "_auto": True}, d
        fx, fy = SIDES[_side(side)]
        if "at" in e:                          # position le long du côté (0 -> 1)
            t = float(e["at"])
            if fy in (0.0, 1.0) and fx == .5:
                fx = t
            elif fx in (0.0, 1.0) and fy == .5:
                fy = t
        return {"node": d["uid"], "fx": fx, "fy": fy}, d

    def auto_pair(a, b, da, db, pa, pb):
        """Choisit les côtés en vis-à-vis pour les extrémités sans côté."""
        ax, ay = center(da) if da else (a.get("x"), a.get("y"))
        bx, by = center(db) if db else (b.get("x"), b.get("y"))
        dx, dy = bx - ax, by - ay
        if abs(dy) >= abs(dx):
            sa, sb = ("bottom", "top") if dy >= 0 else ("top", "bottom")
        else:
            sa, sb = ("right", "left") if dx >= 0 else ("left", "right")
        for e, dn, s in ((a, da, sa), (b, db, sb)):
            if e.pop("_auto", False):
                e["fx"], e["fy"] = SIDES["center"] if dn["type"] == "junction" else SIDES[s]

    a_used = set(used) | {a.get("uid") for a in arrows if isinstance(a, dict) and isinstance(a.get("uid"), int)}
    na = max(a_used | {0}) + 1
    for i, a in enumerate(arrows):
        if not isinstance(a, dict):
            raise FormatError(f"flèche n° {i + 1} : objet JSON attendu")
        where = f"flèche n° {i + 1}"
        src = a.pop("from", None) if "src" not in a else a["src"]
        dst = a.pop("to", None) if "dst" not in a else a["dst"]
        if src is None or dst is None:
            raise FormatError(f"{where} : « from » et « to » obligatoires")
        s, ds = endpoint(src, dst, where, True)
        t, dt = endpoint(dst, src, where, False)
        if s.get("_auto") or t.get("_auto"):
            auto_pair(s, t, ds, dt, src, dst)
        a["src"], a["dst"] = s, t
        a["type"] = "arrow"
        a.pop("id", None)
        if not isinstance(a.get("uid"), int):
            while na in a_used:
                na += 1
            a["uid"] = na
            a_used.add(na)

    for d in nodes:
        for k in ("_ax", "_ay", "_abs", "_key"):
            d.pop(k, None)
    data.setdefault("version", 1)
    return data
