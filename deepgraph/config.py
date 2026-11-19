"""Chargement de la configuration : style global + bibliothèque de blocs (YAML)."""
import os
from dataclasses import dataclass
from typing import Optional

import yaml
from PyQt5.QtGui import QColor, QFont, QFontDatabase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "blocks.yaml")
# blocs ajoutés ou modifiés depuis l'interface : fichier à part, écrit par DeepGraph, ignoré par git
# et jamais touché par les mises à jour
USER_CONFIG = os.path.join(ROOT, "mes_blocs.yaml")
USER_HEADER = """\
# Blocs ajoutés ou modifiés depuis DeepGraph (bouton « Nouveau bloc… », clic droit > Modifier).
# Ce fichier est réécrit par DeepGraph : le plus simple est de passer par l'interface.
# Un bloc qui a le même « id » qu'un bloc de blocks.yaml le remplace.
"""

DEFAULT_STYLE = {
    "font": ["Helvetica", "TeX Gyre Heros", "Nimbus Sans", "Nimbus Sans L", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font_size": 13,              # px
    "text_color": "#000000",
    "border_color": "#2B2B2B",
    "border_width": 1.6,
    "corner_radius": 5,
    "block_width": 110,
    "block_height": 28,
    "operator_size": 18,
    "arrow_color": "#000000",
    "arrow_width": 1.6,
    "arrow_head_length": 8,
    "arrow_head_width": 6.5,
    "arrow_corner_radius": 6,
    "group_fill": "#F2F2F2",
    "group_corner_radius": 12,
    "group_padding": 15,
    "group_label_size": 15,
    "grid_size": 5,
}

BLOCK_FIELDS = {"id", "name", "label", "color", "description", "category", "shape", "symbol",
                "width", "height", "font_size", "text_color", "bold", "stack"}


@dataclass
class BlockDef:
    id: str
    name: str
    label: str
    color: str = "#FFFFFF"
    description: str = ""
    category: str = "Divers"
    shape: str = "rect"          # rect | circle
    symbol: str = "plus"         # pour les cercles (concat, plus, minus, times, dot ou texte)
    width: Optional[float] = None
    height: Optional[float] = None
    font_size: Optional[int] = None
    text_color: Optional[str] = None
    bold: bool = False
    stack: int = 0
    origin: str = "base"         # base (blocks.yaml) | user (mes_blocs.yaml) | override (les deux)


class Config:
    def __init__(self):
        self.path = DEFAULT_CONFIG
        self.user_path = USER_CONFIG
        self.base_blocks = {}
        self.user_raw = []           # contenu brut de mes_blocs.yaml (pour le réécrire)
        self.style = dict(DEFAULT_STYLE)
        self.blocks = {}
        self.categories = []
        self.errors = []
        self._family = None
        self._fonts = {}
        self.family_override = None     # police choisie pour le diagramme (None = celle de la config)

    # ------------------------------------------------------------------ load
    def load(self, path=None):
        self.path = os.path.abspath(path or self.path)
        errors = []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            errors.append(f"Fichier introuvable : {self.path}")
            data = {}
        except yaml.YAMLError as e:
            errors.append(f"Erreur de syntaxe YAML : {e}")
            data = {}
        if not isinstance(data, dict):
            errors.append("Le fichier doit contenir un dictionnaire (clés 'style' et 'blocks').")
            data = {}

        style = dict(DEFAULT_STYLE)
        user_style = data.get("style") or {}
        if isinstance(user_style, dict):
            for k, v in user_style.items():
                if k not in DEFAULT_STYLE:
                    errors.append(f"style : clé inconnue « {k} »")
                    continue
                style[k] = v

        blocks = {}
        for bd in parse_blocks(data.get("blocks"), errors):
            blocks[bd.id] = bd
        self.base_blocks = dict(blocks)

        # blocs de l'utilisateur
        self.user_raw = []
        if self.user_path and os.path.exists(self.user_path):
            uname = os.path.basename(self.user_path)
            try:
                with open(self.user_path, encoding="utf-8") as f:
                    udata = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                errors.append(f"{uname} : erreur de syntaxe YAML : {e}")
                udata = {}
            except OSError as e:
                errors.append(f"{uname} : {e}")
                udata = {}
            raws = udata.get("blocks") if isinstance(udata, dict) else None
            self.user_raw = [r for r in (raws or []) if isinstance(r, dict)]
            for bd in parse_blocks(raws, errors, uname):
                bd.origin = "override" if bd.id in blocks else "user"
                blocks[bd.id] = bd          # même id : remplace le bloc de blocks.yaml, à sa place
        cats = []
        for bd in blocks.values():
            if bd.category not in cats:
                cats.append(bd.category)

        self.style, self.blocks, self.categories, self.errors = style, blocks, cats, errors
        self._family, self._fonts = None, {}
        return not errors

    # ------------------------------------------------------- blocs utilisateur
    def new_id(self, name):
        """Identifiant libre pour un nouveau bloc : le nom, sinon « nom (2) »…"""
        bid, k = name, 2
        while bid in self.blocks or bid.startswith("__"):
            bid, k = f"{name} ({k})", k + 1
        return bid

    def save_user_block(self, raw, old_id=None):
        """Ajoute ou remplace (même id, ou old_id) un bloc dans mes_blocs.yaml."""
        new = str(raw.get("id", raw["name"]))
        key = old_id or new
        out, done = [], False
        for r in self.user_raw:
            if str(r.get("id", r.get("name"))) in (key, new):
                if not done:
                    out.append(raw)
                    done = True
                continue
            out.append(r)
        if not done:
            out.append(raw)
        self._write_user(out)

    def remove_user_block(self, bid):
        """Retire le bloc de mes_blocs.yaml (bloc ajouté : supprimé ; bloc modifié : rétabli)."""
        self._write_user([r for r in self.user_raw if str(r.get("id", r.get("name"))) != bid])

    def _write_user(self, raws):
        text = USER_HEADER + "\n" + yaml.dump({"blocks": raws}, Dumper=_Dumper, allow_unicode=True,
                                               sort_keys=False, width=100, default_flow_style=False)
        tmp = self.user_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.user_path)
        self.user_raw = list(raws)

    # ----------------------------------------------------------------- fonts
    @staticmethod
    def resolve_family(name):
        """Nom de police -> famille Qt installée (ou None si absente).

        Une même police peut être enregistrée sous plusieurs fonderies (« Nimbus Sans [URW ] »,
        « Nimbus Sans [urw] », « Nimbus Sans [UKWN] »), certaines avec une partie seulement de ses
        styles. Prendre la première venue peut donner une fonderie sans style droit : tout le texte
        sort alors en italique. On garde donc la fonderie la plus complète (styles droits et
        italiques, sinon au moins un style droit), la première en cas d'égalité."""
        if not name:
            return None
        db = QFontDatabase()
        fams = db.families()
        c = str(name).lower()
        exact = next((f for f in fams if f.lower() == c), None)
        if exact:
            return exact
        cands = [f for f in fams if f.lower().startswith(c + " [")]
        if not cands:
            return None

        def completeness(fam):
            italic = [db.italic(fam, s) for s in db.styles(fam)]
            return 2 * any(not i for i in italic) + any(italic)

        return max(cands, key=completeness)

    def default_family(self):
        cands = self.style["font"]
        if isinstance(cands, str):
            cands = [cands]
        for c in cands:
            hit = self.resolve_family(c)
            if hit:
                return hit
        return QFont().defaultFamily()

    def font_family(self):
        if self._family is None:
            self._family = self.resolve_family(self.family_override) or self.default_family()
        return self._family

    def set_family_override(self, name):
        if name != self.family_override:
            self.family_override = name
            self._family, self._fonts = None, {}

    def font(self, px, bold=False, italic=False):
        """Police en taille *pixel* : indispensable pour que l'export PDF/SVG ait les mêmes proportions."""
        key = (max(1, int(round(px))), bool(bold), bool(italic))
        if key not in self._fonts:
            f = QFont(self.font_family())
            f.setPixelSize(key[0])
            f.setBold(key[1])
            f.setItalic(key[2])
            f.setHintingPreference(QFont.PreferNoHinting)
            self._fonts[key] = f
        return QFont(self._fonts[key])


def parse_blocks(raws, errors, source=""):
    """Liste YAML de blocs -> BlockDef valides (les problèmes vont dans errors)."""
    out, seen = [], set()
    pre = f"{source} : " if source else ""
    for i, raw in enumerate(raws or []):
        where = f"{pre}blocks[{i}]"
        if not isinstance(raw, dict) or "name" not in raw:
            errors.append(f"{where} : chaque bloc doit être un dictionnaire avec au moins « name »")
            continue
        unknown = set(raw) - BLOCK_FIELDS
        if unknown:
            errors.append(f"{where} ({raw['name']}) : champ(s) inconnu(s) {sorted(unknown)}")
        name = str(raw["name"])
        bid = str(raw.get("id", name))
        if bid in seen:
            errors.append(f"{where} : identifiant « {bid} » en double (ajoute un champ « id »)")
            continue
        color = str(raw.get("color", "#FFFFFF"))
        if not QColor(color).isValid():
            errors.append(f"{where} ({name}) : couleur invalide « {color} »")
            color = "#FFFFFF"
        shape = str(raw.get("shape", "rect"))
        if shape not in ("rect", "circle"):
            errors.append(f"{where} ({name}) : shape doit valoir 'rect' ou 'circle'")
            shape = "rect"
        try:
            bd = BlockDef(
                id=bid, name=name, label=str(raw.get("label", name)), color=color,
                description=str(raw.get("description", "") or "").strip(),
                category=str(raw.get("category", "Divers")), shape=shape,
                symbol=str(raw.get("symbol", "plus")),
                width=float(raw["width"]) if raw.get("width") else None,
                height=float(raw["height"]) if raw.get("height") else None,
                font_size=int(raw["font_size"]) if raw.get("font_size") else None,
                text_color=str(raw["text_color"]) if raw.get("text_color") else None,
                bold=bool(raw.get("bold", False)), stack=int(raw.get("stack", 0) or 0))
        except (TypeError, ValueError) as e:
            errors.append(f"{where} ({name}) : valeur invalide ({e})")
            continue
        seen.add(bid)
        out.append(bd)
    return out


class _Dumper(yaml.SafeDumper):
    pass


def _str_repr(dumper, value):
    # descriptions sur plusieurs lignes : bloc « | » lisible plutôt qu'une chaîne avec des \n
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_repr)


def description_html(text, title=None, summary_only=False):
    """Description (texte brut, plusieurs lignes) -> HTML pour les infobulles et panneaux.
    1re ligne = résumé ; les lignes « Réf. » sont affichées en plus discret."""
    import html
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    if summary_only:
        lines = lines[:1]
    out = []
    if title:
        out.append(f"<b>{html.escape(title)}</b>")
    for i, l in enumerate(lines):
        e = html.escape(l)
        if l.startswith("Réf."):
            e = f"<span style='color:gray'><i>{e}</i></span>"
        elif l.startswith(("Formule", "Formules")) and ":" in l:
            k, v = e.split(":", 1)
            e = f"<u>{k.strip()}</u> :{v}"
        out.append(e)
    return "<br>".join(out)


CFG = Config()
