"""Chargement de la configuration : style global + bibliothèque de blocs (YAML)."""
import os
from dataclasses import dataclass
from typing import Optional

import yaml
from PyQt5.QtGui import QColor, QFont, QFontDatabase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "blocks.yaml")

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


class Config:
    def __init__(self):
        self.path = DEFAULT_CONFIG
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

        blocks, cats = {}, []
        for i, raw in enumerate(data.get("blocks") or []):
            where = f"blocks[{i}]"
            if not isinstance(raw, dict) or "name" not in raw:
                errors.append(f"{where} : chaque bloc doit être un dictionnaire avec au moins « name »")
                continue
            unknown = set(raw) - BLOCK_FIELDS
            if unknown:
                errors.append(f"{where} ({raw['name']}) : champ(s) inconnu(s) {sorted(unknown)}")
            name = str(raw["name"])
            bid = str(raw.get("id", name))
            if bid in blocks:
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
            blocks[bid] = bd
            if bd.category not in cats:
                cats.append(bd.category)

        self.style, self.blocks, self.categories, self.errors = style, blocks, cats, errors
        self._family, self._fonts = None, {}
        return not errors

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
