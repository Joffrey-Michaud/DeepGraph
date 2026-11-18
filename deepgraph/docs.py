"""Textes d'aide affichés dans l'application (menu Aide)."""
import os

from .config import CFG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_DOC = os.path.join(ROOT, "docs", "DGRAPH_POUR_LLM.md")
README = os.path.join(ROOT, "README.md")
ANNEX = "## Annexe"


def blocks_table_md():
    """Table Markdown des blocs utilisables avec la clé "lib" (d'après la configuration chargée)."""
    st = CFG.style
    out = [f"Taille par défaut d'un bloc : {st['block_width']} × {st['block_height']} px ; "
           f"opérateur : diamètre {st['operator_size']} px.", "",
           "| lib | texte affiché | catégorie | couleur | taille (l × h) |", "|---|---|---|---|---|"]
    for bid, b in CFG.blocks.items():
        label = b.label.replace("\n", " / ").replace("|", "\\|")
        if b.shape == "circle":
            size = f"cercle « {b.symbol} »"
        else:
            size = f"{b.width or st['block_width']:g} × {b.height or st['block_height']:g}"
        out.append(f"| `{bid}` | {label} | {b.category} | {b.color} | {size} |")
    return "\n".join(out) + "\n"


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"# Fichier introuvable\n\n`{path}`\n\n{e}\n"


def llm_doc():
    """Documentation pour LLM, avec l'annexe des blocs régénérée depuis la configuration actuelle."""
    text = _read(LLM_DOC)
    i = text.find(ANNEX)
    if i < 0:
        return text
    head, annex = text[:i], text[i:]
    j = annex.find("\n|")                      # début de l'ancienne table
    k = annex.find("Taille par défaut")
    cut = min(x for x in (j, k) if x >= 0) if (j >= 0 or k >= 0) else len(annex)
    intro = annex[:cut].rstrip() + "\n\n"
    intro = intro.replace("Table générée depuis `blocks.yaml` (la régénérer après avoir ajouté des blocs :\n"
                          "`python outils/liste_blocs.py`).",
                          f"Table générée depuis la configuration chargée (`{os.path.basename(CFG.path)}`).")
    return head + intro + blocks_table_md()


def user_guide():
    return _read(README)
