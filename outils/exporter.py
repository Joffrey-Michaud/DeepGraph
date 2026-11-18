"""Exporte un ou plusieurs .dgraph en PDF / SVG / PNG, sans ouvrir l'éditeur.

    python outils/exporter.py schema.dgraph                      # schema.pdf, .svg et .png à côté
    python outils/exporter.py a.dgraph b.dgraph -f pdf           # PDF seulement
    python outils/exporter.py schema.dgraph -o figs/schema       # figs/schema.pdf, .svg, .png
    python outils/exporter.py modeles/*.dgraph -f pdf -d /tmp/x  # un PDF par fichier dans /tmp/x
    python outils/exporter.py schema.dgraph --font "TeX Gyre Termes" --scale 6

Même rendu que Fichier > Exporter (Ctrl+E) : PDF et SVG vectoriels (texte sélectionnable, polices
intégrées), PNG à l'échelle --scale (4 par défaut, fond blanc ou --transparent). Le fichier peut être
au format simplifié (écrit à la main ou par un LLM, voir docs/DGRAPH_POUR_LLM.md) ou au format
complet enregistré par l'éditeur.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv[:1])

from deepgraph import updater  # noqa: E402
from deepgraph.config import CFG, DEFAULT_CONFIG  # noqa: E402
from deepgraph.export import export_scene  # noqa: E402
from deepgraph.scene import DiagramScene  # noqa: E402

FORMATS = ("pdf", "svg", "png")


def export_file(path, out=None, formats=FORMATS, font=None, scale=4.0, transparent=False):
    """Exporte path ; renvoie la liste des fichiers écrits."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "deepgraph":
        raise ValueError("ce n'est pas un fichier DeepGraph (\"format\": \"deepgraph\" absent)")
    if font:
        data.setdefault("settings", {})["font"] = font
    scene = DiagramScene()
    scene.load(data)
    base = out or os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    written = []
    for fmt in formats:
        target = f"{base}.{fmt}"
        export_scene(scene, target, fmt, scale=scale, transparent=transparent)
        written.append(target)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="fichiers .dgraph à exporter")
    ap.add_argument("-o", "--out", help="chemin de sortie sans extension (un seul fichier d'entrée)")
    ap.add_argument("-d", "--dir", help="dossier de sortie (garde le nom de chaque fichier d'entrée)")
    ap.add_argument("-f", "--formats", default=",".join(FORMATS),
                    help="formats séparés par des virgules parmi pdf, svg, png (défaut : les trois)")
    ap.add_argument("--font", help="police de tout le diagramme (remplace celle du fichier)")
    ap.add_argument("--scale", type=float, default=4.0, help="échelle du PNG (défaut : 4)")
    ap.add_argument("--transparent", action="store_true", help="PNG sur fond transparent")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="fichier YAML de blocs (défaut : blocks.yaml)")
    ap.add_argument("--no-update-check", action="store_true",
                    help="ne pas chercher de mise à jour de DeepGraph (ou DEEPGRAPH_NO_UPDATE=1)")
    args = ap.parse_args()

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        ap.error(f"format(s) inconnu(s) : {', '.join(bad)}")
    if args.out and len(args.files) > 1:
        ap.error("-o / --out n'accepte qu'un seul fichier d'entrée (utiliser -d / --dir)")
    if args.out and args.dir:
        ap.error("-o / --out et -d / --dir s'excluent")

    if not args.no_update_check and not updater.disabled():
        updater.run_terminal(ask=True)          # demande seulement dans un terminal interactif
    CFG.load(args.config)
    for err in CFG.errors:
        print("[config]", err, file=sys.stderr)
    status = 0
    for path in args.files:
        try:
            out = args.out
            if args.dir:
                out = os.path.join(args.dir, os.path.splitext(os.path.basename(path))[0])
            for target in export_file(path, out, formats, args.font, args.scale, args.transparent):
                print("->", target)
        except Exception as e:  # noqa: BLE001
            print(f"[erreur] {path} : {e}", file=sys.stderr)
            status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
