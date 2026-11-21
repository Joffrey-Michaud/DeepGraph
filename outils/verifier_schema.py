"""Vérifie un schéma (formes des tenseurs, incohérences) et peut y écrire les tailles, sans l'éditeur.

    python outils/verifier_schema.py schema.dgraph                       # rapport
    python outils/verifier_schema.py schema.dgraph --ecrire fleches      # + schema_tailles.dgraph
    python outils/verifier_schema.py schema.dgraph --ecrire blocs -o out.dgraph --png out.png
    python outils/verifier_schema.py schema.dgraph --python ~/envs/torch/bin/python

Même calcul que Outils > Vérifier le schéma… dans l'éditeur : le schéma est traduit en code PyTorch
(comme Fichier > Exporter en code PyTorch…), exécuté une fois sur des entrées factices avec
l'interpréteur --python (qui doit avoir torch), et chaque bloc note la forme de sa sortie.
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

from deepgraph import shapecheck  # noqa: E402
from deepgraph.config import CFG  # noqa: E402
from deepgraph.export import export_scene  # noqa: E402
from deepgraph.scene import DiagramScene  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("schema", help="fichier .dgraph")
    ap.add_argument("--python", default=sys.executable, help="interpréteur qui a torch (défaut : celui-ci)")
    ap.add_argument("--ecrire", choices=["fleches", "blocs"], help="écrit les tailles le long des flèches ou "
                    "dans les blocs")
    ap.add_argument("-o", "--out", help="fichier .dgraph écrit avec --ecrire (défaut : schema_tailles.dgraph)")
    ap.add_argument("--png", help="aperçu PNG du schéma (avec les tailles si --ecrire)")
    ap.add_argument("--json", action="store_true", help="rapport en JSON")
    a = ap.parse_args(argv)
    CFG.load()
    with open(a.schema, encoding="utf-8") as f:
        data = json.load(f)
    scene = DiagramScene()
    scene.load(data)
    data = scene.to_dict()
    title = os.path.splitext(os.path.basename(a.schema))[0]
    rep = shapecheck.check(data, a.python, title)
    if a.json:
        json.dump({k: rep[k] for k in ("shapes", "issues", "error", "params")}, sys.stdout,
                  ensure_ascii=False, indent=1, default=str)
        print()
    else:
        print(report_text(rep, a.python))
    if a.ecrire:
        n = shapecheck.write_labels(scene, rep, "arrows" if a.ecrire == "fleches" else "blocks")
        out = a.out or os.path.splitext(a.schema)[0] + "_tailles.dgraph"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(scene.to_dict(), f, indent=1, ensure_ascii=False)
        print(f"{out} : {n} taille(s) écrite(s)", file=sys.stderr)
    if a.png:
        export_scene(scene, a.png, "png", scale=3)
    return 1 if any(i["level"] == "erreur" for i in rep["issues"]) or rep["no_torch"] else 0


def report_text(rep, python):
    lines = []
    if rep["no_torch"]:
        return f"torch introuvable pour {python} : donner --python (interpréteur de l'environnement avec torch)"
    n_sh = len(rep["shapes"])
    lines.append(f"{n_sh} forme(s) calculée(s)" + (f", {rep['params']:,} paramètres".replace(",", " ")
                                                     if rep.get("params") else ""))
    for lv in shapecheck.LEVELS:
        items = [i for i in rep["issues"] if i["level"] == lv]
        if items:
            lines.append(f"\n{lv.capitalize()} ({len(items)}) :")
            lines += [f"  - {i['text']}" for i in items]
    if rep["missing"]:
        lines.append(f"\n{len(rep['missing'])} bloc(s) sans forme calculée")
    if not rep["issues"]:
        lines.append("Aucune incohérence trouvée.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
