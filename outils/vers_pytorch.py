#!/usr/bin/env python3
"""Squelette de code PyTorch à partir d'un schéma DeepGraph (l'inverse de depuis_pytorch.py).

    python outils/vers_pytorch.py schema.dgraph                  # écrit schema.py à côté
    python outils/vers_pytorch.py schema.dgraph -o modele.py     # autre fichier de sortie
    python outils/vers_pytorch.py schema.dgraph --essai          # puis l'exécute sur des entrées factices
    python outils/vers_pytorch.py modeles/*.dgraph -d /tmp/code  # un .py par schéma

Les blocs de la bibliothèque, le texte des blocs (« Conv 7×7, 64, /2 », « BN + ReLU »), les flèches,
les opérateurs ⊕ ⊗ et les cadres (« N× », « Bloc 1 », cadre en pointillés) sont traduits ; ce que le
schéma ne dit pas reçoit une valeur par défaut marquée TODO. Voir docs/VERS_PYTORCH.md.

Ni Qt ni torch ne sont nécessaires pour générer le code (torch l'est pour --essai).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deepgraph.codegen import generate  # noqa: E402


def user_defs():
    """Champ « pytorch » des blocs (blocks.yaml, puis mes_blocs.yaml qui le remplace)."""
    try:
        import yaml
    except ImportError:
        return {}
    out = {}
    for name in ("blocks.yaml", "mes_blocs.yaml"):
        try:
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        for raw in data.get("blocks") or []:
            if isinstance(raw, dict) and raw.get("name"):
                bid = str(raw.get("id", raw["name"]))
                if raw.get("pytorch"):
                    out[bid] = str(raw["pytorch"]).strip()
                else:
                    out.pop(bid, None)
    return out


def convert(path, out, defs):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "deepgraph":
        raise ValueError("ce n'est pas un fichier DeepGraph (\"format\": \"deepgraph\" absent)")
    code, info = generate(data, defs, os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(code)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="schémas .dgraph")
    ap.add_argument("-o", "--out", help="fichier .py à écrire (un seul schéma)")
    ap.add_argument("-d", "--dir", help="dossier de sortie (garde le nom de chaque schéma)")
    ap.add_argument("--essai", action="store_true",
                    help="exécute le code produit sur des entrées factices (torch nécessaire)")
    args = ap.parse_args()
    if args.out and len(args.files) > 1:
        ap.error("-o / --out n'accepte qu'un seul schéma (utiliser -d / --dir)")
    if args.out and args.dir:
        ap.error("-o / --out et -d / --dir s'excluent")

    defs = user_defs()
    status = 0
    for path in args.files:
        base = os.path.splitext(os.path.basename(path))[0]
        out = args.out or (os.path.join(args.dir, base + ".py") if args.dir else os.path.splitext(path)[0] + ".py")
        try:
            info = convert(path, out, defs)
        except Exception as e:  # noqa: BLE001
            print(f"[erreur] {path} : {e}", file=sys.stderr)
            status = 1
            continue
        todo = f", {info['todo']} TODO" if info["todo"] else ""
        print(f"-> {out}  ({', '.join(info['classes']) or 'aucune classe'}{todo})")
        if args.essai:
            r = subprocess.run([sys.executable, out], capture_output=True, text=True)
            if r.returncode == 0:
                print("   essai :", r.stdout.strip().replace("\n", "\n   essai : "))
            else:
                err = (r.stderr.strip().splitlines() or ["?"])[-1]
                print(f"   essai : ÉCHEC — {err}", file=sys.stderr)
                status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
