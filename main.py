#!/usr/bin/env python3
"""DeepGraph — éditeur de schémas de réseaux de neurones (style « Attention is all you need »).

Usage :
    python main.py [diagramme.dgraph] [--config blocks.yaml]
"""
import argparse
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from deepgraph import __version__
from deepgraph.config import CFG, DEFAULT_CONFIG
from deepgraph.window import MainWindow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="diagramme .dgraph à ouvrir")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="fichier YAML de blocs (défaut : blocks.yaml)")
    args = ap.parse_args()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("DeepGraph")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    CFG.load(args.config)
    for err in CFG.errors:
        print("[config]", err, file=sys.stderr)

    win = MainWindow()
    win.show()
    if args.file:
        win.open_file(args.file)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
