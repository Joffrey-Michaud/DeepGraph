#!/usr/bin/env python3
"""DeepGraph — éditeur de schémas de réseaux de neurones (style « Attention is all you need »).

Usage :
    python main.py [diagramme.dgraph] [--config blocks.yaml] [--no-update-check]
    python main.py --update        # installe la dernière release (sans fenêtre)
"""
import argparse
import os
import sys

from PyQt5.QtCore import QLibraryInfo, Qt, QTranslator
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from deepgraph import __version__, updater
from deepgraph.config import CFG, DEFAULT_CONFIG
from deepgraph.window import MainWindow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="diagramme .dgraph à ouvrir")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="fichier YAML de blocs (défaut : blocks.yaml)")
    ap.add_argument("--update", action="store_true",
                    help="chercher et installer la dernière release, puis quitter (sans fenêtre)")
    ap.add_argument("--no-update-check", action="store_true",
                    help="ne pas chercher de mise à jour au démarrage (ou DEEPGRAPH_NO_UPDATE=1)")
    ap.add_argument("--version", action="version", version=f"DeepGraph {__version__}")
    args = ap.parse_args()

    if args.update:                      # mode texte : pas besoin de Qt
        sys.exit(0 if updater.run_terminal(apply=True, out=sys.stdout) in ("installed", "uptodate") else 1)
    headless = os.environ.get("QT_QPA_PLATFORM", "") in ("offscreen", "minimal")
    if headless and not args.no_update_check and not updater.disabled():
        updater.run_terminal(ask=True)   # terminal interactif : demande ; sinon installe tout seul

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("DeepGraph")
    app.setApplicationVersion(__version__)
    icon = QIcon()                       # icône de l'application (fenêtres, barre des tâches)
    icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepgraph", "icon")
    for size in (16, 32, 48, 64, 128, 256):
        icon.addFile(os.path.join(icon_dir, f"logo_{size}.png"))
    app.setWindowIcon(icon)
    app.setStyle("Fusion")
    tr = QTranslator(app)                # boutons standard de Qt en français (« Afficher les détails… »)
    if tr.load("qtbase_fr", QLibraryInfo.location(QLibraryInfo.TranslationsPath)):
        app.installTranslator(tr)

    CFG.load(args.config)
    for err in CFG.errors:
        print("[config]", err, file=sys.stderr)

    win = MainWindow()
    win.show()
    if args.file:
        win.open_file(args.file)
    if not headless and not args.no_update_check:
        from deepgraph.update_ui import check_in_background
        check_in_background(win)         # en tâche de fond : ne retarde pas l'ouverture
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
