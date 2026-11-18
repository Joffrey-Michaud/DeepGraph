"""Affiche (en Markdown) les blocs de blocks.yaml utilisables avec la clé "lib" d'un .dgraph.

    python outils/liste_blocs.py            # tableau à copier dans une conversation avec un LLM
(dans l'application : menu Aide > Documentation pour LLM, toujours à jour)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
from deepgraph.config import CFG  # noqa: E402
from deepgraph.docs import blocks_table_md  # noqa: E402

CFG.load()
print(blocks_table_md())
