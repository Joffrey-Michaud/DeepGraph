"""Fenêtre « Importer un modèle PyTorch… » : interface de outils/depuis_pytorch.py.

L'outil tourne dans un processus à part, avec l'interpréteur Python choisi (celui de
l'environnement qui contient torch et le code du modèle) et depuis la racine du projet du
modèle : exactement comme la ligne de commande, dont la fenêtre montre l'équivalent. Une erreur
dans le code du modèle ne peut donc pas fermer l'éditeur.
"""
import ast
import json
import os
import shlex
import sys
import tempfile

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .docs import ROOT

TOOL = os.path.join(ROOT, "outils", "depuis_pytorch.py")
KEY_LAST, KEY_MODELS = "pytorch_import/last", "pytorch_import/models"
TYPES = ["float", "long", "int", "bool"]
SHAPES = [("entree-sortie", "entrée → sortie"), ("sortie", "sortie seulement"), ("aucune", "aucune")]
VIEWS = [("", "normal"), ("deplier", "déplier"), ("replier", "replier")]
MARKERS = (".git", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")
ERR = "#C0392B"


# ------------------------------------------------------------------ lecture du fichier (sans torch)
def scan_file(path):
    """Classes d'un fichier .py lues avec ast (rien n'est exécuté).

    -> (classes [nom], classe proposée, {nom: (init, forward)}) ; init / forward : [(nom, défaut)].
    Les sous-classes de nn.Module d'abord ; la proposée est la dernière qu'aucune autre n'utilise
    (le modèle complet est en général défini après ses blocs)."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return [], None, {}
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            meths = {f.name: f for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - {node.name}
            classes[node.name] = (bases, meths, used)

    def is_module(name, seen=()):
        for b in classes[name][0]:
            if b in classes:
                if b not in seen and is_module(b, seen + (name,)):
                    return True
            elif b.split(".")[-1].endswith(("Module", "Sequential", "ModuleList", "ModuleDict")):
                return True
        return False

    def method(name, meth, seen=()):
        bases, meths, _ = classes[name]
        if meth in meths:
            return meths[meth]
        for b in bases:
            if b in classes and b not in seen:
                r = method(b, meth, seen + (name,))
                if r is not None:
                    return r
        return None

    def params(f):
        if f is None:
            return None
        a = f.args
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + [ast.unparse(d) for d in a.defaults]
        out = [(p.arg, d) for p, d in zip(pos, defaults)][1:]          # sans self
        out += [(p.arg, ast.unparse(d) if d is not None else None) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
        return out

    mods = [c for c in classes if is_module(c)]
    names = mods + [c for c in classes if c not in mods] if mods else list(classes)
    used = set().union(*(classes[c][2] for c in mods)) if mods else set()
    top = [c for c in mods if c not in used]
    best = (top or mods or names or [None])[-1]
    sigs = {c: (params(method(c, "__init__")), params(method(c, "forward"))) for c in classes}
    return names, best, sigs


def project_root(path):
    """Racine du projet d'un fichier : le dossier parent le plus proche qui contient .git,
    pyproject.toml… (là d'où on lance son code, pour que « from models.x import … » marche)."""
    d = os.path.dirname(os.path.abspath(path))
    cur = d
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in MARKERS):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return d
        cur = parent


def guess_type(name):
    n = name.lower()
    return "long" if any(k in n for k in ("token", "ids", "idx", "index", "label")) else "float"


def norm_shape(s):
    s = s.strip().strip("()[]")
    for c in "×x* ;":
        s = s.replace(c, ",")
    return ",".join(p for p in s.split(",") if p)


def fmt_params(n):
    if n is None:
        return ""
    if n >= 1e6:
        return f"{n / 1e6:.2f} M".replace(".", ",")
    if n >= 1e3:
        return f"{n / 1e3:.1f} k".replace(".", ",")
    return str(n)


class InputRow(QWidget):
    def __init__(self, dialog, name="", shape="", kind="float"):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("nom")
        self.shape = QLineEdit(shape)
        self.shape.setPlaceholderText("forme, batch compris : 1,3,224,224")
        self.kind = QComboBox()
        self.kind.addItems(TYPES)
        self.kind.setCurrentText(kind if kind in TYPES else "float")
        self.kind.setToolTip("float : valeurs aléatoires ; long / int : indices (tokens) ; bool : masque")
        rm = QToolButton()
        rm.setText("−")
        rm.setToolTip("Retirer cette entrée")
        rm.clicked.connect(lambda: dialog.remove_input(self))
        h.addWidget(self.name, 2)
        h.addWidget(self.shape, 4)
        h.addWidget(self.kind, 1)
        h.addWidget(rm)
        for w in (self.name, self.shape):
            w.textChanged.connect(dialog.refresh_command)
        self.kind.currentIndexChanged.connect(dialog.refresh_command)

    def value(self):
        return [self.name.text().strip(), norm_shape(self.shape.text()), self.kind.currentText()]


class PytorchImportDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.main = window
        self.settings = window.settings
        self.setWindowTitle("Importer un modèle PyTorch")
        # non modale : on regarde le résultat dans l'éditeur, on change un réglage, on réimporte
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.proc = None
        self._checked_py = None
        self._closing = False
        self._check = None          # vérification de l'interpréteur (import torch) en cours
        self.job = None
        self.out_path = None
        self.sigs = {}
        self.analysis = None        # dernier résultat de --classes (liste des classes)
        self.inputs = []
        self._loading = False
        self.mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._build()
        self._restore()
        hint = self.sizeHint()
        self.resize(max(1080, hint.width()), max(720, hint.height()))

    # ------------------------------------------------------------------ interface
    def _build(self):
        v = QVBoxLayout(self)
        intro = QLabel("Exécute le modèle une fois sur des entrées aléatoires et dessine ce qui s'est passé "
                       "(même outil que <code>outils/depuis_pytorch.py</code> en ligne de commande). "
                       "La fenêtre reste ouverte : changer un réglage et réimporter.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:gray;")
        v.addWidget(intro)
        cols = QHBoxLayout()
        v.addLayout(cols, 1)
        left = QVBoxLayout()
        cols.addLayout(left, 5)

        # --- modèle
        g = QGroupBox("Modèle")
        form = QFormLayout(g)
        self.e_python = QLineEdit()
        self.e_python.setToolTip("L'interpréteur de l'environnement qui contient torch et les dépendances du "
                                 "modèle (conda, venv…) ; pas forcément celui de DeepGraph")
        self.e_python.editingFinished.connect(lambda: self.check_python(only_if_changed=True))
        self.l_python = QLabel()
        form.addRow("Python :", self._with_browse(self.e_python, self.browse_python, self.l_python))
        self.e_cwd = QLineEdit()
        self.e_cwd.setToolTip("Dossier d'où l'outil est lancé, comme ton propre code : en général la racine "
                              "du projet (celle qui contient models/…). Rempli seul au choix du fichier")
        self.e_cwd.textChanged.connect(self.refresh_command)
        form.addRow("Racine du projet :", self._with_browse(self.e_cwd, self.browse_cwd))
        self.e_model = QLineEdit()
        self.e_model.setPlaceholderText("models/unet.py   ou   torchvision.models")
        self.e_model.setToolTip("Fichier .py qui définit la classe, ou module importable (paquet.module)")
        self.e_model.editingFinished.connect(self.model_changed)
        form.addRow("Fichier ou module :", self._with_browse(self.e_model, self.browse_model))
        self.c_class = QComboBox()
        self.c_class.setEditable(True)
        self.c_class.setToolTip("Classe du modèle (ou fonction qui le construit, ex. resnet18)")
        self.c_class.currentTextChanged.connect(self.class_changed)
        form.addRow("Classe :", self.c_class)
        self.e_args = QLineEdit()
        self.e_args.setToolTip('Arguments du constructeur en JSON, ex. {"channels": [32, 64], "dropout": 0.1}')
        self.e_args.textChanged.connect(self.refresh_command)
        self.l_init = QLabel()
        self.l_init.setWordWrap(True)
        self.l_init.setStyleSheet("color:gray;")
        form.addRow("Arguments (JSON) :", self.e_args)
        form.addRow("", self.l_init)
        left.addWidget(g)

        # --- entrées
        g = QGroupBox("Entrées du forward (dans l'ordre)")
        gv = QVBoxLayout(g)
        self.inputs_box = QVBoxLayout()
        gv.addLayout(self.inputs_box)
        row = QHBoxLayout()
        add = QPushButton("+ Entrée")
        add.clicked.connect(lambda: self.add_input())
        row.addWidget(add)
        row.addStretch(1)
        gv.addLayout(row)
        left.addWidget(g)

        # --- mise en page
        g = QGroupBox("Schéma")
        grid = QGridLayout(g)
        self.s_depth = QSpinBox()
        self.s_depth.setRange(1, 9)
        self.s_depth.setToolTip("1 = les sous-modules directs du modèle, 2 = un niveau de plus… "
                                "(les ModuleList / Sequential ne comptent pas)")
        self.s_depth.valueChanged.connect(self.refresh_command)
        grid.addWidget(QLabel("Profondeur :"), 0, 0)
        grid.addWidget(self.s_depth, 0, 1)
        self.c_shapes = QComboBox()
        for k, t in SHAPES:
            self.c_shapes.addItem(t, k)
        self.c_shapes.setToolTip("Formes des tenseurs écrites dans chaque bloc")
        self.c_shapes.currentIndexChanged.connect(self.refresh_command)
        grid.addWidget(QLabel("Formes :"), 0, 2)
        grid.addWidget(self.c_shapes, 0, 3)
        self.k_fold = QCheckBox("Regrouper les blocs identiques en « N× »")
        self.k_fold.setChecked(True)
        self.k_fold.toggled.connect(self.refresh_command)
        grid.addWidget(self.k_fold, 1, 0, 1, 4)
        self.c_dest = QComboBox()
        self.c_dest.addItem("un nouveau document", "new")
        self.c_dest.addItem("un nouveau calque du document actuel", "layer")
        grid.addWidget(QLabel("Ouvrir dans :"), 2, 0)
        grid.addWidget(self.c_dest, 2, 1, 1, 3)
        grid.setColumnStretch(4, 1)
        left.addWidget(g)
        left.addStretch(1)

        # --- classes
        g = QGroupBox("Blocs à détailler")
        gv = QVBoxLayout(g)
        row = QHBoxLayout()
        self.b_analyse = QPushButton("Analyser le modèle")
        self.b_analyse.setToolTip("Crée le modèle (et l'exécute si les formes d'entrée sont données) pour "
                                  "lister ses classes de blocs")
        self.b_analyse.clicked.connect(self.analyse)
        row.addWidget(self.b_analyse)
        self.l_analysis = QLabel()
        self.l_analysis.setWordWrap(True)
        row.addWidget(self.l_analysis, 1)
        gv.addLayout(row)
        self.t_classes = QTableWidget(0, 4)
        self.t_classes.setHorizontalHeaderLabels(["Classe", "Nb", "Encart", "Vue d'ensemble"])
        hh = self.t_classes.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.t_classes.verticalHeader().hide()
        self.t_classes.setSelectionMode(QAbstractItemView.NoSelection)
        self.t_classes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t_classes.itemChanged.connect(lambda _: self.refresh_command())
        self.t_classes.horizontalHeaderItem(2).setToolTip(
            "--detail : l'intérieur de la classe est dessiné en encart à droite (un appel typique)")
        self.t_classes.horizontalHeaderItem(3).setToolTip(
            "--deplier : montre l'intérieur directement dans la vue d'ensemble ; "
            "--replier : un seul bloc, même avec une grande profondeur")
        gv.addWidget(self.t_classes, 1)
        hint = QLabel("Encart : le plus proche d'une figure d'article (vue d'ensemble + détail des blocs). "
                      "Les classes en gris n'ont pas été appelées pendant l'exécution.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:gray;")
        gv.addWidget(hint)
        cols.addWidget(g, 4)

        # --- commande et journal
        self.tabs = QTabWidget()
        self.t_cmd = QPlainTextEdit()
        self.t_cmd.setReadOnly(True)
        self.t_cmd.setFont(self.mono)
        self.t_log = QPlainTextEdit()
        self.t_log.setReadOnly(True)
        self.t_log.setFont(self.mono)
        self.tabs.addTab(self.t_cmd, "Commande équivalente")
        self.tabs.addTab(self.t_log, "Journal")
        self.tabs.setMaximumHeight(150)
        v.addWidget(self.tabs)

        row = QHBoxLayout()
        self.l_status = QLabel()
        self.l_status.setWordWrap(True)
        row.addWidget(self.l_status, 1)
        b = QPushButton("Copier la commande")
        b.setToolTip("Pour la relancer dans un terminal, ou l'écrire dans un script / README")
        b.clicked.connect(lambda: QApplication.clipboard().setText(self.command_text()))
        row.addWidget(b)
        b = QPushButton("Aide…")
        b.clicked.connect(self.main.show_pytorch_doc)
        row.addWidget(b)
        self.b_stop = QPushButton("Arrêter")
        self.b_stop.clicked.connect(self.stop)
        self.b_stop.hide()
        row.addWidget(self.b_stop)
        self.b_import = QPushButton("Importer")
        self.b_import.setDefault(True)
        self.b_import.clicked.connect(self.run_import)
        row.addWidget(self.b_import)
        b = QPushButton("Fermer")
        b.clicked.connect(self.close)
        row.addWidget(b)
        v.addLayout(row)

    def _with_browse(self, edit, slot, extra=None):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        if extra is not None:
            h.addWidget(extra)
        b = QToolButton()
        b.setText("…")
        b.clicked.connect(slot)
        h.addWidget(b)
        return w

    # ------------------------------------------------------------------ entrées
    def add_input(self, name=None, shape="", kind=None):
        if name is None:
            name = f"x{len(self.inputs)}" if self.inputs else "x"
        r = InputRow(self, name, shape, kind or guess_type(name))
        self.inputs.append(r)
        self.inputs_box.addWidget(r)
        self.refresh_command()
        return r

    def remove_input(self, r):
        self.inputs.remove(r)
        r.setParent(None)
        r.deleteLater()
        self.refresh_command()

    def set_inputs(self, rows):
        for r in list(self.inputs):
            self.inputs.remove(r)
            r.setParent(None)
            r.deleteLater()
        for name, shape, kind in rows:
            self.add_input(name, shape, kind)

    # ------------------------------------------------------------------ choix de fichiers
    def browse_python(self):
        start = os.path.dirname(self.e_python.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Interpréteur Python (environnement avec torch)", start)
        if path:
            self.e_python.setText(path)
            self.check_python()

    def browse_cwd(self):
        path = QFileDialog.getExistingDirectory(self, "Racine du projet", self.e_cwd.text() or os.path.expanduser("~"))
        if path:
            self.e_cwd.setText(path)

    def browse_model(self):
        cur = self._model_path()
        start = os.path.dirname(cur) if cur else (self.e_cwd.text() or os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Fichier qui définit le modèle", start, "Python (*.py)")
        if path:
            self.e_model.setText(path)
            self.model_changed()

    # ------------------------------------------------------------------ modèle
    def _model_path(self):
        """Chemin absolu du fichier du modèle, ou None si c'est un module importable."""
        t = self.e_model.text().strip()
        if not t or not (t.endswith(".py") or os.sep in t):
            return None
        if not os.path.isabs(t):
            t = os.path.join(self.e_cwd.text().strip() or os.getcwd(), t)
        return os.path.normpath(os.path.expanduser(t))

    def spec(self):
        model, cls = self.e_model.text().strip(), self.c_class.currentText().strip()
        path = self._model_path()
        if path:
            cwd = self.e_cwd.text().strip()
            rel = os.path.relpath(path, cwd) if cwd else path
            model = rel if cwd and not rel.startswith("..") else path
        return f"{model}:{cls}" if model and cls else ""

    def key(self):
        """Clé des réglages mémorisés d'un modèle (chemin absolu : indépendante de la racine)."""
        model, cls = self.e_model.text().strip(), self.c_class.currentText().strip()
        return f"{self._model_path() or model}:{cls}" if model and cls else ""

    def model_changed(self):
        t = self.e_model.text().strip()
        head, sep, tail = t.rpartition(":")
        if sep and tail.isidentifier() and head:          # « models/unet.py:UNet » collé d'un coup
            self.e_model.setText(head)
            self._fill_classes(prefer=tail)
        else:
            self._fill_classes()
        self.refresh_command()

    def _fill_classes(self, prefer=None):
        path = self._model_path()
        if path and os.path.isfile(path):
            # racine du projet devinée depuis le fichier (les réglages mémorisés du modèle priment)
            self.e_model.setText(path)
            self.e_cwd.setText(project_root(path))
        names, best, self.sigs = scan_file(path) if path and os.path.isfile(path) else ([], None, {})
        cur = prefer or self.c_class.currentText().strip()
        self._loading = True
        self.c_class.clear()
        self.c_class.addItems(names)
        self._loading = False
        pick = cur if cur and (cur in names or not names) else best or cur or ""
        self.c_class.setCurrentText(pick)
        self.class_changed(pick)

    def class_changed(self, _=None):
        if self._loading:
            return
        saved = self._saved_models().get(self.key())
        if saved:
            self._apply_model_state(saved)
        else:
            self.analysis = None
            self._fill_table([])
            self.l_analysis.setText("")
            init, fwd = self.sigs.get(self.c_class.currentText().strip(), (None, None))
            self._show_init(init)
            if fwd is not None:
                self._match_inputs([n for n, d in fwd if d is None])
        self.refresh_command()

    def _match_inputs(self, names):
        """Entrées = arguments obligatoires du forward (on garde les formes déjà écrites sous le même nom)."""
        cur = [r.value() for r in self.inputs]
        if not names or [n for n, _, _ in cur] == names:
            return
        old = {n: (sh, k) for n, sh, k in cur}
        self.set_inputs([[n, *old.get(n, ("", guess_type(n)))] for n in names])

    def _show_init(self, init):
        """Signature du constructeur sous le champ des arguments (et modèle de JSON en filigrane)."""
        if not init:
            self.l_init.setText("")
            self.e_args.setPlaceholderText("{}")
            return
        parts = [n if d is None else f"{n}={d}" for n, d in init]
        self.l_init.setText("Constructeur : " + ", ".join(parts))
        req = [n for n, d in init if d is None]
        self.e_args.setPlaceholderText("{" + ", ".join(f'"{n}": …' for n in req) + "}" if req else "{}")

    def _stop_check(self):
        p, self._check = self._check, None
        if p is not None:
            p.blockSignals(True)
            p.kill()
            p.waitForFinished(1000)
            p.deleteLater()

    def check_python(self, only_if_changed=False):
        py = self.e_python.text().strip()
        if only_if_changed and (py == self._checked_py or self._closing):
            return                     # editingFinished arrive aussi à la fermeture (perte du focus)
        self._checked_py = py
        self._stop_check()
        self.l_python.setText("…")
        self.l_python.setStyleSheet("")
        p = self._check = QProcess(self)
        p.finished.connect(lambda code, _st, p=p: self._python_checked(p, code))
        p.errorOccurred.connect(lambda e, p=p: e == QProcess.FailedToStart and self._python_checked(p, -1))
        p.start(py, ["-c", "import torch; print(torch.__version__)"])
        self.refresh_command()

    def _python_checked(self, p, code):
        if p is not self._check:
            return
        self._check = None
        try:
            if code == 0:
                ver = bytes(p.readAllStandardOutput()).decode(errors="replace").strip()
                self.l_python.setText(f"torch {ver}")
                self.l_python.setStyleSheet("color:#2E8B57;")
            else:
                self.l_python.setText("torch introuvable")
                self.l_python.setStyleSheet(f"color:{ERR};")
            p.deleteLater()
        except RuntimeError:                  # fenêtre détruite pendant la vérification
            pass

    # ------------------------------------------------------------------ tableau des classes
    def _fill_table(self, classes, choices=None):
        """classes : liste de --classes ; choices : {classe: {"detail": bool, "view": str}}."""
        if choices is None:
            choices = self._choices()
        self.t_classes.blockSignals(True)
        self.t_classes.setRowCount(0)
        for c in classes:
            r = self.t_classes.rowCount()
            self.t_classes.insertRow(r)
            name = c["name"]
            it = QTableWidgetItem("    " * max(0, c.get("level", 1) - 1) + name)
            it.setData(Qt.UserRole, name)
            tip = f"Exemple : {c.get('example', '')}"
            if c.get("builtin"):
                tip += " (torch.nn)"
            if c.get("called") is False:
                tip += "\nPas appelée pendant l'exécution : ignorée"
                it.setForeground(QColor("gray"))
            it.setToolTip(tip)
            self.t_classes.setItem(r, 0, it)
            n = QTableWidgetItem(str(c.get("count", "")))
            n.setTextAlignment(Qt.AlignCenter)
            self.t_classes.setItem(r, 1, n)
            ch = choices.get(name, {})
            k = QTableWidgetItem()
            k.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            k.setCheckState(Qt.Checked if ch.get("detail") else Qt.Unchecked)
            self.t_classes.setItem(r, 2, k)
            combo = QComboBox()
            for key, text in VIEWS:
                combo.addItem(text, key)
            keys = [key for key, _ in VIEWS]
            combo.setCurrentIndex(keys.index(ch.get("view")) if ch.get("view") in keys else 0)
            combo.currentIndexChanged.connect(self.refresh_command)
            self.t_classes.setCellWidget(r, 3, combo)
        self.t_classes.blockSignals(False)

    def _choices(self):
        out = {}
        for r in range(self.t_classes.rowCount()):
            it = self.t_classes.item(r, 0)
            if it is None:
                continue
            detail = self.t_classes.item(r, 2).checkState() == Qt.Checked
            view = self.t_classes.cellWidget(r, 3).currentData()
            if detail or view:
                out[it.data(Qt.UserRole)] = {"detail": detail, "view": view}
        return out

    # ------------------------------------------------------------------ commande
    def state(self):
        return {"python": self.e_python.text().strip(), "cwd": self.e_cwd.text().strip(),
                "model": self.e_model.text().strip(), "cls": self.c_class.currentText().strip(),
                "args": self.e_args.text().strip(), "inputs": [r.value() for r in self.inputs],
                "depth": self.s_depth.value(), "shapes": self.c_shapes.currentData(),
                "fold": self.k_fold.isChecked(), "dest": self.c_dest.currentData(),
                "classes": self._choices(), "analysis": self.analysis}

    def tool_args(self, out, list_only=False):
        st = self.state()
        a = [TOOL, self.spec()]
        for name, shape, kind in st["inputs"]:
            if shape:
                a += ["-e", f"{name}:{shape}" + ("" if kind == "float" else f":{kind}")]
        if st["args"] and st["args"] != "{}":
            a += ["--args", st["args"]]
        if list_only:
            return a + ["--classes"]
        if st["depth"] != 1:
            a += ["-p", str(st["depth"])]
        for c, ch in st["classes"].items():
            if ch["detail"]:
                a += ["--detail", c]
        for c, ch in st["classes"].items():
            if ch["view"]:
                a += [f"--{ch['view']}", c]
        if st["shapes"] != "entree-sortie":
            a += ["--formes", st["shapes"]]
        if not st["fold"]:
            a.append("--sans-repetitions")
        return a + ["-o", out]

    def command_text(self):
        cwd = self.e_cwd.text().strip()
        cmd = shlex.join([self.e_python.text().strip() or "python"] + self.tool_args("brouillon.dgraph"))
        return f"cd {shlex.quote(cwd)} && {cmd}" if cwd else cmd

    def refresh_command(self, *_):
        if not hasattr(self, "t_cmd"):
            return
        self.t_cmd.setPlainText(self.command_text())

    def problems(self, list_only=False):
        st = self.state()
        if not st["python"]:
            return "Choisir l'interpréteur Python (environnement qui contient torch)."
        if not st["model"] or not st["cls"]:
            return "Choisir le fichier (ou module) et la classe du modèle."
        path = self._model_path()
        if path and not os.path.isfile(path):
            return f"Fichier introuvable : {path}"
        if st["cwd"] and not os.path.isdir(st["cwd"]):
            return f"Dossier introuvable : {st['cwd']}"
        try:
            if not isinstance(json.loads(st["args"] or "{}"), dict):
                raise ValueError
        except ValueError:
            return 'Arguments du constructeur : un objet JSON est attendu, ex. {"dim": 256}.'
        if not list_only and not any(s for _, s, _ in st["inputs"]):
            return "Donner la forme d'au moins une entrée du forward (batch compris, ex. 1,3,224,224)."
        for name, shape, _ in st["inputs"]:
            if shape and not all(p.strip().isdigit() for p in shape.split(",")):
                return f"Forme de « {name} » : des entiers séparés par des virgules (ex. 1,3,224,224)."
        return None

    # ------------------------------------------------------------------ exécution
    def _start(self, job, args):
        self.job = job
        self.t_log.clear()
        self.proc = QProcess(self)
        cwd = self.e_cwd.text().strip()
        if cwd:
            self.proc.setWorkingDirectory(cwd)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardError.connect(self._read_err)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self._stdout = b""
        self._busy(True)
        self._status("Analyse du modèle…" if job == "list" else "Import en cours…")
        self.proc.start(self.e_python.text().strip(), args)

    def _busy(self, on):
        self.b_import.setEnabled(not on)
        self.b_analyse.setEnabled(not on)
        self.b_stop.setVisible(on)
        QApplication.setOverrideCursor(Qt.BusyCursor) if on else QApplication.restoreOverrideCursor()

    def _status(self, text, error=False):
        self.l_status.setText(text)
        self.l_status.setStyleSheet(f"color:{ERR};" if error else "")

    def _read_err(self):
        txt = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        self.t_log.moveCursor(self.t_log.textCursor().End)
        self.t_log.insertPlainText(txt)

    def _proc_error(self, err):
        if err == QProcess.FailedToStart:
            self._done()
            self._status(f"Impossible de lancer « {self.e_python.text().strip()} » : vérifier l'interpréteur "
                         "Python.", True)

    def stop(self):
        if self.proc is not None:
            self.proc.kill()

    def _done(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.deleteLater()
        self._busy(False)

    def _finished(self, code, status):
        proc, job = self.proc, self.job
        if proc is None:
            return
        self._read_err()
        out = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._done()
        if status != QProcess.NormalExit:
            self._status("Arrêté.", True)
            self._cleanup()
            return
        if job == "list":
            self._listed(code, out)
        else:
            self._imported(code)

    def _last_error(self):
        lines = [ln for ln in self.t_log.toPlainText().strip().splitlines() if ln.strip()]
        return lines[-1] if lines else "erreur inconnue"

    def analyse(self):
        err = self.problems(list_only=True)
        if err:
            self._status(err, True)
            return
        self._start("list", self.tool_args("", list_only=True))

    def _listed(self, code, out):
        res = None
        for line in reversed(out.strip().splitlines()):        # le modèle peut aussi écrire (print…)
            if line.startswith("{"):
                try:
                    res = json.loads(line)
                    break
                except ValueError:
                    pass
        if code != 0 or res is None:
            self.tabs.setCurrentWidget(self.t_log)
            self._status(f"Analyse impossible : {self._last_error()}", True)
            return
        if res.get("init") is not None:
            self._show_init([(p["name"], p["default"]) for p in res["init"]])
        if res.get("erreur"):
            self._status(f"Création du modèle impossible : {res['erreur']}", True)
            return
        fwd = res.get("forward")
        if fwd is not None:
            self._match_inputs([p["name"] for p in fwd if p["default"] is None])
        self.analysis = res["classes"]
        self._fill_table(self.analysis)
        n = len(self.analysis)
        self.l_analysis.setText(f"{fmt_params(res.get('params'))} paramètres · {n} classe(s) de blocs"
                                if n else f"{fmt_params(res.get('params'))} paramètres · aucun bloc composé")
        if res.get("erreur_forward"):
            self._status(f"Le modèle est créé, mais son exécution échoue : {res['erreur_forward']} "
                         "(vérifier l'ordre et la forme des entrées)", True)
        elif not any(r.value()[1] for r in self.inputs):
            self._status("Modèle analysé. Donner la forme des entrées, puis Importer.")
        else:
            self._status("Modèle analysé : il s'exécute sur ces entrées. Cocher les blocs à détailler, puis "
                         "Importer.")
        self._remember()

    def run_import(self):
        err = self.problems()
        if err:
            self._status(err, True)
            return
        fd, self.out_path = tempfile.mkstemp(suffix=".dgraph", prefix="deepgraph_import_")
        os.close(fd)
        self._remember()
        self._start("import", self.tool_args(self.out_path))

    def _imported(self, code):
        path = self.out_path
        if code != 0 or not path or not os.path.getsize(path):
            self.tabs.setCurrentWidget(self.t_log)
            self._status(f"Import impossible : {self._last_error()}", True)
            self._cleanup()
            return
        title = self.c_class.currentText().strip()
        try:
            ok = self.main.use_model(path, title, self.c_dest.currentData())
        finally:
            self._cleanup()
        if ok:
            log = self.t_log.toPlainText().strip().splitlines()
            warn = [ln for ln in log if ln.startswith("[")]
            msg = log[-1].split(" : ", 1)[-1] if log else ""
            self._status(f"Importé ({msg})." + (f" {len(warn)} avertissement(s), voir le journal." if warn else ""),
                         bool(warn))
            if warn:
                self.tabs.setCurrentWidget(self.t_log)

    def _cleanup(self):
        if self.out_path and os.path.exists(self.out_path):
            try:
                os.remove(self.out_path)
            except OSError:
                pass
        self.out_path = None

    # ------------------------------------------------------------------ mémoire des réglages
    def _saved_models(self):
        try:
            return json.loads(self.settings.value(KEY_MODELS, "{}") or "{}")
        except ValueError:
            return {}

    def _remember(self):
        st = self.state()
        self.settings.setValue(KEY_LAST, json.dumps(st, ensure_ascii=False))
        if st["python"]:
            self.settings.setValue("pytorch/python", st["python"])     # partagé avec Vérifier le schéma
        key = self.key()
        if key:
            models = self._saved_models()
            models[key] = {k: st[k] for k in ("cwd", "args", "inputs", "depth", "shapes", "fold", "classes", "analysis")}
            self.settings.setValue(KEY_MODELS, json.dumps(models, ensure_ascii=False))

    def _apply_model_state(self, st):
        if st.get("cwd") and self._model_path():
            self.e_cwd.setText(st["cwd"])
        self.e_args.setText(st.get("args", ""))
        self.set_inputs(st.get("inputs") or [["x", "", "float"]])
        self.s_depth.setValue(int(st.get("depth", 1)))
        i = self.c_shapes.findData(st.get("shapes", "entree-sortie"))
        self.c_shapes.setCurrentIndex(max(0, i))
        self.k_fold.setChecked(st.get("fold", True))
        self.analysis = st.get("analysis")
        self._fill_table(self.analysis or [], st.get("classes", {}))
        init, _ = self.sigs.get(self.c_class.currentText().strip(), (None, None))
        self._show_init(init)
        self.l_analysis.setText("")

    def _restore(self):
        try:
            st = json.loads(self.settings.value(KEY_LAST, "") or "{}")
        except ValueError:
            st = {}
        self.e_python.setText(st.get("python") or self.settings.value("pytorch/python", "") or sys.executable)
        self.e_cwd.setText(st.get("cwd", ""))
        self.e_model.setText(st.get("model", ""))
        i = self.c_dest.findData(st.get("dest", "new"))
        self.c_dest.setCurrentIndex(max(0, i))
        if st.get("model"):
            self._fill_classes(prefer=st.get("cls"))
        if not self.inputs:
            self.add_input("x")
        self.refresh_command()
        QTimer.singleShot(0, self.check_python)

    def showEvent(self, e):
        self._closing = False
        super().showEvent(e)

    def closeEvent(self, e):
        self._closing = True
        self._stop_check()
        if self.proc is not None:
            self.proc.kill()
        self._remember()
        super().closeEvent(e)
