"""Fenêtre « Vérifier le schéma » : formes des tenseurs, incohérences, tailles écrites sur le schéma.

Le calcul (deepgraph/shapecheck.py) exécute le code PyTorch du schéma dans un processus à part,
avec l'interpréteur Python choisi (celui qui a torch, comme pour l'import d'un modèle).
"""
import os
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from . import shapecheck
from .config import CFG

KEY_PYTHON = "pytorch/python"
COLORS = {"erreur": "#C0392B", "attention": "#C77700", "hypothèse": "#808080"}
TITLES = {"erreur": "Erreurs", "attention": "Incohérences", "hypothèse": "Hypothèses (tailles supposées)"}


def python_setting(settings):
    """Interpréteur Python qui a torch : celui choisi (import ou vérification), sinon celui de DeepGraph."""
    return settings.value(KEY_PYTHON, "") or sys.executable


class CheckDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.main = window
        self.scene = window.scene
        self.settings = window.settings
        self.setWindowTitle("Vérifier le schéma")
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.proc = None
        self.report = None
        self.gen = None
        self.paths = None
        self._writing = False
        self._build()
        self.resize(620, 520)
        self.scene.modified.connect(self._scene_changed)

    def _build(self):
        v = QVBoxLayout(self)
        intro = QLabel("Le schéma est traduit en code PyTorch (comme « Exporter en code PyTorch »), exécuté une "
                       "fois sur des entrées factices, et chaque bloc note la forme de sa sortie.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:gray;")
        v.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Python :"))
        self.e_python = QLineEdit(python_setting(self.settings))
        self.e_python.setToolTip("Interpréteur d'un environnement qui contient torch (le même que pour "
                                 "Fichier > Importer un modèle PyTorch…)")
        self.e_python.editingFinished.connect(
            lambda: self.settings.setValue(KEY_PYTHON, self.e_python.text().strip()))
        row.addWidget(self.e_python, 1)
        b = QToolButton()
        b.setText("…")
        b.clicked.connect(self._browse_python)
        row.addWidget(b)
        v.addLayout(row)

        self.l_status = QLabel()
        self.l_status.setWordWrap(True)
        v.addWidget(self.l_status)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setWordWrap(True)
        self.tree.itemClicked.connect(self._show_item)
        v.addWidget(self.tree, 1)
        hint = QLabel("Cliquer sur une ligne sélectionne le bloc dans le schéma. Les formes ne comptent pas la "
                      "dimension du batch ; C×H×W, C/4… : bloc détaillé appelé avec plusieurs tailles.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:gray;")
        v.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Écrire les tailles"))
        self.c_mode = QComboBox()
        self.c_mode.addItem("le long des flèches", "arrows")
        self.c_mode.addItem("dans les blocs (dernière ligne)", "blocks")
        row.addWidget(self.c_mode)
        self.b_write = QPushButton("Écrire")
        self.b_write.setToolTip("Ajoute les formes calculées au schéma (Ctrl+Z pour annuler). Les tailles déjà "
                                "écrites par l'auteur à côté d'un bloc sont laissées telles quelles")
        self.b_write.clicked.connect(self.write)
        row.addWidget(self.b_write)
        self.b_remove = QPushButton("Retirer")
        self.b_remove.setToolTip("Retire les tailles écrites le long des flèches par une vérification")
        self.b_remove.clicked.connect(self.remove)
        row.addWidget(self.b_remove)
        row.addStretch(1)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addStretch(1)
        self.b_run = QPushButton("Vérifier")
        self.b_run.setDefault(True)
        self.b_run.clicked.connect(self.run)
        row.addWidget(self.b_run)
        b = QPushButton("Fermer")
        b.clicked.connect(self.close)
        row.addWidget(b)
        v.addLayout(row)
        self._sync_buttons()

    def _browse_python(self):
        start = os.path.dirname(self.e_python.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Interpréteur Python (environnement avec torch)", start)
        if path:
            self.e_python.setText(path)
            self.settings.setValue(KEY_PYTHON, path)

    def _sync_buttons(self):
        busy = self.proc is not None
        self.b_run.setEnabled(not busy)
        self.b_write.setEnabled(not busy and bool(self.report and self.report["shapes"]))

    def _status(self, text, color=None):
        self.l_status.setText(text)
        self.l_status.setStyleSheet(f"color:{color};" if color else "")

    # ------------------------------------------------------------------ calcul
    def run(self):
        if self.proc is not None:
            return
        self.scene.checkpoint()
        data = self.scene.to_dict()
        title = os.path.splitext(os.path.basename(self.main.current_file))[0] if self.main.current_file else ""
        defs = {bid: bd.pytorch for bid, bd in CFG.blocks.items() if bd.pytorch}
        try:
            code, self.gen = shapecheck.probe_code(data, defs, title)
        except Exception as e:  # noqa: BLE001
            self._status(f"Traduction en code impossible : {type(e).__name__} : {e}", COLORS["erreur"])
            return
        self.paths = shapecheck.write_temp(code)
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)
        self._status("Vérification…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._sync_buttons()
        self.proc.start(self.e_python.text().strip() or sys.executable, shapecheck.runner_args(*self.paths))

    def _error(self, err):
        if err == QProcess.FailedToStart:
            self._done()
            self._status(f"Impossible de lancer « {self.e_python.text().strip()} » : choisir l'interpréteur "
                         "Python d'un environnement qui contient torch.", COLORS["erreur"])

    def _done(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.deleteLater()
        QApplication.restoreOverrideCursor()
        stderr = ""
        if proc is not None:
            stderr = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        res = shapecheck.read_result(self.paths[1], stderr) if self.paths else None
        for f in self.paths or ():
            try:
                os.remove(f)
            except OSError:
                pass
        self.paths = None
        self._sync_buttons()
        return res

    def _finished(self, *_):
        if self.proc is None:
            return
        res = self._done()
        rep = shapecheck.analyse(self.gen, res)
        self.report = rep
        self._sync_buttons()
        if rep["no_torch"]:
            self._status(f"torch est introuvable pour {self.e_python.text().strip()} : choisir l'interpréteur "
                         "Python d'un environnement qui contient torch (« … »).", COLORS["erreur"])
            self.tree.clear()
            return
        self._fill(rep)

    def _fill(self, rep):
        self.tree.clear()
        n_err = sum(1 for i in rep["issues"] if i["level"] == "erreur")
        n_att = sum(1 for i in rep["issues"] if i["level"] == "attention")
        parts = [f"{len(rep['shapes'])} forme(s) calculée(s)"]
        if rep.get("params"):
            parts.append(f"{rep['params']:,} paramètres".replace(",", " "))
        if n_err or n_att:
            parts.append(f"{n_err + n_att} problème(s)")
            color = COLORS["erreur"] if n_err else COLORS["attention"]
        else:
            parts.append("aucune incohérence")
            color = "#2E8B57"
        self._status(" · ".join(parts), color)
        for lv in shapecheck.LEVELS:
            items = [i for i in rep["issues"] if i["level"] == lv]
            if not items:
                continue
            top = QTreeWidgetItem([f"{TITLES[lv]} ({len(items)})"])
            top.setForeground(0, QBrush(QColor(COLORS[lv])))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for i in items:
                it = QTreeWidgetItem([i["text"]])
                it.setToolTip(0, i["text"])
                it.setData(0, Qt.UserRole, (i.get("uid"), i.get("arrow")))
                top.addChild(it)
            top.setExpanded(lv != "hypothèse" or len(items) <= 6)
        if rep["missing"]:
            it = QTreeWidgetItem([f"{len(rep['missing'])} bloc(s) sans forme calculée"
                                  + (" (après l'erreur)" if n_err else "")])
            it.setForeground(0, QBrush(QColor("gray")))
            self.tree.addTopLevelItem(it)

    # ------------------------------------------------------------------ schéma
    def _show_item(self, item, _col=0):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        uid, arrow = data
        target = None
        if arrow is not None:
            target = next((a for a in self.scene.arrows() if str(a.uid) == str(arrow)), None)
        elif uid is not None:
            target = next((n for n in self.scene.nodes() if str(n.uid) == str(uid)), None)
        if target is None:
            return
        self.scene.clearSelection()
        target.setSelected(True)
        self.main.view.centerOn(target.sceneBoundingRect().center())

    def write(self):
        if not self.report:
            return
        self._writing = True
        try:
            n = shapecheck.write_labels(self.scene, self.report, self.c_mode.currentData())
            self.scene.checkpoint()
        finally:
            self._writing = False
        self.main.statusBar().showMessage(f"{n} taille(s) écrite(s) — Ctrl+Z pour annuler", 5000)

    def remove(self):
        self._writing = True
        try:
            n = shapecheck.remove_labels(self.scene)
            self.scene.checkpoint()
        finally:
            self._writing = False
        self.main.statusBar().showMessage(f"{n} taille(s) retirée(s)", 4000)

    def _scene_changed(self):
        if self._writing or self.report is None or not self.isVisible():
            return
        if "a changé" not in self.l_status.text():
            self._status(self.l_status.text() + " — le schéma a changé : « Vérifier » pour mettre à jour", "gray")

    def closeEvent(self, e):
        if self.proc is not None:
            self.proc.kill()
        super().closeEvent(e)
