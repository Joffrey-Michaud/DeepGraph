"""Mise à jour, côté fenêtre : vérification en tâche de fond au démarrage, dialogues, redémarrage."""
import os
import sys
import threading

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

from . import __version__, updater


class _Bridge(QObject):
    done = pyqtSignal(object)


_bridges = []          # garde les ponts en vie pendant la vérification


def check_in_background(win, manual=False):
    """Lance la vérification sans bloquer l'interface ; la réponse arrive dans le fil principal."""
    if not manual and updater.disabled():
        return
    bridge = _Bridge()
    _bridges.append(bridge)

    def finished(res):
        _bridges.remove(bridge)
        handle(win, res, manual)

    bridge.done.connect(finished, Qt.QueuedConnection)
    if manual:
        win.statusBar().showMessage("Recherche de mises à jour…")
    threading.Thread(target=lambda: bridge.done.emit(updater.check()), daemon=True).start()


def handle(win, res, manual=False):
    sb = win.statusBar()
    if res.status == "available":
        ask_install(win, res.release)
    elif res.status in ("noauth", "denied"):
        box = QMessageBox(QMessageBox.Warning, "Mises à jour non vérifiées",
                          "Impossible de vérifier si une nouvelle version de DeepGraph existe.", parent=win)
        box.setInformativeText(res.message)
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.exec_()
    elif res.status in ("offline", "error"):
        if manual:
            QMessageBox.information(win, "Mises à jour", f"Vérification impossible :\n{res.message}")
        else:
            sb.showMessage(f"Mises à jour non vérifiées : {res.message}", 8000)
    else:
        if manual:
            QMessageBox.information(win, "Mises à jour", f"DeepGraph est à jour (version {__version__}).")
        else:
            sb.showMessage(f"DeepGraph {__version__} — à jour", 4000)


def ask_install(win, release):
    mode = updater.install_mode()
    how = ("clone git : avance de la branche main jusqu'à la release" if mode == "git"
           else "téléchargement de la release et remplacement des fichiers "
                "(un blocks.yaml modifié est conservé)")
    box = QMessageBox(QMessageBox.Question, "Mise à jour disponible",
                      f"DeepGraph {release.label} est disponible (version actuelle : {__version__}).\n"
                      "Installer maintenant ?", parent=win)
    box.setInformativeText(f"Installation : {how}.")
    if release.notes.strip():
        box.setDetailedText(release.notes.strip())
    yes = box.addButton("Mettre à jour", QMessageBox.AcceptRole)
    box.addButton("Plus tard", QMessageBox.RejectRole)
    box.setDefaultButton(yes)
    box.exec_()
    if box.clickedButton() is not yes:
        win.statusBar().showMessage(f"Mise à jour {release.label} reportée (proposée au prochain démarrage)", 6000)
        return
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        ok, msg = updater.install(release, log=lambda s: (win.statusBar().showMessage(s),
                                                          QApplication.processEvents()))
    finally:
        QApplication.restoreOverrideCursor()
    if not ok:
        QMessageBox.warning(win, "Mise à jour non installée", msg)
        return
    box = QMessageBox(QMessageBox.Information, "Mise à jour installée", msg, parent=win)
    box.setInformativeText("Redémarrer DeepGraph maintenant pour utiliser la nouvelle version ?")
    restart = box.addButton("Redémarrer", QMessageBox.AcceptRole)
    box.addButton("Plus tard", QMessageBox.RejectRole)
    box.setDefaultButton(restart)
    box.exec_()
    if box.clickedButton() is restart and win.maybe_save():
        args = [sys.executable] + sys.argv
        if win.current_file and win.current_file not in sys.argv:
            args = [sys.executable, sys.argv[0], win.current_file]
        win.saved_state = win.scene.current_state()       # déjà enregistré : pas de seconde question
        os.execv(sys.executable, args)
