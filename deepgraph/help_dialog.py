"""Fenêtre d'aide : affiche un texte Markdown, recherche, copie et enregistrement."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QTextDocument
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QShortcut, QTextBrowser, QVBoxLayout)


class DocDialog(QDialog):
    def __init__(self, title, markdown, parent=None, hint="", save_name="documentation.md"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.markdown = markdown
        self.save_name = save_name
        self.resize(900, 760)
        v = QVBoxLayout(self)
        if hint:
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet("color:gray;")
            v.addWidget(h)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher dans la documentation… (Entrée = suivant, Maj+Entrée = précédent)")
        self.search.returnPressed.connect(lambda: self.find(backward=bool(
            QApplication.keyboardModifiers() & Qt.ShiftModifier)))
        row.addWidget(self.search, 1)
        v.addLayout(row)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(False)              # liens relatifs (fichiers du projet) : ignorés
        self.view.anchorClicked.connect(self._link)
        self.view.setMarkdown(markdown)
        self.view.document().setDefaultStyleSheet("code { background:#EEEEEE; }")
        v.addWidget(self.view, 1)
        btns = QHBoxLayout()
        copy = QPushButton("Copier tout (Markdown)")
        copy.setToolTip("Copie le texte complet dans le presse-papiers, prêt à coller dans une conversation")
        copy.clicked.connect(self.copy_all)
        save = QPushButton("Enregistrer sous…")
        save.clicked.connect(self.save_as)
        close = QPushButton("Fermer")
        close.clicked.connect(self.close)
        self.status = QLabel()
        self.status.setStyleSheet("color:#2E8B57;")
        btns.addWidget(copy)
        btns.addWidget(save)
        btns.addWidget(self.status, 1)
        btns.addWidget(close)
        v.addLayout(btns)
        QShortcut(QKeySequence.Find, self, activated=lambda: (self.search.setFocus(), self.search.selectAll()))

    def _link(self, url):
        if url.scheme() in ("http", "https"):
            from PyQt5.QtGui import QDesktopServices
            QDesktopServices.openUrl(url)

    def find(self, backward=False):
        t = self.search.text()
        if not t:
            return
        flags = QTextDocument.FindBackward if backward else QTextDocument.FindFlags()
        if not self.view.find(t, flags):            # reprise au début / à la fin
            cur = self.view.textCursor()
            cur.movePosition(cur.End if backward else cur.Start)
            self.view.setTextCursor(cur)
            if not self.view.find(t, flags):
                self.status.setText("Introuvable")
                return
        self.status.setText("")

    def copy_all(self):
        QApplication.clipboard().setText(self.markdown)
        self.status.setText(f"Copié ({len(self.markdown)} caractères)")

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la documentation", self.save_name,
                                              "Markdown (*.md);;Texte (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.markdown)
        except OSError as e:
            QMessageBox.critical(self, "Enregistrement impossible", str(e))
            return
        self.status.setText("Enregistré")
