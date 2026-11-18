"""Panneau de droite : propriétés de la sélection."""
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
                             QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                             QScrollArea, QSpinBox,
                             QVBoxLayout, QWidget)

from .config import CFG, description_html
from .library import BUILTINS

HELP = """<b>Raccourcis</b><br>
• Glisser un bloc depuis la bibliothèque (ou double-clic)<br>
• Double-clic sur un bloc : modifier le texte<br>
• <b>A</b> : mode flèche — glisser d'un bloc à un autre<br>
&nbsp;&nbsp;clic sur une flèche existante = embranchement<br>
• <b>V</b> / Échap : mode sélection<br>
• <b>1 / 2 / 3</b> : flèche droite / 1 angle / 2 angles<br>
• Flèche sélectionnée : tirer le carré bleu pour placer le coude<br>
• Ctrl+G / Ctrl+Maj+G : grouper / dégrouper<br>
• Un groupe se saisit par son bord ou son étiquette<br>
• Lignes roses : guides d'alignement pendant un déplacement<br>
• Maj : désactive le magnétisme (grille et guides)<br>
• Ctrl+molette : zoom · clic molette : déplacer la vue<br>
• Origine : recentrer la vue · Ctrl+1 : tout afficher<br>
• Flèches du clavier : déplacer (Maj = 1 px)"""


class PropertiesPanel(QWidget):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(6, 6, 6, 6)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:bold; font-size:13px; padding-bottom:4px;")
        self.v.addWidget(self.title)
        # description du bloc (tirée de la configuration)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setTextFormat(Qt.RichText)
        self.desc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.desc.setStyleSheet("padding:4px;")
        self.desc_box = QScrollArea()
        self.desc_box.setWidget(self.desc)
        self.desc_box.setWidgetResizable(True)
        self.desc_box.setFrameShape(QFrame.StyledPanel)
        self.desc_box.setMaximumHeight(230)
        self.desc_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.desc_box.hide()
        self.v.addWidget(self.desc_box)
        self.host = QWidget()
        self.v.addWidget(self.host)
        self.v.addStretch(1)
        self.help = QLabel(HELP)
        self.help.setWordWrap(True)
        self.help.setStyleSheet("color:gray; font-size:11px;")
        self.v.addWidget(self.help)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.scene.checkpoint)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self.rebuild)
        scene.selectionChanged.connect(self._rebuild_timer.start)
        scene.itemsEdited.connect(self._rebuild_timer.start)
        scene.settingsChanged.connect(self._rebuild_timer.start)
        self.rebuild()

    def rebuild(self):
        try:
            sel = self.scene.selectedItems()
        except RuntimeError:
            return
        new = QWidget()
        self.v.replaceWidget(self.host, new)
        self.host.hide()
        self.host.deleteLater()
        self.host = new
        form = QFormLayout(new)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        items = [i for i in sel if getattr(i, "PROPS", None)]
        self.help.setVisible(not items)          # l'aide n'est utile que sans sélection
        if not items:
            self.title.setText("Aucune sélection")
            self.desc_box.hide()
            return
        first = items[0]
        self._show_description(first)
        same = [i for i in items if type(i) is type(first)]
        title = first.label_fr + (f" (×{len(same)})" if len(same) > 1 else "")
        if len(items) > len(same):
            title += "  — autres types ignorés"
        self.title.setText(title)
        for spec in first.PROPS:
            name, label, typ = spec[:3]
            opt = spec[3] if len(spec) > 3 else None
            form.addRow(label, self._editor(same, name, typ, opt, getattr(first, name)))

    def _show_description(self, item):
        name, text = self._description_of(item)
        if text:
            self.desc.setText(description_html(text, name))
            w = max(150, self.desc_box.viewport().width() or self.width() - 20)
            self.desc_box.setMinimumHeight(min(230, self.desc.heightForWidth(w) + 6))
            self.desc_box.show()
            self.desc_box.verticalScrollBar().setValue(0)
        else:
            self.desc_box.hide()

    @staticmethod
    def _description_of(item):
        bd = CFG.blocks.get(getattr(item, "def_id", None) or "")
        if bd is not None:
            return bd.name, bd.description
        kind = getattr(item, "kind", "")
        builtin = {"text": "__text__", "junction": "__junction__", "group": "__group__"}.get(kind)
        for bid, name, desc in BUILTINS:
            if bid == builtin:
                return name, desc
        return None, ""

    def _apply(self, items, name, value):
        for it in items:
            try:
                it.set_prop(name, value)
            except RuntimeError:
                pass
        self._timer.start()

    def _editor(self, items, name, typ, opt, value):
        apply = lambda v: self._apply(items, name, v)
        if typ == "text":
            w = QPlainTextEdit()
            w.setPlainText(str(value))
            w.setFixedHeight(56)
            w.textChanged.connect(lambda: apply(w.toPlainText()))
            return w
        if typ == "line":
            w = QLineEdit(str(value))
            w.textChanged.connect(apply)
            return w
        if typ == "color":
            w = QPushButton()
            cur = [str(value)]

            def paint_btn():
                c = QColor(cur[0])
                fg = "#000" if c.lightness() > 120 else "#FFF"
                w.setText(cur[0].upper())
                w.setStyleSheet(f"background:{cur[0]}; color:{fg}; border:1px solid #777; padding:3px;")

            def pick():
                c = QColorDialog.getColor(QColor(cur[0]), self, "Choisir une couleur")
                if c.isValid():
                    cur[0] = c.name().upper()
                    paint_btn()
                    apply(cur[0])

            paint_btn()
            w.clicked.connect(pick)
            return w
        if typ == "int":
            w = QSpinBox()
            lo, hi = opt or (0, 999)
            w.setRange(lo, hi)
            w.setValue(int(value))
            w.valueChanged.connect(apply)
            return w
        if typ == "float":
            w = QDoubleSpinBox()
            lo, hi, step = opt or (0, 9999, 1)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(1)
            w.setValue(float(value))
            w.valueChanged.connect(apply)
            return w
        if typ == "width_opt":
            # épaisseur individuelle ; None = valeur globale du diagramme (opt = clé globale)
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            sb = QDoubleSpinBox()
            sb.setRange(0.1, 12)
            sb.setSingleStep(0.1)
            sb.setDecimals(2)
            btn = QPushButton("Défaut")
            btn.setToolTip("Revenir à l'épaisseur globale du diagramme (barre d'outils)")
            glob = float(getattr(self.scene, opt))

            def show(is_default):
                sb.setSuffix("  (défaut)" if is_default else "")
                btn.setEnabled(not is_default)

            sb.setValue(glob if value is None else float(value))
            show(value is None)

            def changed(v):
                show(False)
                apply(v)

            def reset():
                apply(None)
                sb.blockSignals(True)
                sb.setValue(float(getattr(self.scene, opt)))
                sb.blockSignals(False)
                show(True)

            sb.valueChanged.connect(changed)
            btn.clicked.connect(reset)
            lay.addWidget(sb, 1)
            lay.addWidget(btn)
            return box
        if typ == "margin_opt":
            # marge d'un côté ; None = valeur « Marges » commune aux quatre côtés
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            sb = QDoubleSpinBox()
            sb.setRange(-1, 1000)
            sb.setDecimals(1)
            sb.setSingleStep(1)
            sb.setSpecialValueText("= Marges")      # affiché pour -1, c.-à-d. None
            sb.setValue(-1 if value is None else float(value))
            btn = QPushButton("Défaut")
            btn.setToolTip("Revenir à la valeur « Marges » (identique sur les quatre côtés)")
            btn.setEnabled(value is not None)

            def changed(v):
                btn.setEnabled(v >= 0)
                apply(None if v < 0 else v)

            sb.valueChanged.connect(changed)
            btn.clicked.connect(lambda: sb.setValue(-1))
            lay.addWidget(sb, 1)
            lay.addWidget(btn)
            return box
        if typ == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(apply)
            return w
        if typ == "choice":
            w = QComboBox()
            for val, lab in opt:
                w.addItem(lab, val)
            i = w.findData(value)
            w.setCurrentIndex(max(0, i))
            w.currentIndexChanged.connect(lambda i: apply(w.itemData(i)))
            return w
        if typ == "choice_edit":
            w = QComboBox()
            w.setEditable(True)
            labels = {lab: val for val, lab in opt}
            for val, lab in opt:
                w.addItem(lab, val)
            i = w.findData(value)
            if i >= 0:
                w.setCurrentIndex(i)
            else:
                w.setEditText(str(value))
            w.currentTextChanged.connect(lambda t: apply(labels.get(t, t.strip()) or "plus"))
            return w
        return QLabel(str(value))
