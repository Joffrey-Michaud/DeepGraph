"""Fenêtre « Nouveau bloc / Modifier le bloc » : remplit un formulaire au lieu d'éditer le YAML."""
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QFrame, QGraphicsScene, QGraphicsView, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
                             QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .config import CFG
from .items import SYMBOLS, BlockItem, OperatorItem

# couleurs proposées d'office (pastels lisibles en noir), complétées par celles de la bibliothèque
BASE_COLORS = ["#FFFFFF", "#F0F0F0", "#F9D9DB", "#FCE5CD", "#FFF2CC", "#D9EAD3",
               "#D0E0E3", "#CFE2F3", "#D9D2E9", "#EAD1DC", "#E6B8AF", "#B6D7A8"]


def _btn_style(color):
    return (f"QToolButton {{ background:{color}; border:1px solid #777; border-radius:3px; }}"
            "QToolButton:checked { border:2px solid #1E78E6; }")


class ColorPicker(QWidget):
    """Grille de pastilles + « Autre couleur… »."""

    def __init__(self, color, extra=(), parent=None):
        super().__init__(parent)
        self.color = QColor(color).name().upper()
        colors = []
        for c in list(BASE_COLORS) + list(extra):
            c = QColor(c).name().upper()
            if c not in colors:
                colors.append(c)
        colors = colors[:24]
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(3)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.swatches = {}
        per_row = 12
        for i, c in enumerate(colors):
            b = QToolButton()
            b.setCheckable(True)
            b.setFixedSize(QSize(20, 20))
            b.setStyleSheet(_btn_style(c))
            b.setToolTip(c)
            b.clicked.connect(lambda _=False, c=c: self.set_color(c))
            self.group.addButton(b)
            self.swatches[c] = b
            g.addWidget(b, i // per_row, i % per_row)
        row = (len(colors) - 1) // per_row + 1
        h = QHBoxLayout()
        self.current = QLabel()
        self.current.setFixedSize(QSize(44, 20))
        other = QPushButton("Autre couleur…")
        other.clicked.connect(self._choose)
        h.addWidget(QLabel("Choisie :"))
        h.addWidget(self.current)
        h.addWidget(other)
        h.addStretch(1)
        g.addLayout(h, row, 0, 1, per_row)
        self.changed = None
        self.set_color(self.color)

    def _choose(self):
        c = QColorDialog.getColor(QColor(self.color), self, "Couleur du bloc")
        if c.isValid():
            self.set_color(c.name().upper())

    def set_color(self, c):
        self.color = QColor(c).name().upper()
        self.current.setStyleSheet(f"background:{self.color}; border:1px solid #777; border-radius:3px;")
        self.current.setToolTip(self.color)
        b = self.swatches.get(self.color)
        if b is not None:
            b.setChecked(True)
        else:                                   # couleur hors palette : aucune pastille cochée
            self.group.setExclusive(False)
            for x in self.swatches.values():
                x.setChecked(False)
            self.group.setExclusive(True)
        if self.changed:
            self.changed()


class BlockDialog(QDialog):
    """values : dict au format de blocks.yaml (name, label, color, category, shape…).
    mode : "new" (ajout) ou "edit" (bloc existant, son id est gardé)."""

    def __init__(self, parent=None, values=None, mode="new", origin="user"):
        super().__init__(parent)
        v = dict(values or {})
        self.mode, self.origin = mode, origin
        self.edit_id = v.get("id") if mode == "edit" else None
        self.setWindowTitle("Modifier le bloc" if mode == "edit" else "Nouveau bloc")
        self.setMinimumWidth(620)
        self.resize(660, 0)
        st = CFG.style

        root = QVBoxLayout(self)
        if mode == "edit" and origin in ("base", "override"):
            note = QLabel("Bloc fourni avec DeepGraph : tes changements sont enregistrés à part. "
                          "Clic droit > « Rétablir l'original » pour les annuler.")
            note.setWordWrap(True)
            note.setStyleSheet("color: palette(mid);")
            root.addWidget(note)

        # --- formulaire -----------------------------------------------------------------
        form_box = QWidget()
        form = QFormLayout(form_box)
        form.setContentsMargins(0, 0, 0, 0)
        self.form = form
        root.addWidget(form_box)

        self.name = QLineEdit(v.get("name", ""))
        self.name.setPlaceholderText("ex. Conv 3×3, Attention, LSTM…")
        form.addRow("Nom *", self.name)

        self.label = QLineEdit(v.get("label", "") if v.get("label") != v.get("name") else "")
        self.label.setPlaceholderText("identique au nom")
        self.label.setToolTip("Texte écrit dans le bloc. Formules entre $…$, ex. $W^Q$ ; "
                              "\\n force un retour à la ligne.")
        form.addRow("Texte dans le bloc", self.label)

        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.addItems(CFG.categories or ["Divers"])
        self.category.setCurrentText(v.get("category") or "Mes blocs")
        self.category.lineEdit().setPlaceholderText("choisir ou taper une nouvelle catégorie")
        self.category.setToolTip("Dossier de la bibliothèque. Taper un nouveau nom crée la catégorie.")
        form.addRow("Catégorie", self.category)

        shape_row = QHBoxLayout()
        self.r_rect = QRadioButton("Rectangle (couche)")
        self.r_circle = QRadioButton("Cercle (opération ⊕, ⊗…)")
        (self.r_circle if v.get("shape") == "circle" else self.r_rect).setChecked(True)
        shape_row.addWidget(self.r_rect)
        shape_row.addWidget(self.r_circle)
        shape_row.addStretch(1)
        form.addRow("Forme", shape_row)

        self.symbol = QComboBox()
        self.symbol.setEditable(True)
        for key, lab in SYMBOLS:
            self.symbol.addItem(lab, key)
        sym = v.get("symbol", "plus")
        i = self.symbol.findData(sym)
        if i >= 0:
            self.symbol.setCurrentIndex(i)
        else:
            self.symbol.setEditText(str(sym))
        self.symbol.setToolTip("Un symbole de la liste, ou un texte court (ex. σ, C)")
        form.addRow("Symbole", self.symbol)

        extra = [bd.color for bd in CFG.blocks.values()]
        self.color = ColorPicker(v.get("color", "#FFFFFF"), extra)
        form.addRow("Couleur", self.color)

        txt_row = QHBoxLayout()
        self.text_color = QColor(v.get("text_color") or st["text_color"])
        self.b_text_color = QPushButton()
        self.b_text_color.setToolTip("Couleur du texte")
        self.b_text_color.clicked.connect(self._choose_text_color)
        self.bold = QCheckBox("Gras")
        self.bold.setChecked(bool(v.get("bold")))
        self.font_size = QSpinBox()
        self.font_size.setRange(0, 72)
        self.font_size.setSpecialValueText("taille globale")
        self.font_size.setSuffix(" px")
        self.font_size.setMinimumWidth(130)
        self.font_size.setValue(int(v.get("font_size") or 0))
        self.l_tc = QLabel("couleur")
        self.l_fs = QLabel("taille")
        txt_row.addWidget(self.l_tc)
        txt_row.addWidget(self.b_text_color)
        txt_row.addSpacing(10)
        txt_row.addWidget(self.l_fs)
        txt_row.addWidget(self.font_size)
        txt_row.addSpacing(10)
        txt_row.addWidget(self.bold)
        txt_row.addStretch(1)
        self.l_text = QLabel("Texte")
        form.addRow(self.l_text, txt_row)

        size_row = QHBoxLayout()
        self.width = QSpinBox()
        self.height = QSpinBox()
        for sp, d in ((self.width, st["block_width"]), (self.height, st["block_height"])):
            sp.setRange(0, 2000)
            sp.setSingleStep(5)
            sp.setSpecialValueText(f"défaut ({d:g})")
            sp.setSuffix(" px")
        self.width.setValue(int(v.get("width") or 0))
        self.height.setValue(int(v.get("height") or 0))
        size_row.addWidget(self.width)
        self.l_times = QLabel("×")
        size_row.addWidget(self.l_times)
        size_row.addWidget(self.height)
        size_row.addStretch(1)
        self.l_size = QLabel("Taille")
        form.addRow(self.l_size, size_row)

        self.stack = QSpinBox()
        self.stack.setRange(0, 20)
        self.stack.setSpecialValueText("aucune")
        self.stack.setValue(int(v.get("stack") or 0))
        self.stack.setToolTip("Copies dessinées derrière le bloc (effet « ×h » du Multi-Head Attention)")
        form.addRow("Copies empilées", self.stack)

        self.pytorch = QLineEdit(v.get("pytorch", ""))
        self.pytorch.setPlaceholderText("facultatif, ex. nn.Conv2d({in}, 64, 3, padding=1)")
        self.pytorch.setToolTip("Module PyTorch utilisé pour ce bloc par Fichier > Exporter en code PyTorch.\n"
                                "{in} est remplacé par la taille d'entrée du bloc (canaux ou caractéristiques).\n"
                                "Vide : DeepGraph le déduit du texte du bloc, ou laisse un TODO.")
        form.addRow("Code PyTorch", self.pytorch)

        # --- aperçu ---------------------------------------------------------------------
        prev = QGroupBox("Aperçu")
        pv = QVBoxLayout(prev)
        self.pscene = QGraphicsScene(self)
        self.pview = QGraphicsView(self.pscene)
        self.pview.setRenderHint(QPainter.Antialiasing)
        self.pview.setRenderHint(QPainter.TextAntialiasing)
        self.pview.setInteractive(False)
        self.pview.setFrameShape(QFrame.NoFrame)
        self.pview.setBackgroundBrush(QColor("#FFFFFF"))
        self.pview.setFixedHeight(120)
        self.pview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        pv.addWidget(self.pview)
        root.addWidget(prev)

        # --- description ----------------------------------------------------------------
        root.addWidget(QLabel("Description (facultatif)"))
        self.desc = QPlainTextEdit(v.get("description", ""))
        self.desc.setPlaceholderText("1re ligne : résumé affiché dans la bibliothèque.\n"
                                     "Lignes suivantes : détails (formule, rôle…), visibles dans Propriétés.\n"
                                     "Réf. Auteur et al., 2017")
        self.desc.setFixedHeight(90)
        root.addWidget(self.desc)

        self.warn = QLabel()
        self.warn.setStyleSheet("color:#B06000;")
        self.warn.setWordWrap(True)
        root.addWidget(self.warn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok = bb.button(QDialogButtonBox.Ok)
        self.ok.setText("Enregistrer" if mode == "edit" else "Ajouter à la bibliothèque")
        bb.button(QDialogButtonBox.Cancel).setText("Annuler")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        # --- liaisons -------------------------------------------------------------------
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._refresh)
        later = self._timer.start
        for w in (self.name, self.label):
            w.textChanged.connect(later)
        self.symbol.currentTextChanged.connect(later)
        self.r_rect.toggled.connect(later)
        self.bold.toggled.connect(later)
        for sp in (self.font_size, self.width, self.height, self.stack):
            sp.valueChanged.connect(later)
        self.color.changed = later
        self._update_text_color_button()
        self._refresh()
        self.name.setFocus()

    # ------------------------------------------------------------------ outils
    def _set_row_visible(self, field, on):
        lab = self.form.labelForField(field)
        if lab is not None:
            lab.setVisible(on)
        field.setVisible(on)

    def _choose_text_color(self):
        c = QColorDialog.getColor(self.text_color, self, "Couleur du texte")
        if c.isValid():
            self.text_color = c
            self._update_text_color_button()
            self._timer.start()

    def _update_text_color_button(self):
        c = self.text_color.name().upper()
        fg = "#FFFFFF" if self.text_color.lightness() < 110 else "#000000"
        self.b_text_color.setText(c)
        self.b_text_color.setStyleSheet(f"background:{c}; color:{fg}; border:1px solid #777; "
                                        "border-radius:3px; padding:2px;")

    def _symbol_value(self):
        txt = self.symbol.currentText().strip()
        i = self.symbol.findText(txt)
        if i >= 0:
            return self.symbol.itemData(i)
        return txt or "plus"

    def _refresh(self):
        circle = self.r_circle.isChecked()
        self._set_row_visible(self.symbol, circle)
        for w in (self.label, self.stack, self.pytorch):
            self._set_row_visible(w, not circle)
        # ligne « Texte » (couleur, gras, taille) : rectangles seulement
        for w in (self.b_text_color, self.bold, self.font_size, self.l_tc, self.l_fs):
            w.setVisible(not circle)
        self.l_text.setVisible(not circle)
        self.height.setVisible(not circle)
        self.l_times.setVisible(not circle)
        self.l_size.setText("Diamètre" if circle else "Taille")
        self.width.setSpecialValueText(f"défaut ({CFG.style['operator_size' if circle else 'block_width']:g})")

        name = self.name.text().strip()
        dup = [bd for bd in CFG.blocks.values() if bd.name == name and bd.id != self.edit_id]
        self.warn.setText(f"Un bloc « {name} » existe déjà (catégorie {dup[0].category}). Les deux seront "
                          "gardés." if dup else "")
        self.ok.setEnabled(bool(name))
        self._draw_preview()

    def _draw_preview(self):
        self.pscene.clear()
        v = self.values()
        if v.get("shape") == "circle":
            it = OperatorItem(v.get("symbol", "plus"), v["color"], v.get("width"))
        else:
            it = BlockItem(v.get("label", v.get("name") or "Bloc"), v["color"], v.get("width"), v.get("height"))
            if v.get("font_size"):
                it.font_size = int(v["font_size"])
            if v.get("text_color"):
                it.text_color = v["text_color"]
            it.bold, it.stack = bool(v.get("bold")), int(v.get("stack") or 0)
            if not v.get("width") or not v.get("height"):
                need_w, need_h = it.text_need()        # comme « ajuster au texte » sur le schéma
                it.set_size(max(it.w, need_w) if not v.get("width") else it.w,
                            max(it.h, need_h) if not v.get("height") else it.h)
        self.pscene.addItem(it)
        r = it.sceneBoundingRect().adjusted(-10, -10, 10, 10)
        self.pscene.setSceneRect(r)
        vw, vh = max(50, self.pview.viewport().width()), max(50, self.pview.viewport().height())
        k = min(2.5, vw / r.width(), vh / r.height())     # grossi, sans dépasser
        self.pview.resetTransform()
        self.pview.scale(k, k)
        self.pview.centerOn(r.center())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._timer.start()

    # ----------------------------------------------------------------- résultat
    def values(self):
        """Dict au format de blocks.yaml, sans les valeurs par défaut."""
        name = self.name.text().strip()
        v = {"name": name or "Bloc"}
        circle = self.r_circle.isChecked()
        if not circle:
            lab = self.label.text().strip()
            if lab and lab != name:
                v["label"] = lab
        v["category"] = self.category.currentText().strip() or "Divers"
        v["color"] = self.color.color
        if circle:
            v["shape"] = "circle"
            v["symbol"] = self._symbol_value()
        else:
            if self.text_color.name().upper() != QColor(CFG.style["text_color"]).name().upper():
                v["text_color"] = self.text_color.name().upper()
            if self.bold.isChecked():
                v["bold"] = True
            if self.font_size.value():
                v["font_size"] = self.font_size.value()
            if self.height.value():
                v["height"] = self.height.value()
            if self.stack.value():
                v["stack"] = self.stack.value()
            if self.pytorch.text().strip():
                v["pytorch"] = self.pytorch.text().strip()
        if self.width.value():
            v["width"] = self.width.value()
        d = self.desc.toPlainText().strip()
        if d:
            v["description"] = d
        return v

    def result_raw(self):
        """Bloc à écrire dans mes_blocs.yaml (avec son id)."""
        v = self.values()
        bid = self.edit_id or CFG.new_id(v["name"])
        raw = {"id": bid} if bid != v["name"] else {}
        raw.update(v)
        return raw


def values_from_def(bd):
    """BlockDef -> dict pour pré-remplir la fenêtre."""
    v = {"id": bd.id, "name": bd.name, "category": bd.category, "color": bd.color,
         "description": bd.description, "shape": bd.shape}
    if bd.label != bd.name:
        v["label"] = bd.label
    if bd.shape == "circle":
        v["symbol"] = bd.symbol
    for k in ("width", "height", "font_size", "text_color", "stack", "pytorch"):
        if getattr(bd, k):
            v[k] = getattr(bd, k)
    if bd.bold:
        v["bold"] = True
    return v


def values_from_item(it):
    """Bloc ou opérateur du schéma -> dict pour pré-remplir la fenêtre (« Ajouter à la bibliothèque »)."""
    bd = CFG.blocks.get(getattr(it, "def_id", None) or "")
    v = {"category": bd.category if bd else "Mes blocs", "color": it.color}
    if bd is not None and bd.description:
        v["description"] = bd.description
    if isinstance(it, OperatorItem):
        v.update(shape="circle", symbol=it.symbol, name=bd.name if bd else "Opération")
        if abs(it.w - CFG.style["operator_size"]) > 0.5:
            v["width"] = int(round(it.w))
        return v
    text = str(it.text or "").replace("\n", " ").strip()
    # nom de la bibliothèque : le texte du bloc, sans la syntaxe des formules
    v["name"] = " ".join(text.replace("$", "").replace("\\", "").replace("{", "").replace("}", "").split()) \
        or (bd.name if bd else "Bloc")
    if v["name"] != it.text:
        v["label"] = it.text
    if it.text_color and QColor(it.text_color) != QColor(CFG.style["text_color"]):
        v["text_color"] = QColor(it.text_color).name().upper()
    if it.bold:
        v["bold"] = True
    if it.font_size:
        v["font_size"] = int(it.font_size)
    if it.stack:
        v["stack"] = int(it.stack)
    w, h = it.base_size()
    if abs(w - CFG.style["block_width"]) > 0.5:
        v["width"] = int(round(w))
    if abs(h - CFG.style["block_height"]) > 0.5:
        v["height"] = int(round(h))
    return v
