"""Boîte de dialogue « Empiler » : copies sans texte dessinées derrière un bloc (style « ×h »)."""
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QGridLayout, QLabel, QSpinBox, QToolButton, QVBoxLayout,
                             QWidget)

# (clé, flèche, infobulle, signe x, signe y)
DIRECTIONS = [
    ("ul", "↖", "Vers le haut à gauche", -1, -1),
    ("ur", "↗", "Vers le haut à droite", +1, -1),
    ("dl", "↙", "Vers le bas à gauche", -1, +1),
    ("dr", "↘", "Vers le bas à droite", +1, +1),
]
FIELDS = ("stack", "stack_dx", "stack_dy", "stack_fade")
DEFAULT_GAP = 6.0


class StackDialog(QDialog):
    """Modifie les blocs en direct (aperçu) ; Annuler remet les valeurs d'origine."""

    def __init__(self, blocks, parent=None, view=None):
        super().__init__(parent)
        self.setWindowTitle("Empiler")
        # non modale (on peut zoomer / se déplacer dans le schéma) et toujours au-dessus de l'éditeur
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.view = view
        self._drag = None
        self.blocks = blocks
        self._orig = [{k: getattr(b, k) for k in FIELDS} for b in blocks]
        first = blocks[0]
        stacked = first.stack > 0

        v = QVBoxLayout(self)
        intro = QLabel("Ajoute derrière le bloc des copies sans texte,\nde plus en plus transparentes.\n"
                       "Le schéma se met à jour en direct ; tirer le fond\nde cette fenêtre pour la déplacer.")
        intro.setStyleSheet("color:gray;")
        v.addWidget(intro)
        form = QFormLayout()
        v.addLayout(form)

        self.count = QSpinBox()
        self.count.setRange(1, 51)
        self.count.setValue(first.stack + 1 if stacked else 3)
        self.count.setToolTip("Nombre total de blocs, bloc d'origine compris (3 = 2 copies ajoutées)")
        form.addRow("Nombre de blocs :", self.count)

        # 4 directions en carré, comme les coins d'un pavé
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.dirs = QButtonGroup(self)
        self.dirs.setExclusive(True)
        sx = 1 if first.stack_dx >= 0 else -1
        sy = 1 if first.stack_dy > 0 else -1
        if not stacked:
            sx, sy = 1, -1
        for i, (key, arrow, tip, dx, dy) in enumerate(DIRECTIONS):
            b = QToolButton()
            b.setText(arrow)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setFixedSize(38, 32)
            f = b.font()
            f.setPointSizeF(f.pointSizeF() * 1.5)
            b.setFont(f)
            b.setProperty("sign", (dx, dy))
            self.dirs.addButton(b, i)
            grid.addWidget(b, i // 2, i % 2)
            if (dx, dy) == (sx, sy):
                b.setChecked(True)
        grid.setColumnStretch(2, 1)
        form.addRow("Direction :", box)

        def gap_box(val):
            sb = QDoubleSpinBox()
            sb.setRange(0, 200)
            sb.setDecimals(1)
            sb.setSingleStep(1)
            sb.setSuffix(" px")
            sb.setValue(val)
            return sb

        self.gx = gap_box(abs(first.stack_dx) if stacked else DEFAULT_GAP)
        self.gy = gap_box(abs(first.stack_dy) if stacked else DEFAULT_GAP)
        self.gx.setToolTip("Décalage horizontal entre deux blocs successifs")
        self.gy.setToolTip("Décalage vertical entre deux blocs successifs")
        form.addRow("Espacement horizontal :", self.gx)
        form.addRow("Espacement vertical :", self.gy)

        self.fade = QCheckBox("Opacité de plus en plus faible")
        self.fade.setChecked(first.stack_fade if stacked else True)
        form.addRow("", self.fade)

        if len(blocks) > 1:
            n = QLabel(f"S'applique aux {len(blocks)} blocs sélectionnés.")
            n.setStyleSheet("color:gray;")
            v.addWidget(n)

        bb = QDialogButtonBox()
        ok = bb.addButton("Confirmer", QDialogButtonBox.AcceptRole)
        bb.addButton("Annuler", QDialogButtonBox.RejectRole)
        ok.setDefault(True)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self.count.valueChanged.connect(self.preview)
        self.dirs.buttonClicked.connect(self.preview)
        self.gx.valueChanged.connect(self.preview)
        self.gy.valueChanged.connect(self.preview)
        self.fade.toggled.connect(self.preview)
        self.preview()

    def values(self):
        b = self.dirs.checkedButton()
        sx, sy = b.property("sign") if b is not None else (1, -1)
        return {"stack": self.count.value() - 1, "stack_dx": sx * self.gx.value(),
                "stack_dy": sy * self.gy.value(), "stack_fade": self.fade.isChecked()}

    def _set(self, block, vals):
        try:
            for k, val in vals.items():
                if getattr(block, k) != val:
                    block.set_prop(k, val)
        except RuntimeError:          # bloc supprimé entre-temps
            pass

    def preview(self, *_):
        vals = self.values()
        for b in self.blocks:
            self._set(b, vals)
        self._refresh()

    def _refresh(self):
        sc = self.view.scene() if self.view is not None else None
        if sc is not None:
            sc.refit_groups()
            sc.update()
            self.view.viewport().update()

    # ------------------------------------------------- placement / déplacement
    def place_beside_blocks(self):
        """Place la fenêtre à côté des blocs (à droite si possible), sans les recouvrir."""
        if self.view is None:
            return
        self.adjustSize()
        rect = None
        for b in self.blocks:
            r = b.mapRectToScene(b.layout_rect())
            rect = r if rect is None else rect.united(r)
        vp = self.view.viewport()
        tl = vp.mapToGlobal(self.view.mapFromScene(rect.topLeft()))
        br = vp.mapToGlobal(self.view.mapFromScene(rect.bottomRight()))
        target = QRect(tl, br)
        screen = (QApplication.screenAt(target.center()) or QApplication.primaryScreen()).availableGeometry()
        w, h, gap = self.width(), self.height(), 40
        y = max(screen.top(), min(target.center().y() - h // 2, screen.bottom() - h))
        if target.right() + gap + w <= screen.right():
            pos = QPoint(target.right() + gap, y)
        elif target.left() - gap - w >= screen.left():
            pos = QPoint(target.left() - gap - w, y)
        else:                                       # en dessous, ou à défaut au-dessus
            x = max(screen.left(), min(target.center().x() - w // 2, screen.right() - w))
            pos = QPoint(x, min(target.bottom() + gap, screen.bottom() - h))
        self.move(pos)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self._drag)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = None
        super().mouseReleaseEvent(e)

    def reject(self):
        for b, vals in zip(self.blocks, self._orig):
            self._set(b, vals)
        self._refresh()
        super().reject()
