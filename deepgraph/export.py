"""Export PDF / SVG / PNG (vectoriel pour PDF et SVG)."""
import math

from PyQt5.QtCore import Qt, QMarginsF, QRectF, QSize, QSizeF
from PyQt5.QtGui import QImage, QPageLayout, QPageSize, QPainter, QPdfWriter
from PyQt5.QtSvg import QSvgGenerator

from .arrow import ArrowItem
from .items import NodeItem, descendants


def _collect(scene, only):
    if only is None:
        return [i for i in scene.items() if isinstance(i, (NodeItem, ArrowItem)) and i is not scene._temp
                and i.isVisible()], None
    keep = set()
    for it in only:
        keep.add(it)
        if isinstance(it, NodeItem):
            keep.update(descendants(it))
    for a in scene.arrows():
        if a.src.node in keep and a.dst.node in keep:
            keep.add(a)
    keep = {i for i in keep if i.isVisible()}
    return list(keep), keep


def _render(scene, painter, target, source):
    painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    scene.render(painter, target, source, Qt.IgnoreAspectRatio)


def export_scene(scene, path, fmt, scale=4.0, transparent=False, only=None, margin=4.0):
    items, keep = _collect(scene, only)
    if not items:
        raise ValueError("Rien à exporter.")
    selected = scene.selectedItems()
    scene.clearSelection()
    scene.exporting = True
    scene.export_only = keep
    try:
        rect = QRectF()
        for it in items:
            r = it.mapRectToScene(it.content_rect())
            rect = r if rect.isNull() else rect.united(r)
        rect.adjust(-margin, -margin, margin, margin)
        w, h = rect.width(), rect.height()
        if fmt == "svg":
            gen = QSvgGenerator()
            gen.setFileName(path)
            gen.setResolution(72)
            gen.setSize(QSize(math.ceil(w), math.ceil(h)))
            gen.setViewBox(QRectF(0, 0, w, h))
            gen.setTitle("DeepGraph")
            p = QPainter(gen)
            _render(scene, p, QRectF(0, 0, w, h), rect)
            p.end()
        elif fmt == "pdf":
            wr = QPdfWriter(path)
            wr.setCreator("DeepGraph")
            wr.setPageSize(QPageSize(QSizeF(w, h), QPageSize.Point, "", QPageSize.ExactMatch))
            wr.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Point)
            wr.setResolution(720)
            p = QPainter(wr)
            _render(scene, p, QRectF(0, 0, wr.width(), wr.height()), rect)
            p.end()
        elif fmt == "png":
            iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
            img = QImage(iw, ih, QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent if transparent else Qt.white)
            dpm = int(round(scale * 72 / 0.0254))
            img.setDotsPerMeterX(dpm)
            img.setDotsPerMeterY(dpm)
            p = QPainter(img)
            _render(scene, p, QRectF(0, 0, iw, ih), rect)
            p.end()
            if not img.save(path, "PNG"):
                raise IOError(f"Impossible d'écrire {path}")
        else:
            raise ValueError(f"Format inconnu : {fmt}")
    finally:
        scene.exporting = False
        scene.export_only = None
        for it in selected:
            it.setSelected(True)
