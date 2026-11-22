"""Calques et groupes de calques (partie « modèle » de la scène).

self.ltree : liste, de l'arrière-plan vers le premier plan, de nœuds
    calque : {"id", "name", "visible", "locked", "opacity"}
    groupe : idem + "group": True, "children": [...] (même ordre)
Un groupe masqué / verrouillé masque / verrouille tout son contenu ; les opacités se multiplient.
Seuls les éléments du calque courant (ou des calques du groupe courant) sont modifiables.
"""
import copy
import json

from PyQt5.QtCore import QMimeData
from PyQt5.QtWidgets import QApplication, QGraphicsItem

from .arrow import ArrowItem
from .items import GroupItem, NodeItem, ancestors, descendants

CLIP_MIME = "application/x-deepgraph"


class LayersMixin:
    LAYER_STEP = 100

    def _init_layers(self):
        self.ltree = [self._new_layer(1, "Calque 1")]
        self.current_layer = 1      # id d'un calque ou d'un groupe
        self._loading = False
        self._linfo = None

    # ------------------------------------------------------------ structure
    @staticmethod
    def _new_layer(lid, name, group=False):
        d = {"id": lid, "name": name, "visible": True, "locked": False, "opacity": 1.0}
        if group:
            d["group"] = True
            d["children"] = []
        return d

    def _walk(self, lst=None, anc=()):
        """(nœud, liste qui le contient, ancêtres), de l'arrière vers l'avant."""
        lst = self.ltree if lst is None else lst
        for n in lst:
            yield n, lst, anc
            if n.get("group"):
                yield from self._walk(n["children"], anc + (n,))

    @property
    def layers(self):
        """Calques (sans les groupes), de l'arrière-plan vers le premier plan."""
        return [n for n, _, _ in self._walk() if not n.get("group")]

    def find_layer(self, nid):
        for n, lst, anc in self._walk():
            if n["id"] == nid:
                return n, lst, anc
        return None, None, ()

    def lnode(self, nid):
        return self.find_layer(nid)[0]

    def layer(self, lid):
        n = self.lnode(lid)
        return n if n is not None and not n.get("group") else None

    @staticmethod
    def leaves_of(node):
        if node is None:
            return []
        if not node.get("group"):
            return [node]
        out = []
        for c in node["children"]:
            out.extend(LayersMixin.leaves_of(c))
        return out

    def layer_index(self, lid):
        return next((i for i, l in enumerate(self.layers) if l["id"] == lid), 0)

    def _next_lid(self):
        return max([0] + [n["id"] for n, _, _ in self._walk()]) + 1

    def _free_name(self, base):
        names = {n["name"] for n, _, _ in self._walk()}
        k = 1
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def effective(self, nid):
        """(visible, verrouillé, opacité) en tenant compte des groupes parents."""
        n, _, anc = self.find_layer(nid)
        if n is None:
            return True, False, 1.0
        chain = anc + (n,)
        op = 1.0
        for x in chain:
            op *= float(x["opacity"])
        return all(x["visible"] for x in chain), any(x["locked"] for x in chain), op

    def active_layers(self):
        return {l["id"] for l in self.leaves_of(self.lnode(self.current_layer))}

    def target_layer(self):
        """Calque qui reçoit les nouveaux éléments."""
        n = self.lnode(self.current_layer)
        if n is None:
            n = self.layers[-1]
            self.current_layer = n["id"]
        if n.get("group"):
            leaves = self.leaves_of(n)
            if leaves:
                return leaves[-1]["id"]
            L = self._new_layer(self._next_lid(), self._free_name("Calque"))
            n["children"].append(L)
            self._linfo = None
            self.layersChanged.emit()
            return L["id"]
        return n["id"]

    def _info(self):
        if self._linfo is None:
            act = self.active_layers()
            self._linfo = {}
            for i, L in enumerate(self.layers):
                vis, locked, op = self.effective(L["id"])
                self._linfo[L["id"]] = (i, vis, locked, op, L["id"] in act)
        return self._linfo

    # -------------------------------------------------- application aux éléments
    def top_of(self, it):
        while it.parentItem() is not None:
            it = it.parentItem()
        return it

    def layer_of(self, it):
        return self.top_of(it).layer

    def addItem(self, item):
        if isinstance(item, (NodeItem, ArrowItem)):
            if self.layer(getattr(item, "layer", None)) is None:
                item.layer = self.target_layer()
            if not self._loading:
                self._make_usable(item.layer)
        super().addItem(item)
        if isinstance(item, (NodeItem, ArrowItem)):
            self.apply_item_layer(item)

    def _make_usable(self, lid):
        """Un élément créé dans un calque masqué / verrouillé : on ré-affiche et déverrouille."""
        n, _, anc = self.find_layer(lid)
        if n is None:
            return
        changed = False
        for x in anc + (n,):
            if not x["visible"] or x["locked"]:
                x["visible"], x["locked"] = True, False
                changed = True
        if changed:
            self._linfo = None
            self.apply_layers()
            self.layersChanged.emit()

    @staticmethod
    def _base_z(it):
        if isinstance(it, ArrowItem):
            return 10
        if isinstance(it, GroupItem):
            return -10 if it.parentItem() is None else -1
        return 0

    ORDER_EPS = 1e-5               # rang -> petite fraction du z (rang < 100000)

    def _next_order(self):
        return 1 + max((n.order for n in self.items()
                        if isinstance(n, NodeItem) and n.order is not None), default=-1)

    def _order_z(self, x):
        if not isinstance(x, NodeItem):
            return 0.0
        if x.order is None:
            x.order = self._next_order()
        return min(float(x.order), 99999.0) * self.ORDER_EPS

    def apply_item_layer(self, it):
        """Ordre d'affichage, opacité, visibilité et verrouillage d'un élément et de ses enfants."""
        info = self._info()
        top = self.top_of(it)
        if top.layer not in info:          # calque créé depuis le dernier calcul ?
            self._linfo = None
            info = self._info()
        if top.layer not in info:
            top.layer = self.layers[0]["id"]
        idx, vis, locked, op, active = info[top.layer]
        free = vis and not locked and active
        group = [top] + (descendants(top) if isinstance(top, NodeItem) else [])
        for x in group:
            x.layer = top.layer
            if not free and x.isSelected():
                x.setSelected(False)
            x.setFlag(QGraphicsItem.ItemIsSelectable, free)
            if isinstance(x, NodeItem):
                x.setFlag(QGraphicsItem.ItemIsMovable, free)
            if x is top:
                x.setZValue(idx * self.LAYER_STEP + self._base_z(x) + self._order_z(x))
                x.setOpacity(op)            # les enfants en héritent
                x.setVisible(vis)
            else:
                x.setZValue(self._base_z(x) + self._order_z(x))
                x.setOpacity(1.0)
                x.setVisible(True)

    def apply_layers(self):
        self._linfo = None
        for it in self.items():
            if isinstance(it, (NodeItem, ArrowItem)) and it.parentItem() is None and it is not self._temp:
                self.apply_item_layer(it)

    def items_in(self, nid):
        """Éléments de premier niveau (nœuds sans parent et flèches) d'un calque ou d'un groupe."""
        ids = {l["id"] for l in self.leaves_of(self.lnode(nid))}
        return [it for it in self.items()
                if isinstance(it, (NodeItem, ArrowItem)) and it.parentItem() is None
                and it is not self._temp and it.layer in ids]

    # ------------------------------------------------------- (dé)sérialisation
    def _parse_tree(self, lst, seen):
        out = []
        for l in lst or []:
            try:
                lid = int(l["id"])
                n = self._new_layer(lid, str(l.get("name", "Calque")), bool(l.get("group")))
                n.update(visible=bool(l.get("visible", True)), locked=bool(l.get("locked", False)),
                         opacity=min(1.0, max(0.0, float(l.get("opacity", 1.0)))))
                if n.get("group"):
                    n["children"] = self._parse_tree(l.get("children"), seen)
                out.append(n)
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _apply_layers_data(self, data):
        self.ltree = self._parse_tree(data.get("layers"), set())
        self._fix_ids()
        if not self.layers:
            self.ltree.append(self._new_layer(self._next_lid(), self._free_name("Calque")))
        if self.lnode(self.current_layer) is None:
            self.current_layer = self.layers[-1]["id"]
        self._linfo = None

    def _fix_ids(self):
        """Identifiants en double : les calques (qui portent le contenu) gardent le leur,
        les groupes et les doublons suivants en reçoivent un neuf."""
        leaves, used = set(), set()
        for n, _, _ in self._walk():
            if not n.get("group"):
                if n["id"] in leaves:
                    n["id"] = None
                else:
                    leaves.add(n["id"])
        used |= leaves
        nxt = max([0] + list(used)) + 1
        for n, _, _ in self._walk():
            if n.get("group"):
                if n["id"] in used:
                    n["id"] = None
                else:
                    used.add(n["id"])
        for n, _, _ in self._walk():
            if n["id"] is None:
                n["id"] = nxt
                used.add(nxt)
                nxt += 1

    def layers_data(self):
        return copy.deepcopy(self.ltree)

    # --------------------------------------------------------------- opérations
    def _layers_done(self, checkpoint=True):
        if not self.layers:
            self.ltree.append(self._new_layer(self._next_lid(), self._free_name("Calque")))
        if self.lnode(self.current_layer) is None:
            self.current_layer = self.layers[-1]["id"]
        self.apply_layers()
        if checkpoint:
            self.checkpoint()
        self.layersChanged.emit()

    def set_current_layer(self, nid):
        if self.lnode(nid) is not None and nid != self.current_layer:
            self.current_layer = nid
            self.apply_layers()          # seuls les éléments de ce calque restent modifiables
            self.currentLayerChanged.emit()

    def add_layer(self, name=None, group=False):
        """Nouveau calque (ou groupe) juste au-dessus de la ligne sélectionnée, au même niveau qu'elle :
        groupe sélectionné -> à côté du groupe (en dehors) ; calque d'un groupe -> dans ce groupe."""
        node = self._new_layer(self._next_lid(), name or self._free_name("Groupe" if group else "Calque"), group)
        cur, lst, _ = self.find_layer(self.current_layer)
        if cur is None:
            self.ltree.append(node)
        else:
            lst.insert(lst.index(cur) + 1, node)
        if group:                       # un groupe neuf contient un calque, prêt à l'emploi
            node["children"].append(self._new_layer(self._next_lid(), self._free_name("Calque")))
        self.current_layer = node["id"]
        self._layers_done()
        return node["id"]

    def delete_layers(self, ids):
        """Supprime des calques / groupes ET leur contenu (Ctrl+Z pour revenir)."""
        ids = [i for i in ids if self.lnode(i) is not None]
        if not ids:
            return
        doomed = []
        for nid in ids:
            doomed.extend(self.items_in(nid))
        self.delete_items(doomed)
        for nid in ids:
            n, lst, _ = self.find_layer(nid)
            if n is not None:
                lst.remove(n)
        self._layers_done()

    def merge_down(self, lid):
        """Fusionne un calque avec le calque situé juste en dessous."""
        leaves = self.layers
        i = self.layer_index(lid)
        if self.layer(lid) is None or i == 0:
            return
        dest = leaves[i - 1]["id"]
        for it in self.items_in(lid):
            it.layer = dest
        n, lst, _ = self.find_layer(lid)
        lst.remove(n)
        self.current_layer = dest
        self._layers_done()

    def move_layer(self, nid, delta):
        n, lst, _ = self.find_layer(nid)
        if n is None:
            return
        i = lst.index(n)
        j = i + delta
        if not 0 <= j < len(lst):
            return
        lst[i], lst[j] = lst[j], lst[i]
        self._layers_done()

    def set_layer_prop(self, nid, key, value, checkpoint=True):
        n = self.lnode(nid)
        if n is None or n.get(key) == value:
            return
        n[key] = value
        self._layers_done(checkpoint)

    def _order_key(self):
        return {n["id"]: k for k, (n, _, _) in enumerate(self._walk())}

    def group_layers(self, ids, name=None):
        """Regroupe des calques / groupes dans un nouveau groupe, à la place du plus haut d'entre eux."""
        nodes = [self.lnode(i) for i in ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return None
        # on ignore ceux déjà contenus dans un autre nœud choisi
        sel = {id(n) for n in nodes}
        nodes = [n for n in nodes if not any(id(a) in sel for a in self.find_layer(n["id"])[2])]
        order = self._order_key()
        nodes.sort(key=lambda n: order[n["id"]])
        top = nodes[-1]
        _, lst, _ = self.find_layer(top["id"])
        pos = lst.index(top) - sum(1 for n in nodes if n is not top and any(n is x for x in lst[:lst.index(top)]))
        # identifiant et nom choisis AVANT de sortir les calques de l'arbre (sinon doublon d'identifiant)
        g = self._new_layer(self._next_lid(), name or self._free_name("Groupe"), group=True)
        for n in nodes:
            _, l2, _ = self.find_layer(n["id"])
            l2.remove(n)
        g["children"] = nodes
        lst.insert(max(0, pos), g)
        self.current_layer = g["id"]
        self._layers_done()
        return g["id"]

    def ungroup_layers(self, gid):
        g, lst, _ = self.find_layer(gid)
        if g is None or not g.get("group"):
            return
        i = lst.index(g)
        lst[i:i + 1] = g["children"]
        if self.current_layer == gid:
            self.current_layer = g["children"][-1]["id"] if g["children"] else None
        self._layers_done()

    def move_out_of_group(self, ids):
        """Sort des calques / groupes de leur groupe : ils se placent juste au-dessus de celui-ci."""
        moved = False
        for nid in ids:
            n, lst, anc = self.find_layer(nid)
            if n is None or not anc:
                continue
            g = anc[-1]
            _, glst, _ = self.find_layer(g["id"])
            lst.remove(n)
            glst.insert(glst.index(g) + 1, n)
            moved = True
        if moved:
            self._layers_done()

    def in_group(self, nid):
        return bool(self.find_layer(nid)[2])

    def set_layer_tree(self, tree):
        """Nouvelle organisation venue du panneau (glisser-déposer)."""
        self.ltree = tree
        self._layers_done()

    def duplicate_layer(self, nid):
        """Copie un calque (ou un groupe et tous ses calques) avec son contenu, juste au-dessus."""
        n, lst, _ = self.find_layer(nid)
        if n is None:
            return None
        remap = {}
        base = self._next_lid()

        def clone(node):
            c = {k: v for k, v in node.items() if k != "children"}
            c["id"] = base + len(remap)
            remap[node["id"]] = c["id"]
            if node.get("group"):
                c["children"] = [clone(ch) for ch in node["children"]]
            return c

        new = clone(n)
        new["name"] = n["name"] + " (copie)"
        lst.insert(lst.index(n) + 1, new)
        items = self.items_in(nid)
        if items:
            data = self.to_dict(items, external=True)
            data["arrows"] = [a for a in data["arrows"] if a.get("layer") in remap]
            for d in data["nodes"] + data["arrows"]:
                if d.get("layer") in remap:
                    d["layer"] = remap[d["layer"]]
            self._loading = True
            try:
                self.from_dict(data, new_ids=True, external=True)
            finally:
                self._loading = False
        self.current_layer = new["id"]
        self._layers_done()
        return new["id"]

    def copy_layer_content(self, nid):
        items = self.items_in(nid)
        if not items:
            return False
        md = QMimeData()
        md.setData(CLIP_MIME, json.dumps(self.to_dict(items)).encode("utf-8"))
        QApplication.clipboard().setMimeData(md)
        self._paste_n = -1          # collé au même endroit
        return True

    def cut_layer_content(self, nid):
        if self.copy_layer_content(nid):
            self.delete_items(self.items_in(nid))
            self.checkpoint()

    def paste_into_layer(self, nid):
        self.set_current_layer(nid)
        self.paste()

    def select_layer_content(self, nid):
        self.set_current_layer(nid)
        self.clearSelection()
        for it in self.items_in(nid):
            if it.flags() & QGraphicsItem.ItemIsSelectable:
                it.setSelected(True)

    def show_only(self, nid):
        n, _, anc = self.find_layer(nid)
        if n is None:
            return
        keep = {id(x) for x in anc} | {id(x) for x, _, _ in self._walk([n])}
        for x, _, _ in self._walk():
            x["visible"] = id(x) in keep
        self._layers_done()

    def show_all_layers(self):
        for x, _, _ in self._walk():
            x["visible"] = True
        self._layers_done()

    def move_selection_to_layer(self, nid):
        """Envoie la sélection sur un calque. Un élément pris dans un groupe d'un autre calque en sort."""
        n = self.lnode(nid)
        if n is None:
            return
        if n.get("group"):
            leaves = self.leaves_of(n)
            if not leaves:
                return
            nid = leaves[-1]["id"]
        sel = [i for i in self.selectedItems() if isinstance(i, (NodeItem, ArrowItem))]
        if not sel:
            return
        s = set(sel)
        for it in sel:
            if isinstance(it, NodeItem) and any(a in s for a in ancestors(it)):
                continue                          # suit son groupe sélectionné
            if isinstance(it, NodeItem) and it.parentItem() is not None:
                if self.layer_of(it) == nid:
                    continue
                self.reparent(it, None)
            it.layer = nid
            self.apply_item_layer(it)
        self.checkpoint()
        self.layersChanged.emit()
