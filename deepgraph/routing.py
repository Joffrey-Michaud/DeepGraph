"""Routage orthogonal automatique des flèches : contourne les blocs (A* sur les couloirs libres).

Les coordonnées candidates sont celles des bords des obstacles (agrandis d'une marge), des
points de départ / d'arrivée et des milieux des couloirs entre obstacles. Le chemin le plus court
est cherché sur cette grille irrégulière, chaque virage coûtant une pénalité : on obtient des
tracés avec peu de coudes, à distance des blocs.
"""
import heapq

from PyQt5.QtCore import QPointF

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _uniq(vals):
    out = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > 0.5:
            out.append(v)
    return out


def route(p1, d1, p2, d2, rects, margin=12.0, bend=None, window=260.0):
    """Polyligne orthogonale de p1 à p2 (QPointF) évitant les rectangles (QRectF, déjà sans les
    extrémités qu'on traverse). d1 / d2 : direction de sortie de chaque ancre ("up", … ou None).
    Renvoie None si aucun chemin n'est trouvé."""
    bend = 3.0 * margin if bend is None else bend
    x1, y1, x2, y2 = p1.x(), p1.y(), p2.x(), p2.y()

    def port(x, y, d):
        if d in DIRS:
            dx, dy = DIRS[d]
            return x + dx * margin, y + dy * margin
        return x, y

    s = port(x1, y1, d1)
    t = port(x2, y2, d2)
    # seuls les obstacles proches comptent
    lo_x, hi_x = min(x1, x2) - window, max(x1, x2) + window
    lo_y, hi_y = min(y1, y2) - window, max(y1, y2) + window
    obs = []
    for r in rects:
        if r.right() < lo_x or r.left() > hi_x or r.bottom() < lo_y or r.top() > hi_y:
            continue
        obs.append((r.left() - margin, r.top() - margin, r.right() + margin, r.bottom() + margin))

    xs = {s[0], t[0]}
    ys = {s[1], t[1]}
    for (l, tp, r, b) in obs:
        xs.update((l, r))
        ys.update((tp, b))
    xs, ys = _uniq(xs), _uniq(ys)
    # milieux des couloirs : un tracé passe à égale distance de deux blocs voisins
    xs = _uniq(xs + [(a + b) / 2 for a, b in zip(xs, xs[1:])])
    ys = _uniq(ys + [(a + b) / 2 for a, b in zip(ys, ys[1:])])
    xs.append(min(xs) - margin * 2)
    xs.append(max(xs) + margin * 2)
    ys.append(min(ys) - margin * 2)
    ys.append(max(ys) + margin * 2)
    xs, ys = _uniq(xs), _uniq(ys)

    def inside(x, y):
        for (l, tp, r, b) in obs:
            if l + 1e-6 < x < r - 1e-6 and tp + 1e-6 < y < b - 1e-6:
                return True
        return False

    def nearest(vals, v):
        return min(range(len(vals)), key=lambda i: abs(vals[i] - v))

    si, sj = nearest(xs, s[0]), nearest(ys, s[1])
    ti, tj = nearest(xs, t[0]), nearest(ys, t[1])
    want_end = None
    if d2 in DIRS:                        # on arrive sur p2 en allant vers lui : sens opposé à d2
        dx, dy = DIRS[d2]
        want_end = (-dx, -dy)
    start_dir = DIRS.get(d1)

    blocked = {}

    def free(i, j):
        k = (i, j)
        if k not in blocked:
            blocked[k] = inside(xs[i], ys[j]) and k not in ((si, sj), (ti, tj))
        return not blocked[k]

    def h(i, j):
        return abs(xs[i] - xs[ti]) + abs(ys[j] - ys[tj])

    start = (si, sj, start_dir)
    best = {start: 0.0}
    prev = {}
    heap = [(h(si, sj), 0.0, start)]
    goal = None
    while heap:
        f, g, st = heapq.heappop(heap)
        if g > best.get(st, 1e18) + 1e-9:
            continue
        i, j, d = st
        if (i, j) == (ti, tj):
            goal = st
            break
        for nd in DIRS.values():
            if d is not None and nd == (-d[0], -d[1]):
                continue                     # pas de demi-tour
            ni, nj = i + nd[0], j + nd[1]
            if not (0 <= ni < len(xs) and 0 <= nj < len(ys)) or not free(ni, nj):
                continue
            mx, my = (xs[i] + xs[ni]) / 2, (ys[j] + ys[nj]) / 2
            if inside(mx, my):
                continue
            cost = g + abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j])
            if d is not None and nd != d:
                cost += bend
            if (ni, nj) == (ti, tj) and want_end is not None and nd != want_end:
                cost += bend * (2 if nd == (-want_end[0], -want_end[1]) else 1)
            ns = (ni, nj, nd)
            if cost < best.get(ns, 1e18) - 1e-9:
                best[ns] = cost
                prev[ns] = st
                heapq.heappush(heap, (cost + h(ni, nj), cost, ns))
    if goal is None:
        return None
    cells = []
    st = goal
    while st is not None:
        cells.append(QPointF(xs[st[0]], ys[st[1]]))
        st = prev.get(st)
    cells.reverse()
    pts = [QPointF(x1, y1)]
    if d1 in DIRS:
        pts.append(QPointF(*s))
    for q in cells:
        _ortho_to(pts, q)
    if d2 in DIRS:
        _ortho_to(pts, QPointF(*t))
    _ortho_to(pts, QPointF(x2, y2))
    return pts


def _ortho_to(pts, q):
    """Ajoute q en restant orthogonal (insère un coude si le point n'est pas aligné)."""
    a = pts[-1]
    if abs(a.x() - q.x()) > 1e-6 and abs(a.y() - q.y()) > 1e-6:
        pts.append(QPointF(a.x(), q.y()))
    pts.append(QPointF(q))
