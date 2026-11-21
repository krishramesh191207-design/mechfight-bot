"""Mech Royale bot v2: aim-gated fire, wall-aware navigation, loot/cube farming, zone + retreat logic."""
import heapq
import json
import math
import struct
import time


class Nav:
    """2 m walkability grid built from the map files, with A* pathing."""
    C = 2.0
    INF = 1e18

    def __init__(self, map_dir):
        self.ok = False
        t0 = time.time()
        world = json.load(open(f"{map_dir}/v2/world.json"))
        tm = world["terrain"]
        nx, ny = int(tm["nx"]), int(tm["ny"])
        with open(f"{map_dir}/v2/terrain.i16", "rb") as f:
            self.tz = struct.unpack(f"<{nx * ny}h", f.read())
        self.tnx, self.tny = nx, ny
        self.tminx, self.tminy, self.tstep = tm["minX"], tm["minY"], tm["step"]
        b = world["bounds"]
        C = self.C
        self.minx, self.miny = b["minX"], b["minY"]
        self.W = int((b["maxX"] - b["minX"]) / C) + 1
        self.H = int((b["maxY"] - b["minY"]) / C) + 1
        W, H = self.W, self.H
        solid = bytearray(W * H)
        for p in world.get("prisms", []):
            self._raster(p, solid)
        self.solid = solid
        blocked = bytearray(solid)
        # steep terrain
        tz, tnx, tny = self.tz, nx, ny
        for j in range(H):
            yc = self.miny + (j + 0.5) * C
            tj = int(round((yc - self.tminy) / self.tstep))
            if tj < 1 or tj >= tny - 1:
                continue
            row = j * W
            for i in range(W):
                xc = self.minx + (i + 0.5) * C
                ti = int(round((xc - self.tminx) / self.tstep))
                if ti < 1 or ti >= tnx - 1:
                    continue
                k = tj * tnx + ti
                gx = (tz[k + 1] - tz[k - 1]) * 0.005
                gy = (tz[k + tnx] - tz[k - tnx]) * 0.005
                if gx * gx + gy * gy > 1.44:  # > ~50 deg
                    blocked[row + i] = 1
        self.blocked0 = bytearray(blocked)  # raw: solids + steep terrain, no clearance
        # inflate by one cell (mech radius 2.5 m)
        infl = bytearray(blocked)
        for j in range(H):
            row = j * W
            for i in range(W):
                if blocked[row + i]:
                    for dj in (-1, 0, 1):
                        jj = j + dj
                        if jj < 0 or jj >= H:
                            continue
                        for di in (-1, 0, 1):
                            ii = i + di
                            if 0 <= ii < W:
                                infl[jj * W + ii] = 1
        # map edge is a wall
        for j in range(H):
            for i in range(W):
                if i < 2 or j < 2 or i >= W - 2 or j >= H - 2:
                    infl[j * W + i] = 1
        self.blocked = infl
        self.g = [self.INF] * (W * H)
        self.parent = [-1] * (W * H)
        self.ok = True
        self.build_s = time.time() - t0

    # ---- terrain / grid helpers
    def ground(self, x, y):
        ti = int(round((x - self.tminx) / self.tstep))
        tj = int(round((y - self.tminy) / self.tstep))
        ti = max(0, min(self.tnx - 1, ti))
        tj = max(0, min(self.tny - 1, tj))
        return self.tz[tj * self.tnx + ti] * 0.01

    def to_cell(self, x, y):
        C = self.C
        i = int((x - self.minx) / C)
        j = int((y - self.miny) / C)
        return max(0, min(self.W - 1, i)), max(0, min(self.H - 1, j))

    def center(self, i, j):
        return self.minx + (i + 0.5) * self.C, self.miny + (j + 0.5) * self.C

    def _raster(self, p, solid):
        kind = p.get("kind")
        if kind not in ("building", "stadium", "bridge"):
            return
        rings = [p["ring"]] + [h for h in (p.get("holes") or [])]
        edges = []
        miny_p, maxy_p = 1e18, -1e18
        for r in rings:
            if not r or isinstance(r[0], (list, tuple)):
                continue
            n = len(r) // 2
            for k in range(n):
                x0, y0 = r[2 * k], r[2 * k + 1]
                x1, y1 = r[2 * ((k + 1) % n)], r[2 * ((k + 1) % n) + 1]
                edges.append((x0, y0, x1, y1))
                miny_p = min(miny_p, y0)
                maxy_p = max(maxy_p, y0)
        if not edges:
            return
        C, W, H = self.C, self.W, self.H
        base, top = p["base"], p["top"]
        j0 = max(0, int((miny_p - self.miny) / C) - 1)
        j1 = min(H - 1, int((maxy_p - self.miny) / C) + 1)
        for j in range(j0, j1 + 1):
            yc = self.miny + (j + 0.5) * C
            xs = []
            for x0, y0, x1, y1 in edges:
                if (y0 <= yc < y1) or (y1 <= yc < y0):
                    xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - self.minx) / C - 0.5)))
                i1 = min(W - 1, int(math.floor((xs[a + 1] - self.minx) / C - 0.5)))
                for i in range(i0, i1 + 1):
                    xc = self.minx + (i + 0.5) * C
                    gz = self.ground(xc, yc)
                    if base - gz < 9.5 and top - gz > 1.2:
                        solid[j * W + i] = 1
        # thin walls can slip between cell centres: also mark every cell an outline passes through
        for x0, y0, x1, y1 in edges:
            L = math.hypot(x1 - x0, y1 - y0)
            n = int(L / 0.7) + 1
            for k in range(n + 1):
                f = k / n
                xs_, ys_ = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                i = int((xs_ - self.minx) / C)
                j = int((ys_ - self.miny) / C)
                if 0 <= i < W and 0 <= j < H and not solid[j * W + i]:
                    gz = self.ground(xs_, ys_)
                    if base - gz < 9.5 and top - gz > 1.2:
                        solid[j * W + i] = 1

    def is_blocked(self, x, y):
        i, j = self.to_cell(x, y)
        return self.blocked[j * self.W + i] != 0

    def is_solid(self, x, y):
        i, j = self.to_cell(x, y)
        return self.solid[j * self.W + i] != 0

    def ray_free(self, x0, y0, x1, y1, grid=None, step=1.0, skip=0.0):
        """True if no blocked cell on the segment (samples every `step` m, skipping the first `skip` m)."""
        if grid is None:
            grid = self.blocked
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return True
        n = int(d / step)
        W = self.W
        C = self.C
        ux, uy = dx / d, dy / d
        s = skip
        while s <= d:
            x, y = x0 + ux * s, y0 + uy * s
            i = int((x - self.minx) / C)
            j = int((y - self.miny) / C)
            if i < 0 or j < 0 or i >= W or j >= self.H or grid[j * W + i]:
                return False
            s += step
        return True

    def shot_clear(self, x0, y0, x1, y1):
        """Rough line of fire check on raw building footprints (>=2 solid samples in a row blocks)."""
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d < 4:
            return True
        ux, uy = dx / d, dy / d
        run = 0
        s = 3.0
        W, C, H = self.W, self.C, self.H
        solid = self.solid
        while s < d - 1.0:
            i = int((x0 + ux * s - self.minx) / C)
            j = int((y0 + uy * s - self.miny) / C)
            if 0 <= i < W and 0 <= j < H and solid[j * W + i]:
                run += 1
                if run >= 2:
                    return False
            else:
                run = 0
            s += 1.0
        return True

    def near_free(self, i, j, maxr=8, grid=None):
        W, H = self.W, self.H
        if grid is None:
            grid = self.blocked
        if grid[j * W + i] == 0:
            return i, j
        for r in range(1, maxr + 1):
            best = None
            bd = 1e18
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < W and 0 <= jj < H and grid[jj * W + ii] == 0:
                        dd = di * di + dj * dj
                        if dd < bd:
                            bd, best = dd, (ii, jj)
            if best:
                return best
        return None

    # ---- A*
    def find_path_robust(self, sx, sy, gx, gy, tblock=None, node_cap=5000, time_cap=0.02):
        """A* on the clearance grid; if that can't reach the goal (gap narrower than the margin), retry on the raw grid."""
        pts, found, exhausted = self._astar(sx, sy, gx, gy, tblock, node_cap, time_cap, self.blocked)
        if found or not exhausted:
            return pts, False
        pts2, found2, _ = self._astar(sx, sy, gx, gy, tblock, node_cap, time_cap, self.blocked0)
        if found2:
            return pts2, True
        return pts, False

    def find_path(self, sx, sy, gx, gy, tblock=None, node_cap=5000, time_cap=0.02):
        return self._astar(sx, sy, gx, gy, tblock, node_cap, time_cap, self.blocked)[0]

    def _astar(self, sx, sy, gx, gy, tblock, node_cap, time_cap, blocked):
        W, H = self.W, self.H
        si, sj = self.to_cell(sx, sy)
        gi, gj = self.to_cell(gx, gy)
        s = self.near_free(si, sj, 6, blocked)
        g = self.near_free(gi, gj, 12, blocked)
        if s is None or g is None:
            return None, False, False
        si, sj = s
        gi, gj = g
        start = sj * W + si
        goal = gj * W + gi
        gs, parent = self.g, self.parent
        touched = [start]
        gs[start] = 0.0
        parent[start] = -1
        D = 1.41421356
        t0 = time.perf_counter()

        def h(i, j):
            dx, dy = abs(i - gi), abs(j - gj)
            return (dx + dy) + (D - 2) * min(dx, dy)

        openh = [(h(si, sj) * 1.15, start)]
        best_idx, best_h = start, h(si, sj)
        nodes = 0
        found = False
        capped = False
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D))
        while openh:
            f, cur = heapq.heappop(openh)
            if cur == goal:
                found = True
                break
            cj, ci = divmod(cur, W)
            gcur = gs[cur]
            if f > (gcur + h(ci, cj)) * 1.15 + 1e-6:
                continue
            nodes += 1
            if nodes > node_cap or (nodes & 255) == 0 and time.perf_counter() - t0 > time_cap:
                capped = True
                break
            for di, dj, cost in nbrs:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= W or nj >= H:
                    continue
                nidx = nj * W + ni
                if blocked[nidx]:
                    continue
                if di and dj and (blocked[cj * W + ni] or blocked[nj * W + ci]):
                    continue
                if tblock and nidx in tblock:
                    continue
                ng = gcur + cost
                if ng < gs[nidx]:
                    if gs[nidx] >= self.INF:
                        touched.append(nidx)
                    gs[nidx] = ng
                    parent[nidx] = cur
                    hh = h(ni, nj)
                    if hh < best_h:
                        best_h, best_idx = hh, nidx
                    heapq.heappush(openh, (ng + hh * 1.15, nidx))
        end = goal if found else best_idx
        cells = []
        k = end
        while k != -1:
            cells.append(k)
            k = parent[k]
        cells.reverse()
        for k in touched:
            gs[k] = self.INF
            parent[k] = -1
        pts = []
        for k in cells:
            j, i = divmod(k, W)
            pts.append(self.center(i, j))
        # string-pull
        out = []
        a = 0
        n = len(pts)
        if n == 0:
            return None, False, False
        out.append(pts[0])
        while a < n - 1:
            b = n - 1
            while b > a + 1 and not self.ray_free(pts[a][0], pts[a][1], pts[b][0], pts[b][1], grid=blocked, step=1.0):
                b -= 1
            out.append(pts[b])
            a = b
        return out, found, (not found and not capped)


# =====================================================================
#  Bot logic
# =====================================================================
import random
import threading
import mechfight as mf

TAU = math.tau
PI = math.pi


def angle_diff(a, b):
    d = (a - b) % TAU
    return d - TAU if d > PI else d


def hyp(a, b):
    return math.sqrt(a * a + b * b)


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def d3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def unit(vx, vy):
    l = math.hypot(vx, vy)
    if l < 1e-9:
        return 0.0, 0.0
    return vx / l, vy / l


def rot(vx, vy, a):
    c, s = math.cos(a), math.sin(a)
    return vx * c - vy * s, vx * s + vy * c


UPGRADE_PRIORITY = [
    "armor", "plating", "gun_damage", "laser_damage", "lightning_chain", "gun_rate",
    "gyros", "nanites", "mortar_reload", "laser_charge", "lightning_efficiency",
    "lightning_range", "servos", "gun_mag", "capacitor", "jets", "reactor",
    "gun_velocity", "mortar_splash", "mortar_velocity", "laser_pierce", "sensors",
]

# how much we like carrying each secondary weapon
WEAPON_VALUE = {"lightning": 3.0, "laser": 2.7, "mortar": 1.5}


def pick_upgrade(choices):
    best, best_rank = 0, 10 ** 6
    for i, c in enumerate(choices):
        cid = c.get("id", "")
        rank = UPGRADE_PRIORITY.index(cid) if cid in UPGRADE_PRIORITY else 500
        if rank < best_rank:
            best, best_rank = i, rank
    return best


class Planner:
    """Runs A* in a background thread so a decision never waits on pathfinding."""

    def __init__(self, nav):
        self.nav = nav
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.req = None
        self.res = None
        self.req_id = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self, sx, sy, gx, gy, tblock):
        with self.cv:
            self.req_id += 1
            self.req = (self.req_id, sx, sy, gx, gy, dict(tblock) if tblock else None)
            self.cv.notify()
        return self.req_id

    def poll(self):
        with self.lock:
            r, self.res = self.res, None
            return r

    def _loop(self):
        while True:
            with self.cv:
                while self.req is None:
                    self.cv.wait()
                req, self.req = self.req, None
            rid, sx, sy, gx, gy, tb = req
            try:
                path, tight = self.nav.find_path_robust(sx, sy, gx, gy, tblock=tb, node_cap=60000, time_cap=0.4)
            except Exception:
                path, tight = None, False
            with self.lock:
                self.res = (rid, path, (gx, gy), tight)


class Bot:
    name = "TactBot2"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.w = cfg["weapons"]
        self.mech = cfg["mech"]
        self.rng = random.Random(hello.get("seed", 1))
        self.nav = None
        self.planner = None
        try:
            self.nav = Nav(hello["map"]["dir"])
            self.planner = Planner(self.nav)
            mf.log(f"nav built in {self.nav.build_s:.2f}s")
        except Exception as e:
            mf.log(f"nav failed: {e!r}")
        self.last_t = 0.0
        self.target_id = None
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.last_damage_t = -99.0
        self.last_damage_bearing = None
        self.last_enemy_seen_t = -99.0
        self.enemy_hist = {}
        self.sound_dir = None
        self.sound_t = -99.0
        self.path = None
        self.path_idx = 0
        self.path_goal = None
        self.path_tight = False
        self.path_req_t = -99.0
        self.pending_rid = None
        self.tblock = {}
        self.pos_hist = []
        self.unstick_until = 0.0
        self.unstick_dir = (1.0, 0.0)
        self.slow_logged = 0
        self.mode = "init"
        self.laser_hold = 0
        self.take_cooldown = {}

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        try:
            act = self._act(obs)
        except Exception as e:  # never crash: fall back to something sane
            mf.log(f"act error tick {obs.get('tick')}: {e!r}")
            act = self._fallback(obs)
        dt = time.perf_counter() - t0
        if dt > 0.035 and self.slow_logged < 10:
            self.slow_logged += 1
            mf.log(f"slow act {dt * 1000:.1f} ms at tick {obs.get('tick')}")
        return act

    def _fallback(self, obs):
        me = obs["self"]
        x, y = me["pos"][0], me["pos"][1]
        zc = obs["zone"]["next_center"]
        ux, uy = unit(zc[0] - x, zc[1] - y)
        act = {"tick": obs["tick"], "move": [ux, uy], "boost": not me.get("in_zone", True)}
        if obs.get("offer"):
            act["pick"] = pick_upgrade(obs["offer"]["choices"])
        return act

    # ------------------------------------------------------------ utilities
    def _slot_of(self, me, kind):
        for i, w in enumerate(me["weapons"]):
            if w is not None and w["kind"] == kind:
                return i
        return -1

    def _ready(self, w):
        if w is None:
            return False
        if w.get("cooldown", 0) > 0.06:
            return False
        if w["kind"] == "gun" and (w.get("reload", 0) > 0 or w.get("ammo", 1) <= 0):
            return False
        return True

    def free_dir(self, x, y, dx, dy, length=7.0):
        """True if moving along (dx,dy) for `length` m stays off blocked cells (ignores the first 2 m)."""
        if not self.nav:
            return True
        return self.nav.ray_free(x, y, x + dx * length, y + dy * length, skip=2.5, step=1.5)

    def adjust_dir(self, x, y, dx, dy, length=7.0):
        """Return the direction closest to (dx,dy) that doesn't run into a wall."""
        if self.free_dir(x, y, dx, dy, length):
            return dx, dy
        for a in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
            rx, ry = rot(dx, dy, a)
            if self.free_dir(x, y, rx, ry, length):
                return rx, ry
        return dx, dy

    # ----------------------------------------------------------- navigation
    def steer(self, obs, goal, stop=2.0):
        """Unit move vector towards goal using the path planner; None when arrived."""
        me = obs["self"]
        x, y = me["pos"][0], me["pos"][1]
        t = obs["t"]
        gx, gy = goal
        dist = math.hypot(gx - x, gy - y)
        if dist < stop:
            return None
        ux, uy = (gx - x) / dist, (gy - y) / dist
        nav = self.nav
        if nav is None or not nav.ok:
            return ux, uy
        # clear straight shot -> just go
        if dist < 120 and nav.ray_free(x, y, gx, gy, step=1.5, skip=2.0):
            self.path = None
            return ux, uy
        # collect finished plan
        if self.planner:
            r = self.planner.poll()
            if r is not None and r[0] == self.pending_rid:
                self.path = r[1]
                self.path_idx = 0
                self.path_goal = r[2]
                self.path_tight = r[3]
        need = (self.path is None or self.path_goal is None
                or math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 12.0
                or (t - self.path_req_t > 3.0))
        if need and self.planner and t - self.path_req_t > 0.4:
            self.pending_rid = self.planner.request(x, y, gx, gy, self._live_tblock(t))
            self.path_req_t = t
        if not self.path:
            return self.adjust_dir(x, y, ux, uy)
        # follow the path: skip passed waypoints, aim at the farthest one we can walk to directly
        path = self.path
        while self.path_idx < len(path) - 1 and math.hypot(path[self.path_idx][0] - x, path[self.path_idx][1] - y) < 4.0:
            self.path_idx += 1
        tgt = path[self.path_idx]
        for k in ([] if self.path_tight else range(min(len(path) - 1, self.path_idx + 4), self.path_idx, -1)):
            px, py = path[k]
            if math.hypot(px - x, py - y) < 60 and nav.ray_free(x, y, px, py, step=1.5, skip=2.0):
                tgt = path[k]
                self.path_idx = max(self.path_idx, k - 1)
                break
        vx, vy = unit(tgt[0] - x, tgt[1] - y)
        if vx == 0 and vy == 0:
            return ux, uy
        return vx, vy

    def _live_tblock(self, t):
        if not self.tblock:
            return None
        self.tblock = {k: v for k, v in self.tblock.items() if v > t}
        return self.tblock

    def check_stuck(self, obs, moving_cmd):
        """Detect being wedged against something and pick a way out."""
        me = obs["self"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        self.pos_hist.append((t, x, y))
        while self.pos_hist and t - self.pos_hist[0][0] > 2.0:
            self.pos_hist.pop(0)
        if t < self.unstick_until:
            return True
        if not moving_cmd or len(self.pos_hist) < 15 or t - self.pos_hist[0][0] < 1.8:
            return False
        moved = math.hypot(x - self.pos_hist[0][1], y - self.pos_hist[0][2])
        if moved < 1.5:
            self.unstick_until = t + 1.0
            a = self.rng.choice((1.4, -1.4, 2.4, -2.4))
            self.unstick_dir = rot(*(self.last_move or (1.0, 0.0)), a)
            if self.nav and self.last_move:
                mx_, my_ = unit(*self.last_move)
                for dd in (3.0, 4.5, 6.0):
                    i, j = self.nav.to_cell(x + mx_ * dd, y + my_ * dd)
                    self.tblock[j * self.nav.W + i] = t + 20.0
            self.path = None
            self.path_req_t = -99.0
            self.pos_hist = []
            return True
        return False

    # ------------------------------------------------------------------ zone
    def zone_goal(self, obs, me):
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        c, r = z["center"], z["radius"]
        nc, nr = z["next_center"], z["next_radius"]
        pre = t >= z["shrink_start"] - 14.0
        if pre:
            cx, cy, rad = nc[0], nc[1], nr
        else:
            cx, cy, rad = c[0], c[1], r
        safe = max(rad * 0.72, 8.0)
        dist = math.hypot(x - cx, y - cy)
        inside = dist <= safe
        if inside:
            goal = (x, y)
        else:
            ux, uy = unit(x - cx, y - cy)
            if dist < 1e-6:
                ux, uy = 0.0, 0.0
            goal = (cx + ux * safe * 0.8, cy + uy * safe * 0.8)
        urgent = (not me.get("in_zone", True)) or (not inside and t >= z["shrink_start"] - 4.0)
        return goal, inside, urgent, (cx, cy, rad)

    def in_safe_region(self, x, y, t, z):
        """Is a point somewhere we won't get zone-burned soon?"""
        c, r = z["center"], z["radius"]
        nc, nr = z["next_center"], z["next_radius"]
        if t >= z["shrink_start"] - 20.0:
            return math.hypot(x - nc[0], y - nc[1]) < nr * 0.92
        return math.hypot(x - c[0], y - c[1]) < r * 0.92

    # --------------------------------------------------------------- aiming
    def lead(self, src, tpos, tvel, speed, tz_off):
        dx, dy = tpos[0] - src[0], tpos[1] - src[1]
        dz = tpos[2] + tz_off - src[2]
        T = math.sqrt(dx * dx + dy * dy + dz * dz) / max(speed, 1.0)
        px, py, pz = tpos[0], tpos[1], tpos[2]
        for _ in range(3):
            px = tpos[0] + tvel[0] * T
            py = tpos[1] + tvel[1] * T
            pz = tpos[2] + tvel[2] * T
            dx, dy = px - src[0], py - src[1]
            dz = pz + tz_off - src[2]
            T = math.sqrt(dx * dx + dy * dy + dz * dz) / max(speed, 1.0)
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, max(math.hypot(dx, dy), 0.1))
        return yaw, pitch, T

    def mortar_solution(self, src, tpos, tvel, speed, g, tz_off):
        dx, dy = tpos[0] - src[0], tpos[1] - src[1]
        T = math.hypot(dx, dy) / max(speed * 0.7, 1.0)
        for _ in range(4):
            px = tpos[0] + tvel[0] * T
            py = tpos[1] + tvel[1] * T
            pz = tpos[2] + tvel[2] * T
            dx, dy = px - src[0], py - src[1]
            R = math.hypot(dx, dy)
            dz = pz + tz_off - src[2]
            v2 = speed * speed
            disc = v2 * v2 - g * (g * R * R + 2 * dz * v2)
            if disc < 0 or R < 1.0:
                return None
            th = math.atan2(v2 - math.sqrt(disc), g * R)
            T = R / max(speed * math.cos(th), 1.0)
        return math.atan2(dy, dx), th, T

    def attack(self, obs, me, tgt, is_mech, farm=False):
        """Pick aim + trigger state against one target. Returns (aim, fire, info)."""
        mx, my, mz = me["pos"]
        stats = me.get("stats", {})
        muz = stats.get("muzzle_height", self.mech["muzzle_height"])
        src = (mx, my, mz + muz)
        tp = tgt["pos"]
        tv = tgt.get("vel", [0.0, 0.0, 0.0])
        if is_mech:
            toff = 4.5
        elif tgt.get("_crate"):
            toff = 1.2
        else:
            toff = 1.7
        dist = d3(src, (tp[0], tp[1], tp[2] + toff))
        dh = math.hypot(tp[0] - mx, tp[1] - my)
        weapons = me["weapons"]
        gun = weapons[0] if weapons[0] and weapons[0]["kind"] == "gun" else None
        sec = weapons[1]
        cur_yaw, cur_pitch = me["aim"]
        fire = [False, False]
        clear = True
        if self.nav and self.nav.ok:
            clear = self.nav.shot_clear(mx, my, tp[0], tp[1])

        # gun lead (always computed; it is the default aim)
        gspeed = gun["stats"]["speed"] if gun and "stats" in gun else self.w["gun"]["speed"]
        grange = gun["stats"]["range"] if gun and "stats" in gun else self.w["gun"]["range"]
        gy, gp, _ = self.lead(src, tp, tv, gspeed, toff)
        aim = (gy, gp)

        tol = max(0.022, min(0.14, math.atan2(2.1, max(dist, 5.0))))
        info = "gun"

        # secondary weapon
        sec_kind = sec["kind"] if sec else None
        tspeed = math.hypot(tv[0], tv[1])
        if sec_kind == "mortar":
            msp = sec.get("stats", {}).get("speed", self.w["mortar"]["speed"])
            if (self._ready(sec) and 28.0 < dh < 190.0 and (tspeed < 7.0 or dh < 90.0)
                    and not (is_mech and dist < 45 and gun and self._ready(gun))):
                sol = self.mortar_solution((mx, my, mz + muz + 2.5), tp, tv, msp, self.w["mortar"]["gravity"], toff)
                if sol:
                    my_, mp_, _ = sol
                    mp_ = max(self.mech["min_pitch"], min(self.mech["max_pitch"], mp_))
                    aim = (my_, mp_)
                    info = "mortar"
                    ey = abs(angle_diff(cur_yaw, my_))
                    ep = abs(cur_pitch - mp_)
                    if ey < 0.05 and ep < 0.05:
                        fire[1] = True
        elif sec_kind == "lightning":
            lrange = sec.get("stats", {}).get("range", self.w["lightning"]["range"])
            if dh < lrange - 2.0 and self._ready(sec) and me.get("energy", 0) > 6 and clear:
                ey = abs(angle_diff(cur_yaw, math.atan2(tp[1] - my, tp[0] - mx)))
                if ey < 0.2:
                    fire[1] = True
                    info = "lightning"
        elif sec_kind == "laser":
            lrange = sec.get("stats", {}).get("range", self.w["laser"]["range"])
            full = sec.get("stats", {}).get("charge", self.w["laser"]["charge"])
            charge = sec.get("charge", 0.0) or 0.0
            ly, lp, _ = self.lead((mx, my, mz + muz), tp, tv, 1e6, toff)
            # hitscan: aim at where the target will be when the torso gets there
            ly = math.atan2(tp[1] + tv[1] * 0.12 - my, tp[0] + tv[0] * 0.12 - mx)
            lp = math.atan2(tp[2] + toff + tv[2] * 0.12 - (mz + muz), max(dh, 0.1))
            ey = abs(angle_diff(cur_yaw, ly))
            ep = abs(cur_pitch - lp)
            ltol = max(0.02, min(0.10, math.atan2(2.2, max(dist, 5.0))))
            if charge > 0.0 or self.laser_hold > 0:
                # we are charging: release when lined up and charged, else keep holding
                aim = (ly, lp)
                info = "laser"
                self.laser_hold += 1
                aligned = ey < ltol and ep < ltol * 1.5
                if (charge >= 0.9 * full and aligned and clear) or charge >= full * 2.5 or not clear or dist > lrange:
                    fire[1] = False
                    self.laser_hold = 0
                else:
                    fire[1] = True
            elif self._ready(sec) and dist < lrange - 5.0 and clear and is_mech and ey < 0.35:
                aim = (ly, lp)
                info = "laser"
                fire[1] = True
                self.laser_hold = 1

        # gun
        if gun is not None and info != "mortar":
            eyg = abs(angle_diff(cur_yaw, gy))
            epg = abs(cur_pitch - gp)
            if info == "laser" and fire[1]:
                pass  # aim belongs to the laser; gun fires only if it happens to line up
            if self._ready(gun) and dist < grange * 0.95 and clear and eyg < tol and epg < tol * 1.3:
                fire[0] = True
        elif gun is not None and info == "mortar":
            eyg = abs(angle_diff(cur_yaw, gy))
            epg = abs(cur_pitch - gp)
            if self._ready(gun) and dist < grange * 0.95 and clear and eyg < tol and epg < tol * 1.3:
                fire[0] = True
        return aim, fire, info

    # --------------------------------------------------------------- targets
    def pick_target(self, me, mechs):
        best, best_s = None, 1e18
        for m in mechs:
            d = d3(m["pos"], me["pos"])
            s = d + m.get("hp", 1000) * 0.04
            if m["id"] == self.target_id:
                s -= 25.0
            if self.nav and self.nav.ok and not self.nav.shot_clear(me["pos"][0], me["pos"][1], m["pos"][0], m["pos"][1]):
                s += 40.0
            if s < best_s:
                best, best_s = m, s
        return best

    def desired_range(self, me, tgt):
        sec = me["weapons"][1]
        k = sec["kind"] if sec else None
        ew = tgt.get("weapons", []) or []
        if k == "lightning":
            return 70.0 if "mortar" in ew else 30.0
        if k == "laser":
            return 95.0 if "lightning" in ew else 80.0
        if k == "mortar":
            return 55.0
        return 70.0 if "lightning" in ew else 60.0

    # ------------------------------------------------------------------ loot
    def want_pickup(self, me, kind):
        if kind == "gun":
            return False
        held = {w["kind"] for w in me["weapons"] if w is not None}
        if kind in held:
            return False
        sec = me["weapons"][1]
        v = WEAPON_VALUE.get(kind, 0.0)
        if sec is None:
            return v > 0
        sv = WEAPON_VALUE.get(sec["kind"], 0.0)
        if sec["kind"] == "laser" and (sec.get("charge", 0) or 0) > 0:
            return False
        return v > sv + 0.6

    def pick_loot(self, obs, me, hp_frac, enemies_near):
        """Return (kind, (x,y), obj) for the best thing worth walking to, or None."""
        x, y, z = me["pos"]
        t = obs["t"]
        zone = obs["zone"]
        best, best_s = None, 0.0

        def consider(kind, pos, base, maxd, obj, stop):
            nonlocal best, best_s
            dd = math.hypot(pos[0] - x, pos[1] - y)
            if dd > maxd:
                return
            if not self.in_safe_region(pos[0], pos[1], t, zone):
                return
            s = base / (dd + 25.0)
            if s > best_s:
                best_s = s
                best = (kind, (pos[0], pos[1]), obj, stop)

        for p in obs.get("pickups", []):
            if self.want_pickup(me, p["kind"]):
                if abs(p["pos"][2] - z) < 3.5:  # skip rooftops we can't walk to
                    consider("pickup", p["pos"], 120.0 * WEAPON_VALUE.get(p["kind"], 1) / 3.0 + 40, 170.0, p, 1.8)
        for c in obs.get("cubes", []):
            if abs(c["pos"][2] - z) < 3.5:
                consider("cube", c["pos"], 100.0, 110.0, c, 0.8)
        if hp_frac > 0.5:
            for c in obs.get("crates", []):
                if abs(c["pos"][2] - z) < 10.0:
                    c2 = dict(c)
                    c2["_crate"] = True
                    c2["vel"] = [0, 0, 0]
                    consider("crate", c["pos"], 55.0, 70.0, c2, 28.0)
            if hp_frac > 0.6:
                for n in obs["mechs"]:
                    if n["kind"] == "npc":
                        consider("npc", n["pos"], 40.0, 60.0, n, 30.0)
        for dr in obs.get("drops", []):
            if dr["holder"] is not None and dr["holder"] != me["id"]:
                continue
            dd = math.hypot(dr["pos"][0] - x, dr["pos"][1] - y)
            eta = dd / max(self.mech["walk_speed"], 1.0)
            if dd < 220 and eta < dr["lands_in"] + 12 and hp_frac > 0.35:
                crowd = sum(1 for m in obs["mechs"] if m["kind"] == "mech"
                            and math.hypot(m["pos"][0] - dr["pos"][0], m["pos"][1] - dr["pos"][1]) < 40)
                if crowd == 0:
                    consider("drop", dr["pos"], 200.0, 230.0, dr, 1.5)
        return best

    # -------------------------------------------------------------- look dir
    def look_yaw(self, obs, me, t, move):
        if t - self.last_damage_t < 2.5 and self.last_damage_bearing is not None:
            return self.last_damage_bearing
        if self.sound_dir is not None and t - self.sound_t < 1.5:
            return self.sound_dir
        base = me["aim"][0]
        if move is not None and (move[0] or move[1]):
            base = math.atan2(move[1], move[0])
        return base + 1.1 * math.sin(t * 0.9)

    # ============================================================ act proper
    def _act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        dt = max(0.0, t - self.last_t)
        self.last_t = t
        hp, maxhp = me["hp"], me["max_hp"]
        hp_frac = hp / max(maxhp, 1.0)
        energy = me.get("energy", 100)
        act = {"tick": obs["tick"]}
        self.last_move = getattr(self, "last_move", None)

        # ---- events / sounds -> awareness
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged" and ev.get("weapon") != "zone" and ev.get("bearing") is not None:
                self.last_damage_t = t
                self.last_damage_bearing = ev["bearing"]
        for s in obs.get("sounds", []):
            if s.get("kind") in ("gun", "mortar", "laser", "laser_charge", "lightning", "explosion", "step", "boost") \
                    and s.get("dist") in ("near", "mid"):
                self.sound_dir = s["bearing"]
                self.sound_t = t

        if obs.get("offer"):
            act["pick"] = pick_upgrade(obs["offer"]["choices"])

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if mechs:
            self.last_enemy_seen_t = t
        # nearest enemy
        nearest = None
        nd = 1e9
        for m in mechs:
            dd = d3(m["pos"], me["pos"])
            if dd < nd:
                nd, nearest = dd, m

        # ---- weapon pickup (always, cheap)
        slot1 = me["weapons"][1]
        for p in obs.get("pickups", []):
            if self.want_pickup(me, p["kind"]) and d2(p["pos"], me["pos"]) < 3.6 and abs(p["pos"][2] - z) < 3.5:
                if self.take_cooldown.get(p["id"], -9) < t - 0.8:
                    self.take_cooldown[p["id"]] = t
                    act["take"] = p["id"]
                    if slot1 is not None:
                        act["drop"] = 1
                    break

        # ---- reload when idle
        gun = me["weapons"][0]
        if gun and gun["kind"] == "gun":
            if (gun.get("ammo", 99) == 0 and gun.get("reload", 0) == 0) or \
                    (not mechs and gun.get("ammo", 99) < gun.get("stats", {}).get("mag", 12) * 0.6 and gun.get("reload", 0) == 0):
                act["reload"] = True

        zgoal, zinside, zurgent, zinfo = self.zone_goal(obs, me)

        # ---- choose what to do
        target = None
        is_mech = False
        loot = None
        retreat = False
        if mechs:
            target = self.pick_target(me, mechs)
            is_mech = True
            self.target_id = target["id"]
            # retreat if badly hurt and losing the exchange
            if hp_frac < 0.3 and nearest is not None:
                ehp = nearest.get("hp", 1400)
                if ehp > hp * 0.7:
                    retreat = True
        else:
            self.target_id = None

        near_enemy = nd < 100.0 and nearest is not None
        if not near_enemy or nd > 140.0:
            loot = self.pick_loot(obs, me, hp_frac, near_enemy)
        # a cube/weapon right at our feet is worth a detour even mid-fight
        if loot is None and mechs and nd > 40.0:
            l2 = self.pick_loot(obs, me, hp_frac, True)
            if l2 and l2[0] in ("cube", "pickup") and math.hypot(l2[1][0] - x, l2[1][1] - y) < 18.0:
                loot = l2

        # ---- aim / fire
        aim_yaw, aim_pitch = me["aim"]
        fire = [False, False]
        info = ""
        atk_target = None
        atk_is_mech = False
        if target is not None:
            atk_target, atk_is_mech = target, True
        elif loot is not None and loot[0] in ("crate", "npc") and \
                math.hypot(loot[1][0] - x, loot[1][1] - y) < 55:
            atk_target, atk_is_mech = loot[2], False
        elif loot is not None and loot[0] == "cube":
            pass
        if atk_target is not None:
            (aim_yaw, aim_pitch), fire, info = self.attack(obs, me, atk_target, atk_is_mech)
            if gun is not None and self.laser_hold > 0 and not (me["weapons"][1] and me["weapons"][1]["kind"] == "laser"):
                self.laser_hold = 0
            mf.draw.line([x, y, z + 7], [atk_target["pos"][0], atk_target["pos"][1], atk_target["pos"][2] + 3], "#ff4040")
        else:
            self.laser_hold = 0

        # ---- movement
        move = None
        want_boost = False
        mode = "idle"

        if not zinside and zurgent:
            mode = "zone"
            move = self.steer(obs, zgoal, stop=3.0)
            want_boost = (not me.get("in_zone", True)) or energy > 50
        elif retreat and nearest is not None:
            mode = "retreat"
            ex, ey_ = nearest["pos"][0], nearest["pos"][1]
            ax, ay = unit(x - ex, y - ey_)
            gx, gy = unit(zgoal[0] - x, zgoal[1] - y) if not zinside else (0.0, 0.0)
            vx, vy = unit(ax + gx * 0.7, ay + gy * 0.7)
            vx, vy = self.adjust_dir(x, y, vx, vy, 9.0)
            move = (vx, vy)
            want_boost = energy > 25
        elif target is not None:
            mode = "fight"
            tx, ty = target["pos"][0], target["pos"][1]
            dist2 = math.hypot(tx - x, ty - y)
            want = self.desired_range(me, target)
            crowd = sum(1 for m in mechs if d3(m["pos"], me["pos"]) < 150.0)
            if crowd >= 3:
                want = max(want, 85.0)  # don't walk into a 3-on-1: hold range, let them come
            lo, hi = want * 0.75, want * 1.3
            if crowd >= 3:
                hi = 1e9
            toward = unit(tx - x, ty - y)
            if t > self.strafe_until:
                self.strafe_dir = -self.strafe_dir if self.rng.random() < 0.8 else self.strafe_dir
                self.strafe_until = t + 0.7 + self.rng.random() * 1.1
            sx, sy = rot(toward[0], toward[1], PI / 2 * self.strafe_dir)
            if dist2 > 130.0 and crowd < 3:
                st = self.steer(obs, (tx, ty), stop=hi)
                move = st if st else (sx, sy)
                want_boost = energy > 60
            else:
                radial = 0.0
                if dist2 < lo:
                    radial = -1.0
                elif dist2 > hi:
                    radial = 1.0
                vx = sx * 0.9 + toward[0] * radial
                vy = sy * 0.9 + toward[1] * radial
                vx, vy = unit(vx, vy)
                vx, vy = self.adjust_dir(x, y, vx, vy, 7.0)
                if not self.free_dir(x, y, vx, vy, 7.0):
                    self.strafe_dir = -self.strafe_dir
                    vx, vy = rot(sx, sy, PI)
                    vx, vy = self.adjust_dir(x, y, vx, vy, 7.0)
                move = (vx, vy)
                if me["weapons"][1] and me["weapons"][1]["kind"] == "laser" and (me["weapons"][1].get("charge", 0) or 0) > 0:
                    pass
            # keep inside the safe zone while fighting
            if not zinside and zurgent is False:
                zx, zy = unit(zgoal[0] - x, zgoal[1] - y)
                move = unit(move[0] * 0.6 + zx * 0.6, move[1] * 0.6 + zy * 0.6)
            # lightning mains want energy: don't burn it on boosting
            if me["weapons"][1] and me["weapons"][1]["kind"] == "lightning":
                want_boost = False
            else:
                want_boost = dist2 > 110 and energy > 60
        elif loot is not None:
            mode = "loot:" + loot[0]
            lx, ly = loot[1]
            st = self.steer(obs, (lx, ly), stop=loot[3])
            if st is None:
                move = (0.0, 0.0)
            else:
                move = st
                want_boost = energy > 70 and math.hypot(lx - x, ly - y) > 60
        elif not zinside:
            mode = "zone"
            move = self.steer(obs, zgoal, stop=3.0)
            want_boost = energy > 60 and zurgent
        elif hp_frac < 0.85 or (t - self.last_damage_t) < 6.0:
            mode = "heal"
            move = (0.0, 0.0)
        else:
            mode = "hold"
            move = (0.0, 0.0)

        # standing on a supply drop: stay inside the ring until we bank it
        for dr in obs.get("drops", []):
            if dr["lands_in"] <= 0 and math.hypot(dr["pos"][0] - x, dr["pos"][1] - y) < dr["radius"] - 1.5:
                others = [m for m in mechs if math.hypot(m["pos"][0] - dr["pos"][0], m["pos"][1] - dr["pos"][1]) < dr["radius"]]
                if not others and (dr["holder"] in (None, me["id"])) and not retreat:
                    move = (0.0, 0.0)
                    want_boost = False
                    mode = "capture"

        # ---- projectile dodging (only things actually on course for us)
        if move is not None and mode in ("fight", "retreat", "loot:crate", "loot:npc", "idle", "hold", "heal", "loot:cube", "loot:pickup", "zone") \
                and obs.get("projectiles"):
            dvx, dvy, hit = 0.0, 0.0, False
            for pr in obs["projectiles"]:
                px, py, pz = pr["pos"]
                vx, vy, vz = pr["vel"]
                sp2 = vx * vx + vy * vy
                if sp2 < 25:
                    continue
                tc = -((px - x) * vx + (py - y) * vy) / sp2
                if tc < 0 or tc > 0.9:
                    continue
                cx, cy = px + vx * tc - x, py + vy * tc - y
                if math.hypot(cx, cy) < 5.0:
                    sp = math.sqrt(sp2)
                    nx_, ny_ = -vy / sp, vx / sp
                    side = cx * nx_ + cy * ny_
                    sgn = -1.0 if side > 0.3 else (1.0 if side < -0.3 else float(self.strafe_dir))
                    dvx += nx_ * sgn
                    dvy += ny_ * sgn
                    hit = True
            if hit:
                ux, uy = unit(dvx, dvy)
                ux, uy = self.adjust_dir(x, y, ux, uy, 6.0)
                move = unit((move[0] if move else 0.0) * 0.4 + ux * 1.0, (move[1] if move else 0.0) * 0.4 + uy * 1.0)

        # ---- being charged by a laser: break the line
        for m in mechs:
            if m.get("charging") and "laser" in (m.get("weapons") or []):
                ba = math.atan2(y - m["pos"][1], x - m["pos"][0])
                if abs(angle_diff(m["aim"][0], ba)) < 0.12 and d3(m["pos"], me["pos"]) < 280:
                    sx, sy = rot(*unit(x - m["pos"][0], y - m["pos"][1]), PI / 2 * self.strafe_dir)
                    sx, sy = self.adjust_dir(x, y, sx, sy, 8.0)
                    move = (sx, sy)
                    break

        # ---- unstick
        moving_cmd = move is not None and (abs(move[0]) + abs(move[1]) > 0.3)
        stuck = self.check_stuck(obs, moving_cmd and mode not in ("capture",))
        if stuck:
            move = self.unstick_dir
            act["jump"] = bool(me.get("grounded")) and energy > 25
            want_boost = energy > 25

        if move is None:
            move = (0.0, 0.0)
        self.last_move = move if (move[0] or move[1]) else self.last_move

        # ---- look direction when nothing to shoot
        if atk_target is None:
            aim_yaw = self.look_yaw(obs, me, t, move)
            aim_pitch = 0.0

        aim_pitch = max(self.mech["min_pitch"], min(self.mech["max_pitch"], aim_pitch))
        act["aim"] = [aim_yaw, aim_pitch]
        act["fire"] = fire
        act["move"] = [float(move[0]), float(move[1])]
        act["boost"] = bool(want_boost and energy > 8)
        self.mode = mode
        mf.draw.label(f"{mode} hp{hp:.0f} e{energy:.0f} {info}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
