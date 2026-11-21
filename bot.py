"""
Tactical mech bot for Mech Royale.
Strategy: zone-aware survival, lead-aim shooting, weapon pickup, evasion, smart upgrades.
"""
import math
import struct
import json
import random
import mechfight as mf

TAU = math.tau
PI = math.pi

def angle_diff(a, b):
    """Signed difference a - b, wrapped to [-pi, pi]."""
    d = (a - b) % TAU
    return d - TAU if d > PI else d

def vec2_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def vec3_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vec2_norm(v):
    l = vec2_len(v)
    if l < 1e-9:
        return [1.0, 0.0]
    return [v[0]/l, v[1]/l]

def dist2d(a, b):
    return math.dist(a[:2], b[:2])

def dist3d(a, b):
    return math.dist(a[:3], b[:3])

def bearing_to(src, dst):
    return math.atan2(dst[1]-src[1], dst[0]-src[0])

def predict_pos(pos, vel, dt):
    return [pos[0]+vel[0]*dt, pos[1]+vel[1]*dt, pos[2]+vel[2]*dt]

def lead_aim(my_pos, my_vel, target_pos, target_vel, speed):
    """Return (yaw, pitch, travel_time) to intercept a moving target."""
    # Iteratively solve for intercept time
    dt = dist3d(my_pos, target_pos) / max(speed, 1.0)
    for _ in range(4):
        tp = predict_pos(target_pos, target_vel, dt)
        d = dist3d(my_pos, tp)
        dt = d / max(speed, 1.0)

    dx = tp[0] - my_pos[0]
    dy = tp[1] - my_pos[1]
    # Aim at mech center of mass (pos.z + 4.5)
    dz = (tp[2] + 4.5) - (my_pos[2] + 7.0)  # target CoM - my muzzle
    horiz = math.sqrt(dx*dx + dy*dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(horiz, 0.1))
    return yaw, pitch, dt

def mortar_aim(my_pos, target_pos, gravity, speed):
    """Compute mortar yaw+pitch for a ballistic trajectory."""
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    dz = (target_pos[2] + 4.5) - (my_pos[2] + 9.5)
    horiz = math.sqrt(dx*dx + dy*dy)
    yaw = math.atan2(dy, dx)

    if horiz < 1.0:
        return yaw, 0.8

    # Ballistic: z = horiz*tan(theta) - g*horiz^2 / (2*v^2*cos^2(theta))
    # Solve: v^2*sin(2*theta) / g = R  (flat approximation, ignoring dz for initial guess)
    v2 = speed * speed
    g = gravity
    disc = v2*v2 - g*(g*horiz*horiz + 2*dz*v2)
    if disc < 0:
        # Out of range: use 45 degrees
        return yaw, PI/4
    # Two solutions: use the lower angle (faster, more direct)
    pitch = math.atan2(v2 - math.sqrt(disc), g * horiz)
    pitch = max(0.05, min(PI/2 - 0.05, pitch))
    return yaw, pitch


UPGRADE_PRIORITY = [
    "sensors",      # vision + bearing: more info
    "speed",        # mobility: zone survival
    "jet",          # jet boost: rooftop access
    "armor",        # damage reduction
    "gun_rate",     # more DPS
    "gun_damage",
    "laser_damage",
    "mortar_damage",
    "lightning_damage",
    "regen",
    "energy",
    "gun_range",
    "gun_reload",
]

def pick_upgrade(choices, weapons):
    """Choose the best upgrade given current weapons."""
    held = set()
    for w in weapons:
        if w is not None:
            held.add(w["kind"])

    for pref in UPGRADE_PRIORITY:
        for i, c in enumerate(choices):
            cid = c["id"]
            if pref in cid:
                return i
    return 0


class MapData:
    """Preloaded map for navigation hints."""
    def __init__(self, map_dir):
        self.prisms = []
        self.terrain_data = None
        self.terrain_meta = None
        try:
            world = json.load(open(f"{map_dir}/v2/world.json"))
            self.prisms = world.get("prisms", [])
            t = world.get("terrain", {})
            self.terrain_meta = t
            with open(f"{map_dir}/v2/terrain.i16", "rb") as f:
                raw = f.read()
            nx, ny = int(t["nx"]), int(t["ny"])
            self.terrain_data = struct.unpack(f"<{nx*ny}h", raw)
            self.terrain_nx = nx
            self.terrain_ny = ny
            self.terrain_minX = t["minX"]
            self.terrain_minY = t["minY"]
            self.terrain_step = t["step"]
            mf.log(f"Map loaded: {len(self.prisms)} prisms, {nx}x{ny} terrain")
        except Exception as e:
            mf.log(f"Map load failed: {e}")

    def ground_z(self, x, y):
        """Bilinear terrain height at (x, y)."""
        if self.terrain_data is None:
            return 0.0
        t = self.terrain_meta
        nx = self.terrain_nx
        ny = self.terrain_ny
        fi = (x - self.terrain_minX) / self.terrain_step
        fj = (y - self.terrain_minY) / self.terrain_step
        i0 = max(0, min(nx-2, int(fi)))
        j0 = max(0, min(ny-2, int(fj)))
        di = fi - i0
        dj = fj - j0
        def h(i, j):
            return self.terrain_data[j*nx + i] * 0.01
        return (h(i0,j0)*(1-di)*(1-dj) + h(i0+1,j0)*di*(1-dj) +
                h(i0,j0+1)*(1-di)*dj + h(i0+1,j0+1)*di*dj)


class Bot:
    name = "TactBot"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.gun_speed = cfg["weapons"]["gun"]["speed"]
        self.gun_range = cfg["weapons"]["gun"]["range"]
        self.mortar_speed = cfg["weapons"]["mortar"]["speed"]
        self.mortar_gravity = cfg["weapons"]["mortar"]["gravity"]
        self.mortar_range = cfg["weapons"]["mortar"].get("splash_radius", 7.0)
        self.laser_range = cfg["weapons"]["laser"]["range"]
        self.lightning_range = cfg["weapons"]["lightning"]["range"]
        self.pickup_radius = cfg["loot"]["pickup_radius"]
        self.walk_speed = cfg["mech"]["walk_speed"]
        self.boost_speed = cfg["mech"]["boost_speed"]
        self.vision_range = cfg["vision"]["range"]
        self.near_range = cfg["vision"]["near_range"]
        self.map = MapData(hello["map"]["dir"])
        self.rng = random.Random(hello["seed"])
        # State tracking
        self.last_enemies = {}  # id -> last known obs
        self.memory_s = cfg["vision"].get("memory_s", 5.0)
        self.strafe_phase = 0.0
        self.strafe_dir = 1
        self.dodge_until = 0.0
        self.dodge_vec = [1.0, 0.0]
        self.target_id = None
        self.last_t = 0.0
        self.laser_charging = False
        self.laser_charge_start = 0.0
        self.wanted_weapon = None  # "mortar", "laser", "lightning"
        self.weapon_pref = ["lightning", "laser", "mortar"]
        self.boost_ok = True

    def _weapon_slot(self, obs, kind):
        for i, w in enumerate(obs["self"]["weapons"]):
            if w is not None and w["kind"] == kind:
                return i
        return -1

    def _has_weapon(self, obs, kind):
        return self._weapon_slot(obs, kind) >= 0

    def _weapon_ready(self, obs, slot):
        w = obs["self"]["weapons"][slot]
        if w is None:
            return False
        if w["cooldown"] > 0.05:
            return False
        if w["kind"] == "gun":
            if w.get("reload", 0) > 0:
                return False
            if w.get("ammo", 1) == 0:
                return False
        return True

    def _choose_target(self, obs):
        """Pick the best target: weakest visible mech."""
        me = obs["self"]
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if not mechs:
            return None
        # Prefer mechs we've been targeting (momentum), else lowest HP then closest
        def score(m):
            d = dist3d(m["pos"], me["pos"])
            hp = m.get("hp", 1400)
            return hp * 0.5 + d * 0.5
        return min(mechs, key=score)

    def _pick_weapon(self, obs):
        """Find best nearby weapon to pick up."""
        me = obs["self"]
        x, y, z = me["pos"]
        weapons = me["weapons"]

        # What do we have?
        held_kinds = {w["kind"] for w in weapons if w is not None}

        # What do we want?
        for want in self.weapon_pref:
            if want in held_kinds:
                continue
            for p in obs["pickups"]:
                if p["kind"] != want:
                    continue
                h_dist = dist2d(p["pos"], me["pos"])
                v_dist = abs(p["pos"][2] - z)
                if h_dist < self.pickup_radius and v_dist < 4.0:
                    # Which slot to use?
                    if weapons[1] is None:
                        return p["id"], None
                    # Drop slot 1 if it's not the gun and not better
                    slot1_kind = weapons[1]["kind"] if weapons[1] else None
                    if slot1_kind != "gun":
                        return p["id"], 1
            break
        return None, None

    def _zone_move(self, obs):
        """Movement direction to stay in zone."""
        me = obs["self"]
        zone = obs["zone"]
        x, y = me["pos"][:2]
        cx, cy = zone["center"]
        r = zone["radius"]

        dist_to_center = dist2d([x, y], [cx, cy])
        # Move toward next zone center if getting close to edge
        nx_c, ny_c = zone["next_center"]

        if dist_to_center > r * 0.75 or not me["in_zone"]:
            # Head to next zone center
            yaw = bearing_to([x, y], [nx_c, ny_c])
            return [math.cos(yaw), math.sin(yaw)], True  # urgent
        elif dist_to_center > r * 0.5:
            # Drift toward next center
            yaw = bearing_to([x, y], [nx_c, ny_c])
            return [math.cos(yaw)*0.5, math.sin(yaw)*0.5], False
        return None, False

    def _evasion(self, obs):
        """Compute evasion vector from incoming projectiles."""
        me = obs["self"]
        x, y, z = me["pos"]
        evade = [0.0, 0.0]

        for proj in obs["projectiles"]:
            px, py, pz = proj["pos"]
            vx, vy, vz = proj["vel"]
            # Time to closest approach
            dx, dy = px - x, py - y
            speed = vec2_len([vx, vy])
            if speed < 1.0:
                continue
            # Project: will it hit near me?
            t_close = max(0, -(dx*vx + dy*vy) / (vx*vx + vy*vy + 1e-9))
            cx2 = px + vx*t_close - x
            cy2 = py + vy*t_close - y
            close_dist = math.sqrt(cx2*cx2 + cy2*cy2)
            if close_dist < 8.0 and t_close < 1.5:
                # Perpendicular dodge
                norm = max(speed, 1.0)
                evade[0] -= vy / norm
                evade[1] += vx / norm

        # Also dodge based on damage events
        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                b = ev["bearing"]
                # Move perpendicular to damage direction
                evade[0] += math.cos(b + PI/2) * 0.5
                evade[1] += math.sin(b + PI/2) * 0.5

        return evade

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        vx, vy, vz = me["vel"]
        t = obs["t"]
        dt = t - self.last_t
        self.last_t = t

        act: mf.Action = {"tick": obs["tick"]}

        # -- Handle upgrade offer --
        if obs["offer"]:
            act["pick"] = pick_upgrade(obs["offer"]["choices"], me["weapons"])

        # -- Try to pick up a weapon --
        take_id, drop_slot = self._pick_weapon(obs)
        if take_id is not None:
            act["take"] = take_id
            if drop_slot is not None:
                act["drop"] = drop_slot

        # -- Gun reload --
        gun_slot = self._weapon_slot(obs, "gun")
        if gun_slot >= 0:
            gun = me["weapons"][gun_slot]
            if gun and gun.get("ammo", 12) == 0 and gun.get("reload", 0) == 0:
                act["reload"] = True

        # -- Identify target --
        target = self._choose_target(obs)

        # -- Aim and fire --
        aim_yaw = me["aim"][0]
        aim_pitch = me["aim"][1]
        fire = [False, False]

        # Update enemy memory
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                self.last_enemies[m["id"]] = (m, t)

        if target:
            tid = target["id"]
            tp = target["pos"]
            tv = target["vel"]
            td = dist3d(me["pos"], tp)

            # Determine best weapon to use
            best_slot = -1
            best_aim = None

            # Slot 0 is gun, slot 1 is secondary
            weapons = me["weapons"]

            # Try secondary weapon first if advantageous
            if weapons[1] is not None:
                sec = weapons[1]
                sec_kind = sec["kind"]

                if sec_kind == "lightning" and td < self.lightning_range * 0.9:
                    # Lightning: aim and fire if cooldown ready
                    if self._weapon_ready(obs, 1):
                        yaw, pitch, _ = lead_aim(me["pos"], [vx,vy,0], tp, tv, 999.0)
                        best_slot = 1
                        best_aim = (yaw, pitch)
                        fire[1] = True

                elif sec_kind == "laser" and td < self.laser_range:
                    yaw, pitch, _ = lead_aim(me["pos"], [vx,vy,0], tp, tv, 999.0)
                    if self.laser_charging:
                        charge_t = t - self.laser_charge_start
                        # Fire when fully charged or min charge met
                        min_charge = self.cfg["weapons"]["laser"].get("min_charge", 0.15)
                        max_charge = self.cfg["weapons"]["laser"].get("charge", 0.75)
                        if charge_t >= max_charge or (charge_t >= min_charge and td < self.laser_range * 0.5):
                            fire[1] = False  # let go = fire: stop holding
                            self.laser_charging = False
                        else:
                            fire[1] = True  # keep holding
                    else:
                        # Start charging
                        if self._weapon_ready(obs, 1):
                            fire[1] = True
                            self.laser_charging = True
                            self.laser_charge_start = t
                    best_slot = 1
                    best_aim = (yaw, pitch)

                elif sec_kind == "mortar" and td > 25.0 and td < 220.0:
                    yaw, pitch = mortar_aim(me["pos"], tp, self.mortar_gravity, self.mortar_speed)
                    if self._weapon_ready(obs, 1):
                        best_slot = 1
                        best_aim = (yaw, pitch)
                        fire[1] = True

            # Gun as fallback or primary
            if best_slot == -1 and td < self.gun_range:
                yaw, pitch, flight = lead_aim(me["pos"], [vx,vy,0], tp, tv, self.gun_speed)
                best_aim = (yaw, pitch)
                if self._weapon_ready(obs, 0):
                    fire[0] = True
            elif best_slot == -1 and best_aim is None:
                # Just aim at target even if out of range
                yaw, pitch, _ = lead_aim(me["pos"], [vx,vy,0], tp, tv, self.gun_speed)
                best_aim = (yaw, pitch)

            if best_aim:
                aim_yaw, aim_pitch = best_aim

            mf.draw.line([x, y, z+7], [tp[0], tp[1], tp[2]+4.5], "#ff4040")
        else:
            # No visible target: scan by turning
            aim_yaw += 0.3

        # Clamp pitch
        min_p = self.cfg["mech"].get("min_pitch", -0.6)
        max_p = self.cfg["mech"].get("max_pitch", 0.6)
        aim_pitch = max(min_p, min(max_p, aim_pitch))
        act["aim"] = [aim_yaw, aim_pitch]
        act["fire"] = fire

        # -- Movement --
        zone_move, urgent = self._zone_move(obs)
        evade = self._evasion(obs)

        move = [0.0, 0.0]

        if urgent:
            move = zone_move
        else:
            # Strafe around target while moving toward zone
            if target:
                tx, ty = target["pos"][:2]
                to_target_yaw = bearing_to([x, y], [tx, ty])
                td2d = dist2d([x, y], [tx, ty])

                # Optimal engagement distance: not too close (mortar splash), not too far
                optimal_dist = 60.0
                if td2d < optimal_dist * 0.6:
                    # Back up
                    move[0] = math.cos(to_target_yaw + PI) * 0.8
                    move[1] = math.sin(to_target_yaw + PI) * 0.8
                elif td2d > optimal_dist * 1.4:
                    # Close in
                    move[0] = math.cos(to_target_yaw) * 0.8
                    move[1] = math.sin(to_target_yaw) * 0.8
                else:
                    # Strafe perpendicular
                    self.strafe_phase += dt * 1.5
                    if self.strafe_phase > PI:
                        self.strafe_phase -= TAU
                        self.strafe_dir *= -1
                    strafe_yaw = to_target_yaw + PI/2 * self.strafe_dir
                    move[0] = math.cos(strafe_yaw)
                    move[1] = math.sin(strafe_yaw)
            elif zone_move:
                move = zone_move
            else:
                # Wander toward zone center
                zone = obs["zone"]
                cx2, cy2 = zone["next_center"]
                y2 = bearing_to([x, y], [cx2, cy2])
                move = [math.cos(y2)*0.5, math.sin(y2)*0.5]

        # Apply evasion on top
        evade_len = vec2_len(evade)
        if evade_len > 0.1:
            move[0] += evade[0] / evade_len * 0.6
            move[1] += evade[1] / evade_len * 0.6

        # Normalize
        ml = vec2_len(move)
        if ml > 1.0:
            move = [move[0]/ml, move[1]/ml]

        act["move"] = move

        # -- Boost when fleeing zone or chasing --
        energy = me["energy"]
        should_boost = False
        if not me["in_zone"]:
            should_boost = True
        elif urgent and energy > 30:
            should_boost = True
        elif target and dist3d(me["pos"], target["pos"]) > 80 and energy > 50:
            should_boost = True
        act["boost"] = should_boost

        # -- Jump occasionally to get unstuck or onto ledges --
        if me["grounded"] and obs["tick"] % 50 == 0 and energy > 20:
            act["jump"] = True

        # -- Supply drop capture --
        for drop in obs["drops"]:
            if drop["lands_in"] == 0:
                dd = dist2d([x, y], drop["pos"][:2])
                if dd < drop["radius"]:
                    # Stay put to capture; override movement
                    if drop["holder"] == me["id"] or drop["holder"] is None:
                        # Is anyone else contesting it?
                        contesters = [m for m in obs["mechs"] if m["kind"] == "mech" and dist2d(m["pos"][:2], drop["pos"][:2]) < drop["radius"]]
                        if not contesters:
                            act["move"] = [0.0, 0.0]
                            act["boost"] = False

        # -- Cube pickup --
        if obs["cubes"]:
            nearest_cube = min(obs["cubes"], key=lambda c: dist2d(c["pos"], me["pos"]))
            cd = dist2d(nearest_cube["pos"], me["pos"])
            cube_pickup_r = self.cfg["cubes"].get("pickup_radius", 3.5)
            if cd < cube_pickup_r * 3 and not target:
                # Walk to cube
                cy_b = bearing_to([x, y], nearest_cube["pos"][:2])
                act["move"] = [math.cos(cy_b), math.sin(cy_b)]

        # Debug
        zone = obs["zone"]
        mf.draw.label(f"hp{me['hp']:.0f} e{me['energy']:.0f} {'[+]' if target else ''}")
        mf.draw.point([zone["next_center"][0], zone["next_center"][1], z], "#40c0ff", 6)

        return act


mf.run(Bot)
