"""Toy 5v5 match simulator to compare game plans head-to-head.

NOT the real AWS engine — a simple physics/probability model (movement speeds,
tackles near the carrier, pass interceptions along the passing lane, shot
success by distance and keeper position). Good for catching broken tactics and
comparing plans relative to each other; absolute scores mean little.
"""
import sys, os, math, random, importlib.util, copy

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "lib"))
import state as st
import fallback as fb

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

v2 = load_module("tactics_v2", os.path.join(REPO, "lib", "tactics.py"))
v1 = load_module("tactics_v1_sim", os.path.join(REPO, "lib", "tactics_v1.py"))

ROLES = ["GK", "DEF", "MID", "FWD1", "FWD2"]
CFG = {"GK": fb.GK_CONFIG, "DEF": fb.DEF_CONFIG, "MID": fb.MID_CONFIG, "FWD1": fb.FWD1_CONFIG, "FWD2": fb.FWD2_CONFIG}
FB = {r: fb.build_fallback(c) for r, c in CFG.items()}
AGG = {r: fb.build_fallback(copy.replace(c, press_distance=60, shoot_threshold=30, press_intensity=1.0)
                            if r != "GK" else c) for r, c in CFG.items()}

def pol_sample(gs, tid, pid, role):          # the sample team's rule layer (proxy for its LLM)
    return FB[role](gs, tid, pid)

def pol_aggressive(gs, tid, pid, role):      # everyone presses, shoots early
    return AGG[role](gs, tid, pid)

def pol_v1(gs, tid, pid, role):              # our first upgrade: instant rules + sample fallback off-ball
    fn = {"GK": v1.gk_rules, "DEF": v1.def_rules, "MID": v1.mid_rules}.get(role, v1.fwd_rules)
    out = fn(gs, tid, pid)
    return out if out else FB[role](gs, tid, pid)

def pol_v2(gs, tid, pid, role):              # adaptive game plan (LLM assumed to follow the plan)
    cmds, _ = v2.decide(gs, tid, pid, role)
    return cmds or FB[role](gs, tid, pid)

def pol_defensive(gs, tid, pid, role):       # park the bus: everyone behind the ball, counter
    c = v2.Ctx(gs, tid, pid)
    if role == "GK" or c.i_have:
        return FB[role](gs, tid, pid)
    if c.free and v2.dist(c.pos, c.ball.get("position", {})) < 12:
        return [v2._cmd("INTERCEPT", pid, tid, {"aggressive": True})]
    if c.they_have and v2.dist(c.pos, c.ball.get("position", {})) < 8:
        return [v2._cmd("PRESS_BALL", pid, tid, {"intensity": 0.8})]
    X = {"DEF": -40, "MID": -30, "FWD1": -20, "FWD2": -15}[role]
    return v2._move(c, X, c.bY * 0.5 + {"DEF": 0, "MID": -6, "FWD1": 6, "FWD2": -4}[role])

POLICIES = {"sample": pol_sample, "aggressive": pol_aggressive, "defensive": pol_defensive,
            "v1": pol_v1, "v2": pol_v2}

# ---------------------------------------------------------------------------
KICKOFF = {"GK": (-50, 0), "DEF": (-30, 0), "MID": (-12, 0), "FWD1": (-3, -8), "FWD2": (-3, 8)}

class Match:
    def __init__(self, home, away, seed, decision_s=2.0, duration_s=180):
        self.rng = random.Random(seed); random.seed(seed)
        self.pol = [POLICIES[home], POLICIES[away]]
        self.score = [0, 0]; self.t = 0.0; self.dt = 0.1
        self.decision_s, self.duration_s = decision_s, duration_s
        self.stats = {"shots": [0, 0], "passes": [0, 0], "pass_ok": [0, 0], "tackles": [0, 0]}
        self.kickoff(0)

    def kickoff(self, team_with_ball):
        self.pos, self.stam, self.cmd = {}, {}, {}
        for t in (0, 1):
            f = 1 if t == 0 else -1
            for i, r in enumerate(ROLES):
                x, y = KICKOFF[r]
                self.pos[(t, i)] = [x * f, y]; self.stam[(t, i)] = self.stam.get((t, i), 1.0)
        self.holder = (team_with_ball, 2); self.ball = [0.0, 0.0]
        self.pos[(team_with_ball, 2)] = [0.0, 0.0]

    def gs(self):
        players = []
        for t in (0, 1):
            for i in range(5):
                players.append({"agentId": f"agentId_{i}", "teamCode": "home" if t == 0 else "away",
                                "position": {"x": self.pos[(t, i)][0], "y": self.pos[(t, i)][1]},
                                "stamina": self.stam[(t, i)]})
        ball = {"position": {"x": self.ball[0], "y": self.ball[1]}, "isFree": self.holder is None}
        if self.holder:
            ball["possessionAgentId"] = f"agentId_{self.holder[1]}"   # ambiguous on purpose, like the server
        return {"gameTime": self.t, "score": {"home": self.score[0], "away": self.score[1]},
                "ball": ball, "players": players}

    # --- physics helpers ---
    def step_to(self, k, tx, ty, sprint):
        speed = (8.0 if sprint and self.stam[k] > 0.15 else 5.5) * (0.85 if self.holder == k else 1.0)
        if sprint: self.stam[k] = max(0.0, self.stam[k] - 0.004)
        else: self.stam[k] = min(1.0, self.stam[k] + 0.0015)
        x, y = self.pos[k]; dx, dy = tx - x, ty - y; d = math.hypot(dx, dy)
        if d < 1e-6: return
        s = min(d, speed * self.dt)
        self.pos[k] = [max(-55, min(55, x + dx / d * s)), max(-34, min(34, y + dy / d * s))]
        if self.holder == k: self.ball = list(self.pos[k])

    def resolve_pass(self, k, target_idx, ptype):
        t = k[0]; tgt = (t, target_idx)
        if tgt == k or tgt not in self.pos: return
        self.stats["passes"][t] += 1
        sx, sy = self.pos[k]; ex, ey = self.pos[tgt]; L = math.hypot(ex - sx, ey - sy)
        for o in range(5):
            ox, oy = self.pos[(1 - t, o)]
            if ptype == "AERIAL":
                dline = math.hypot(ox - ex, oy - ey) + 0.5   # only contest at the landing spot
            else:
                u = max(0, min(1, ((ox - sx) * (ex - sx) + (oy - sy) * (ey - sy)) / max(L * L, 1e-6)))
                dline = math.hypot(ox - (sx + u * (ex - sx)), oy - (sy + u * (ey - sy)))
            if dline < 2.5 and self.rng.random() < (0.55 if ptype != "AERIAL" else 0.4):
                self.holder = (1 - t, o); self.ball = list(self.pos[(1 - t, o)]); return
        ok = 0.97 - L * 0.006 - (0.08 if ptype == "AERIAL" else 0)
        if self.rng.random() < ok:
            self.stats["pass_ok"][t] += 1
            self.holder = tgt; self.ball = list(self.pos[tgt])
        else:
            self.holder = None
            self.ball = [ex + self.rng.uniform(-5, 5), ey + self.rng.uniform(-5, 5)]

    def resolve_shot(self, k, aim):
        t = k[0]; f = 1 if t == 0 else -1; self.stats["shots"][t] += 1
        sx, sy = self.pos[k]; d = math.hypot(55 - sx * f, sy)
        p = max(0.02, min(0.6, 0.62 - 0.02 * max(0, d - 8)))
        blockers = sum(1 for o in range(1, 5) if (self.pos[(1 - t, o)][0] * f) > sx * f
                       and abs(self.pos[(1 - t, o)][1] - sy) < 3 and abs(self.pos[(1 - t, o)][0] - sx) < 10)
        p *= 0.6 ** blockers
        gx, gy = self.pos[(1 - t, 0)]
        aim_y = 2.5 if aim in ("TL", "BL") else -2.5 if aim in ("TR", "BR") else 0
        if abs(gx * f - 55) < 6 and abs(gy - aim_y) < 2.5: p *= 0.45
        if aim in ("BL", "BR"): p *= 1.05
        if self.rng.random() < p:
            self.score[t] += 1; self.kickoff(1 - t)
        else:
            self.holder = (1 - t, 0); self.ball = list(self.pos[(1 - t, 0)])

    def tick_decisions(self):
        g = self.gs()
        for t in (0, 1):
            for i, r in enumerate(ROLES):
                try:
                    cmds = self.pol[t](g, t, i, r) or []
                except Exception as e:
                    raise RuntimeError(f"policy error team {t} {r}: {e}")
                self.cmd[(t, i)] = cmds[0] if cmds else None
        # one-shot actions resolve immediately (in random order)
        keys = list(self.cmd); self.rng.shuffle(keys)
        for k in keys:
            c = self.cmd.get(k)
            if not c or self.holder != k: continue
            ct, p = c["commandType"], c.get("parameters", {})
            if ct == "SHOOT": self.resolve_shot(k, p.get("aim_location", "CENTER")); self.cmd[k] = None
            elif ct in ("PASS", "GK_DISTRIBUTE"):
                self.resolve_pass(k, p.get("target_player_id", 1),
                                  "AERIAL" if p.get("method") == "KICK" else p.get("type", "GROUND"))
                self.cmd[k] = None

    def physics(self):
        for k, c in self.cmd.items():
            if not c: continue
            ct, p = c["commandType"], c.get("parameters", {})
            t = k[0]; f = 1 if t == 0 else -1
            if ct == "MOVE_TO":
                self.step_to(k, p.get("target_x", 0), p.get("target_y", 0), p.get("sprint", False))
            elif ct in ("PRESS_BALL", "INTERCEPT", "SLIDE_TACKLE"):
                self.step_to(k, self.ball[0], self.ball[1], True)
            elif ct == "MARK":
                o = (1 - t, p.get("target_player_id", 1)); ox, oy = self.pos[o]
                self.step_to(k, ox - 2 * f, oy, True)
        # possession contests
        if self.holder is None:
            near = [(math.hypot(self.pos[k][0] - self.ball[0], self.pos[k][1] - self.ball[1]), k) for k in self.pos]
            d, k = min(near)
            if d < 1.2: self.holder = k; self.ball = list(self.pos[k])
        else:
            h = self.holder
            for o in range(5):
                k = (1 - h[0], o)
                c = self.cmd.get(k)
                if math.hypot(self.pos[k][0] - self.ball[0], self.pos[k][1] - self.ball[1]) < 1.6:
                    rate = 0.10 if c and c["commandType"] in ("PRESS_BALL", "SLIDE_TACKLE", "INTERCEPT") else 0.03
                    if self.rng.random() < rate:
                        self.stats["tackles"][k[0]] += 1
                        self.holder = None
                        self.ball = [self.ball[0] + self.rng.uniform(-3, 3), self.ball[1] + self.rng.uniform(-3, 3)]
                        break

    def play(self):
        next_dec = 0.0
        while self.t < self.duration_s:
            if self.t >= next_dec:
                self.tick_decisions(); next_dec += self.decision_s
            self.physics(); self.t += self.dt
        return self.score

def series(a, b, n=60):
    w = d = l = gf = ga = 0; shots = [0, 0]
    for s in range(n):
        if s % 2 == 0: m = Match(a, b, s); sa, sb = m.play(); sh = m.stats["shots"]
        else: m = Match(b, a, s); sb, sa = m.play(); sh = m.stats["shots"][::-1]
        gf += sa; ga += sb; shots[0] += sh[0]; shots[1] += sh[1]
        w += sa > sb; d += sa == sb; l += sa < sb
    return w, d, l, gf / n, ga / n, shots[0] / n, shots[1] / n

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    for ours in ("sample", "v1", "v2"):
        for opp in ("sample", "aggressive", "defensive"):
            w, d, l, gf, ga, sf, sa = series(ours, opp, n)
            print(f"{ours:7s} vs {opp:10s}  W{w:3d} D{d:3d} L{l:3d}  goals {gf:4.2f}-{ga:4.2f}  shots {sf:4.1f}-{sa:4.1f}")
