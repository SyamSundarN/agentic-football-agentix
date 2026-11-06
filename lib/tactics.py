"""Team "Agentix" game plan — adaptive tactics shared by all 5 players.

Every player reads the same game state, so every player computes the same
team MODE each tick and the team shifts shape together without talking:

  COUNTER   opponent has 3+ outfielders in our half (aggressive / high press)
            -> stay compact, win it, hit long balls to the striker left high
  ATTACK    we are losing -> push everyone up, shoot from further out
  PROTECT   we lead by 2+ -> extra defender, keep shape
  BALANCED  default

decide() returns (commands, instant):
  instant=True  -> send immediately (on-ball, loose ball, goalkeeper, emergencies)
  instant=False -> a positioning suggestion; the LLM sees it and may adjust it,
                   and it is used as-is if the LLM is slow or fails.
"""

from __future__ import annotations
import math
import random

import tactics_v1
from fallback import build_fallback, GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG
from state import get_possession_info, holder_is_me, dist, _player_idx, _is_my_team

GOAL_X = 55.0
FIELD_Y = 33.0
OPEN_RADIUS = 6.0
PRESSURE_RADIUS = 5.0
SHOOT_RANGE = 28.0          # always shoot inside this distance (30 when chasing the game)
BETTER_MARGIN = 6.0         # pass if a free teammate is this much closer to goal
FWD_PASS_WHEN_BETTER = False# forwards also play the simple pass to a better-placed teammate
FWD_HOLD_X = 22.0           # BALANCED: where forwards wait when they have the ball
FWD_PRESS_R = 20.0          # BALANCED: forwards press a carrier this close
FWD_RUN_X = 30.0            # where forwards run when we have the ball
MID_PRESS_R = 13.0
BALANCED_USES_CLASSIC = True   # normal play = the v1 plan that beat Benchmark FC
COUNTER_LINE = 10.0
BLOCKED_PASS = True         # pass instead of dribbling into a defender standing in front


# ---------------------------------------------------------------------------
# Helpers — "attack frame": X > 0 always means toward the opponent goal
# ---------------------------------------------------------------------------

def _cmd(t, pid, tid, params, duration=0):
    return {"commandType": t, "playerId": pid, "teamId": tid,
            "parameters": params, "duration": duration}


class Ctx:
    def __init__(self, gs, team_id, pid):
        self.tid, self.pid = team_id, pid
        self.f = 1 if team_id == 0 else -1
        self.ball = gs.get("ball", {})
        self.players = gs.get("players", [])
        bp = self.ball.get("position", {"x": 0, "y": 0})
        self.bX, self.bY = self.X(bp), bp.get("y", 0)
        mates = [p for p in self.players if _is_my_team(p, team_id)]
        self.opps = [p for p in self.players if not _is_my_team(p, team_id)]
        self.me = next((p for p in mates if _player_idx(p) == pid), None)
        self.mates = [p for p in mates if _player_idx(p) != pid]
        self.pos = (self.me or {}).get("position", {"x": 0, "y": 0})
        self.mX, self.mY = self.X(self.pos), self.pos.get("y", 0)
        self.stam = (self.me or {}).get("stamina", 1.0)
        _, status, self.we_have = get_possession_info(self.ball, self.players, team_id)
        self.free = status == "free"
        self.they_have = not self.we_have and not self.free and status != "unknown"
        self.i_have = holder_is_me(self.ball, self.players, team_id, pid)
        score = gs.get("score", {})
        home, away = score.get("home", 0), score.get("away", 0)
        self.diff = (home - away) if team_id == 0 else (away - home)
        self.mode = self._mode()

    def X(self, pos):
        return pos.get("x", 0) * self.f

    def world(self, X, Y):
        X = max(-GOAL_X + 2, min(GOAL_X - 2, X))
        Y = max(-FIELD_Y, min(FIELD_Y, Y))
        return round(X * self.f, 1), round(Y, 1)

    def _mode(self):
        if self.diff < 0:
            return "ATTACK"
        if self.diff >= 2:
            return "PROTECT"
        opp_in_our_half = sum(1 for o in self.opps
                              if _player_idx(o) != 0 and self.X(o.get("position", {})) < COUNTER_LINE)
        return "COUNTER" if opp_in_our_half >= 3 else "BALANCED"

    def nearest_opp_dist(self, p):
        pos = p.get("position", {})
        return min((dist(pos, o.get("position", {})) for o in self.opps), default=99.0)

    def nearest_opp_to_me(self):
        return min(self.opps, key=lambda o: dist(self.pos, o.get("position", {})), default=None)

    def dist_goal(self, pos):
        return math.hypot(GOAL_X - self.X(pos), pos.get("y", 0))

    def closest_outfielder_to_ball(self):
        bp = self.ball.get("position", {"x": 0, "y": 0})
        cands = [p for p in self.players if _is_my_team(p, self.tid) and _player_idx(p) != 0]
        return min(cands, key=lambda p: (dist(p.get("position", {}), bp), _player_idx(p)), default=None)

    def opp_attackers_by_danger(self):
        """Opponent outfielders sorted by closeness to our goal."""
        outs = [o for o in self.opps if _player_idx(o) != 0]
        return sorted(outs, key=lambda o: self.X(o.get("position", {})))


# ---------------------------------------------------------------------------
# On-ball decisions (instant)
# ---------------------------------------------------------------------------

def _shoot(c):
    gk = next((o for o in c.opps if _player_idx(o) == 0), None)
    gk_y = (gk or {}).get("position", {}).get("y", 0)
    aims = ["BL", "BR"] if abs(gk_y) < 1.5 else ["BL", "BR", "TL", "TR"]
    return [_cmd("SHOOT", c.pid, c.tid, {"aim_location": random.choice(aims), "power": 0.95})]


def _pass_score(c, p):
    pp = p.get("position", {})
    openness = min(c.nearest_opp_dist(p), 15.0)
    progress = c.X(pp) - c.mX
    length = dist(pp, c.pos)
    if length > 45 or openness < 2.5:
        return None
    score = openness * 1.5 + progress * 0.8 - length * 0.15
    if c.mode == "COUNTER":
        score += progress * 0.5          # favour the ball over the press
    return score


def _best_pass(c, exclude_gk=True):
    best, best_s = None, -1e9
    for p in c.mates:
        if exclude_gk and _player_idx(p) == 0:
            continue
        s = _pass_score(c, p)
        if s is not None and s > best_s:
            best, best_s = p, s
    return best


def _pass(c, target):
    tp = target.get("position", {})
    ahead = c.X(tp) - c.mX
    length = dist(tp, c.pos)
    if length > 28 and c.mX < 0:
        ptype = "AERIAL"                 # long ball out of our half, over the press
    elif ahead > 10:
        ptype = "THROUGH"
    else:
        ptype = "GROUND"
    return [_cmd("PASS", c.pid, c.tid, {"target_player_id": _player_idx(target), "type": ptype})]


def _dribble(c):
    # Head for goal, sidestepping the nearest opponent in front of us.
    ty = c.mY * 0.5
    o = c.nearest_opp_to_me()
    if o:
        op = o.get("position", {})
        if 0 < c.X(op) - c.mX < 8 and abs(op.get("y", 0) - c.mY) < 5:
            ty = c.mY + (6 if op.get("y", 0) <= c.mY else -6)
    tx, ty = c.world(GOAL_X - 12, ty)
    return [_cmd("MOVE_TO", c.pid, c.tid, {"target_x": tx, "target_y": ty, "sprint": c.stam > 0.25})]


def _on_ball(c, role):
    shoot_range = SHOOT_RANGE + (4.0 if c.mode == "ATTACK" else 0.0)
    d_goal = c.dist_goal(c.pos)
    pressured = c.nearest_opp_dist(c.me) < PRESSURE_RADIUS

    if role in ("MID", "FWD") and d_goal < shoot_range:
        return _shoot(c)

    target = _best_pass(c)
    if role == "DEF" or (c.mX < -20):
        # Our defensive third: never dribble — move it on (long if they press).
        if target:
            return _pass(c, target)
        return _dribble(c)

    if target:
        t_open = c.nearest_opp_dist(target) > OPEN_RADIUS
        t_better = c.dist_goal(target.get("position", {})) < d_goal - BETTER_MARGIN
        blocked = BLOCKED_PASS and _blocked_ahead(c)
        if t_open and (pressured or blocked
                       or (t_better and (role == "MID" or FWD_PASS_WHEN_BETTER))
                       or (c.mode == "COUNTER" and c.X(target.get("position", {})) > c.mX + 10)):
            return _pass(c, target)
    return _dribble(c)


def _blocked_ahead(c):
    for o in c.opps:
        op = o.get("position", {})
        if 0 < c.X(op) - c.mX < 7 and abs(op.get("y", 0) - c.mY) < 4:
            return True
    return False


# ---------------------------------------------------------------------------
# Off-ball positioning (suggestions)
# ---------------------------------------------------------------------------

def _move(c, X, Y, sprint=None):
    tx, ty = c.world(X, Y)
    far = dist(c.pos, {"x": tx, "y": ty}) > 8
    sp = (far and c.stam > 0.3) if sprint is None else (sprint and c.stam > 0.3)
    return [_cmd("MOVE_TO", c.pid, c.tid, {"target_x": tx, "target_y": ty, "sprint": sp})]


def _press(c, intensity=0.9):
    return [_cmd("PRESS_BALL", c.pid, c.tid, {"intensity": intensity}, duration=2)]


def _mark(c, o):
    return [_cmd("MARK", c.pid, c.tid, {"target_player_id": _player_idx(o), "tightness": "TIGHT"}, duration=2)]


def _def_off(c):
    d_ball = dist(c.pos, c.ball.get("position", {}))
    if c.they_have:
        if c.bX < 0 and d_ball < 9:
            return _press(c, 1.0), True                       # danger: engage now
        attackers = c.opp_attackers_by_danger()
        if attackers and c.X(attackers[0].get("position", {})) < -25:
            return _mark(c, attackers[0]), False
        X = max(-42, min(c.bX - 10, -12))
        return _move(c, X, c.bY * 0.5), False
    # We have it / loose
    top = 0 if c.mode == "ATTACK" else -15
    return _move(c, min(c.bX - 22, top), c.bY * 0.3, sprint=False), False


def _mid_off(c):
    d_ball = dist(c.pos, c.ball.get("position", {}))
    if c.they_have:
        if d_ball < MID_PRESS_R:
            return _press(c, 0.9), False
        X = c.bX - 7 if c.mode != "ATTACK" else c.bX - 3
        if c.mode in ("COUNTER", "PROTECT"):
            X = min(X, -5)
        return _move(c, X, c.bY * 0.5), False
    # Support: behind-and-beside the ball for a safe pass, on the open side.
    side = -8 if c.bY > 0 else 8
    X = c.bX - 6 if c.mode != "ATTACK" else c.bX + 4
    return _move(c, X, c.bY + side), False


def _fwd_off(c, lane):
    d_ball = dist(c.pos, c.ball.get("position", {}))
    striker = lane < 0          # FWD1 = left lane = the outlet striker
    if c.we_have:
        # Runs into the channels, away from the nearest defender.
        X = 37 if c.mode != "PROTECT" else 30
        Y = lane
        o = c.nearest_opp_to_me()
        if o and dist(c.pos, o.get("position", {})) < 6:
            Y = lane + (5 if o.get("position", {}).get("y", 0) < c.mY else -5)
        return _move(c, X, Y, sprint=True), False

    if c.they_have:
        if c.mode == "ATTACK" or (c.bX > 0 and d_ball < 11):
            return _press(c, 1.0), False                    # high press in their half

        if c.mode in ("COUNTER", "PROTECT") and not striker:
            # FWD2 drops in as an extra defender: mark the 2nd most dangerous attacker.
            attackers = c.opp_attackers_by_danger()
            if len(attackers) >= 2 and c.X(attackers[1].get("position", {})) < 10:
                return _mark(c, attackers[1]), False
            return _move(c, min(c.bX - 6, -8), lane * 0.5), False
        # Striker waits on the halfway line for the counter-attack ball.
        return _move(c, 2 if striker else max(c.bX + 6, -5), lane, sprint=False), False

    return _move(c, max(c.bX, 0), lane), False


def _gk(c):
    if c.i_have:
        t = _best_pass(c) or (c.mates[0] if c.mates else None)
        tid_ = _player_idx(t) if t else 1
        far = t is not None and dist(t.get("position", {}), c.pos) > 25
        return [_cmd("GK_DISTRIBUTE", c.pid, c.tid,
                     {"target_player_id": tid_, "method": "KICK" if far else "THROW"})]
    ball_to_goal = c.bX + GOAL_X
    if c.free and ball_to_goal < 15 and abs(c.bY) < 15:
        return [_cmd("INTERCEPT", c.pid, c.tid, {"aggressive": True}, duration=2)]
    if c.they_have and ball_to_goal < 12 and dist(c.pos, c.ball.get("position", {})) < 9:
        return _press(c, 1.0)
    # Narrow the angle: step off the line as the ball approaches.
    step = 4.5 if ball_to_goal > 35 else 5.5
    return _move(c, -GOAL_X + step, max(-5, min(5, c.bY * 0.35)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide(game_state, team_id, pid, role):
    """Return (commands, instant) or (None, False) if the player is missing."""
    c = Ctx(game_state, team_id, pid)
    if not c.me:
        return None, False
    if c.mode == "BALANCED" and BALANCED_USES_CLASSIC:
        r = _V1[role](game_state, team_id, pid)
        if r:
            return r, True
        return _CLASSIC[role](game_state, team_id, pid), False
    if role == "GK":
        return _gk(c), True
    if c.i_have:
        return _on_ball(c, role), True
    if c.free:
        closest = c.closest_outfielder_to_ball()
        if closest and _player_idx(closest) == pid:
            return [_cmd("INTERCEPT", pid, team_id, {"aggressive": True}, duration=2)], True
    if role == "DEF":
        return _def_off(c)
    if role == "MID":
        return _mid_off(c)
    return _fwd_off(c, -8 if role == "FWD1" else 8)


_V1 = {"GK": tactics_v1.gk_rules, "DEF": tactics_v1.def_rules, "MID": tactics_v1.mid_rules,
       "FWD1": tactics_v1.fwd_rules, "FWD2": tactics_v1.fwd_rules}
_CLASSIC = {"GK": build_fallback(GK_CONFIG), "DEF": build_fallback(DEF_CONFIG), "MID": build_fallback(MID_CONFIG),
            "FWD1": build_fallback(FWD1_CONFIG), "FWD2": build_fallback(FWD2_CONFIG)}


def team_mode(game_state, team_id):
    return Ctx(game_state, team_id, -1).mode


def make_rules(role):
    def rules(gs, tid, pid):
        cmds, instant = decide(gs, tid, pid, role)
        if cmds is None:
            return None
        return cmds if instant else {"suggest": cmds, "mode": team_mode(gs, tid)}
    return rules


gk_rules = make_rules("GK")
def_rules = make_rules("DEF")
mid_rules = make_rules("MID")
fwd1_rules = make_rules("FWD1")
fwd2_rules = make_rules("FWD2")
fwd_rules = fwd1_rules  # backward compatibility
