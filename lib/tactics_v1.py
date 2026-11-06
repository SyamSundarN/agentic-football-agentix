"""Version 1 rules (beat Benchmark FC) — used by tactics.py in BALANCED mode.

Each rules function returns a list of commands, or None to let the LLM decide.
Rules run BEFORE the LLM, so on-ball actions (shoot / pass / dribble) and
loose-ball chases are instant and never time out.
"""

from __future__ import annotations
import random

from state import (
    get_goal_positions,
    get_possession_info,
    holder_is_me,
    dist,
    _player_idx,
    _is_my_team,
)

SHOOT_RANGE = 26.0      # distance to opponent goal line where we always shoot
PRESSURE_RADIUS = 5.0   # opponent this close = we are under pressure
OPEN_RADIUS = 6.0       # teammate with no opponent this close = open


def _cmd(cmd_type, pid, tid, params, duration=0):
    return {"commandType": cmd_type, "playerId": pid, "teamId": tid,
            "parameters": params, "duration": duration}


def _ctx(game_state, team_id, my_pid):
    ball = game_state.get("ball", {})
    players = game_state.get("players", [])
    mates = [p for p in players if _is_my_team(p, team_id)]
    opps = [p for p in players if not _is_my_team(p, team_id)]
    me = next((p for p in mates if _player_idx(p) == my_pid), None)
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    return {
        "ball": ball,
        "ball_pos": ball.get("position", {"x": 0, "y": 0}),
        "players": players,
        "mates": [p for p in mates if _player_idx(p) != my_pid],
        "opps": opps,
        "me": me,
        "pos": (me or {}).get("position", {"x": 0, "y": 0}),
        "stam": (me or {}).get("stamina", 1.0),
        "my_goal_x": my_goal_x,
        "opp_goal_x": opp_goal_x,
        "fwd": 1 if opp_goal_x > 0 else -1,   # +1 means attacking toward +x
    }


def _nearest_opp_dist(p, opps):
    pos = p.get("position", {})
    return min((dist(pos, o.get("position", {})) for o in opps), default=99.0)


def _dist_to_opp_goal(pos, c):
    return dist(pos, {"x": c["opp_goal_x"], "y": 0})


def _best_pass_target(c, candidates, my_pos):
    """Most useful open teammate: open > advanced > close."""
    best, best_score = None, -1e9
    for p in candidates:
        pp = p.get("position", {})
        openness = min(_nearest_opp_dist(p, c["opps"]), 15.0)
        progress = (pp.get("x", 0) - my_pos.get("x", 0)) * c["fwd"]
        length = dist(pp, my_pos)
        if length > 40:          # too long, likely intercepted
            continue
        score = openness * 1.5 + progress * 0.8 - length * 0.2
        if score > best_score:
            best, best_score = p, score
    return best


def _shoot(pid, tid, c):
    # Aim at the side of the goal away from the keeper, low corners.
    opp_gk = next((o for o in c["opps"] if _player_idx(o) == 0), None)
    gk_y = (opp_gk or {}).get("position", {}).get("y", 0)
    if abs(gk_y) < 1.0:
        aim = random.choice(["BL", "BR"])
    else:
        aim = random.choice(["BL", "BR", "TL", "TR"])
    return [_cmd("SHOOT", pid, tid, {"aim_location": aim, "power": 0.95})]


def _dribble(pid, tid, c):
    pos = c["pos"]
    tx = c["opp_goal_x"] - c["fwd"] * 12
    ty = pos.get("y", 0) * 0.5   # drift toward the middle for a better angle
    return [_cmd("MOVE_TO", pid, tid,
                 {"target_x": tx, "target_y": ty, "sprint": c["stam"] > 0.25})]


def _pass(pid, tid, target, c):
    pp = target.get("position", {})
    ahead = (pp.get("x", 0) - c["pos"].get("x", 0)) * c["fwd"]
    ptype = "THROUGH" if ahead > 10 else "GROUND"
    return [_cmd("PASS", pid, tid,
                 {"target_player_id": _player_idx(target), "type": ptype})]


def _i_am_closest_to_ball(c):
    my_d = dist(c["pos"], c["ball_pos"])
    return all(dist(m.get("position", {}), c["ball_pos"]) >= my_d
               for m in c["mates"] if _player_idx(m) != 0)


# ---------------------------------------------------------------------------
# Role rules
# ---------------------------------------------------------------------------

def gk_rules(game_state, team_id, my_pid):
    """Goalkeeper is fully rule-based: positioning is pure geometry."""
    c = _ctx(game_state, team_id, my_pid)
    if not c["me"]:
        return None
    ball, bp, pos = c["ball"], c["ball_pos"], c["pos"]

    if holder_is_me(ball, c["players"], team_id, my_pid):
        target = _best_pass_target(c, c["mates"], pos) or (c["mates"][0] if c["mates"] else None)
        tid_ = _player_idx(target) if target else 1
        far = target and dist(target.get("position", {}), pos) > 25
        return [_cmd("GK_DISTRIBUTE", my_pid, team_id,
                     {"target_player_id": tid_, "method": "KICK" if far else "THROW"})]

    ball_to_goal = abs(bp.get("x", 0) - c["my_goal_x"])
    _, status, we_have = get_possession_info(ball, c["players"], team_id)

    # Loose ball in our box: claim it.
    if status == "free" and ball_to_goal < 15 and abs(bp.get("y", 0)) < 15:
        return [_cmd("INTERCEPT", my_pid, team_id, {"aggressive": True}, duration=2)]

    # Attacker bearing down one-on-one: close the angle.
    if not we_have and ball_to_goal < 12 and dist(pos, bp) < 9:
        return [_cmd("PRESS_BALL", my_pid, team_id, {"intensity": 1.0}, duration=2)]

    # Default: on the line, shading toward the ball.
    tx = c["my_goal_x"] + c["fwd"] * 4.5
    ty = max(-5.0, min(5.0, bp.get("y", 0) * 0.35))
    far = dist(pos, {"x": tx, "y": ty}) > 6
    return [_cmd("MOVE_TO", my_pid, team_id, {"target_x": tx, "target_y": ty, "sprint": far})]


def _outfield_rules(game_state, team_id, my_pid, role):
    c = _ctx(game_state, team_id, my_pid)
    if not c["me"]:
        return None
    ball, pos = c["ball"], c["pos"]

    # ---- I have the ball: decide instantly ----
    if holder_is_me(ball, c["players"], team_id, my_pid):
        d_goal = _dist_to_opp_goal(pos, c)
        under_pressure = _nearest_opp_dist(c["me"], c["opps"]) < PRESSURE_RADIUS

        if role in ("MID", "FWD") and d_goal < SHOOT_RANGE:
            return _shoot(my_pid, team_id, c)

        outfield = [m for m in c["mates"] if _player_idx(m) != 0]
        if role == "DEF":
            # Defender: move it on quickly to the best open teammate.
            target = _best_pass_target(c, outfield, pos)
            if target:
                return _pass(my_pid, team_id, target, c)
            return _dribble(my_pid, team_id, c)

        # MID / FWD: pass if pressed and a teammate is better placed, else dribble.
        target = _best_pass_target(c, outfield, pos)
        if target:
            tpos = target.get("position", {})
            t_closer = _dist_to_opp_goal(tpos, c) < d_goal - 5
            t_open = _nearest_opp_dist(target, c["opps"]) > OPEN_RADIUS
            if t_open and (under_pressure or (role == "MID" and t_closer)):
                return _pass(my_pid, team_id, target, c)
        return _dribble(my_pid, team_id, c)

    # ---- Loose ball and I'm our nearest outfielder: go win it ----
    _, status, _ = get_possession_info(ball, c["players"], team_id)
    if status == "free" and _i_am_closest_to_ball(c):
        return [_cmd("INTERCEPT", my_pid, team_id, {"aggressive": True}, duration=2)]

    # Everything else (positioning, pressing, marking): let the LLM decide.
    return None


def def_rules(gs, tid, pid):
    return _outfield_rules(gs, tid, pid, "DEF")


def mid_rules(gs, tid, pid):
    return _outfield_rules(gs, tid, pid, "MID")


def fwd_rules(gs, tid, pid):
    return _outfield_rules(gs, tid, pid, "FWD")
