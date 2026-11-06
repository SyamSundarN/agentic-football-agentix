"""Base agent factory for AI soccer position agents."""

import asyncio
import json
import os
from typing import Callable
from strands import Agent
from strands.models import BedrockModel

from parsing import parse_commands
from state import summarize_state
from fallback import FallbackConfig, build_last_resort


def create_agent(system_prompt: str, model_id: str = "us.amazon.nova-micro-v1:0",
                 max_tokens: int | None = None, temperature: float | None = None) -> Agent:
    """Create a Strands Agent with the given system prompt."""
    kwargs = {"model_id": model_id}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    model = BedrockModel(**kwargs)
    return Agent(model=model, system_prompt=system_prompt)


def parse_prompt(payload) -> dict:
    """Parse the invocation payload's prompt, accepting either str or dict.

    The main path has always accepted both, but the error handler used to
    re-parse with a bare json.loads(), which raises TypeError on a dict. That
    exception was caught by the outer bare except, so the rule-based fallback
    was skipped entirely and control fell through to the last resort.
    """
    prompt = payload.get("prompt", "{}")
    if isinstance(prompt, dict):
        return prompt
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def team_id_from(prompt_data: dict, log, label: str) -> int:
    """teamId from the payload; HOME only as a loudly logged last resort.

    teamId is authoritative input, not a guessable default. Silently
    defaulting to 0 makes an away agent emit commands tagged HOME, which the
    server attributes to the opposing team — and leaves nothing in the logs
    to explain why the away side played as though it were the home side.
    """
    team_id = prompt_data.get("teamId")
    if team_id is None:
        log.error(
            f"{label}: payload contained no teamId; falling back to HOME (0). "
            "Commands this tick may be attributed to the wrong team."
        )
        return 0
    return team_id


# ---------------------------------------------------------------------------
# Pitch scaling. The real game pitch is ~±6.4 wide in x; all tactics, prompts
# and fallbacks were written for ±55. Scale incoming positions up and outgoing
# targets back down so everything stays consistent.
# ---------------------------------------------------------------------------
REAL_GOAL_X = float(os.environ.get("REAL_GOAL_X", "6.4"))


def _pitch_scale(game_state: dict) -> float:
    xs = [abs(p.get("position", {}).get("x", 0)) for p in game_state.get("players", [])]
    if xs and max(xs) > 15:          # already in the ±55 frame (sample data)
        return 1.0
    if not xs:
        return 1.0
    return 55.0 / REAL_GOAL_X


def _scale_state(game_state: dict, k: float) -> dict:
    if k == 1.0:
        return game_state
    gs = json.loads(json.dumps(game_state))
    def sc(d):
        if isinstance(d, dict):
            for key in ("x", "y"):
                if isinstance(d.get(key), (int, float)):
                    d[key] = d[key] * k
    ball = gs.get("ball", {})
    sc(ball.get("position")); sc(ball.get("velocity"))
    for p in gs.get("players", []):
        sc(p.get("position")); sc(p.get("velocity"))
    return gs


def _unscale_cmds(cmds, k: float):
    if k == 1.0 or not isinstance(cmds, list):
        return cmds
    out = []
    for c in cmds:
        c = dict(c); prm = dict(c.get("parameters", {}) or {})
        for key in ("target_x", "target_y", "distance"):
            if isinstance(prm.get(key), (int, float)):
                prm[key] = round(prm[key] / k, 3)
        c["parameters"] = prm
        out.append(c)
    return out


def _latest_coach_message(game_state: dict, prompt_data: dict) -> str:
    """Best-effort: find the coach's latest shout wherever the server puts it."""
    for src in (prompt_data, game_state):
        for key in ("coachMessage", "coachPrompt", "coachMessages", "teamChat", "chat"):
            val = src.get(key) if isinstance(src, dict) else None
            if not val:
                continue
            if isinstance(val, list):
                val = val[-1]
            if isinstance(val, dict):
                val = val.get("message") or val.get("text") or val.get("content") or json.dumps(val)
            text = str(val).strip()
            if text:
                return text[:200]
    return ""


def create_invoke_handler(
    app,
    agent: Agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    rules_fn: Callable[[dict, int, int], list[dict] | None] | None = None,
    stateless: bool = False,
    llm_timeout_s: float | None = None,
):
    """Create and register the @app.entrypoint invoke handler.

    Three layers of error handling, from best to worst:
      1. LLM response → parse into commands
      2. fallback_fn(game_state, team_id, my_player_id) → rule-based commands
      3. last-resort command from fallback_cfg → single safe command
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        # Parsed once, outside the try, so the error handler below can reuse
        # it instead of re-parsing and throwing a second exception.
        prompt_data = parse_prompt(payload)
        if os.environ.get("LOG_RAW_PAYLOAD", "0") == "1":
            try:
                log.info("RAWPAYLOAD " + json.dumps(payload, default=str)[:6000])
            except Exception:
                pass
        team_id = team_id_from(prompt_data, log, position_label)
        k = 1.0
        try:
            game_state = prompt_data.get("gameState", {})
            k = _pitch_scale(game_state)
            game_state = _scale_state(game_state, k)

            # Honor myPlayers from payload if present, otherwise use configured player ID
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id

            # Layer 0: team game plan. A list = act instantly; a dict = a
            # positioning suggestion the LLM may adjust (and our safety net).
            suggestion = None
            mode = None
            if rules_fn is not None:
                try:
                    ruled = rules_fn(game_state, team_id, effective_pid)
                except Exception as rule_err:
                    log.warn(f"{position_label} rules error: {rule_err}")
                    ruled = None
                if isinstance(ruled, list) and ruled:
                    log.info(f"{position_label} RULE: {[c.get('commandType') for c in ruled]}")
                    yield json.dumps(_unscale_cmds(ruled, k))
                    return
                if isinstance(ruled, dict):
                    suggestion = ruled.get("suggest")
                    mode = ruled.get("mode")

            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )
            coach = _latest_coach_message(game_state, prompt_data)
            if coach:
                state_summary += f"\n\nCOACH SAYS: {coach}"
            if suggestion:
                s0 = suggestion[0]
                state_summary += (
                    f"\n\nTEAM MODE: {mode}. Game-plan move for you: "
                    f"{s0.get('commandType')} {json.dumps(s0.get('parameters', {}))}. "
                    "Use it unless the state or the coach clearly calls for something better."
                )
            safe_cmds = suggestion or None
            log.info(f"{position_label} agent invoked for team {team_id}, controlling player {effective_pid}")

            # Stateless: a fresh agent each tick, so old ticks don't pile up in
            # the context (slower calls, stale positions confusing the model).
            run_agent = (
                Agent(model=agent.model, system_prompt=agent.system_prompt, callback_handler=None)
                if stateless else agent
            )
            if llm_timeout_s:
                try:
                    response = await asyncio.wait_for(
                        run_agent.invoke_async(state_summary), timeout=llm_timeout_s
                    )
                except asyncio.TimeoutError:
                    log.warn(f"{position_label} LLM timed out after {llm_timeout_s}s, using game plan")
                    yield json.dumps(_unscale_cmds(safe_cmds or fallback_fn(game_state, team_id, effective_pid), k))
                    return
            else:
                response = run_agent(state_summary)
            response_text = str(response)

            def on_recovered(raw: str) -> None:
                # The model wrote Python-flavoured JSON (usually `True`/`False`/`None`).
                # We recovered it rather than dropping the command and falling back —
                # logged so you can see how often your model does this.
                log.warn(f"{position_label} recovered malformed JSON from the model: {raw[:200]}")

            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)

            if commands:
                log.info(f"LLM returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                yield json.dumps(_unscale_cmds(commands, k))
            else:
                log.warn(f"LLM parse failed, using game plan. Response: {response_text[:200]}")
                commands = safe_cmds or fallback_fn(game_state, team_id, effective_pid)
                log.info(f"Fallback returned {len(commands)} commands")
                yield json.dumps(_unscale_cmds(commands, k))

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                # Reuse the payload parsed at entry. Re-parsing here is what
                # broke this path: json.loads() on an already-decoded dict
                # raises TypeError, the bare except below swallowed it, and
                # the rule-based fallback never ran.
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                _raw = prompt_data.get("gameState", {})
                k = _pitch_scale(_raw)
                commands = fallback_fn(
                    _scale_state(_raw, k),
                    team_id,
                    effective_pid,
                )
                yield json.dumps(_unscale_cmds(commands, k))
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                cmd = dict(last_resort)
                # team_id came from the payload at entry, so the last resort
                # is no longer hardcoded to HOME.
                cmd["teamId"] = team_id
                yield json.dumps(_unscale_cmds([cmd], k))

    return invoke
