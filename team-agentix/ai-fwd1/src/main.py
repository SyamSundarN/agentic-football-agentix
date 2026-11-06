"""
Forward 1 Agent — Controls ONLY player 3. Team "Agentix".
Hybrid: instant rules for on-ball / loose-ball moments, Nova Micro LLM for positioning.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD1_CONFIG
from tactics import fwd1_rules

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "1.0"))

# --- System Prompt (off-ball decisions only; on-ball is handled by rules) ---

SYSTEM_PROMPT = f"""You control ONLY player {MY_PLAYER_ID} (Forward 1 (left striker, the outlet for counter-attacks)) for team Agentix in a 5v5 soccer match. One command per tick.
Your job: Score: run the left channel when we have the ball, press their defenders in their half, wait near halfway for long balls when they attack.
Field x -55..55, y -35..35. The state tells you where your goal and the opponent goal are.

Each tick you get TEAM MODE (BALANCED / COUNTER / ATTACK / PROTECT) and a game-plan move computed for the whole team.
Follow the game-plan move by default — it keeps the team's shape. Change it only if it is clearly wrong for the state,
or if a COACH SAYS line asks for something different (the coach outranks the game plan).

Commands: MOVE_TO(target_x, target_y, sprint) | PRESS_BALL(intensity 0-1) | MARK(target_player_id, tightness "TIGHT"|"LOOSE")
| INTERCEPT(aggressive) | PASS(target_player_id, type "GROUND"|"THROUGH"|"AERIAL") and SHOOT(aim_location "BL"|"BR"|"TL"|"TR", power) only if hasBall=True.

Reply with ONLY a JSON array, nothing else:
[{{"commandType":"MOVE_TO","playerId":{MY_PLAYER_ID},"parameters":{{"target_x":10,"target_y":0,"sprint":true}},"duration":0}}]"""


# --- Fallback ---

fallback_commands = build_fallback(FWD1_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0", max_tokens=150, temperature=0.2)
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD1_CONFIG,
    rules_fn=fwd1_rules,
    stateless=True,
    llm_timeout_s=LLM_TIMEOUT_S,
)

if __name__ == "__main__":
    app.run()
