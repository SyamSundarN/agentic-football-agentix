# Agentix — AI Agents That Play 5v5 Soccer

My team for the **AWS Agentic Soccer workshop** (AI/ML Club × AWS, University of Toledo, Nov 6, 2026).

Five AI agents, one per player, each deployed on **Amazon Bedrock AgentCore**. Every couple of seconds the
match server sends each agent the full game state, and each agent answers with one command
(`MOVE_TO`, `PASS`, `SHOOT`, `PRESS_BALL`, `MARK`, `INTERCEPT`, ...) for its player.

Built from the workshop's sample agents ([aws-samples/sample-ai-possibilities](https://github.com/aws-samples/sample-ai-possibilities/tree/main/agentic-football-sample-agents)),
then debugged and extended into an adaptive, hybrid rules + LLM team.

## How the team thinks

Each player runs a **hybrid decision loop**:

1. **Instant rules (no LLM)** for moments where speed matters: shoot when in range, pass when pressed
   or blocked, chase loose balls, goalkeeper positioning and distribution.
2. **Shared game plan** — every agent reads the same state, so all five pick the same team mode
   without talking to each other:

   | Mode | Trigger | Behaviour |
   |---|---|---|
   | BALANCED | default | Shoot early, pass under pressure, press near the ball |
   | COUNTER | 3+ opponents pushed into our half | Stay compact, one striker waits high, long balls over the press |
   | ATTACK | we are losing | Shoot from further out, everyone pushes up |
   | PROTECT | we lead by 2+ | Extra defender, keep shape |

3. **LLM (Amazon Nova Micro via Strands Agents)** for off-ball positioning. It gets the game state, the
   team mode, the game plan's suggested move, and the coach's latest shout, and may adjust the move.
   If it is slower than 1 s or returns bad output, the game-plan move is used instead.

## Bugs I found and fixed in the sample code

Found by reading the live match reports and CloudWatch logs, then logging the raw server payload:

- **Wrong pitch scale.** The sample code assumes goals at x = ±55; the real pitch is about ±6.4.
  Every position-based rule (keeper position, shooting range, runs) was off by ~8.6×.
  `lib/agent_base.py` now scales incoming state to the ±55 frame and scales outgoing targets back.
- **Ball carrier misidentified.** Player IDs (`agentId_0`..`agentId_4`) repeat on both teams, so agents
  often thought an opponent's ball was theirs (match report: 12 SHOOT commands, 1 actual shot, 0 passes).
  `lib/state.py` now picks the carrier by distance to the ball and current action.
- **Slow, forgetful agents.** The midfielder used a slow model, and every agent kept the entire match
  history in its context, so calls got slower each tick. All players now use Nova Micro with a fresh
  context per tick and a hard timeout.
- **Coach shouts ignored.** The state summary dropped the team chat; the latest shout is now passed to the LLM.

## Simulator

`simulator/sim.py` is a simple 5v5 match simulator (movement, tackles, pass interception, shot odds) I
wrote to compare game plans head-to-head before deploying. It is **not** the real game engine — it is for
relative comparisons and catching broken tactics.

```bash
python simulator/sim.py 100   # 100 matches per pairing
```

`sample` = the workshop's sample rules, `v1` = my first upgrade, `v2` = the final Agentix plan.
Opponents: `sample` (balanced), `aggressive` (everyone presses), `defensive` (park the bus).

| Agentix (final) vs | W-D-L over 100 sim matches |
|---|---|
| Balanced | 43-15-42 |
| Aggressive | 41-11-48 |
| Defensive | 93-6-1 |

## Live results

- Sample agents: lost to The Benchmark FC.
- First upgrade (instant rules, faster model): **beat The Benchmark FC**.
- Lost 0–1 to Total Attack United (aggressive). Its match report led to finding the pitch-scale and ball-carrier bugs above.

## Project layout

```
lib/                  shared code used by all 5 agents
  agent_base.py       request handler: scaling, rules -> LLM -> fallback, timeouts
  tactics.py          adaptive team game plan (modes, on-ball and off-ball decisions)
  tactics_v1.py       first-upgrade rules, used in BALANCED mode
  state.py            game-state parsing, ball-carrier detection, LLM state summary
  fallback.py         rule-based safety net per position
  parsing.py          extracts JSON commands from LLM output
team-agentix/
  ai-gk, ai-def, ai-mid, ai-fwd1, ai-fwd2   one agent per player (src/main.py = role prompt + wiring)
  deploy-all.sh       deploys all 5 agents to AgentCore
simulator/sim.py      offline match simulator
```

## Run it

Requires Python 3.10+, the AWS CLI, `uv`, and AWS credentials with Bedrock access (us-east-1).

```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
pip install strands-agents bedrock-agentcore-starter-toolkit
pip install -r team-agentix/ai-gk/requirements.txt

cd team-agentix
for p in gk def mid fwd1 fwd2; do python ai-$p/test_local.py; done   # offline tests
python ai-fwd1/test_local.py --llm                                     # one real Bedrock call
./deploy-all.sh                                                        # deploy all 5 agents
```

Then register the five runtime ARNs in the match portal (formation 1-1-2: GK, DEF, MID, FWD, FWD).

## Tech

Amazon Bedrock AgentCore Runtime · Amazon Bedrock (Nova Micro) · Strands Agents · Python · CloudWatch / OpenTelemetry

## License

MIT No Attribution — see [LICENSE](LICENSE). Based on AWS sample code © Amazon.com, Inc. or its affiliates.
