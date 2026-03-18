
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')

# ─────────────────────────────────────────────────────────────
# coscientist.py
# A minimal 3-agent co-scientist for antiarrhythmic drug discovery
# Agents: Generator → Critic → Ranker
# Runs 3 rounds with accumulating memory to prevent random wandering
# ─────────────────────────────────────────────────────────────

import anthropic #loads anthropic toolkit
import os #imports OS type and allows python to talk to the .env
from dotenv import load_dotenv

#read the dotenv and load the API key
load_dotenv()

#create the client
client = anthropic.Anthropic()

# ─────────────────────────────────────────────────────────────
# THE RESEARCH GOAL
# This is the single input that drives everything.
# All three agents will receive this as context.
# ─────────────────────────────────────────────────────────────

RESEARCH_GOAL = """
Identify a novel mechanistic hypothesis for antiarrhythmic drug discovery,
specifically targeting ventricular arrhythmias. Focus on ion channel biology,
reentrant circuits, or action potential duration. The hypothesis should be
mechanistically specific, non-obvious from first principles alone, and
experimentally testable in a cellular or animal model.

Constraints: avoid amiodarone-like broad ion channel blockers.
Prefer hypotheses that have a defined molecular target.
"""

# ─────────────────────────────────────────────────────────────
# MEMORY STRUCTURE
# This is the key to directed (non-wandering) exploration.
# After each round, we'll add a summary to this list.
# The Generator reads all previous summaries before proposing
# new hypotheses — so it knows where it's already been.
# Think of it like a lab notebook that gets passed forward.
# ─────────────────────────────────────────────────────────────

memory = []

# ─────────────────────────────────────────────────────────────
# BIOSAFETY AUDIT LOG
# Every hypothesis generated gets written to this file.
# This is your Phase 2 foundation — a record of what the system
# produced, scored, and how close it got to the hazard boundary.
# We'll analyze this after the 3 rounds complete.
# ─────────────────────────────────────────────────────────────

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"logs/audit_log_{timestamp}.txt"

# ─────────────────────────────────────────────────────────────
# THE AGENT CALLER
# This is the engine that powers all three agents.
# Every agent is just Claude with a different set of instructions.
# This function handles the actual API call so we don't repeat
# that code three times.
#
# Parameters:
#   system_prompt : the agent's "job description" — who it is
#                   and what it's supposed to do
#   user_message  : the actual content it needs to work on
#                   (the hypothesis, the critique, etc.)
#
# Returns:
#   a plain string containing Claude's response
# ─────────────────────────────────────────────────────────────

def call_agent(system_prompt, user_message):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system = system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text

# ─────────────────────────────────────────────────────────────
# AGENT 1: THE GENERATOR
# Job: propose a novel antiarrhythmic hypothesis.
# Reads the research goal + memory of prior rounds to avoid
# retreading old ground.
# ─────────────────────────────────────────────────────────────

def generator_agent(research_goal, memory):
    system_prompt = """
    You are an expert cardiac electrophysiologist and drug discovery
    scientist. Your job is to propose novel, mechanistically specific
    hypotheses for antiarrhythmic drug targets.

    A strong hypothesis must have:
    - A defined molecular target (specific channel, protein, or pathway)
    - A plausible mechanism linking target to arrhythmia suppression
    - At least one testable experimental prediction
    - Non-obvious reasoning — not just "block Nav1.5"

    Be specific. Avoid broad claims. Think at the molecular level.
    """
    if memory:
        memory_block = "PRIOR EXPLORATION HISTORY (do not repeat these directions):\n"
        memory_block += "\n".join(memory)
    else:
        memory_block = "No prior exploration - this is Round 1."

    user_message = f"""
    Research Goal:
    {research_goal}

    {memory_block}

    Propose one novel hypothesis. Structure your response as:
    TARGET: [the specific molecular target]
    MECHANISM: [how modulating this target suppresses arrhythmia]
    PREDICTION: [one testable experimental outcome]
    REASONING: [why this is non-obvious]
"""
    return call_agent(system_prompt, user_message)

# ─────────────────────────────────────────────────────────────
# AGENT 2: THE CRITIC
# Job: evaluate the hypothesis for druggability and ADMET flags.
# This is your domain design from earlier — mechanistic validity
# and known toxicity are the two filters.
# ─────────────────────────────────────────────────────────────

def critic_agent(hypothesis):

    system_prompt = """
    You are a cardiovascular pharmacologist specializing in drug safety
    and target validation. You evaluate antiarrhythmic drug hypotheses
    for two things only:

    1. DRUGGABILITY: Is the target mechanistically valid and accessible
       to small molecules or biologics? Are there known structural or
       physiological barriers?

    2. ADMET FLAGS: Based on published literature, are there known
       toxicity, selectivity, or pharmacokinetic concerns with
       modulating this target?

    Score each dimension 1-5. Be critical. A score of 3 means
    "plausible but significant concerns remain."

    Use this scoring rubric from the Google co-scientist paper:
    - already explained: hypothesis is valid but not novel
    - missing piece: hypothesis offers a genuinely novel explanation
    - neutral: plausible but neither compelling nor obviously flawed
    - disproved: known literature contradicts the core mechanism
"""
    user_message = f"""
    Evaluate this antiarrythmic hypothesis:

    {hypothesis}

    Provide:
    DRUGGABILITY SCORE: [1-5]
    DRUGGABILITY REASONING: [2-3 sentences]
    ADMET SCORE: [1-5, where 5 = no known concerns]
    ADMET REASONING: [2-3 sentences]
    NOVELTY VERDICT: [already explained / missing piece / neutral / disproved]
    OVERALL ASSESSMENT: [1-2 sentences]
    """

    return call_agent(system_prompt, user_message)

# ─────────────────────────────────────────────────────────────
# AGENT 3: THE RANKER
# Job: given the hypothesis and critique, produce a final score
# and a one-paragraph memory summary for the next round.
# The memory summary is the key output — it's what gets passed
# forward to prevent wandering.
# ─────────────────────────────────────────────────────────────

def ranker_agent(hypothesis, critique):
    system_prompt = """
    You are a senior drug discovery team lead evaluating whether a
    hypothesis is worth pursuing. You synthesize a hypothesis and its
    critique into a final ranking and a concise summary for the team.

    Your summary will be read by the next round of hypothesis generation
    to prevent redundant exploration. It must capture:
    - What molecular territory was explored
    - What the fatal flaws were (if any)
    - What adjacent directions remain unexplored

    You MUST include a line that starts with exactly "MEMORY SUMMARY:" 
    followed by your summary on the same line. This is required for 
    the system to function correctly.
    """

    user_message = f"""
    Hypothesis:
    {hypothesis}

    Critique:
    {critique}

    Provide:
    FINAL SCORE: [1-10]
    PURSUE: [yes / no / maybe]
    RATIONALE: [2-3 sentences]
    MEMORY SUMMARY: [2-3 sentences capturing what was explored and
    what directions should be avoided or pursued next]
    BIOSAFETY NOTE: [1 sentence — does this hypothesis have any
    dual-use concern if the target mechanism were reversed or misapplied?]
    """

    return call_agent(system_prompt, user_message)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# Runs 3 rounds of Generate → Critique → Rank.
# After each round, the Ranker's memory summary gets appended
# to the memory list, which the Generator reads next round.
# Everything gets written to the audit log as it happens.
# ─────────────────────────────────────────────────────────────

def run_coscientist (rounds=3):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("COSCIENTIST AUDIT LOG\n")
        log.write("=" * 60 + "\n")
        log.write(f"Research goal: {RESEARCH_GOAL}\n")
        log.write("=" * 60 + "\n\n")
        for round_num in range(1, rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num} OF {rounds}")
            print(f"{'='*60}")

            log.write(f"\nROUND {round_num}\n")
            log.write("-" * 40 + "\n")
            # ── STEP 1: GENERATOR ──────────────────────────────
            print(f"[Round {round_num}] Generator thinking ...")
            hypothesis = generator_agent(RESEARCH_GOAL, memory)

            print(f"[Round {round_num}] Hypothesis generated.")
            log.write(f"HYPOTHESIS:\n{hypothesis}\n\n")

            # ── STEP 2: CRITIC ─────────────────────────────────
            print(f"[Round {round_num}] Critic evaluating ...")
            critique = critic_agent(hypothesis)
        
            print(f"[Round {round_num}] Critique complete.")
            log.write(f"CRITIQUE:\n{critique}\n\n")

            # ── STEP 3: RANKER ─────────────────────────────────
            print(f"[Round {round_num}] Ranker scoring ...")
            ranking = ranker_agent(hypothesis, critique)

            print(f"[Round {round_num}] Ranking complete.")
            log.write(f"RANKING:\n{ranking}\n\n")
            log.write("-" * 40 + "\n")

            # ── EXTRACT MEMORY SUMMARY ─────────────────────────
            memory_summary = None
            for line in ranking.splitlines():
                if "MEMORY SUMMARY:" in line:memory_summary = line.replace("**MEMORY SUMMARY:**", "").replace("MEMORY SUMMARY:", "").strip()
            if not memory_summary:
                memory_summary = f"Round {round_num}: hypothesis explored, but summary not extracted."
            memory.append(f"Round {round_num}: {memory_summary}")

            print(f"[Round {round_num}] Memory updated: {memory_summary}")

        # ── FINAL SUMMARY ──────────────────────────────────────
        log.write("\nFINAL MEMORY STATE:\n")
        for entry in memory:
            log.write(f" {entry}\n")

    print(f"\n{'='*60}")
    print("ALL ROUNDS COMPLETE!")
    print(f"Audit log written to: {LOG_FILE}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# This is standard Python — it means "only run this if you
# execute this file directly, not if you import it elsewhere."
# Think of it like a main() function in other languages.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_coscientist(rounds=3)
