# feedbackllm.py
import os
import json
from groq import Groq
from typing import Dict, Any, List
from retrieval import retrieve_docs
from kg_reasoner import KnowledgeGraph

# Replace with your actual Groq endpoint if different
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "<set_in_env>")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq-1-small")  # or 'groq-1-large' if available
client = Groq(api_key=GROQ_API_KEY)
kg = KnowledgeGraph("knowledge_graph.json")

def build_prompt(bm, miss_dir,handedness) -> str:
    kg_results = kg.evaluate_all(bm, handedness)

    matched            = kg_results["matched_shooters"]
    styles             = kg_results["active_styles"]
    deviations         = kg_results["local_deviations"]
    preserve           = kg_results["preserve_constraints"]
    preserve_conflicts = kg_results["preserve_conflicts"]


    angles_rel = bm["angles_at_release"]
    angles_dip = bm["angles_at_dip"]
    rom = bm["range_of_motion"]
    timing = bm["timing"]
    coord = bm["coordination"]
    cons = bm["consistency"]

    shot_summary = (
        f"elbow_release={angles_rel['elbow']:.1f}, "
        f"knee_release={angles_rel['knee']:.1f}, "
        f"shoulder_release={angles_rel['shoulder']:.1f}, "
        f"elbow_dip={angles_dip['elbow']:.1f}, "
        f"knee_dip={angles_dip['knee']:.1f}, "
        f"elbow_rom={rom['elbow_extension']:.1f}, "
        f"knee_rom={rom['knee_extension']:.1f}, "
        f"dip_time={timing['dip_duration_s']:.2f}, "
        f"drive_time={timing['drive_duration_s']:.2f}, "
        f"knee_to_elbow_delay={coord['knee_to_elbow_delay_s']:.3f}, "
        f"elbow_smoothness={cons['elbow_smoothness']:.3f}"
    )

    rag_query = (
        f"Biomechanics: {shot_summary}\n"
        f"Miss-direction: {miss_dir}\n"
        f"Matched shooter styles: {styles}\n"
        f"Local deviations: {[d['deviation'] for d in deviations]}\n"
        f"Preserve constraints: {list(preserve.keys())}\n"
    )

    rag_context = retrieve_docs(rag_query)

    header = (
        "You are a professional basketball shooting coach specializing in biomechanics.\n"
        f"The shooter is **{handedness}**.\n"
        "Your job is to diagnose the user's shooting mechanics and give clear, "
        "actionable, prioritized coaching insights.\n\n"
    )

    biomech_section = (
        "=== BIOMECHANICAL SUMMARY ===\n"
        f"{shot_summary}\n\n"
    )

    kg_section = (
        "=== SHOOTER STYLE MATCHING (KNOWLEDGE GRAPH) ===\n"
        f"Top matched elite shooters (distance-based): {matched}\n"
        f"Active shooter styles: {styles}\n\n"

        "=== LOCAL DEVIATIONS WITHIN STYLE ===\n"
        "Each deviation includes the affected feature, severity, and a suggested coaching cue.\n"
        "Prioritize HIGH severity first, then MODERATE, then LOW.\n"
        f"{json.dumps(deviations, indent=2)}\n\n"

        "=== MECHANICS TO PRESERVE ===\n"
        "The following mechanics are already sound for this shooter's style. "
        "Do NOT give advice that degrades them.\n"
        f"{json.dumps(preserve, indent=2)}\n\n"

        "=== CORRECTION / PRESERVATION CONFLICTS ===\n"
        "Where a correction overlaps with a preserved mechanic, frame the coaching cue carefully "
        "to improve the deviation WITHOUT disrupting the preserved quality.\n"
        f"{json.dumps(preserve_conflicts, indent=2)}\n\n"

        "IMPORTANT:\n"
        "- Preserve the shooter's matched style traits.\n"
        "- Only correct the listed local deviations.\n"
        "- Do NOT suggest textbook form changes.\n"
        "- Use the coaching_cue from each deviation as a starting point, but adapt to the full context.\n\n"
    )

    rag_section = (
        "=== RETRIEVED EVIDENCE FROM KNOWLEDGE BASE ===\n"
        f"{rag_context}\n\n"
    )

    miss_section = (
        f"=== USER MISS PATTERNS ===\n"
        f"{miss_dir}\n\n"
    )

    instructions = (
        "=== TASK ===\n"
        "Provide:\n"
        "A) The top 3 highest-priority corrections (short, clear, actionable).\n"
        "B) 2 drills with short step-by-step execution.\n"
        "C) A positive 1-paragraph encouragement.\n"
        "D) One quantifiable goal for the next session.\n"
        "Keep tone supportive, expert, and direct.\n"
    )

    final_prompt = (
        header +
        biomech_section +
        miss_section +
        kg_section +
        rag_section +
        instructions
    )

    return final_prompt

def query_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    completion = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
    temperature=temperature,
    top_p=1,
    reasoning_effort="medium",
    )
    return completion.choices[0].message.content

def generate_coaching(shot_metrics: List[Dict], miss_dir, handedness, extra_context: str = "") -> Dict[str,Any]:
    """
    Compose prompt, call Groq, return textual coaching. For RAG, we rely on RAG_DOC_URL being available to Groq infra.
    """
    prompt = build_prompt(shot_metrics, miss_dir, handedness)
    text = query_groq(prompt)
    return {"coach_text": text, "prompt": prompt}
