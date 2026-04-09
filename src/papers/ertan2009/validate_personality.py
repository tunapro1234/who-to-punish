"""
Personality validation: assign BFI-2-Expanded, measure with Mini-IPIP via EDSL.

Cross-instrument validation — uses two DIFFERENT psychometric instruments
so the LLM can't just parrot back the assignment.

Assignment:   BFI-2-Expanded sentences (Soto & John, 2017)
Measurement:  Mini-IPIP 20-item inventory (Donnellan et al., 2006)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from edsl import Agent, AgentList, Model, QuestionLinearScale, Survey

from src.framework.personalities import PersonalityFactory, PROFILES, DOMAINS, build_description


# ── Mini-IPIP items (Donnellan et al., 2006) ─────────────────────────
# 20 items, 4 per domain, 1-5 Likert scale

MINI_IPIP = [
    # Extraversion
    {"id": 1,  "text": "Am the life of the party",                        "domain": "extraversion",       "reverse": False},
    {"id": 2,  "text": "Don't talk a lot",                                "domain": "extraversion",       "reverse": True},
    {"id": 3,  "text": "Talk to a lot of different people at parties",     "domain": "extraversion",       "reverse": False},
    {"id": 4,  "text": "Keep in the background",                          "domain": "extraversion",       "reverse": True},
    # Agreeableness
    {"id": 5,  "text": "Sympathize with others' feelings",                "domain": "agreeableness",      "reverse": False},
    {"id": 6,  "text": "Am not interested in other people's problems",    "domain": "agreeableness",      "reverse": True},
    {"id": 7,  "text": "Feel others' emotions",                           "domain": "agreeableness",      "reverse": False},
    {"id": 8,  "text": "Am not really interested in others",              "domain": "agreeableness",      "reverse": True},
    # Conscientiousness
    {"id": 9,  "text": "Get chores done right away",                      "domain": "conscientiousness",  "reverse": False},
    {"id": 10, "text": "Often forget to put things back in their proper place", "domain": "conscientiousness", "reverse": True},
    {"id": 11, "text": "Like order",                                      "domain": "conscientiousness",  "reverse": False},
    {"id": 12, "text": "Make a mess of things",                           "domain": "conscientiousness",  "reverse": True},
    # Neuroticism
    {"id": 13, "text": "Have frequent mood swings",                       "domain": "neuroticism",        "reverse": False},
    {"id": 14, "text": "Am relaxed most of the time",                     "domain": "neuroticism",        "reverse": True},
    {"id": 15, "text": "Get upset easily",                                "domain": "neuroticism",        "reverse": False},
    {"id": 16, "text": "Seldom feel blue",                                "domain": "neuroticism",        "reverse": True},
    # Openness
    {"id": 17, "text": "Have a vivid imagination",                        "domain": "openness",           "reverse": False},
    {"id": 18, "text": "Am not interested in abstract ideas",             "domain": "openness",           "reverse": True},
    {"id": 19, "text": "Have difficulty understanding abstract ideas",     "domain": "openness",           "reverse": True},
    {"id": 20, "text": "Do not have a good imagination",                  "domain": "openness",           "reverse": True},
]


def build_mini_ipip_survey():
    """Build an EDSL survey with all 20 Mini-IPIP items as LinearScale questions."""
    questions = []
    for item in MINI_IPIP:
        q = QuestionLinearScale(
            question_name=f"ipip_{item['id']}",
            question_text=(
                f"Rate how accurately this describes you: \"{item['text']}\""
            ),
            question_options=[1, 2, 3, 4, 5],
            option_labels={
                1: "Very inaccurate",
                2: "Moderately inaccurate",
                3: "Neutral",
                4: "Moderately accurate",
                5: "Very accurate",
            },
        )
        questions.append(q)
    return Survey(questions=questions)


def score_results(df):
    """Score Mini-IPIP responses into Big Five domain scores (1-5)."""
    scores = {}
    for domain in DOMAINS:
        items = [i for i in MINI_IPIP if i["domain"] == domain]
        total = 0
        for item in items:
            col = f"answer.ipip_{item['id']}"
            raw = df[col].iloc[0]
            if raw is None or str(raw) == "nan":
                raw = 3  # neutral fallback
            else:
                raw = float(raw)
            if item["reverse"]:
                total += (6 - raw)
            else:
                total += raw
        scores[domain] = total / len(items)
    return scores


def level_from_score(score):
    """Convert 1-5 score to high/average/low."""
    if score >= 3.75:
        return "high"
    elif score <= 2.25:
        return "low"
    return "average"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate BFI-2 personality via Mini-IPIP")
    parser.add_argument("--model", default="stepfun/step-3.5-flash")
    parser.add_argument("--profiles", nargs="*", default=None)
    args = parser.parse_args()

    profiles_to_test = args.profiles or list(PROFILES.keys())
    model = Model(args.model, service_name="open_router", max_tokens=100000)
    survey = build_mini_ipip_survey()
    factory = PersonalityFactory()

    print(f"\n{'='*65}", flush=True)
    print("PERSONALITY VALIDATION TEST", flush=True)
    print(f"Assignment: BFI-2-Expanded | Measurement: Mini-IPIP (20 items)", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"{'='*65}\n", flush=True)

    all_results = []

    for profile_name in profiles_to_test:
        agent = factory.create_profile(profile_name, n=1)[0]
        assigned = PROFILES[profile_name]

        # Map weights to high/low/average for comparison
        assigned_levels = {}
        for d, w in assigned.items():
            if w >= 1.5:
                assigned_levels[d] = "high"
            elif w <= -1.5:
                assigned_levels[d] = "low"
            else:
                assigned_levels[d] = "average"

        print(f"Profile: {profile_name}", flush=True)
        print(f"  Assigned: {', '.join(f'{d[:3]}={l}' for d, l in assigned_levels.items())}", flush=True)

        r = survey.by(agent).by(model).run()
        df = r.to_pandas()

        scores = score_results(df)
        measured_levels = {d: level_from_score(s) for d, s in scores.items()}
        matches = sum(1 for d in DOMAINS if assigned_levels[d] == measured_levels[d])

        scores_str = " ".join(f"{d[:3]}={s:.1f}" for d, s in scores.items())
        print(f"  Measured: {scores_str}", flush=True)
        print(f"  Levels:   {', '.join(f'{d[:3]}={l}' for d, l in measured_levels.items())}", flush=True)
        print(f"  Match:    {matches}/5", flush=True)
        print(flush=True)

        all_results.append({
            "profile": profile_name,
            "assigned_levels": assigned_levels,
            "measured_scores": scores,
            "measured_levels": measured_levels,
            "matches": matches,
        })

    # Summary
    print(f"{'='*65}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"{'Profile':<12s} {'EXT':>5s} {'AGR':>5s} {'CON':>5s} {'NEU':>5s} {'OPE':>5s} {'Match':>6s}", flush=True)
    print("-" * 52, flush=True)
    for r in all_results:
        s = r["measured_scores"]
        print(
            f"{r['profile']:<12s} "
            f"{s['extraversion']:5.2f} {s['agreeableness']:5.2f} "
            f"{s['conscientiousness']:5.2f} {s['neuroticism']:5.2f} "
            f"{s['openness']:5.2f} "
            f"{r['matches']}/5",
            flush=True,
        )

    total = sum(r["matches"] for r in all_results)
    possible = len(all_results) * 5
    print(f"\nOverall: {total}/{possible} ({total/possible*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
