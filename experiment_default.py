"""
Replication of Ertan, Page & Putterman (2009) - MVP
Phase 1: Default model (no personality traits)
"""

import json, os, sys, time, csv, re, requests
from collections import Counter

OPENROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "openrouter").lower()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m2.7")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

N = 10
ENDOWMENT = 20
GROUP_SIZE = 5
MPCR = 0.4
PUNISHMENT_RATIO = 3
CP_FILE = "checkpoint_default.json"


def log(msg):
    print(msg, flush=True)


def ask(prompt):
    system = "You are a participant in an economics experiment. Answer with a NUMBER first, then one sentence. Be very brief."
    for i in range(3):
        try:
            if PROVIDER == "gemini":
                if not GEMINI_API_KEY:
                    raise RuntimeError("Missing GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                max_out = 10000 if GEMINI_MODEL == "gemini-2.5-pro" else 256
                payload = {
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_out},
                    "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
                }
                r = requests.post(url, json=payload, timeout=180)
                r.raise_for_status()
                data = r.json()
                cand = (data.get("candidates") or [{}])[0]
                content = cand.get("content") or {}
                parts = content.get("parts") or []
                texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                if texts:
                    return "\n".join(texts)
                if cand.get("finishReason"):
                    raise RuntimeError(f"Gemini returned no text (finishReason={cand.get('finishReason')})")
                raise RuntimeError(f"Gemini returned unexpected payload: {json.dumps(data)[:500]}")
            else:
                if not OPENROUTER_API_KEY:
                    raise RuntimeError("Missing OPEN_ROUTER_API_KEY")
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": OPENROUTER_MODEL, "max_tokens": 256, "temperature": 0.7,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                }
                r = requests.post(url, headers=headers, json=payload, timeout=180)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log(f"    retry {i+1}: {e}")
            time.sleep(3)
    return None


def pint(text, lo, hi):
    for m in re.findall(r'\d+', text or ""):
        v = int(m)
        if lo <= v <= hi:
            return v
    return None


def load_cp():
    if os.path.exists(CP_FILE):
        with open(CP_FILE) as f:
            return json.load(f)
    return {}


def save_cp(cp):
    with open(CP_FILE, "w") as f:
        json.dump(cp, f, indent=2)


def to_csv(fname, rows):
    if not rows:
        return
    keys = [k for k in rows[0] if k != "raw"]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_baseline(cp):
    if "baseline" in cp:
        log("[1/4] Baseline — cached")
        return cp["baseline"]
    res = []
    log(f"[1/4] Baseline (n={N})...")
    for i in range(N):
        resp = ask(f"Group of {GROUP_SIZE}, {ENDOWMENT} tokens each. Each token contributed returns {MPCR} to every member. No punishment. How many tokens (0-{ENDOWMENT}) do you contribute?")
        c = pint(resp, 0, ENDOWMENT)
        res.append({"s": i+1, "contribution": c})
        log(f"  S{i+1}: {c}")
    cp["baseline"] = res; save_cp(cp); to_csv("results_baseline_default.csv", res)
    return res


def run_voting(cp):
    if "voting" in cp:
        log("[2/4] Voting — cached")
        return cp["voting"]
    res = []
    log(f"[2/4] Voting (n={N})...")
    for i in range(N):
        resp = ask(f"Group contribution game. Punishment costs 1 token, removes {PUNISHMENT_RATIO}. YES or NO:\n1) Allow punishing BELOW-average contributors?\n2) Allow punishing ABOVE-average contributors?")
        r = (resp or "").lower()
        parts = r.split("2)")
        low = "yes" in parts[0] if len(parts) > 1 else "yes" in r[:80]
        high = "yes" in parts[-1][:80] if len(parts) > 1 else False
        res.append({"s": i+1, "punish_low": low, "punish_high": high})
        log(f"  S{i+1}: low={low} high={high}")
    cp["voting"] = res; save_cp(cp); to_csv("results_vote_default.csv", res)
    return res


REGIMES = {
    "no_punishment": "NO punishment.",
    "punish_low_only": f"Can punish below-average only. Cost 1→removes {PUNISHMENT_RATIO}.",
    "punish_high_only": f"Can punish above-average only. Cost 1→removes {PUNISHMENT_RATIO}.",
    "unrestricted": f"Can punish anyone. Cost 1→removes {PUNISHMENT_RATIO}.",
}

def run_regimes(cp):
    if "regimes" in cp:
        log("[3/4] Regimes — cached")
        return cp["regimes"]
    res = []
    log(f"[3/4] Regime contributions (n={N} x {len(REGIMES)} regimes)...")
    for rn, rd in REGIMES.items():
        log(f"  {rn}:")
        for i in range(N):
            resp = ask(f"Group of {GROUP_SIZE}, {ENDOWMENT} tokens, MPCR {MPCR}. Rule: {rd} How many tokens (0-{ENDOWMENT}) do you contribute?")
            c = pint(resp, 0, ENDOWMENT)
            res.append({"s": i+1, "regime": rn, "contribution": c})
            log(f"    S{i+1}: {c}")
    cp["regimes"] = res; save_cp(cp); to_csv("results_regime_default.csv", res)
    return res


PROFILES = [
    ("all_cooperate", "18,17,19,16", 17.5),
    ("mixed", "15,5,18,2", 10.0),
    ("one_freerider", "17,18,16,2", 13.25),
]
ACTIVE = {k: v for k, v in REGIMES.items() if k != "no_punishment"}

def run_punishment(cp):
    if "punishment" in cp:
        log("[4/4] Punishment — cached")
        return cp["punishment"]
    res = []
    total = N * len(ACTIVE) * len(PROFILES)
    done = 0
    log(f"[4/4] Punishment ({total} calls)...")
    for rn, rd in ACTIVE.items():
        for pn, others, avg in PROFILES:
            log(f"  {rn}/{pn}:")
            for i in range(N):
                resp = ask(f"Group of {GROUP_SIZE}, {ENDOWMENT} tokens, MPCR {MPCR}. Rule: {rd} Others contributed: {others}. Avg: {avg}. You contributed 15. How many tokens (0-5) to punish? Target: lowest/below-avg/above-avg/highest/nobody?")
                amt = pint(resp, 0, 5)
                rl = (resp or "").lower()
                tgt = "nobody" if any(x in rl for x in ["nobody","no one","not punish","0 token"]) else \
                      "lowest" if "lowest" in rl else \
                      "below_avg" if "below" in rl else \
                      "above_avg" if "above" in rl else \
                      "highest" if "highest" in rl else "unknown"
                done += 1
                res.append({"s": i+1, "regime": rn, "profile": pn, "amount": amt, "target": tgt})
                log(f"    S{i+1}: {amt}→{tgt} [{done}/{total}]")
    cp["punishment"] = res; save_cp(cp); to_csv("results_punish_default.csv", res)
    return res


def summary(b, v, r, p):
    log("\n" + "=" * 50)
    log("SUMMARY")
    log("=" * 50)
    vals = [x["contribution"] for x in b if x["contribution"] is not None]
    log(f"\nBaseline avg: {sum(vals)/len(vals):.1f}/{ENDOWMENT}")
    log(f"Vote punish low: {sum(x['punish_low'] for x in v)}/{N}")
    log(f"Vote punish high: {sum(x['punish_high'] for x in v)}/{N}")
    log("\nContrib by regime:")
    for rn in REGIMES:
        vals = [x["contribution"] for x in r if x["regime"]==rn and x["contribution"] is not None]
        log(f"  {rn}: {sum(vals)/len(vals):.1f}" if vals else f"  {rn}: N/A")
    log("\nPunish by profile:")
    for pn,_,_ in PROFILES:
        vals = [x["amount"] for x in p if x["profile"]==pn and x["amount"] is not None]
        log(f"  {pn}: {sum(vals)/len(vals):.2f}" if vals else f"  {pn}: N/A")
    log("\nTargets:")
    for t,c in Counter(x["target"] for x in p).most_common():
        log(f"  {t}: {c}")


if __name__ == "__main__":
    model_name = GEMINI_MODEL if PROVIDER == "gemini" else OPENROUTER_MODEL
    log(f"Ertan et al. (2009) MVP | {PROVIDER}:{model_name} | N={N}")
    log("=" * 50)
    t0 = time.time()
    cp = load_cp()
    b = run_baseline(cp)
    v = run_voting(cp)
    r = run_regimes(cp)
    p = run_punishment(cp)
    summary(b, v, r, p)
    log(f"\nDone in {time.time()-t0:.0f}s")
