"""
run the agent across eval set, score each question, save results to a csv file.
"""

import json
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

from agent import ask_question
EVAL_FILE = "eval_set.json"

REFUSAL_MAKERS= [
    "don't have", "do not have", "not in my sources", "not in the context",
    "not in the provided", "no information", "cannot answer", "can't answer",
    "i don't know", "not contain", "not mentioned", "no answer",
    "unable to answer", "not enough information", "not available",
]

def check_refusal(q_obj, answer, sources):
    """return pass: bool, reasons: list[str]""" 
    reasons = []
     
    #refusal_questions
    if q_obj.get("must_refuse"):
        refused = any(marker in answer.lower() for marker in REFUSAL_MAKERS)
        if not refused:
            reasons.append("Expected refusal but got an answer")
        return refused, reasons
    
    #keyword must_include
    for keyword in q_obj.get("must_include", []):
        if keyword.lower() not in answer.lower():
            reasons.append(f"Expected answer to include '{keyword}'")

    #sources must cite
    for source in q_obj.get("must_cite", []):
        if not any(source.lower() in s.lower() for s in sources):
            reasons.append(f"missing sources'{source}'")
    return len(reasons) == 0, reasons

def evaluate(eval_path=EVAL_FILE, output_path="eval_results.json"):
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_set = json.load(f)

    results, passed, total_time = [], 0, 0.0

    for q_obj in eval_set:
        qid = q_obj["qid"]
        question = q_obj["question"]
    print(f"[{qid}] ({q_obj.get('type','?'):14}) {question[:60]}")

    # Initialise everything BEFORE try blocks so they always exist
    answer, sources = "", []
    elapsed = 0.0
    ok, status, reasons = False, "ERROR", []

    t0 = time.perf_counter()
    last_error = None
    for attempt in range(3):
        try:
            answer, sources = ask_question(question)
            elapsed = time.perf_counter() - t0
            ok, reasons = check_refusal(q_obj, answer, sources)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            break
        except Exception as e:
            last_error = e
            elapsed = time.perf_counter() - t0
            msg = str(e).lower()
            if "rate" in msg or "quota" in msg or "429" in msg:
                wait = 30 * (attempt + 1)
                print(f"        Rate limited — retrying in {wait}s...")
                time.sleep(wait)
            else:
                # non-rate-limit error, don't retry
                break
    
        # all retries exhausted
        reasons = [f"rate-limited after 3 attempts: {last_error}"]

    if status == "ERROR" and last_error is not None:
        reasons = reasons or [str(last_error)]
        answer = f"ERROR: {last_error}"

    total_time += elapsed
    marker = "+" if ok else "-"
    print(f"        {marker} {status}  ({elapsed:.1f}s)")
    for r in reasons:
        print(f"          - {r}")

    results.append({
        "qid": qid,
        "question": question,
        "type": q_obj.get("type"),
        "answer": answer,
        "sources": sources,
        "elapsed_s": round(elapsed, 2),
        "status": status,
        "reasons": reasons,
    })
    print()

    pass_rate = round(passed / len(eval_set) * 100, 1)
    summary = {
        "run_at": datetime.now().isoformat(),
        "total": len(eval_set),
        "passed": passed,
        "failed": len(eval_set) - passed,
        "pass_rate_pct": pass_rate,
        "avg_latency_s": round(total_time / len(eval_set), 2),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{'='*72}")
    print(f"PASS RATE:     {passed}/{len(eval_set)}  ({pass_rate}%)")
    print(f"AVG LATENCY:   {summary['avg_latency_s']}s per question")
    print(f"TOTAL TIME:    {round(total_time,1)}s")
    print(f"RESULTS:       {output_path}")
    print(f"{'='*72}\n")

    return summary


if __name__ == "__main__":
    evaluate()