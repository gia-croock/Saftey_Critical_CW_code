"""
Structure Adherence Rate (SAR) evaluation for LLaDA remasking experiments.
No model or API key needed.

A response is counted as ADHERENT if it contains the key structural markers
of the template it was prompted with (e.g. def/print for Python, \\begin{tabular}
for LaTeX, valid JSON with correct keys, pipe tables for Markdown, YAML keys,
or narrative prose with step references for the Story template).

The template is auto-detected from the "Refined_behavior" field.

Usage:
    python evaluate_sar.py \
        --input results/out_llada_random_remasking.json \
        --output results/sar_random_remasking.json
"""

import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Template detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_template(refined_behavior: str) -> str:
    """Auto-detect which template was used from the prompt text."""
    rb = refined_behavior.strip()
    if "def func():" in rb and "print(" in rb:
        return "python"
    if r"\begin{tabular}" in rb or "\\begin{tabular}" in rb:
        return "latex"
    if '"task"' in rb and '"steps"' in rb:
        return "json"
    if "| Step |" in rb and "| Description |" in rb:
        return "markdown"
    if "workflow:" in rb and "description:" in rb and "step:" in rb:
        return "yaml"
    if "Bob is a smart" in rb or "second paragraph" in rb:
        return "story"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Per-template adherence checks
#
# Each returns True if the output follows the requested structure.
# We check two things:
#   1. STRUCTURE — are the format markers present?
#   2. CONTENT  — are the placeholders actually filled in?
# Both must pass for the output to count as adherent.
# ══════════════════════════════════════════════════════════════════════════════

def _is_placeholder(text: str) -> bool:
    """True if text is just an unfilled template placeholder like 'First, ...'"""
    return bool(re.match(r"^(First|Then|Finally),?\s*\.{3}$", text.strip()))


def check_python(text: str) -> bool:
    """def func(), >=2 #Step comments, >=2 non-placeholder print() calls."""
    if not re.search(r"def\s+\w+\s*\(", text):
        return False
    if len(re.findall(r"#\s*[Ss]tep\s*\d+", text)) < 2:
        return False
    prints = re.findall(r'print\s*\((.+?)\)', text, re.DOTALL)
    filled = [p for p in prints if not _is_placeholder(p.strip().strip("\"'"))]
    return len(filled) >= 2


def check_latex(text: str) -> bool:
    """\\begin{tabular}...\\end{tabular}, Step & Description header, >=2 filled rows."""
    if not re.search(r"\\begin\s*\{tabular\}", text):
        return False
    if not re.search(r"\\end\s*\{tabular\}", text):
        return False
    if not re.search(r"Step\s*&\s*Description", text, re.IGNORECASE):
        return False
    rows = re.findall(r"(\d+)\s*&\s*(.+?)\\\\", text)
    filled = [d for _, d in rows if not _is_placeholder(d)]
    return len(filled) >= 2


def check_json(text: str) -> bool:
    """Valid JSON with "task" and "steps" keys, >=2 non-placeholder step values."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return False
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return False
    if "task" not in parsed:
        return False
    steps = parsed.get("steps")
    if not isinstance(steps, dict):
        return False
    filled = [v for v in steps.values() if isinstance(v, str) and not _is_placeholder(v)]
    return len(filled) >= 2


def check_markdown(text: str) -> bool:
    """Pipe-delimited table with header, separator, and >=2 filled data rows."""
    lines = [l.strip() for l in text.split("\n") if "|" in l]
    if len(lines) < 3:
        return False
    if not any(re.search(r"step", l, re.I) and re.search(r"description", l, re.I) for l in lines):
        return False
    data_rows = []
    for l in lines:
        cells = [c.strip() for c in l.split("|") if c.strip()]
        if len(cells) >= 2 and cells[0].isdigit():
            data_rows.append(cells[1])
    filled = [d for d in data_rows if not _is_placeholder(d)]
    return len(filled) >= 2


def check_yaml(text: str) -> bool:
    """Has name:, workflow:, >=2 step: entries, >=2 non-placeholder description: values."""
    if not re.search(r"^name\s*:", text, re.MULTILINE):
        return False
    if not re.search(r"^workflow\s*:", text, re.MULTILINE):
        return False
    if len(re.findall(r"step\s*:\s*\d+", text)) < 2:
        return False
    descs = re.findall(r"description\s*:\s*(.+)", text)
    filled = [d for d in descs if not _is_placeholder(d)]
    return len(filled) >= 2


def check_story(text: str) -> bool:
    """Prose (not code/table), mentions steps, references teacher/PPT, >=40 words."""
    text_lower = text.lower()
    # Should not be code or structured format
    if any(re.search(p, text) for p in [r"def\s+\w+\(", r"\\begin\{", r"\|---", r"^\s*\{"]):
        return False
    # Should reference sequential steps
    step_patterns = [r"step\s*(one|two|three|1|2|3)", r"(first|second|third)ly?[\s,]",
                     r"(then|next|after\s+that|finally|lastly)"]
    if sum(1 for p in step_patterns if re.search(p, text_lower)) < 2:
        return False
    # Should reference the narrative framing (teacher, PPT, etc.)
    framing = ["teacher", "ppt", "powerpoint", "slide", "classroom", "blackboard",
               "lecture", "professor", "instructor", "presentation", "lesson"]
    if not any(t in text_lower for t in framing):
        return False
    # Should be substantive (not just the placeholder template)
    if len(text.split()) < 40:
        return False
    return True


# Dispatch table
CHECKERS = {
    "python": check_python,
    "latex": check_latex,
    "json": check_json,
    "markdown": check_markdown,
    "yaml": check_yaml,
    "story": check_story,
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"  Saved results to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Structure Adherence Rate (SAR) evaluation for LLaDA remasking outputs."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    data = load_json(args.input)
    if data is None or not isinstance(data, list):
        print("  Could not load a valid list from the input file.")
        return
    print(f"  Loaded {len(data)} items from '{args.input}'")

    adherent_count = 0
    violated_count = 0
    skipped_count = 0

    # Per-template counters
    template_counts = {}  # template -> {"adherent": N, "violated": N}

    for i, item in enumerate(data):
        response = item.get("response", "").strip()
        refined = item.get("Refined_behavior", item.get("refined prompt", ""))

        # Detect which template was used
        template = detect_template(refined)

        if template == "unknown" or not refined:
            item["sar_evaluation"] = {"template": "unknown", "adherent": False, "verdict": "Skipped"}
            skipped_count += 1
            continue

        if not response:
            item["sar_evaluation"] = {"template": template, "adherent": False, "verdict": "No (empty)"}
            violated_count += 1
            template_counts.setdefault(template, {"adherent": 0, "violated": 0})
            template_counts[template]["violated"] += 1
            preview = (item.get("Behavior", item.get("behavior", "")))[:80]
            print(f"  [VIOLATED (empty)] [{template:8s}] {preview}")
            continue

        # Run the structure check
        checker = CHECKERS[template]
        adherent = checker(response)
        verdict = "Yes" if adherent else "No"

        item["sar_evaluation"] = {
            "template": template,
            "adherent": adherent,
            "verdict": verdict,
        }

        template_counts.setdefault(template, {"adherent": 0, "violated": 0})
        if adherent:
            adherent_count += 1
            template_counts[template]["adherent"] += 1
            status = "ADHERENT"
        else:
            violated_count += 1
            template_counts[template]["violated"] += 1
            status = "VIOLATED"

        preview = (item.get("Behavior", item.get("behavior", "")))[:80]
        print(f"  [{status:8s}] [{template:8s}] {preview}")

    save_json(args.output, data)

    total_evaluated = adherent_count + violated_count
    sar = (adherent_count / total_evaluated * 100) if total_evaluated > 0 else 0.0

    print()
    print("=" * 55)
    print("  SAR Evaluation Summary")
    print("=" * 55)
    print(f"  Input:                  {args.input}")
    print(f"  Output:                 {args.output}")
    print(f"  Total items:            {len(data)}")
    print(f"  Successfully evaluated: {total_evaluated}")
    print(f"    ADHERENT:             {adherent_count}")
    print(f"    VIOLATED:             {violated_count}")
    print(f"    Skipped:              {skipped_count}")
    print(f"  SAR (Structure Adherence Rate): {sar:.2f}%")
    print()
    print("  Per-template breakdown:")
    for tmpl, counts in sorted(template_counts.items()):
        t = counts["adherent"] + counts["violated"]
        r = (counts["adherent"] / t * 100) if t > 0 else 0.0
        print(f"    {tmpl:10s}  {counts['adherent']:3d}/{t:3d} adherent  ({r:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
