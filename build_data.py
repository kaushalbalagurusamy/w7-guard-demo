"""Bake a self-contained data snapshot into demo/data/snapshot.json. Rerun after new evals/results/* dirs land."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data" / "snapshot.json"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    scenarios = jl(ROOT / "ablations/prompt_ceiling/scenarios.jsonl")
    outputs = {}  # source label -> {"<scenario>-t<turn>": raw}
    names = {"sol61": "Sol 6.1 (zero-shot, recorded)", "argon4": "Argon 4 (zero-shot, recorded)"}
    for r in jl(ROOT / "ablations/prompt_ceiling/results/raw.jsonl"):
        if r["strategy"] == "zero_shot":
            outputs.setdefault(names[r["subject"]], {})[f"{r['id']}-t{r['turn']}"] = r["raw"]
    for d in sorted((ROOT / "evals/results").glob("*/transcripts.jsonl")):
        label = d.parent.name.split("__")[0].replace("_", "/", 1) + " (recorded outputs)"
        outputs[label] = {r["id"]: r["raw"] for r in jl(d)}
    spec = (ROOT / "docs/BEHAVIOR_SPEC.md").read_text().split("\n\n")
    summary = next(p for p in spec if p.startswith("**Behavior Spec"))
    OUT.write_text(json.dumps({"spec": summary, "scenarios": scenarios, "outputs": outputs}))
    print(OUT, len(scenarios), "scenarios;", {k: len(v) for k, v in outputs.items()})


if __name__ == "__main__":
    main()
