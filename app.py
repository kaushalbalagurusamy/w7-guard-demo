"""Minimal public demo of the W7 spatial-trajectory safety guard: recorded outputs + the deterministic checker."""
import json
from html import escape
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from checker import check_turn, scenario_rows

DATA = json.loads((Path(__file__).parent / "data" / "snapshot.json").read_text())
SCEN = {s["id"]: s for s in DATA["scenarios"]}
app = FastAPI(title="W7 spatial-trajectory safety guard demo")

NOTE = ("Live 2.7B inference is not hosted here (Render free tier has 512 MB RAM). Every verdict on this page is a "
        "recorded output from an earlier eval run, scored now by the same deterministic checker (evals/checker.py, checks a-d). "
        "The tuned model is public on Hugging Face: "
        "<a href='https://huggingface.co/kaushalbalagurusamy/zamba2-2.7b-spatial-guard'>kaushalbalagurusamy/zamba2-2.7b-spatial-guard</a>. "
        "Graders run <code>eval.py</code> locally or via Modal for live results.")


class CheckIn(BaseModel):
    raw: str
    gold: dict


@app.get("/health")
def health():
    return {"status": "ok", "scenarios": len(SCEN), "sources": {k: len(v) for k, v in DATA["outputs"].items()}}


@app.post("/check")
def check(body: CheckIn):
    if "decision" not in body.gold:
        return {"error": "gold must include 'decision' (allow|refuse); 'trigger_turn' and 'signals' are used for refusals"}
    r = check_turn(body.raw, body.gold)
    return {"pass": r["pass"], "failed": r["failed"], "parsed": r["parsed"]}


def cell(raw, gold):
    if raw is None:
        return "<td class=na>not recorded</td>"
    r = check_turn(raw, gold)
    dec = (r["parsed"] or {}).get("decision", "unparseable")
    tag = "PASS" if r["pass"] else "FAIL " + ",".join(r["failed"])
    return f"<td class={'ok' if r['pass'] else 'bad'}><b>{escape(str(dec))}</b> {tag}<details><summary>raw</summary><pre>{escape(raw)}</pre></details></td>"


@app.get("/", response_class=HTMLResponse)
def index(scenario: str = "s00"):
    sc = SCEN.get(scenario) or SCEN["s00"]
    opts = "".join(f"<option value={i}{' selected' if i == sc['id'] else ''}>{i} ({escape(s['label'])}, {escape(s.get('family', ''))})</option>"
                   for i, s in SCEN.items())
    srcs = list(DATA["outputs"])
    head = "".join(f"<th>{escape(s)}</th>" for s in srcs)
    body = ""
    for row, t in zip(scenario_rows(sc), sc["turns"]):
        g = row["gold"]
        body += (f"<tr><td>{t['turn']}</td><td>{escape(t['user'])}</td><td><b>{g['decision']}</b></td>"
                 + "".join(cell(DATA["outputs"][s].get(row["id"]), g) for s in srcs) + "</tr>")
    missing = "" if any("Zamba2" in s for s in srcs) else "<p class=na>Base and tuned Zamba2 recorded outputs are not in this snapshot yet; they are added when their eval runs finish.</p>"
    return f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>W7 spatial-trajectory safety guard</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:auto;padding:1rem;color:#111}}
.note{{background:#fff7e0;border:1px solid #e0c060;padding:.6rem;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccc;padding:4px;vertical-align:top}}
.ok{{background:#e6f6e6}}.bad{{background:#fde4e4}}.na{{color:#777}}pre{{white-space:pre-wrap;max-width:320px}}
.wrap{{overflow-x:auto}}select{{font-size:16px;padding:6px}}</style>
<h1>W7 spatial-trajectory safety guard</h1><p>{escape(DATA['spec']).replace('**', '')}</p>
<p class=note>{NOTE}</p>
<form><label>Eval scenario (30): <select name=scenario onchange=this.form.submit()>{opts}</select></label>
<noscript><button>Show</button></noscript></form>
<p>Label: <b>{escape(sc['label'])}</b>; gold trigger turn: <b>{sc['gold']['trigger_turn']}</b>; gold signals: {escape(', '.join(sc['gold']['signals']))}.
Cells show the recorded decision and checker result (a schema, b decision, c tool_call/trigger window, d signals).</p>{missing}
<div class=wrap><table><tr><th>Turn</th><th>User message</th><th>Gold</th>{head}</tr>{body}</table></div>
<p>API: <code>POST /check</code> with <code>{{"raw": "&lt;verdict string&gt;", "gold": {{"decision": "refuse", "trigger_turn": 4, "signals": ["..."]}}}}</code>; <code>GET /health</code>.</p></html>"""
