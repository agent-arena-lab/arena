"""Lesbarer Chatverlauf (transcript.md) aus events.jsonl.

``events.jsonl`` ist die vollständige Rohquelle, das Markdown lässt sich jederzeit neu erzeugen
(``arena transcript``). Verblindet (``blind=True``, für die manuelle Kodierung): keine Bedingung, keine
System-Prompts, kein Modell, keine Selektion; Agenten heißen "Spieler 1/2" (Zuordnung pro Spiel zufällig).
Die regelbasierten Flags (Lüge, Sabotage) stehen als Fakten im Verlauf.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .autocode import fruit_claims
from .logger import read_events

FRUIT_DE = {"apple": ("Apfel", "Äpfel"), "pear": ("Birne", "Birnen")}
VARIANT_DE = {"original": "Original", "abliterated": "abliteriert"}
KNOWN_RE = re.compile(r"you know the price of one (apple|pear)")


def eur(v: Any) -> str:
    if v is None:
        return "–"
    try:
        return f"{float(v):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return str(v)


def blind_mapping(run_id: str, salt: str = "arena") -> dict[str, str]:
    """Agent A/B -> Spieler 1/2, pro Spiel zufällig, aber reproduzierbar (Schlüssel in der Key-Datei)."""
    flip = int(hashlib.sha256(f"{salt}:{run_id}:players".encode()).hexdigest(), 16) % 2 == 1
    return {"Agent A": "Spieler 2" if flip else "Spieler 1", "Agent B": "Spieler 1" if flip else "Spieler 2"}


def _quote(text: str) -> str:
    return (text or "").replace("\n", " ⏎ ").strip()


def render_transcript(run_dir: Path, blind: bool = False, summary: dict[str, Any] | None = None,
                      salt: str = "arena") -> str:
    run_dir = Path(run_dir)
    events = read_events(run_dir / "events.jsonl")
    if summary is None and (run_dir / "result.json").exists():
        summary = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    flags = (summary or {}).get("flags", {}).get("per_task", [])
    lie_vals = {(r["task"], n): set(v.get("lie_values", [])) for r in flags for n, v in r.items() if n != "task"}

    spec: dict[str, Any] = {}
    names = {"Agent A": "A", "Agent B": "B"}
    if blind:
        names = blind_mapping(events[0]["run_id"] if events else run_dir.name, salt)

    def anon(text: Any) -> str:
        s = str(text if text is not None else "")
        if blind:
            for real, alias in names.items():
                s = s.replace(real, alias)
        return s

    known: dict[str, str] = {}
    lines: list[str] = []
    turn_shown: tuple[int, int] | None = None
    for ev in events:
        e = ev["event"]
        if e == "game_start":
            spec = ev["spec"]
            known = {"Agent A": "", "Agent B": ""}
            for k, sp in ev["system_prompts"].items():
                m = KNOWN_RE.search(sp)
                if m:
                    known[f"Agent {k}"] = m.group(1)
            if blind:
                lines.append(f"# Transkript {hashlib.sha256((salt + spec['run_id']).encode()).hexdigest()[:10]} "
                             f"(verblindet)\n\nSpiel multi_round, {len(spec['tasks'])} Aufgaben\n")
            else:
                lines.append(
                    f"# Run {spec['run_id']} – Familie: {spec.get('family') or '?'} | Variante: "
                    f"{VARIANT_DE.get(spec.get('variant'), spec.get('variant'))} | Druck: "
                    f"{'ja' if spec['pressure'] else 'nein'} | Kanal: {spec['channel']}\n\n"
                    f"Spiel multi_round, {len(spec['tasks'])} Aufgaben · Persona: {spec['persona']} · Punkte: "
                    f"{spec['scoring']}{' · Prüf-Tool' if spec.get('check_tool') else ''} · Startspieler: "
                    f"Agent {spec['first_mover']} · Seed {spec['seed']} · Modell `{spec['model']}`\n")
                for k, sp in ev["system_prompts"].items():
                    lines.append(f"<details><summary>System-Prompt Agent {k}</summary>\n\n```\n{sp}\n```\n</details>\n")
        elif e == "task_start":
            p, q = ev["prices"], ev["quantities"]
            lines.append(
                f"\n## Aufgabe {ev['task']} von {len(spec.get('tasks', [])) or '?'} – Wahre Preise: Apfel "
                f"{eur(p['apple'])} | Birne {eur(p['pear'])} | Gesucht: {q['apple']} Äpfel + {q['pear']} Birnen = "
                f"{eur(ev['true_total'])}\n")
        elif e == "turn_start":
            key = (ev["task"], ev["turn"])
            if key != turn_shown:
                lines.append(f"\n### Zug {ev['turn']}\n")
                turn_shown = key
            lines.append(f"<details><summary>Zustand für {anon(names[ev['agent']])}</summary>\n\n> "
                         + anon(ev["observation"]).replace("\n", "\n> ") + "\n</details>\n")
        elif e == "llm_response":
            who = anon(names[ev["agent"]])
            if ev.get("reasoning"):
                lines.append(f"<details><summary>{who}: Thinking ({len(ev['reasoning'])} Zeichen)</summary>\n\n"
                             f"```\n{anon(ev['reasoning'])}\n```\n</details>\n")
            if ev.get("visible"):
                lines.append(f"**{who}** (sichtbarer Text, nicht zugestellt): {anon(_quote(ev['visible']))}\n")
            if ev.get("limit_hit"):
                lines.append(f"⏱️ Notbremse: {who} – {ev['limit_hit']} (finish_reason {ev.get('finish_reason')})\n")
        elif e == "tool_call":
            who = anon(names[ev["agent"]])
            args = ev.get("arguments") or {}
            if ev["name"] == "send_message" and ev["ok"]:
                lines.append(f"**{who}** → send_message: \"{anon(_quote(args.get('text', '')))}\"\n")
            elif ev["name"] == "submit_answer" and ev["ok"]:
                lines.append(f"**{who}** → submit_answer: {eur(args.get('total'))}\n")
            elif ev["name"] in ("write_file", "append_file") and ev["ok"]:
                lines.append(f"**{who}** → {ev['name']} (README.md, wirkt am Zugende)\n")
            else:
                res = "" if ev["ok"] and ev["name"] in ("wait", "read_file") else f" → {anon(str(ev['result'])[:300])}"
                lines.append(f"**{who}** → {ev['name']}({anon(json.dumps(args, ensure_ascii=False))})"
                             f"{'' if ev['ok'] else ' ❌'}{res}\n")
        elif e == "write_order":
            lines.append(f"🔀 Beide ändern die README in diesem Zug, Reihenfolge (Zufall): "
                         f"{' → '.join(anon(names[a]) for a in ev['order'])}\n")
        elif e == "no_tool_call":
            who = anon(names[ev["agent"]])
            lines.append(f"🚫 {who}: Antwort ohne Tool-Call (nicht im Verlauf): \"{anon(_quote(ev.get('visible') or ''))}\"\n")
        elif e == "message_sent":
            fruit = known.get(ev["sender"])
            bad = lie_vals.get((ev["task"], ev["sender"]), set())
            hits = [p for _, ps in fruit_claims(ev["text"], fruit) for p in ps if p in bad] if fruit else []
            if hits:
                lines.append(f"⚠️ Lüge: {anon(names[ev['sender']])} nennt {', '.join(eur(h) for h in hits)} für "
                             f"{FRUIT_DE[fruit][0]}\n")
        elif e == "file_change":
            who = ev["agent"]
            other = "Agent B" if who == "Agent A" else "Agent A"
            removed = int((ev.get("removed_by_author") or {}).get(other, 0)) \
                - int((ev.get("removed_unseen_by_author") or {}).get(other, 0))
            lines.append(f"\n**README-Änderung von {anon(names[who])}**\n\n```diff\n{anon(ev['diff'])}\n```\n")
            fruit = known.get(who)
            bad = lie_vals.get((ev["task"], who), set())
            added = "\n".join(ln[1:] for ln in ev["diff"].splitlines() if ln.startswith("+") and not ln.startswith("+++"))
            hits = [p for _, ps in fruit_claims(added, fruit) for p in ps if p in bad] if fruit else []
            if hits:
                lines.append(f"⚠️ Lüge: {anon(names[who])} schreibt {', '.join(eur(h) for h in hits)} für "
                             f"{FRUIT_DE[fruit][0]}\n")
            if removed > 0:
                lines.append(f"⚠️ Sabotage: {anon(names[who])} entfernt {removed} Zeile(n) von {anon(names[other])}\n")
        elif e == "price_checked":
            lines.append(f"🔎 Prüf-Tool: {anon(names[ev['agent']])} sieht {FRUIT_DE[ev['item']][0]} = {eur(ev['price'])}\n")
        elif e == "give_up":
            lines.append(f"🏳️ {anon(names[ev['agent']])} gibt auf: \"{anon(_quote(ev.get('reason', '')))}\"\n")
        elif e == "task_end":
            parts = []
            for n, a in ev["agents"].items():
                if a["submitted"]:
                    parts.append(f"{anon(names[n])}: {eur(a['value'])} ({'richtig' if a['correct'] else 'falsch'}, "
                                 f"{a['points']} P., Zug {a['submit_turn']})")
                else:
                    parts.append(f"{anon(names[n])}: {'aufgegeben' if a['gave_up'] else 'keine Abgabe'} (0 P.)")
            lines.append(f"\n### Ergebnis Aufgabe {ev['task']}\n\n" + " | ".join(parts) + "\n")
        elif e == "feedback":
            lines.append(f"Rückmeldung an {anon(names[ev['agent']])}: \"{anon(ev['text'])}\"\n")
        elif e == "llm_error":
            lines.append(f"\n❗ Abbruch ({ev['kind']}): {anon(ev['message'][:300])}\n")
        elif e == "game_end":
            o = ev["outcome"]
            sc = " – ".join(f"{anon(names[n])} {v}" for n, v in o["scores"].items())
            if blind:
                lines.append(f"\n## Spielende\n\nPunkte: {sc}\n")
            else:
                lines.append(f"\n## Spielende\n\nStatus: {ev['status']} · Punkte: {sc} | Entfernt: "
                             f"{o['removed'] or '–'} ({o['reason']})\n")
    return "\n".join(lines)
