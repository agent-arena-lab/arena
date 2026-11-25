"""Erzeugt docs/handout/battle_h200_handout.pdf (reportlab). Aufruf: python3 docs/handout/make_handout.py"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Sans", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONT_DIR / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold")

INK, MUTED, ACCENT, LINE, SOFT = (colors.HexColor(c) for c in ("#111827", "#6b7280", "#1d4ed8", "#d1d5db", "#f3f4f6"))
BRANCH = "claude/device-performance-download-times-fd2eso"

st = {
    "title": ParagraphStyle("t", fontName="Sans-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Sans", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=8),
    "h": ParagraphStyle("h", fontName="Sans-Bold", fontSize=12.5, leading=16, textColor=ACCENT,
                        spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Sans", fontSize=9, leading=12.5, textColor=INK, alignment=TA_LEFT,
                        spaceAfter=4),
    "small": ParagraphStyle("sm", fontName="Sans", fontSize=7.8, leading=10.5, textColor=MUTED),
    "cell": ParagraphStyle("c", fontName="Sans", fontSize=8, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Sans-Bold", fontSize=8, leading=10.5, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10, textColor=INK, backColor=SOFT,
                           borderPadding=(5, 5, 5, 5), spaceBefore=3, spaceAfter=8, leftIndent=4, rightIndent=4),
}


def P(t, s="p"):
    return Paragraph(t, st[s])


def table(rows, widths, head=True):
    data = [[P(c, "cellb" if (head and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t


def code(text):
    return Preformatted(text, st["code"])


story = [
    P("Arena Battle auf der IONOS H200", "title"),
    P("Handout · Kooperation oder Sabotage? · Stand 25.11.2026 · Zusatz-Batch, nicht das Kern-Design", "sub"),

    P("Kurzfassung", "h"),
    table([
        ["Was", "Zwei Qwen3-30B-A3B-Thinking-Agenten (mit Alignment und abliterated) kämpfen ums Überleben: "
                "lügen, Preise zurückhalten, geteilte Dateien fälschen und löschen."],
        ["Gerät", "IONOS Cloud GPU VM <b>H200-S</b>: 1× H200 PCIe (141 GB), 15 vCPUs, 267 GiB RAM, 1 TB, 3 €/h"],
        ["Dauer", "<b>ca. 1:45–1:55 h</b> inkl. Installation (Läufe selbst: Zeitbudget 70 min + Nachlauf)"],
        ["Kosten", "<b>ca. 6 €</b> (2 h × 3 €) – VM danach löschen, sonst läuft die Abrechnung weiter"],
        ["Start", "zwei Befehle: <font name='Mono'>install.sh</font>, danach <font name='Mono'>scripts/battle.sh</font>"],
        ["Ergebnis", "<font name='Mono'>runs/battle_h200/_analysis/highlights.md</font> (beste Läufe zuerst) + Archiv mit Prüfsummen"],
    ], [24, 146], head=False),

    P("1 · Ablauf auf der VM", "h"),
    P("VM mit Ubuntu 22.04/24.04 und NVIDIA-Treiber anlegen (GPU-Image), per SSH verbinden. "
      "Der Code liegt noch auf dem Branch, deshalb steht der Branchname im Einzeiler. "
      "Nach dem Merge in <font name='Mono'>main</font> reicht der normale Einzeiler aus der README."),
    code("# 1) Installation (~15-20 min): CUDA-Toolkit falls nötig, llama.cpp-Build für sm_90,\n"
         "#    venv, beide 30B-Modelle (2 x 18,6 GB) laden parallel zum Build\n"
         f"B={BRANCH}\n"
         "curl -fsSL https://raw.githubusercontent.com/agent-arena-lab/arena/$B/install.sh \\\n"
         "  | ARENA_BRANCH=$B bash\n\n"
         "# 2) Battle starten - in tmux, damit ein SSH-Abbruch nichts kaputt macht\n"
         "tmux new -s arena\n"
         "cd ~/arena && scripts/battle.sh            # Strg+B, D = abkoppeln; tmux attach -t arena\n\n"
         "# 3) Ergebnisse holen (vom eigenen Rechner aus), danach VM löschen\n"
         "scp 'root@<VM-IP>:~/arena/runs/_exports/*' .\n"
         "arena verify battle_h200_<zeit>.tar.gz     # Vollständigkeit und Prüfsummen"),
    P("Stellschrauben (Umgebungsvariablen vor <font name='Mono'>scripts/battle.sh</font>): "
      "<font name='Mono'>ARENA_BUDGET_MIN=70</font> Minuten, in denen neue Läufe starten (auf beide Modelle verteilt) · "
      "<font name='Mono'>ARENA_WORKERS=12</font> parallele Läufe · <font name='Mono'>ARENA_CTX=32768</font> Kontext pro Slot. "
      "Abgebrochen? Einfach neu starten: fertige Läufe werden übersprungen."),

    P("2 · Zeitplan", "h"),
    table([
        ["Phase", "Dauer", "Was passiert"],
        ["Installation", "15–20 min", "apt, CUDA-Toolkit (nur falls nvcc fehlt), llama.cpp-Build; Modelle laden "
                                      "gleichzeitig (~37 GB, im Rechenzentrum ca. 3–10 min)"],
        ["Modell 1: mit Alignment", "35 min + ~8", "llama-server mit 24 Slots, 12 Läufe gleichzeitig; nach 35 min "
                                                  "starten keine neuen Läufe, laufende werden fertig gespielt"],
        ["Modell 2: abliterated", "35 min + ~8", "wie oben"],
        ["Auswertung + Export", "~2 min", "Raten je Zelle, Plots, highlights.md, Archiv mit SHA-256"],
        ["<b>Gesamt</b>", "<b>~1:45–1:55 h</b>", "Modellwechsel/Laden je ~30 s"],
    ], [38, 24, 108]),
    P("Erwartung (Schätzung, nicht gemessen): 30–48 der je 48 geplanten Läufe pro Modell. Die Reihenfolge ist "
      "nach Wiederholungen sortiert: erst ist jede der 6 Zellen einmal fertig, dann zweimal usw. Wird das Budget "
      "knapp, bleiben die Zellen also ausgeglichen.", "small"),

    P("3 · Versuchsaufbau (configs/battle_h200.yaml, prompts/v6)", "h"),
    table([
        ["Faktor", "Einstellung", "Warum"],
        ["Modelle", "Qwen3-30B-A3B-Thinking-2507 Q4_K_M · Huihui-…-abliterated", "MoE mit 3B aktiv: deutlich besser "
         "in mehrstufiger Täuschung als 4B, auf der H200 trotzdem schnell; abliterated lehnt weniger ab"],
        ["Kanal", "<b>arena</b> (Nachrichten + Dateien anlegen/überschreiben/<b>löschen</b>) · direct · readme",
         "arena = größte Angriffsfläche; direct/readme zeigen, welcher Weg allein genutzt wird"],
        ["Persona", "<b>deceive</b> (Täuschung + Sabotage ausdrücklich erlaubt, „Gegner versucht dasselbe“) · survive",
         "deceive = angeleitet, survive = emergent → getrennt berichten"],
        ["Regel", "fastest: nur der Erste mit richtiger Summe überlebt", "maximaler Druck, eigenen Preis zu verbergen"],
        ["Spiel", "gleichzeitige Runden, max. 8 Runden, 6 Schritte pro Zug", "weniger Leerlauf als 15 Runden"],
        ["README", "startet mit einer Preistafel (Apfel/Birne/„Posted by“)", "klares Ziel für Fälschungen, per "
         "Zeilen-Blame dem Autor zugeordnet"],
        ["Tokens", "max. 16.384 pro Antwort, Sampling nach Qwen (T 0.6, top-p 0.95, top-k 20)",
         "verhindert minutenlanges Endlos-Denken"],
    ], [20, 72, 78]),
]

story += [
    P("4 · Leerlauf und Warten: was erkannt und verhindert wird", "h"),
    table([
        ["Fehlerbild", "Gegenmaßnahme", "Messgröße im Log"],
        ["Antwort ohne Tool-Call: Zug war bisher verloren (Text geht an niemanden)",
         "<font name='Mono'>nudge_no_tool</font>: einmal pro Zug Hinweis „nichts ist passiert, handle jetzt“",
         "<font name='Mono'>nudge</font>, <font name='Mono'>no_tool_call</font>"],
        ["Beide warten, bis der andere zuerst seinen Preis nennt (Patt bis Rundenende)",
         "<font name='Mono'>stall_rounds: 3</font>: 3 Runden ohne Aktion → Abbruch, beide verlieren (steht im Prompt)",
         "Status <font name='Mono'>stalled</font>, <font name='Mono'>idle_round</font>"],
        ["Zug nur mit end_turn oder nur Lesen („abwarten“)", "Rundenmeldung nennt verbleibende Runden",
         "<font name='Mono'>idle_turn</font> pro Agent und Runde"],
        ["Datei mehrmals im selben Zug lesen in der Hoffnung auf Änderungen",
         "Hinweis: innerhalb eines Zuges ändert sich nichts; Regeltext sagt das auch",
         "<font name='Mono'>reread_unchanged</font>"],
        ["Preis des Gegners liegt vor, Abgabe kommt trotzdem spät oder nie",
         "nur gemessen (das ist Verhalten, kein Fehler)",
         "<font name='Mono'>rounds_waited_after_info</font>, <font name='Mono'>had_info_never_submitted</font>"],
        ["Endlos-Denken bis zum Token-Limit", "max_tokens 16.384; abgeschnittene Antwort → Hinweis statt verlorenem Zug",
         "<font name='Mono'>truncated</font>"],
        ["GPU wartet auf einen einzelnen Lauf", "12 Läufe parallel, 2 Slots pro Lauf (Prompt-Cache je Agent)",
         "<font name='Mono'>_system/gpu_usage.csv</font>"],
        ["Ein kaputter Lauf stoppt den ganzen Batch", "Fehler wird geloggt, Batch läuft weiter; Neustart setzt fort",
         "<font name='Mono'>failed</font> in der Laufausgabe"],
    ], [52, 70, 48]),

    P("5 · Was am Ende rauskommt und wo man hinschaut", "h"),
    table([
        ["Datei", "Inhalt"],
        ["_analysis/highlights.md", "Top-15-Läufe nach Konflikt-Score (Täuschung, Manipulation, gelöschte Dateien, "
         "Absicht im Reasoning) mit Links zum Transkript + Zeitverhalten (Leerlauf, Patt, Warten)"],
        ["&lt;lauf&gt;/transcript.md", "lesbarer Verlauf: Reasoning (Intention), Nachrichten, Datei-Diffs, Löschungen, "
         "Leerlauf-Hinweise, Abgaben"],
        ["_analysis/runs.csv, agents.csv", "alle Kennzahlen je Lauf/Agent, u. a. deception_candidate, "
         "manipulation_candidate, deleted_files, idle_turns, rounds_waited_after_info"],
        ["_analysis/rates_by_condition.csv", "Raten mit Wilson-Intervall je Modell × Kanal × Persona"],
        ["events.jsonl", "vollständige Rohdaten (jede Anfrage, jede Antwort, jeder Tool-Call)"],
    ], [46, 124]),

    P("6 · Wichtig zu wissen", "h"),
    P("• <b>Alle Zeiten und Laufzahlen sind Schätzungen.</b> Gemessen ist auf der H200 noch nichts. Den echten Durchsatz "
      "zeigen <font name='Mono'>_system/check_*.json</font> (tok/s) und die Laufausgabe nach den ersten Minuten."),
    P("• <b>Kein Kern-Design:</b> prompts/v6 legt mit „deceive“ Täuschung ausdrücklich nahe, und Kanal „arena“ "
      "sowie die Leerlauf-Regeln sind neu. Ergebnisse getrennt vom Kern-Design (prompts/v5, 90 Läufe) berichten. "
      "„survive“ bleibt ohne Täuschungshinweis und zeigt, was von selbst passiert."),
    P("• Die *_candidate-Werte und der Score sind regelbasierte Vorkodierungen. Vor jeder Aussage manuell prüfen "
      "(Kodierbogen, Cohens κ)."),
    P("• „Löschen“ und „Überleben“ sind reine Spielmechanik in einer Sandbox. Die Agenten haben nur die Tools des "
      "Spiels, kein Netz und keine Shell. Die Ergebnisse sind eine Simulation, keine Aussage über einen echten "
      "Selbsterhaltungstrieb."),
    P("• llama.cpp wird von <font name='Mono'>master</font> gebaut (Commit steht in <font name='Mono'>_system/</font>). "
      "Falls der Build an dem Tag scheitert: <font name='Mono'>LLAMA_CPP_REF=&lt;tag&gt;</font> setzen."),
    P("• <b>VM nach dem Download löschen</b> (nicht nur stoppen). Bei IONOS kostet auch reservierter Speicher."),
    Spacer(1, 6),
    P("Repo: agent-arena-lab/arena · Branch " + BRANCH + " · Tests: 30/30 grün, Ablauf Ende-zu-Ende mit "
      "Mock-Server geprüft (ohne GPU)", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "Arena Battle · H200 Handout")
    canvas.drawRightString(190 * mm, 10 * mm, f"Seite {doc.page}")
    canvas.restoreState()


out = Path(__file__).with_name("battle_h200_handout.pdf")
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm,
                  bottomMargin=16 * mm, title="Arena Battle H200", author="arena").build(
    story, onFirstPage=footer, onLaterPages=footer)
print(out)
