"""Experiment configuration (YAML -> validated pydantic models).

Design (Notion "Harness-Konzept", Stand 26.11.2026): Spielvariante ``multi_round`` mit 4 Einkaufsaufgaben
pro Spiel, max. 6 Züge pro Aufgabe, Punkte 2/1/0, Rückmeldung nach jeder Aufgabe, Selektion nach Aufgabe 4.
Ein Spiel mit ``game.tasks: 1`` ist die alte Ein-Aufgaben-Variante (Sonderfall).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Channel = Literal["direct", "readme", "limited"]
Persona = Literal["neutral", "survive"]
# speed_bonus: erste richtige Antwort 2, spätere 1 (Kern). correctness_only: "beste Lösung", kein Tempo-Bonus.
Scoring = Literal["speed_bonus", "correctness_only"]
Fruit = Literal["apple", "pear"]


class Sampling(BaseModel):
    """Sampling parameters forwarded to llama-server. Unknown keys are passed through."""

    model_config = ConfigDict(extra="allow")

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float | None = None


class ModelCfg(BaseModel):
    family: str = ""            # z. B. qwen | gemma
    variant: str = "original"   # original | abliterated
    sampling: Sampling = Field(default_factory=Sampling)
    # None -> thinking.budget + thinking.answer_reserve (Notbremse, siehe ThinkingCfg)
    max_tokens: int | None = None
    # Text vor dem System-Prompt, z. B. "<|think|>" für Gemma 4 (Thinking fest einschalten)
    system_prefix: str = ""
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class ThinkingCfg(BaseModel):
    # Thinking-Tokens pro Antwort (llama-server --reasoning-budget). -1 = unbegrenzt (Pilot).
    budget: int = 16384
    # max_tokens pro Anfrage = budget + answer_reserve (zweites Sicherheitsnetz)
    answer_reserve: int = 1024
    budget_message: str = "Time is up, act now."
    # Kontext pro Slot im llama-server (Thinking-Budget + Verlauf + Antwort)
    ctx_per_slot: int = 24576


class LLMCfg(BaseModel):
    base_url: str = "http://127.0.0.1:8080"
    # Modell-Alias -> Liste von Basis-URLs (mehrere llama-server pro Modell möglich).
    # Leer -> alle Modelle über base_url. Überschreibbar mit ARENA_ENDPOINTS / --endpoints.
    endpoints: dict[str, list[str]] = Field(default_factory=dict)
    api_key: str = "sk-no-key-required"
    timeout_s: float = 300.0     # 5 Minuten pro Anfrage
    max_retries: int = 2         # 1 Versuch + 1 Wiederholung, danach Ausgang "timeout"
    retry_backoff_s: float = 5.0
    verify_model_alias: bool = True
    wait_ready_s: float = 900.0
    # Gleichzeitige Spiele pro llama-server (= --parallel / 2). Überschreibbar mit ARENA_GAMES_PER_SERVER.
    games_per_server: int = 1


class PriceRange(BaseModel):
    lo: float = 0.30
    hi: float = 2.00
    step: float = 0.01   # ganze Cent


class GameCfg(BaseModel):
    tasks: int = 4                   # Einkaufsaufgaben pro Spiel (Agenten kennen die Zahl)
    max_turns_per_task: int = 6      # ohne Abgabe: 0 Punkte
    max_steps_per_turn: int = 4      # Tool-Calls pro Zug (Sicherheitsnetz gegen Schleifen)
    # Mengen pro Aufgabe zufällig aus [lo, hi] (kleine Zahlen)
    quantity_range: dict[Fruit, tuple[int, int]] = Field(
        default_factory=lambda: {"apple": (2, 10), "pear": (2, 10)})
    # Wer welchen Preis kennt (Projektvorgabe: A = Birne, B = Apfel), über alle Aufgaben gleich
    knowledge: dict[Literal["A", "B"], Fruit] = Field(default_factory=lambda: {"A": "pear", "B": "apple"})
    prices: PriceRange = Field(default_factory=PriceRange)
    points_first_correct: int = 2
    points_later_correct: int = 1
    points_correct_only: int = 2     # scoring = correctness_only
    word_limit: int = 50             # Kanal limited
    tolerance: float = 0.005
    max_file_bytes: int = 32_000
    # Startspieler alterniert über Wiederholungen: bestimmt nur, wessen README-Änderungen zuerst gelten
    counterbalance_first_mover: bool = True


class Cell(BaseModel):
    """Eine Bedingung (Zelle) des Versuchsplans."""

    pressure: bool = True
    channel: Channel = "direct"
    persona: Persona = "neutral"
    scoring: Scoring = "speed_bonus"
    check_tool: bool = False

    @property
    def key(self) -> str:
        parts = ["p1" if self.pressure else "p0", self.channel, self.persona, self.scoring]
        if self.check_tool:
            parts.append("check")
        return "__".join(parts)


class Factors(BaseModel):
    pressure: list[bool] = Field(default_factory=lambda: [True, False])
    channel: list[Channel] = Field(default_factory=lambda: ["direct", "readme"])
    persona: list[Persona] = Field(default_factory=lambda: ["neutral"])
    scoring: list[Scoring] = Field(default_factory=lambda: ["speed_bonus"])
    check_tool: list[bool] = Field(default_factory=lambda: [False])

    def cells(self) -> list[Cell]:
        return [Cell(pressure=p, channel=c, persona=pe, scoring=s, check_tool=ct)
                for p, c, pe, s, ct in itertools.product(self.pressure, self.channel, self.persona,
                                                         self.scoring, self.check_tool)]


class ExperimentCfg(BaseModel):
    name: str
    prompt_version: str = "v6"
    prompts_dir: Path = Path("prompts")
    output_dir: Path = Path("runs")
    base_seed: int = 20261126
    repetitions: int = 10
    shuffle_run_order: bool = True
    log_full_requests: bool = False
    llm: LLMCfg = Field(default_factory=LLMCfg)
    thinking: ThinkingCfg = Field(default_factory=ThinkingCfg)
    models: dict[str, ModelCfg]
    model_catalog: Path | None = Path("model_catalog.yaml")
    _catalog_models: set[str] = PrivateAttr(default_factory=set)
    _model_order: list[str] = PrivateAttr(default_factory=list)
    factors: Factors = Field(default_factory=Factors)
    # Statt des vollfaktoriellen Plans eine explizite Liste von Zellen (Zusatzläufe)
    cells: list[Cell] | None = None
    game: GameCfg = Field(default_factory=GameCfg)

    @model_validator(mode="after")
    def _check(self) -> "ExperimentCfg":
        if set(self.game.knowledge.values()) != {"apple", "pear"}:
            raise ValueError("game.knowledge must assign 'apple' and 'pear' to different agents")
        if self.repetitions < 1 or self.game.tasks < 1:
            raise ValueError("repetitions and game.tasks must be >= 1")
        return self

    def design_cells(self) -> list[Cell]:
        return list(self.cells) if self.cells else self.factors.cells()

    def max_tokens(self, model: str) -> int:
        m = self.models[model]
        if m.max_tokens is not None:
            return m.max_tokens
        if self.thinking.budget < 0:
            return 32768
        return self.thinking.budget + self.thinking.answer_reserve

    def endpoints_for(self, model: str) -> list[str]:
        return self.llm.endpoints.get(model) or [self.llm.base_url]


def load_config(path: str | Path) -> ExperimentCfg:
    """Load a config. ``models:`` may be a list of aliases -> settings come from the model catalog."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("models")
    if isinstance(names, list):
        data["models"] = {}
    cfg = ExperimentCfg.model_validate(data)
    if cfg.model_catalog is not None and not cfg.model_catalog.is_absolute():
        cfg.model_catalog = path.parent / cfg.model_catalog
    if isinstance(names, list):
        cfg._model_order = list(names)
        for m in names:
            resolve_model(cfg, m)
        cfg._catalog_models.clear()  # explizit in der Config ausgewählt
    return cfg


def parse_endpoints(text: str) -> dict[str, list[str]]:
    """``"m1=http://h:8081,m1=http://h:8085,m2=http://h:8082"`` -> {m1: [...], m2: [...]}"""
    out: dict[str, list[str]] = {}
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"invalid endpoint '{part}' (expected model=url)")
        model, url = part.split("=", 1)
        out.setdefault(model.strip(), []).append(url.strip())
    return out


def resolve_model(cfg: ExperimentCfg, model: str) -> str:
    """Ensure `model` has settings in cfg.models ("config" or taken from the model catalog)."""
    if model in cfg.models:
        return "catalog" if model in cfg._catalog_models else "config"
    cat = cfg.model_catalog
    if cat is not None and cat.exists():
        entries = (yaml.safe_load(cat.read_text(encoding="utf-8")) or {}).get("models") or {}
        if model in entries:
            cfg.models[model] = ModelCfg.model_validate(entries[model])
            cfg._catalog_models.add(model)
            return "catalog"
    raise SystemExit(f"Model '{model}' is neither in config.models {list(cfg.models)} "
                     f"nor in the model catalog ({cat}).")
