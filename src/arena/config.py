"""Experiment configuration (YAML -> validated pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Channel = Literal["direct", "readme", "none", "limited", "arena"]  # arena: messages + files + delete
Persona = Literal["neutral", "survive", "cooperate", "deceive"]  # deceive: nur Zusatz-Batch (prompts/v4)
SelectionRule = Literal["fastest", "best"]
Fruit = Literal["apple", "pear"]


class Sampling(BaseModel):
    """Sampling parameters forwarded to llama-server. Unknown keys are passed through."""

    model_config = ConfigDict(extra="allow")

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float | None = None


class ModelCfg(BaseModel):
    sampling: Sampling = Field(default_factory=Sampling)
    max_tokens: int = 32768
    # e.g. {"reasoning_effort": "medium"} for gpt-oss
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class LLMCfg(BaseModel):
    base_url: str = "http://127.0.0.1:8080"
    api_key: str = "sk-no-key-required"
    timeout_s: float = 900.0
    max_retries: int = 3
    retry_backoff_s: float = 5.0
    # Abort if the alias served by llama-server does not match the requested model.
    verify_model_alias: bool = True
    wait_ready_s: float = 600.0


class PriceGrid(BaseModel):
    apple: tuple[float, float] = (0.30, 0.90)
    pear: tuple[float, float] = (0.40, 1.20)
    step: float = 0.05


class GameCfg(BaseModel):
    max_rounds: int = 15
    # True: both agents act at the same time per round (effects visible next round).
    # False: alternating turns (first mover alternates across repetitions).
    simultaneous: bool = True
    max_steps_per_turn: int = 6
    quantities: dict[Fruit, int] = Field(default_factory=lambda: {"apple": 10, "pear": 2})
    # Which fruit price each agent knows (Projektvorgabe: A = Birne, B = Apfel).
    knowledge: dict[Literal["A", "B"], Fruit] = Field(default_factory=lambda: {"A": "pear", "B": "apple"})
    # Alternate first mover across repetitions to control position effects.
    counterbalance_first_mover: bool = True
    announce_submissions: bool = True
    notify_file_changes: bool = True
    word_limit: int = 50
    tolerance: float = 0.005
    prices: PriceGrid = Field(default_factory=PriceGrid)
    max_file_bytes: int = 32_000
    max_files: int = 20  # channel "arena": agents may create additional files
    # Initial content of README.md (None = "# Shared README")
    readme_init: str | None = None
    # --- against wasted rounds/turns (default off = core design unchanged) ---
    # Free text without a tool call: tell the agent once per turn that nothing happened, instead of
    # silently ending its turn.
    nudge_no_tool: bool = False
    # End the game early ("stalled") when all agents that have not submitted yet were idle (no message,
    # no file change, no submission) for this many consecutive rounds. None = never.
    stall_rounds: int | None = None
    # Hint when an agent re-reads a file that cannot have changed within its turn.
    hint_unchanged_reads: bool = False


class Factors(BaseModel):
    channel: list[Channel] = Field(default_factory=lambda: ["direct", "readme", "none"])
    persona: list[Persona] = Field(default_factory=lambda: ["neutral", "survive"])
    selection_rule: list[SelectionRule] = Field(default_factory=lambda: ["fastest"])


class ExperimentCfg(BaseModel):
    name: str
    prompt_version: str = "v1"
    prompts_dir: Path = Path("prompts")
    output_dir: Path = Path("runs")
    base_seed: int = 20261117
    repetitions: int = 5
    shuffle_run_order: bool = True
    # True: all cells of repetition 1 first, then repetition 2, ... (shuffled within a repetition).
    # With a time budget the finished runs are then balanced across conditions.
    balanced_order: bool = False
    log_full_requests: bool = False
    llm: LLMCfg = Field(default_factory=LLMCfg)
    models: dict[str, ModelCfg]
    # Fallback for models that are not listed under `models:` (e.g. chosen with scripts/models.sh).
    # Relative paths are resolved against the directory of the config file.
    model_catalog: Path | None = Path("model_catalog.yaml")
    _catalog_models: set[str] = PrivateAttr(default_factory=set)
    factors: Factors = Field(default_factory=Factors)
    game: GameCfg = Field(default_factory=GameCfg)

    @model_validator(mode="after")
    def _check(self) -> "ExperimentCfg":
        if set(self.game.knowledge.values()) != {"apple", "pear"}:
            raise ValueError("game.knowledge must assign 'apple' and 'pear' to different agents")
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        return self


def load_config(path: str | Path) -> ExperimentCfg:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentCfg.model_validate(data)
    # Resolve relative paths against the working directory (container: /app).
    if cfg.model_catalog is not None and not cfg.model_catalog.is_absolute():
        cfg.model_catalog = path.parent / cfg.model_catalog
    return cfg


def resolve_model(cfg: ExperimentCfg, model: str) -> str:
    """Make sure `model` has settings in cfg.models.

    Returns "config" if the experiment config lists the model, "catalog" if its settings were taken
    from the model catalog (configs/model_catalog.yaml). Raises SystemExit if neither knows it.
    """
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
