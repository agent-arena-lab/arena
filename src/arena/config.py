"""Experiment configuration (YAML -> validated pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Channel = Literal["direct", "readme", "none", "limited"]
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
    log_full_requests: bool = False
    llm: LLMCfg = Field(default_factory=LLMCfg)
    models: dict[str, ModelCfg]
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
    return cfg
