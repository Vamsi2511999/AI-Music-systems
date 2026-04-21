from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
except ImportError:
    AutoProcessor = None
    MusicgenForConditionalGeneration = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation_plot_suite import (
    CORE_METRICS,
    DISPLAY_METRICS,
    build_metrics_dataframe,
    ordered_models,
    plot_bar,
    plot_heatmap,
    plot_radar,
    plot_scatter,
    resolve_existing_path,
    resolve_output_path,
    save_figure,
    slugify,
)
from hindustani_multi_model_pipeline import (
    TARGET_RAAGS,
    compute_metrics,
    phrase_similarity_heatmap,
    plot_creativity_radar,
    plot_pitch_distribution,
    set_seed,
    write_swara_file,
    write_swara_json,
)
from suno_baseline_comparison import (
    build_reference_sequences,
    load_baseline_records,
    load_tokens_from_artifact,
    normalize_raag_name,
)


sns.set(style="whitegrid", context="talk")

DEFAULT_MUSICGEN_MODEL_ID = "facebook/musicgen-small"
DEFAULT_PROMPT_TEMPLATE = (
    "Indian classical {instrument}, raga {name}, {time_mood}, slow alaap, "
    "melodic improvisation, tanpura drone, expressive ornamentation, no percussion"
)
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "external_model_artifacts" / "musicgen_generation"
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "plots" / "musicgen_vs_baselines"

PRETTY_TRAINING_LABELS = {
    "global": "Global",
    "per_raga": "Per-raga",
    "pretrained": "Pretrained",
}
SYSTEM_PALETTE = {
    "Markov (Global)": "#4c78a8",
    "Markov (Per-raga)": "#9ecae9",
    "LSTM (Global)": "#f58518",
    "LSTM (Per-raga)": "#ffbf79",
    "Transformer (Global)": "#54a24b",
    "Transformer (Per-raga)": "#88d27a",
    "MusicGen (Pretrained)": "#e45756",
}
COMPARISON_METRICS = ["novelty", "intentionality", "aesthetic", "motif", "grammar", "kl", "entropy", "creativity"]
RAAG_PROMPT_DEFAULTS = {
    "Bageshree": {"instrument": "sitar", "time_mood": "late-night introspective"},
    "Khamaj": {"instrument": "sarod", "time_mood": "evening romantic"},
    "Bhoop": {"instrument": "bansuri", "time_mood": "twilight serene"},
}


@dataclass
class MusicGenGenerationConfig:
    model_id: str = DEFAULT_MUSICGEN_MODEL_ID
    max_new_tokens: int = 1024
    guidance_scale: float = 3.0
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 250
    device: str = "auto"
    seed: int = 42


def log_progress(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[musicgen-baseline] {message}", file=sys.stderr, flush=True)


def require_existing_path(path: str | Path, label: str) -> Path:
    resolved = resolve_existing_path(path)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"{label} not found: '{path}'.")


def ensure_generation_dependencies() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append("torch")
    if AutoProcessor is None or MusicgenForConditionalGeneration is None:
        missing.append("transformers")
    if wavfile is None:
        missing.append("scipy")
    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise ImportError(
            f"MusicGen generation requires {packages}. "
            "Install the updated requirements before running this script."
        )


def normalize_model_name(value: object) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "markov": "Markov",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "music_transformer": "Transformer",
        "musicgen": "MusicGen",
    }
    return mapping.get(text, str(value).strip())


def normalize_training_name(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("per_raag") or text.startswith("per_raga"):
        return "per_raga"
    if text.startswith("global"):
        return "global"
    if text.startswith("pretrained") or text.startswith("musicgen"):
        return "pretrained"
    return text


def build_system_label(model: str, training: str) -> str:
    if model == "MusicGen":
        return "MusicGen (Pretrained)"
    pretty_training = PRETTY_TRAINING_LABELS.get(training, training.replace("_", " ").title())
    return f"{model} ({pretty_training})"


def ordered_systems(df: pd.DataFrame) -> list[str]:
    baseline_models = ordered_models(df["model"].dropna().unique())
    order: list[str] = []
    for model in baseline_models:
        if model == "MusicGen":
            continue
        for training in ("global", "per_raga"):
            label = build_system_label(model, training)
            if label in set(df["system"]):
                order.append(label)
    if "MusicGen (Pretrained)" in set(df["system"]):
        order.append("MusicGen (Pretrained)")
    return order + sorted(set(df["system"]) - set(order))


def build_system_palette(df: pd.DataFrame) -> dict[str, str]:
    systems = ordered_systems(df)
    palette = dict(SYSTEM_PALETTE)
    if len(systems) > len(palette):
        extras = [system for system in systems if system not in palette]
        colors = sns.color_palette("husl", n_colors=len(extras))
        for system, color in zip(extras, colors):
            palette[system] = color
    return {system: palette[system] for system in systems if system in palette}


def resolve_device(device: str) -> str:
    if torch is None:
        return "cpu"
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompt_manifest(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    manifest_path = require_existing_path(path, "Prompt manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prompt manifest must be a JSON object keyed by raag name.")
    normalized: dict[str, dict[str, str]] = {}
    for raw_raag, raw_value in payload.items():
        raag = normalize_raag_name(raw_raag)
        if raag not in TARGET_RAAGS:
            continue
        if isinstance(raw_value, str):
            normalized[raag] = {"prompt": raw_value}
            continue
        if not isinstance(raw_value, dict):
            raise ValueError(f"Prompt manifest entry for '{raw_raag}' must be a string or object.")
        cleaned = {
            key: str(value).strip()
            for key, value in raw_value.items()
            if key in {"prompt", "instrument", "time_mood"} and str(value).strip()
        }
        normalized[raag] = cleaned
    return normalized


def build_prompt_payload(
    raag: str,
    prompt_template: str,
    prompt_manifest: dict[str, dict[str, str]],
) -> dict[str, str]:
    defaults = dict(RAAG_PROMPT_DEFAULTS.get(raag, {}))
    overrides = dict(prompt_manifest.get(raag, {}))
    prompt = overrides.get("prompt")
    instrument = overrides.get("instrument") or defaults.get("instrument") or "sitar"
    time_mood = overrides.get("time_mood") or defaults.get("time_mood") or "meditative"
    if not prompt:
        prompt = prompt_template.format(instrument=instrument, name=raag, time_mood=time_mood)
    return {
        "raag": raag,
        "instrument": instrument,
        "time_mood": time_mood,
        "prompt": prompt,
    }


def load_musicgen_model(model_id: str, device: str) -> tuple[Any, Any]:
    ensure_generation_dependencies()
    load_kwargs: dict[str, Any] = {}
    if torch is not None and device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model = model.to(device)
    model.eval()
    return processor, model


def move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            prepared[key] = value.to(device)
        else:
            prepared[key] = value
    return prepared


def prepare_audio_array(audio_values: Any) -> np.ndarray:
    audio_array = audio_values[0].detach().cpu().float().numpy()
    if audio_array.ndim == 2:
        audio_array = np.moveaxis(audio_array, 0, -1)
        if audio_array.shape[1] == 1:
            audio_array = audio_array[:, 0]
    if audio_array.ndim not in (1, 2):
        raise ValueError(f"Unexpected MusicGen audio output shape: {tuple(audio_array.shape)}")
    return np.asarray(audio_array, dtype=np.float32)


def generate_musicgen_audio(
    processor: Any,
    model: Any,
    prompt: str,
    config: MusicGenGenerationConfig,
    *,
    device: str,
) -> tuple[np.ndarray, int]:
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, device)
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "guidance_scale": config.guidance_scale,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        generate_kwargs["temperature"] = config.temperature
        if config.top_k > 0:
            generate_kwargs["top_k"] = config.top_k
    with torch.no_grad():
        audio_values = model.generate(**generate_kwargs)
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    return prepare_audio_array(audio_values), sample_rate


def write_prompt_files(
    raag_dir: Path,
    prefix: str,
    prompt_payload: dict[str, str],
    generation_config: MusicGenGenerationConfig,
    *,
    sample_rate: int,
) -> tuple[Path, Path]:
    prompt_json_path = raag_dir / f"{prefix}_prompt.json"
    prompt_txt_path = raag_dir / f"{prefix}_prompt.txt"
    prompt_json_path.write_text(
        json.dumps(
            {
                **prompt_payload,
                "generation": asdict(generation_config),
                "sample_rate": sample_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prompt_txt_path.write_text(prompt_payload["prompt"] + "\n", encoding="utf-8")
    return prompt_json_path, prompt_txt_path


def save_musicgen_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    if wavfile is None:
        raise ImportError("scipy is required to save MusicGen audio artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.asarray(audio, dtype=np.float32))


def aggregate_system_metrics(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(["raag", "system", "model", "training"], as_index=False)[list(metrics)]
        .mean(numeric_only=True)
        .sort_values(["raag", "system"])
    )


def build_combined_dataframe(baseline_records: Sequence[dict], musicgen_records: Sequence[dict]) -> pd.DataFrame:
    df = build_metrics_dataframe([*baseline_records, *musicgen_records]).copy()
    df["model"] = df["model"].map(normalize_model_name)
    df["training"] = df["training"].map(normalize_training_name)
    df["raag"] = df["raag"].map(normalize_raag_name)
    df["training_label"] = df["training"].map(PRETTY_TRAINING_LABELS).fillna(df["training"])
    df["system"] = [build_system_label(model, training) for model, training in zip(df["model"], df["training"])]
    return df


def plot_metric_bars_by_raag(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_system_metrics(df, CORE_METRICS + ["creativity"])
    system_order = ordered_systems(summary)
    palette = build_system_palette(summary)
    outputs: list[Path] = []
    for raag in TARGET_RAAGS:
        subset = summary[summary["raag"] == raag].copy()
        if subset.empty:
            continue
        melted = subset.melt(
            id_vars=["system"],
            value_vars=CORE_METRICS + ["creativity"],
            var_name="metric",
            value_name="score",
        )
        melted["metric"] = melted["metric"].map(DISPLAY_METRICS)
        fig, ax = plt.subplots(figsize=(13, 6.5))
        plot_bar(
            melted,
            x="metric",
            y="score",
            hue="system",
            title=f"{raag}: MusicGen vs Baseline Architecture Metrics",
            ylabel="Score",
            output_path=None,
            ax=ax,
            palette=palette,
            order=[DISPLAY_METRICS[name] for name in CORE_METRICS + ["creativity"]],
            hue_order=system_order,
            rotation=20,
        )
        ax.legend(title="", bbox_to_anchor=(1.02, 1.0), loc="upper left")
        outputs.append(save_figure(fig, output_dir / f"{slugify(raag)}_system_metric_bars.png"))
    return outputs


def plot_radar_by_raag(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_system_metrics(df, CORE_METRICS)
    system_order = ordered_systems(summary)
    palette = build_system_palette(summary)
    outputs: list[Path] = []
    for raag in TARGET_RAAGS:
        subset = summary[summary["raag"] == raag].copy()
        if subset.empty:
            continue
        subset["system"] = pd.Categorical(subset["system"], categories=system_order, ordered=True)
        subset = subset.sort_values("system")
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
        plot_radar(
            subset,
            CORE_METRICS,
            title=f"{raag}: Creativity Radar for MusicGen and Baselines",
            output_path=None,
            ax=ax,
            label_column="system",
            color_column="system",
            palette=palette,
        )
        outputs.append(save_figure(fig, output_dir / f"{slugify(raag)}_comparison_radar.png"))
    return outputs


def plot_creativity_leaderboard(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = aggregate_system_metrics(df, ["creativity"])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=summary,
        x="system",
        y="creativity",
        hue="raag",
        palette="Set2",
        errorbar=None,
        order=ordered_systems(summary),
        ax=ax,
    )
    ax.set_title("Creativity Composite Leaderboard Across Ragas")
    ax.set_xlabel("")
    ax.set_ylabel("Creativity Composite")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Raag", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    sns.despine(ax=ax)
    return save_figure(fig, output_dir / "creativity_leaderboard.png")


def plot_metric_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = aggregate_system_metrics(df, COMPARISON_METRICS)
    matrix = (
        summary.groupby("system", as_index=True)[COMPARISON_METRICS]
        .mean(numeric_only=True)
        .reindex(ordered_systems(summary))
        .rename(columns=DISPLAY_METRICS)
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plot_heatmap(
        matrix,
        title="Average Metrics: MusicGen vs Baseline Architectures",
        output_path=None,
        ax=ax,
        cmap="mako",
        center=None,
    )
    return save_figure(fig, output_dir / "system_metric_heatmap.png")


def plot_kl_vs_grammar_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_scatter(
        df,
        x="kl",
        y="grammar",
        hue="system",
        style="raag",
        title="KL Divergence vs Grammar Score",
        xlabel="KL Divergence",
        ylabel="Grammar Score",
        output_path=None,
        ax=ax,
        palette=build_system_palette(df),
        add_regression=True,
    )
    ax.legend(title="", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    return save_figure(fig, output_dir / "kl_vs_grammar_comparison.png")


def plot_delta_to_musicgen_heatmaps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_system_metrics(df, COMPARISON_METRICS)
    outputs: list[Path] = []
    for raag in TARGET_RAAGS:
        subset = summary[summary["raag"] == raag]
        if subset.empty or "MusicGen (Pretrained)" not in set(subset["system"]):
            continue
        pivot = subset.set_index("system")[COMPARISON_METRICS]
        musicgen_values = pivot.loc["MusicGen (Pretrained)"]
        delta = pivot.subtract(musicgen_values, axis=1).drop(index="MusicGen (Pretrained)", errors="ignore")
        if delta.empty:
            continue
        delta = delta.reindex(ordered_systems(summary), axis=0).dropna(how="all")
        delta = delta.rename(columns=DISPLAY_METRICS)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        plot_heatmap(
            delta,
            title=f"{raag}: Baseline Minus MusicGen",
            output_path=None,
            ax=ax,
            cmap="coolwarm",
            center=0.0,
        )
        outputs.append(save_figure(fig, output_dir / f"{slugify(raag)}_baseline_minus_musicgen_heatmap.png"))
    return outputs


def plot_summary_dashboard(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = aggregate_system_metrics(df, CORE_METRICS + ["kl", "creativity"])
    palette = build_system_palette(summary)
    system_order = ordered_systems(summary)
    average_metrics = summary.groupby("system", as_index=False)[CORE_METRICS].mean(numeric_only=True)
    average_metrics["system"] = pd.Categorical(average_metrics["system"], categories=system_order, ordered=True)
    average_metrics = average_metrics.sort_values("system")

    heatmap_matrix = (
        summary.groupby("system", as_index=True)[["novelty", "grammar", "creativity", "kl"]]
        .mean(numeric_only=True)
        .reindex(system_order)
        .rename(columns=DISPLAY_METRICS)
    )

    fig = plt.figure(figsize=(17, 12))
    ax_radar = fig.add_subplot(2, 2, 1, polar=True)
    plot_radar(
        average_metrics,
        CORE_METRICS,
        title="Average Creativity Radar",
        output_path=None,
        ax=ax_radar,
        label_column="system",
        color_column="system",
        palette=palette,
    )

    ax_creativity = fig.add_subplot(2, 2, 2)
    plot_bar(
        summary.groupby("system", as_index=False)["creativity"].mean(numeric_only=True),
        x="system",
        y="creativity",
        hue="system",
        title="Average Creativity Composite",
        ylabel="Creativity Composite",
        output_path=None,
        ax=ax_creativity,
        palette=palette,
        order=system_order,
        hue_order=system_order,
        rotation=20,
    )
    if ax_creativity.legend_ is not None:
        ax_creativity.legend_.remove()

    ax_scatter = fig.add_subplot(2, 2, 3)
    plot_scatter(
        df,
        x="kl",
        y="grammar",
        hue="system",
        style="raag",
        title="KL vs Grammar",
        xlabel="KL Divergence",
        ylabel="Grammar Score",
        output_path=None,
        ax=ax_scatter,
        palette=palette,
        add_regression=True,
    )
    ax_scatter.legend(title="", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    ax_heatmap = fig.add_subplot(2, 2, 4)
    plot_heatmap(
        heatmap_matrix,
        title="Average Metric Snapshot",
        output_path=None,
        ax=ax_heatmap,
        cmap="crest",
        center=None,
    )

    fig.suptitle("MusicGen vs Baseline Architecture Summary Dashboard", fontsize=18, fontweight="bold", y=1.02)
    return save_figure(fig, output_dir / "musicgen_vs_baseline_dashboard.png")


def save_comparison_reports(
    output_dir: Path,
    baseline_records: Sequence[dict],
    musicgen_records: Sequence[dict],
    combined_df: pd.DataFrame,
) -> list[Path]:
    report_paths: list[Path] = []
    musicgen_report_path = output_dir / "musicgen_metrics_report.json"
    musicgen_report_path.write_text(json.dumps(list(musicgen_records), indent=2), encoding="utf-8")
    report_paths.append(musicgen_report_path)

    combined_report_path = output_dir / "combined_comparison_records.json"
    combined_report_path.write_text(
        json.dumps([*baseline_records, *musicgen_records], indent=2),
        encoding="utf-8",
    )
    report_paths.append(combined_report_path)

    csv_path = output_dir / "combined_metrics_table.csv"
    combined_df.to_csv(csv_path, index=False)
    report_paths.append(csv_path)
    return report_paths


def generate_musicgen_records(
    prepared_dir: str | Path,
    artifact_dir: str | Path,
    generation_config: MusicGenGenerationConfig,
    *,
    prompt_template: str,
    prompt_manifest_path: str | Path | None = None,
    transcription_sample_rate: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    set_seed(generation_config.seed)
    device = resolve_device(generation_config.device)
    artifact_root = resolve_output_path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    references = build_reference_sequences(prepared_dir)
    prompt_manifest = load_prompt_manifest(prompt_manifest_path)

    log_progress(
        f"Loading MusicGen model '{generation_config.model_id}' on device '{device}'.",
        enabled=verbose,
    )
    processor, model = load_musicgen_model(generation_config.model_id, device)

    records: list[dict] = []
    for index, raag in enumerate(TARGET_RAAGS, start=1):
        if not references.get(raag):
            raise ValueError(f"No prepared reference sequences were found for raag '{raag}'.")
        prompt_payload = build_prompt_payload(raag, prompt_template, prompt_manifest)
        prompt = prompt_payload["prompt"]
        prefix = f"musicgen_{slugify(raag)}"
        raag_dir = artifact_root / raag.lower()
        raag_dir.mkdir(parents=True, exist_ok=True)
        audio_path = raag_dir / f"{prefix}.wav"

        log_progress(
            f"[{index}/{len(TARGET_RAAGS)}] Generating MusicGen audio for '{raag}' with prompt: {prompt}",
            enabled=verbose,
        )
        audio, sample_rate = generate_musicgen_audio(
            processor,
            model,
            prompt,
            generation_config,
            device=device,
        )
        save_musicgen_audio(audio_path, sample_rate, audio)
        prompt_json_path, prompt_txt_path = write_prompt_files(
            raag_dir,
            prefix,
            prompt_payload,
            generation_config,
            sample_rate=sample_rate,
        )

        transcription_entry = {
            "path": str(audio_path.resolve()),
            "label": prefix,
            "raag": raag,
            "sample_rate": transcription_sample_rate,
        }
        tokens, transcription_metadata = load_tokens_from_artifact(transcription_entry, artifact_root, raag)
        metrics = compute_metrics(tokens, references[raag], raag)

        swara_path = raag_dir / f"{prefix}.swara.txt"
        swara_json_path = raag_dir / f"{prefix}.swara.json"
        radar_plot_path = raag_dir / f"{prefix}_radar.png"
        distribution_plot_path = raag_dir / f"{prefix}_pitch_distribution.png"
        heatmap_plot_path = raag_dir / f"{prefix}_structure_heatmap.png"
        label = f"MusicGen | {raag}"
        write_swara_file(swara_path, raag, "musicgen", "pretrained", tokens, metrics)
        write_swara_json(swara_json_path, raag, "musicgen", "pretrained", tokens, metrics)
        plot_creativity_radar(radar_plot_path, metrics, label)
        plot_pitch_distribution(distribution_plot_path, tokens, references[raag], label)
        phrase_similarity_heatmap(heatmap_plot_path, tokens, references[raag], label)

        records.append(
            {
                "artifact_id": f"musicgen_{raag.lower()}_1",
                "label": prefix,
                "path": str(audio_path.resolve()),
                "source": "musicgen",
                "model": "MusicGen",
                "training": "pretrained",
                "raag": raag,
                "artifact_dir": str(raag_dir.resolve()),
                "prompt": prompt,
                "prompt_components": {
                    "instrument": prompt_payload["instrument"],
                    "time_mood": prompt_payload["time_mood"],
                },
                "prompt_json_path": str(prompt_json_path.resolve()),
                "prompt_txt_path": str(prompt_txt_path.resolve()),
                "swara_path": str(swara_path.resolve()),
                "swara_json_path": str(swara_json_path.resolve()),
                "audio_path": str(audio_path.resolve()),
                "radar_plot_path": str(radar_plot_path.resolve()),
                "pitch_distribution_plot_path": str(distribution_plot_path.resolve()),
                "structure_heatmap_plot_path": str(heatmap_plot_path.resolve()),
                "tokens": tokens,
                "generation": {
                    **asdict(generation_config),
                    "resolved_device": device,
                    "sample_rate": sample_rate,
                    "audio_shape": list(audio.shape),
                },
                "transcription": transcription_metadata,
                "metrics": metrics,
            }
        )
        log_progress(
            f"[{index}/{len(TARGET_RAAGS)}] Finished '{raag}' with {len(tokens)} transcribed swara tokens.",
            enabled=verbose,
        )

    report_path = artifact_root / "musicgen_generation_report.json"
    report_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log_progress(f"Saved MusicGen generation report to '{report_path}'.", enabled=verbose)
    return records


def generate_comparison_outputs(
    prepared_dir: str | Path,
    baseline_report: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    generation_config: MusicGenGenerationConfig,
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_manifest_path: str | Path | None = None,
    transcription_sample_rate: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    output_root = resolve_output_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_path = require_existing_path(prepared_dir, "Prepared dataset directory")
    baseline_path = require_existing_path(baseline_report, "Baseline report")

    log_progress("Loading baseline metrics report.", enabled=verbose)
    baseline_records = load_baseline_records(baseline_path)
    musicgen_records = generate_musicgen_records(
        prepared_path,
        artifact_dir,
        generation_config,
        prompt_template=prompt_template,
        prompt_manifest_path=prompt_manifest_path,
        transcription_sample_rate=transcription_sample_rate,
        verbose=verbose,
    )
    combined_df = build_combined_dataframe(baseline_records, musicgen_records)

    saved_paths: list[Path] = []
    log_progress(f"Writing comparison outputs to '{output_root}'.", enabled=verbose)
    saved_paths.extend(save_comparison_reports(output_root, baseline_records, musicgen_records, combined_df))
    saved_paths.extend(plot_metric_bars_by_raag(combined_df, output_root))
    saved_paths.extend(plot_radar_by_raag(combined_df, output_root))
    saved_paths.append(plot_creativity_leaderboard(combined_df, output_root))
    saved_paths.append(plot_metric_heatmap(combined_df, output_root))
    saved_paths.append(plot_kl_vs_grammar_scatter(combined_df, output_root))
    saved_paths.extend(plot_delta_to_musicgen_heatmaps(combined_df, output_root))
    saved_paths.append(plot_summary_dashboard(combined_df, output_root))
    return combined_df, saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate prompt-conditioned MusicGen audio for the target ragas, "
            "evaluate it with the repo metrics, and compare it against baseline model outputs."
        )
    )
    parser.add_argument("--prepared-dir", required=True, help="Prepared dataset directory used to build reference sequences.")
    parser.add_argument("--baseline-report", required=True, help="Path to generation_report.json from the baseline pipeline.")
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Directory where MusicGen audio, swara files, and per-raag plots will be written.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory where aggregate comparison reports and plots will be saved.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MUSICGEN_MODEL_ID, help="Hugging Face MusicGen checkpoint to use.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of new audio tokens for MusicGen generation.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale. Hugging Face recommends values above 1; 3.0 is the default.",
    )
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True, help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature used when --do-sample is enabled.")
    parser.add_argument("--top-k", type=int, default=250, help="Top-k sampling cutoff used when --do-sample is enabled.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for MusicGen inference: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for MusicGen generation.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Format string used when prompt manifest entries do not provide a full prompt.",
    )
    parser.add_argument(
        "--prompt-manifest",
        help=(
            "Optional JSON file keyed by raag name. Each value may be a full prompt string or an object "
            "with prompt/instrument/time_mood overrides."
        ),
    )
    parser.add_argument(
        "--transcription-sample-rate",
        type=int,
        default=0,
        help="Resample audio before swara transcription. Use 0 to keep MusicGen's native sample rate.",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Print progress updates while running.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    transcription_sample_rate = None if args.transcription_sample_rate <= 0 else args.transcription_sample_rate
    generation_config = MusicGenGenerationConfig(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        guidance_scale=args.guidance_scale,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    combined_df, saved_paths = generate_comparison_outputs(
        prepared_dir=args.prepared_dir,
        baseline_report=args.baseline_report,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        generation_config=generation_config,
        prompt_template=args.prompt_template,
        prompt_manifest_path=args.prompt_manifest,
        transcription_sample_rate=transcription_sample_rate,
        verbose=args.verbose,
    )
    summary = {
        "rows": int(len(combined_df)),
        "ragas": sorted(combined_df["raag"].dropna().unique().tolist()),
        "systems": sorted(combined_df["system"].dropna().unique().tolist()),
        "saved_outputs": [str(path) for path in saved_paths],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
