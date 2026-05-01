from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import librosa
except ImportError:
    librosa = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation_plot_suite import (
    CORE_METRICS,
    DEFAULT_GENERATED_DIR,
    DISPLAY_METRICS,
    annotate_metric_records_with_composites,
    build_metrics_dataframe,
    load_json_records,
    ordered_models,
    parse_iteration_dir,
    plot_bar,
    plot_heatmap,
    plot_radar,
    plot_scatter,
    resolve_existing_path,
    resolve_generated_iteration_dir,
    resolve_output_path,
    save_figure,
    slugify,
)
from hindustani_multi_model_pipeline import (
    RAAG_GRAMMAR,
    SEP_TOKEN,
    SWARA_ORDER,
    TARGET_RAAGS,
    compute_metrics,
    decode_token_ids,
    load_prepared,
    normalize_raag_name as pipeline_normalize_raag_name,
)


sns.set(style="whitegrid", context="talk")

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
    "Suno (Pretrained)": "#b279a2",
}
COMPARISON_METRICS = ["novelty", "intentionality", "aesthetic", "motif", "grammar", "kl", "entropy", "creativity"]
TOKEN_KEYS = ("tokens", "notes", "swaras", "sequence", "token_sequence")
PHRASE_KEYS = ("phrases", "swaras_by_phrase", "note_phrases")
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TEXT_SUFFIXES = {".txt", ".json"}
SUPPORTED_SUNO_SUFFIXES = AUDIO_SUFFIXES | TEXT_SUFFIXES
DEFAULT_SUNO_DIR = Path(__file__).resolve().parent / "artifacts" / "external_model_artifacts"
DEFAULT_PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
SUNO_OUTPUT_DIRNAME = "suno_vs_baselines"


def log_progress(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[suno-baseline] {message}", file=sys.stderr, flush=True)


def require_existing_path(path: str | Path, label: str) -> Path:
    resolved = resolve_existing_path(path)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"{label} not found: '{path}'.")


def resolve_baseline_report_path(
    baseline_report: str | Path | None,
    generated_dir: str | Path,
    iteration: int | None,
) -> tuple[Path, str | None]:
    if baseline_report:
        resolved = require_existing_path(baseline_report, "Baseline report")
        parent_iteration = parse_iteration_dir(resolved.parent)
        iteration_label = f"iteration-{parent_iteration}" if parent_iteration is not None else None
        return resolved, iteration_label

    iteration_dir, iteration_label = resolve_generated_iteration_dir(generated_dir, iteration)
    return iteration_dir / "generation_report.json", iteration_label


def resolve_suno_output_dir(output_dir: str | Path | None, iteration_label: str | None) -> Path:
    base_root = resolve_output_path(output_dir or DEFAULT_PLOTS_ROOT)
    if base_root.name == SUNO_OUTPUT_DIRNAME and (
        iteration_label is None or parse_iteration_dir(base_root.parent) is not None or base_root.parent.name == iteration_label
    ):
        return base_root
    if iteration_label is None:
        if parse_iteration_dir(base_root) is not None:
            return base_root / SUNO_OUTPUT_DIRNAME
        return base_root / SUNO_OUTPUT_DIRNAME
    if parse_iteration_dir(base_root) is not None or base_root.name == iteration_label:
        return base_root / SUNO_OUTPUT_DIRNAME
    return base_root / iteration_label / SUNO_OUTPUT_DIRNAME


def normalize_model_name(value: object) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "markov": "Markov",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "music_transformer": "Transformer",
        "suno": "Suno",
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
    if text.startswith("pretrained") or text.startswith("suno"):
        return "pretrained"
    return text


def normalize_raag_name(value: object) -> str:
    if value is None:
        raise ValueError("A raag name is required for each Suno artifact.")
    normalized = pipeline_normalize_raag_name(str(value))
    text = normalized.strip().lower()
    mapping = {raag.lower(): raag for raag in TARGET_RAAGS}
    compact_mapping = {raag.lower().replace(" ", ""): raag for raag in TARGET_RAAGS}
    return mapping.get(text, compact_mapping.get(text.replace(" ", ""), normalized.strip()))


def build_reference_sequences(prepared_dir: str | Path) -> dict[str, list[list[str]]]:
    prepared_path = require_existing_path(prepared_dir, "Prepared dataset directory")
    prepared = load_prepared(str(prepared_path))
    id_to_token = {int(key): value for key, value in prepared["id_to_token"].items()}
    references = {
        raag: [
            decode_token_ids(track["token_ids"], id_to_token)
            for track in prepared["tracks"]
            if track["raag"] == raag
        ]
        for raag in TARGET_RAAGS
    }
    return references


def flatten_phrase_tokens(phrases: Sequence[Sequence[Any]]) -> list[str]:
    tokens: list[str] = []
    for index, phrase in enumerate(phrases):
        if isinstance(phrase, str):
            raw_parts: Iterable[Any] = re.split(r"[\s,]+", phrase)
        else:
            raw_parts = phrase
        phrase_tokens = [normalize_token(token) for token in raw_parts]
        phrase_tokens = [token for token in phrase_tokens if token is not None]
        tokens.extend(phrase_tokens)
        if index < len(phrases) - 1 and tokens and tokens[-1] != SEP_TOKEN:
            tokens.append(SEP_TOKEN)
    return tokens


def normalize_token(token: object) -> str | None:
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    if text in SWARA_ORDER:
        return text
    upper = text.upper()
    if upper in {"<SEP>", "SEP", "|", "/"}:
        return SEP_TOKEN
    return None


def hz_to_midi_value(freq_hz: np.ndarray | float) -> np.ndarray | float:
    freq = np.maximum(freq_hz, 1e-6)
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def relative_swara_for_frequency(freq_hz: float, tonic_pc: int) -> str | None:
    if not np.isfinite(freq_hz) or freq_hz <= 0:
        return None

    midi = hz_to_midi_value(freq_hz)
    rel = (midi - tonic_pc) % 12

    distances = np.abs(np.arange(12) - rel)
    circular = np.minimum(distances, 12 - distances)

    min_dist = np.min(circular)

    # tolerance for ornamentation
    if min_dist > 0.6:
        return None

    return SWARA_ORDER[int(np.argmin(circular))]


def score_tonic_candidate(frequencies_hz: np.ndarray, raag: str, tonic_pitch_class: int) -> float:
    notes = [note for note in (relative_swara_for_frequency(freq, tonic_pitch_class) for freq in frequencies_hz) if note is not None]
    if not notes:
        return float("-inf")
    grammar = RAAG_GRAMMAR[raag]
    allowed_ratio = sum(note in grammar["allowed_notes"] for note in notes) / len(notes)
    pairs = list(zip(notes[:-1], notes[1:]))
    illegal_ratio = sum(pair in grammar["illegal_transitions"] for pair in pairs) / max(len(pairs), 1)
    tonic_ratio = notes.count("S") / len(notes)
    vadi_ratio = notes.count(grammar["vadi"]) / len(notes)
    samvadi_ratio = notes.count(grammar["samvadi"]) / len(notes)
    ending_window = notes[-max(8, len(notes) // 12) :]
    preferred_ratio = sum(note in grammar["preferred_endings"] for note in ending_window) / max(len(ending_window), 1)
    return (
        2.6 * allowed_ratio
        - 1.8 * illegal_ratio
        + 0.45 * tonic_ratio
        + 0.45 * vadi_ratio
        + 0.35 * samvadi_ratio
        + 0.20 * preferred_ratio
    )


def estimate_tonic_pitch_class(frequencies_hz: np.ndarray, raag: str) -> int:
    if frequencies_hz.size == 0:
        raise ValueError(f"Could not estimate tonic for raag '{raag}' because no voiced audio frames were found.")
    scores = [(score_tonic_candidate(frequencies_hz, raag, pitch_class), pitch_class) for pitch_class in range(12)]
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[0][1]


def compress_frame_notes_to_tokens(note_frames: Sequence[str | None], min_note_frames: int, sep_frames: int) -> list[str]:
    tokens: list[str] = []
    current_note: str | None = None
    current_run = 0
    silence_run = 0

    def flush_current() -> None:
        nonlocal current_note, current_run
        if current_note is not None and current_run >= min_note_frames:
            if not tokens or tokens[-1] != current_note:
                tokens.append(current_note)
        current_note = None
        current_run = 0

    for note in note_frames:
        if note is None:
            silence_run += 1
            flush_current()
            if silence_run >= sep_frames and tokens and tokens[-1] != SEP_TOKEN:
                tokens.append(SEP_TOKEN)
            continue

        silence_run = 0
        if current_note == note:
            current_run += 1
            continue

        flush_current()
        current_note = note
        current_run = 1

    flush_current()
    while tokens and tokens[-1] == SEP_TOKEN:
        tokens.pop()
    return tokens


def extract_suno_audio_tokens(artifact_path: Path, raag: str, entry: dict) -> tuple[list[str], dict]:
    if librosa is None:
        raise ImportError("librosa is required.")

    import numpy as np
    from scipy.signal import medfilt

    # -----------------------
    # Load audio
    # -----------------------
    audio, sr = librosa.load(str(artifact_path), sr=entry.get("sample_rate", 22050), mono=True)

    if audio.size == 0:
        raise ValueError(f"Empty audio: {artifact_path}")

    # -----------------------
    # Harmonic filtering (CRITICAL)
    # -----------------------
    audio = librosa.effects.harmonic(audio)

    # -----------------------
    # Trim silence
    # -----------------------
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # -----------------------
    # Pitch extraction (use pyin if possible)
    # -----------------------
    fmin = entry.get("fmin_hz", 65.0)
    fmax = entry.get("fmax_hz", 1046.5)

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        pitch_method = "pyin"
    except:
        f0 = librosa.yin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        voiced_flag = np.isfinite(f0)
        voiced_prob = None
        pitch_method = "yin_fallback"

    f0 = np.asarray(f0, dtype=np.float64)

    # -----------------------
    # Voiced filtering (IMPORTANT)
    # -----------------------
    if voiced_prob is not None:
        mask = (voiced_prob > 0.8) & np.isfinite(f0)
    else:
        mask = np.isfinite(f0)

    f0[~mask] = np.nan

    if np.sum(np.isfinite(f0)) < 10:
        raise ValueError(f"No valid pitch detected in {artifact_path}")

    # -----------------------
    # Smooth pitch (CRITICAL)
    # -----------------------
    f0 = medfilt(f0, kernel_size=7)

    # -----------------------
    # Stabilize pitch (remove jitter)
    # -----------------------
    f0 = stabilize_pitch(f0)

    # -----------------------
    # Downsample (reduce noise)
    # -----------------------
    f0 = f0[::2]

    voiced_freqs = f0[np.isfinite(f0)]

    # -----------------------
    # Tonic estimation
    # -----------------------
    tonic_hz = entry.get("tonic_hz")

    if tonic_hz:
        tonic_pc = int(round(hz_to_midi_value(tonic_hz))) % 12
    else:
        tonic_pc = estimate_tonic_pitch_class(voiced_freqs, raag)

    # -----------------------
    # Convert to swaras
    # -----------------------
    note_frames = [
        relative_swara_for_frequency(freq, tonic_pc)
        if np.isfinite(freq) else None
        for freq in f0
    ]

    # -----------------------
    # Compress into tokens
    # -----------------------
    tokens = compress_frame_notes_to_tokens(
        note_frames,
        min_note_frames=6,
        sep_frames=15
    )

    if not tokens:
        raise ValueError(f"No tokens extracted for {artifact_path}")

    # -----------------------
    # Metadata
    # -----------------------
    metadata = {
        "pitch_method": pitch_method,
        "sample_rate": sr,
        "frames": len(f0),
        "voiced_frames": int(np.isfinite(f0).sum()),
        "tonic_pitch_class": int(tonic_pc),
        "mean_pitch": float(np.nanmean(voiced_freqs)),
    }

    return tokens, metadata

def stabilize_pitch(f0):
    stable = []
    prev = None

    for val in f0:
        if not np.isfinite(val):
            stable.append(np.nan)
            continue

        if prev is not None and abs(val - prev) < 5:  # Hz threshold
            stable.append(prev)
        else:
            stable.append(val)
            prev = val

    return np.array(stable)


def extract_tokens_from_json_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return [token for token in (normalize_token(item) for item in payload) if token is not None]
        if all(isinstance(item, (list, tuple)) for item in payload):
            return flatten_phrase_tokens(payload)

    if isinstance(payload, dict):
        for key in TOKEN_KEYS:
            if key in payload:
                return extract_tokens_from_json_payload(payload[key])
        for key in PHRASE_KEYS:
            if key in payload and isinstance(payload[key], list):
                return flatten_phrase_tokens(payload[key])

    raise ValueError("Could not extract swara tokens from JSON payload.")


def extract_tokens_from_swara_text(text: str) -> list[str]:
    lines = text.splitlines()
    capture_next = False
    for line in lines:
        stripped = line.strip()
        if capture_next and stripped:
            tokens = [
                token
                for token in (normalize_token(part) for part in re.split(r"[\s,]+", stripped))
                if token is not None
            ]
            if tokens:
                return tokens
        if stripped.lower().startswith("full swara sequence"):
            capture_next = True

    tokens: list[str] = []
    for part in re.split(r"[\s,]+", text):
        token = normalize_token(part)
        if token is not None:
            tokens.append(token)
    if tokens:
        return tokens
    raise ValueError("Could not extract swara tokens from text artifact.")


def load_tokens_from_artifact(entry: dict, manifest_dir: Path, raag: str) -> tuple[list[str], dict]:
    if isinstance(entry.get("tokens"), list):
        tokens = [token for token in (normalize_token(item) for item in entry["tokens"]) if token is not None]
        if tokens:
            return tokens, {"source_type": "tokens"}

    path_value = entry.get("path")
    if not path_value:
        raise ValueError("Each Suno artifact must provide either 'tokens' or a 'path'.")

    artifact_path = Path(path_value)
    if not artifact_path.is_absolute():
        artifact_path = manifest_dir / artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Suno artifact not found: '{artifact_path}'.")

    suffix = artifact_path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return extract_suno_audio_tokens(artifact_path, raag, entry)

    if suffix == ".json":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return extract_tokens_from_json_payload(payload), {"source_type": "json"}

    return extract_tokens_from_swara_text(artifact_path.read_text(encoding="utf-8")), {"source_type": "text"}


def infer_raag_from_filename(path: Path) -> str:
    match = re.match(r"^suno_(.+)$", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not infer raag from '{path.name}'. Expected filenames like 'suno_bageshree.mp3'."
        )
    candidate = match.group(1).replace("_", " ").replace("-", " ")
    raag = normalize_raag_name(candidate)
    if raag not in TARGET_RAAGS:
        raise ValueError(
            f"File '{path.name}' resolved to unsupported raag '{raag}'. "
            f"Supported ragas: {', '.join(TARGET_RAAGS)}."
        )
    return raag


def discover_suno_artifacts(suno_dir: str | Path) -> list[dict]:
    root = require_existing_path(suno_dir, "Suno artifact directory")
    if not root.exists():
        raise FileNotFoundError(
            f"Suno artifact directory not found: '{root}'. "
            "Place files like 'suno_bageshree.mp3' in that folder."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Suno artifact path is not a directory: '{root}'.")

    records: list[dict] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUNO_SUFFIXES:
            continue
        if not path.stem.lower().startswith("suno_"):
            continue
        raag = infer_raag_from_filename(path)
        records.append(
            {
                "raag": raag,
                "label": path.stem,
                "path": str(path.resolve()),
            }
        )

    if not records:
        raise ValueError(
            f"No Suno artifacts were found in '{root}'. "
            "Expected files named like 'suno_bageshree.mp3', 'suno_khamaj.wav', or 'suno_bhoop.flac'."
        )
    return records


def build_system_label(model: str, training: str) -> str:
    if model == "Suno":
        return "Suno (Pretrained)"
    pretty_training = PRETTY_TRAINING_LABELS.get(training, training.replace("_", " ").title())
    return f"{model} ({pretty_training})"


def load_baseline_records(report_path: str | Path) -> list[dict]:
    resolved_report = require_existing_path(report_path, "Baseline report")
    reports = load_json_records(resolved_report)
    baseline_records: list[dict] = []
    for report in reports:
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        model = normalize_model_name(report.get("model"))
        training = normalize_training_name(report.get("scope") or report.get("training"))
        raag = normalize_raag_name(report.get("raag"))
        baseline_records.append(
            {
                "artifact_id": f"{training}_{raag}_{model}".lower(),
                "label": report.get("artifact_dir") or report.get("swara_path") or f"{model}_{raag}",
                "source": "baseline",
                "model": model,
                "training": training,
                "raag": raag,
                "metrics": metrics,
            }
        )
    if not baseline_records:
        raise ValueError(f"No baseline metric rows were found in '{resolved_report}'.")
    return baseline_records


def evaluate_suno_artifacts(
    suno_dir: str | Path,
    prepared_dir: str | Path,
    *,
    transcription_defaults: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[dict]:
    references = build_reference_sequences(prepared_dir)
    artifact_root = require_existing_path(suno_dir, "Suno artifact directory")
    artifact_records = discover_suno_artifacts(artifact_root)
    log_progress(f"Found {len(artifact_records)} Suno artifact(s) in '{artifact_root}'.", enabled=verbose)
    suno_records: list[dict] = []
    defaults = dict(transcription_defaults or {})
    for index, entry in enumerate(artifact_records, start=1):
        artifact_entry = {**defaults, **entry}
        raag = normalize_raag_name(artifact_entry.get("raag"))
        label = artifact_entry.get("label") or Path(artifact_entry.get("path", f"suno_{index}")).stem or f"suno_{raag.lower()}_{index}"
        log_progress(f"[{index}/{len(artifact_records)}] Evaluating '{label}' for raag '{raag}'.", enabled=verbose)
        if raag not in references or not references[raag]:
            raise ValueError(f"No prepared reference sequences were found for raag '{raag}'.")
        tokens, transcription_metadata = load_tokens_from_artifact(artifact_entry, artifact_root, raag)
        metrics = compute_metrics(tokens, references[raag], raag)
        log_progress(
            f"[{index}/{len(artifact_records)}] Finished '{label}' with {len(tokens)} swara tokens.",
            enabled=verbose,
        )
        suno_records.append(
            {
                "artifact_id": f"suno_{raag.lower()}_{index}",
                "label": label,
                "path": artifact_entry.get("path"),
                "source": "suno",
                "model": "Suno",
                "training": "pretrained",
                "raag": raag,
                "tokens": tokens,
                "transcription": transcription_metadata,
                "metrics": metrics,
            }
        )
    return suno_records


def build_combined_dataframe(baseline_records: Sequence[dict], suno_records: Sequence[dict]) -> pd.DataFrame:
    df = build_metrics_dataframe([*baseline_records, *suno_records]).copy()
    df["model"] = df["model"].map(normalize_model_name)
    df["training"] = df["training"].map(normalize_training_name)
    df["raag"] = df["raag"].map(normalize_raag_name)
    df["training_label"] = df["training"].map(PRETTY_TRAINING_LABELS).fillna(df["training"])
    df["system"] = [build_system_label(model, training) for model, training in zip(df["model"], df["training"])]
    return df


def aggregate_system_metrics(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(["raag", "system", "model", "training"], as_index=False)[list(metrics)]
        .mean(numeric_only=True)
        .sort_values(["raag", "system"])
    )


def ordered_systems(df: pd.DataFrame) -> list[str]:
    baseline_models = ordered_models(df["model"].dropna().unique())
    order: list[str] = []
    for model in baseline_models:
        if model == "Suno":
            continue
        for training in ("global", "per_raga"):
            label = build_system_label(model, training)
            if label in set(df["system"]):
                order.append(label)
    if "Suno (Pretrained)" in set(df["system"]):
        order.append("Suno (Pretrained)")
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
            title=f"{raag}: Suno vs Baseline Architecture Metrics",
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
            title=f"{raag}: Creativity Radar for Suno and Baselines",
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
        title="Average Metrics: Suno vs Baseline Architectures",
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


def plot_delta_to_suno_heatmaps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_system_metrics(df, COMPARISON_METRICS)
    outputs: list[Path] = []
    for raag in TARGET_RAAGS:
        subset = summary[summary["raag"] == raag]
        if subset.empty or "Suno (Pretrained)" not in set(subset["system"]):
            continue
        pivot = subset.set_index("system")[COMPARISON_METRICS]
        suno_values = pivot.loc["Suno (Pretrained)"]
        delta = pivot.subtract(suno_values, axis=1).drop(index="Suno (Pretrained)", errors="ignore")
        if delta.empty:
            continue
        delta = delta.reindex(ordered_systems(summary), axis=0).dropna(how="all")
        delta = delta.rename(columns=DISPLAY_METRICS)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        plot_heatmap(
            delta,
            title=f"{raag}: Baseline Minus Suno",
            output_path=None,
            ax=ax,
            cmap="coolwarm",
            center=0.0,
        )
        outputs.append(save_figure(fig, output_dir / f"{slugify(raag)}_baseline_minus_suno_heatmap.png"))
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

    fig = plt.figure(figsize=(20, 13))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.16, 1.0, 1.08],
        width_ratios=[1.0, 0.5, 1.05],
        hspace=0.62,
        wspace=0.18,
        top=0.91,
        bottom=0.08,
        left=0.05,
        right=0.97,
    )

    ax_legend = fig.add_subplot(grid[0, :])
    ax_legend.axis("off")

    ax_radar = fig.add_subplot(grid[1, 0], polar=True)
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
    radar_handles, radar_labels = ax_radar.get_legend_handles_labels()
    if ax_radar.legend_ is not None:
        ax_radar.legend_.remove()
    ax_radar.tick_params(axis="x", pad=12, labelsize=10)
    ax_radar.tick_params(axis="y", labelsize=8)
    if radar_handles:
        ax_legend.legend(
            radar_handles,
            radar_labels,
            title="System",
            loc="center",
            ncol=4,
            frameon=True,
            fontsize=10,
            title_fontsize=10,
            handlelength=2.4,
            columnspacing=1.5,
        )

    ax_creativity = fig.add_subplot(grid[1, 2])
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
    ax_creativity.set_xticks(ax_creativity.get_xticks())
    ax_creativity.set_xticklabels(
        [tick.get_text().replace(" (", "\n(") for tick in ax_creativity.get_xticklabels()],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax_creativity.tick_params(axis="x", pad=6)

    ax_scatter = fig.add_subplot(grid[2, 0])
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
    scatter_handles, scatter_labels = ax_scatter.get_legend_handles_labels()
    if ax_scatter.legend_ is not None:
        ax_scatter.legend_.remove()
    ax_scatter_legend = fig.add_subplot(grid[2, 1])
    ax_scatter_legend.axis("off")
    ax_scatter_legend.legend(
        scatter_handles,
        scatter_labels,
        title="",
        loc="center left",
        frameon=True,
        borderpad=0.5,
        handlelength=1.5,
        handletextpad=0.45,
        labelspacing=0.45,
        fontsize=9,
        markerscale=0.85,
    )

    ax_heatmap = fig.add_subplot(grid[2, 2])
    plot_heatmap(
        heatmap_matrix,
        title="Average Metric Snapshot",
        output_path=None,
        ax=ax_heatmap,
        cmap="crest",
        center=None,
    )
    ax_heatmap.tick_params(axis="x", labelsize=9)
    ax_heatmap.tick_params(axis="y", labelsize=9)

    fig.suptitle("Suno vs Baseline Architecture Summary Dashboard", fontsize=18, fontweight="bold", y=0.985)
    return save_figure(fig, output_dir / "suno_vs_baseline_dashboard.png", tight=False)


def save_suno_transcriptions(output_dir: Path, suno_records: Sequence[dict]) -> list[Path]:
    transcription_root = output_dir / "suno_transcriptions"
    saved_paths: list[Path] = []
    for record in suno_records:
        raag_dir = transcription_root / record["raag"].lower()
        raag_dir.mkdir(parents=True, exist_ok=True)
        prefix = slugify(str(record.get("label") or record["artifact_id"]))

        text_path = raag_dir / f"{prefix}.swara.txt"
        text_lines = [
            f"# Label: {record.get('label', prefix)}",
            f"# Raga: {record['raag']}",
            f"# Source: {record.get('path') or 'inline tokens'}",
            f"# Transcription source type: {record.get('transcription', {}).get('source_type', 'unknown')}",
            "",
            "Full swara sequence:",
            " ".join(record.get("tokens", [])),
        ]
        text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        saved_paths.append(text_path)

        json_path = raag_dir / f"{prefix}.swara.json"
        json_payload = {
            "label": record.get("label"),
            "raag": record["raag"],
            "path": record.get("path"),
            "tokens": record.get("tokens", []),
            "transcription": record.get("transcription", {}),
            "metrics": record.get("metrics", {}),
        }
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        saved_paths.append(json_path)
    return saved_paths


def save_reports(
    output_dir: Path,
    baseline_records: Sequence[dict],
    suno_records: Sequence[dict],
    combined_df: pd.DataFrame,
) -> list[Path]:
    report_paths: list[Path] = []
    suno_report_path = output_dir / "suno_metrics_report.json"
    suno_report_path.write_text(json.dumps(list(suno_records), indent=2), encoding="utf-8")
    report_paths.append(suno_report_path)

    combined_report_path = output_dir / "combined_comparison_records.json"
    combined_report_path.write_text(
        json.dumps([*baseline_records, *suno_records], indent=2),
        encoding="utf-8",
    )
    report_paths.append(combined_report_path)

    csv_path = output_dir / "combined_metrics_table.csv"
    combined_df.to_csv(csv_path, index=False)
    report_paths.append(csv_path)
    report_paths.extend(save_suno_transcriptions(output_dir, suno_records))
    return report_paths


def generate_comparison_outputs(
    prepared_dir: str | Path,
    baseline_report: str | Path | None,
    suno_dir: str | Path,
    output_dir: str | Path | None,
    *,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
    iteration: int | None = None,
    pitch_method: str = "yin",
    sample_rate: int = 22050,
    hop_length: int = 512,
    frame_length: int = 2048,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[Path], Path, Path, str | None]:
    baseline_path, iteration_label = resolve_baseline_report_path(baseline_report, generated_dir, iteration)
    output_root = resolve_suno_output_dir(output_dir, iteration_label)
    output_root.mkdir(parents=True, exist_ok=True)
    log_progress(f"Writing comparison outputs to '{output_root}'.", enabled=verbose)

    prepared_path = require_existing_path(prepared_dir, "Prepared dataset directory")
    suno_path = require_existing_path(suno_dir, "Suno artifact directory")

    log_progress("Loading baseline metrics report.", enabled=verbose)
    baseline_records = load_baseline_records(baseline_path)
    log_progress(
        f"Transcribing Suno artifacts with pitch_method='{pitch_method}', sample_rate={sample_rate or 'native'}, "
        f"hop_length={hop_length}.",
        enabled=verbose,
    )
    suno_records = evaluate_suno_artifacts(
        suno_path,
        prepared_path,
        transcription_defaults={
            "pitch_method": pitch_method,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "frame_length": frame_length,
        },
        verbose=verbose,
    )
    combined_records = annotate_metric_records_with_composites([*baseline_records, *suno_records])
    baseline_records = combined_records[: len(baseline_records)]
    suno_records = combined_records[len(baseline_records) :]
    combined_df = build_combined_dataframe(baseline_records, suno_records)

    saved_paths: list[Path] = []
    log_progress("Writing reports and plots.", enabled=verbose)
    saved_paths.extend(save_reports(output_root, baseline_records, suno_records, combined_df))
    saved_paths.extend(plot_metric_bars_by_raag(combined_df, output_root))
    saved_paths.extend(plot_radar_by_raag(combined_df, output_root))
    saved_paths.append(plot_creativity_leaderboard(combined_df, output_root))
    saved_paths.append(plot_metric_heatmap(combined_df, output_root))
    saved_paths.append(plot_kl_vs_grammar_scatter(combined_df, output_root))
    saved_paths.extend(plot_delta_to_suno_heatmaps(combined_df, output_root))
    saved_paths.append(plot_summary_dashboard(combined_df, output_root))
    return combined_df, saved_paths, baseline_path, output_root, iteration_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Suno swara artifacts with the repo metrics and compare them against baseline model outputs."
    )
    parser.add_argument("--prepared-dir", required=True, help="Prepared dataset directory used to build reference sequences.")
    parser.add_argument(
        "--baseline-report",
        help=(
            "Path to generation_report.json from the baseline pipeline. "
            "If omitted, the script resolves it from --generated-dir and --iteration."
        ),
    )
    parser.add_argument(
        "--generated-dir",
        default=str(DEFAULT_GENERATED_DIR),
        help=(
            "Generated baseline artifacts root or a specific iteration directory. "
            "Used only when --baseline-report is omitted."
        ),
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help=(
            "Baseline iteration number to compare against, for example '--iteration 5'. "
            "If omitted and --baseline-report is not supplied, the latest generated iteration is used."
        ),
    )
    parser.add_argument(
        "--suno-dir",
        default=str(DEFAULT_SUNO_DIR),
        help=(
            "Folder containing Suno artifacts named like 'suno_bageshree.mp3'. "
            "Supported suffixes: .mp3, .wav, .m4a, .flac, .ogg, .txt, .json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Root directory where Suno comparison outputs will be saved. "
            "When an iteration is known, outputs are stored under an iteration-specific folder ending in '/suno_vs_baselines'."
        ),
    )
    parser.add_argument(
        "--pitch-method",
        choices=("yin", "pyin"),
        default="yin",
        help="Pitch extraction method for Suno audio files. 'yin' is faster; 'pyin' is slower but can be more precise.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Resample Suno audio before transcription. Use 0 to keep the original file sample rate.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length used during pitch extraction. Larger values run faster but produce fewer note frames.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=2048,
        help="Frame length used during pitch extraction.",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Print progress updates while evaluating Suno artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    combined_df, saved_paths, baseline_path, output_root, iteration_label = generate_comparison_outputs(
        prepared_dir=args.prepared_dir,
        baseline_report=args.baseline_report,
        suno_dir=args.suno_dir,
        output_dir=args.output_dir,
        generated_dir=args.generated_dir,
        iteration=args.iteration,
        pitch_method=args.pitch_method,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        verbose=args.verbose,
    )
    summary = {
        "rows": int(len(combined_df)),
        "ragas": sorted(combined_df["raag"].dropna().unique().tolist()),
        "systems": sorted(combined_df["system"].dropna().unique().tolist()),
        "baseline_report": str(baseline_path),
        "output_dir": str(output_root),
        "iteration": iteration_label,
        "saved_outputs": [str(path) for path in saved_paths],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
