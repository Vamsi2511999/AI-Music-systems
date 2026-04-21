from __future__ import annotations

import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "legend.frameon": True,
    }
)

SWARA_ORDER = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
TRAINING_ORDER = ["global", "per_raga", "pretrained"]
TRAINING_LABELS = {"global": "Global", "per_raga": "Per-raga", "pretrained": "Pretrained"}
TRAINING_PALETTE = {"global": "#1f77b4", "per_raga": "#ff7f0e", "pretrained": "#9467bd"}
MODEL_ORDER = ["Markov", "LSTM", "Transformer", "Suno"]
MODEL_PALETTE = {"Markov": "#4c78a8", "LSTM": "#f58518", "Transformer": "#54a24b", "Suno": "#b279a2"}
CORE_METRICS = ["novelty", "intentionality", "aesthetic", "motif", "grammar"]
CORRELATION_METRICS = ["novelty", "intentionality", "aesthetic", "motif", "grammar", "kl", "entropy"]
IMPROVEMENT_METRICS = ["novelty", "intentionality", "aesthetic", "motif", "grammar", "kl"]
DISPLAY_METRICS = {
    "novelty": "Novelty",
    "intentionality": "Intentionality",
    "aesthetic": "Aesthetic",
    "motif": "Motif",
    "grammar": "Grammar",
    "kl": "KL Divergence",
    "entropy": "Entropy",
    "creativity": "Creativity Composite",
}
COMPOSITE_SOURCE_METRICS = {
    "novelty_entropy_diff": ("novelty_entropy_diff", "entropy"),
    "novelty_kl_divergence": ("novelty_kl_divergence", "kl"),
    "intentionality": ("intentionality",),
    "aesthetics": ("aesthetics", "aesthetic"),
    "reflection": ("reflection",),
    "motif_recurrence": ("motif_recurrence", "motif"),
    "grammar_score": ("grammar_score", "grammar"),
}
ITERATION_PREFIX = "iteration-"
DEFAULT_GENERATED_DIR = Path(__file__).resolve().parent / "artifacts" / "generated"
DEFAULT_PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
SUNO_OUTPUT_DIRNAME = "suno_vs_baselines"
SUNO_METRICS_REPORT_NAME = "suno_metrics_report.json"


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def parse_iteration_dir(path: Path) -> int | None:
    if not path.is_dir():
        return None
    if not path.name.startswith(ITERATION_PREFIX):
        return None
    suffix = path.name[len(ITERATION_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def list_iteration_dirs(base_dir: Path) -> list[tuple[int, Path]]:
    if not base_dir.exists():
        return []
    iteration_dirs: list[tuple[int, Path]] = []
    for child in base_dir.iterdir():
        iteration = parse_iteration_dir(child)
        if iteration is not None:
            iteration_dirs.append((iteration, child))
    return sorted(iteration_dirs, key=lambda item: item[0])


def iter_path_candidates(path: str | Path) -> list[Path]:
    raw = Path(path).expanduser()
    candidates: list[Path] = [raw]
    if raw.is_absolute():
        return candidates

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    search_roots = [Path.cwd(), script_dir, repo_root]

    for base in search_roots:
        candidates.append(base / raw)

    if raw.parts and raw.parts[0] == script_dir.name:
        stripped = Path(*raw.parts[1:]) if len(raw.parts) > 1 else Path(".")
        for base in search_roots:
            candidates.append(base / stripped)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def resolve_existing_path(path: str | Path) -> Path | None:
    for candidate in iter_path_candidates(path):
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_output_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    best_candidate: Path | None = None
    best_missing_parts: int | None = None

    for candidate in iter_path_candidates(raw):
        current = candidate
        missing_parts = 0
        while not current.exists():
            parent = current.parent
            if parent == current:
                break
            current = parent
            missing_parts += 1

        if best_missing_parts is None or missing_parts < best_missing_parts:
            best_candidate = candidate
            best_missing_parts = missing_parts

    return (best_candidate or raw).resolve()


def find_json_suggestions() -> list[str]:
    suggestions: list[str] = []
    script_dir = Path(__file__).resolve().parent
    search_roots = [Path.cwd(), script_dir, script_dir.parent]
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for candidate in sorted(root.glob("**/*.json")):
            candidate_str = str(candidate)
            if "__pycache__" in candidate_str:
                continue
            if ".git/" in candidate_str:
                continue
            suggestions.append(candidate_str)
            if len(suggestions) >= 8:
                return suggestions
    return suggestions


def ordered_models(models: Iterable[str]) -> list[str]:
    values = [model for model in MODEL_ORDER if model in set(models)]
    remaining = sorted(set(models) - set(values))
    return values + remaining


def normalize_training_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("per_raag") or text.startswith("per_raga"):
        return "per_raga"
    if text.startswith("global"):
        return "global"
    return text or None


def normalize_model_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "markov": "Markov",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "music_transformer": "Transformer",
        "suno": "Suno",
    }
    return mapping.get(text, str(value).strip())


def load_json_records(path: str | Path) -> list[dict]:
    json_path = resolve_existing_path(path) or Path(path)
    if not json_path.exists():
        suggestions = find_json_suggestions()
        attempted = [str(candidate) for candidate in iter_path_candidates(path)]
        suggestion_text = ""
        if suggestions:
            formatted = "\n".join(f"  - {item}" for item in suggestions)
            suggestion_text = f"\nExisting JSON files you could use include:\n{formatted}"
        attempted_text = "\n".join(f"  - {item}" for item in attempted)
        raise FileNotFoundError(
            f"JSON file not found: '{json_path}'. "
            "Replace placeholder examples like 'path/to/metrics.json' with a real file path."
            f"\nPaths checked:\n{attempted_text}"
            f"{suggestion_text}"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "metrics", "reports", "data"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in '{path}'.")


def resolve_generated_iteration_dir(generated_dir: str | Path, iteration: int | None = None) -> tuple[Path, str | None]:
    base_dir = resolve_existing_path(generated_dir) or Path(generated_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Generated artifact directory not found: '{generated_dir}'.")

    explicit_iteration = parse_iteration_dir(base_dir)
    if explicit_iteration is not None:
        if iteration is not None and iteration != explicit_iteration:
            raise ValueError(
                f"Requested iteration {iteration}, but '{base_dir}' already points to "
                f"'{ITERATION_PREFIX}{explicit_iteration}'."
            )
        report_path = base_dir / "generation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing generation_report.json under '{base_dir}'.")
        return base_dir.resolve(), f"{ITERATION_PREFIX}{explicit_iteration}"

    if iteration is not None:
        candidate = base_dir / f"{ITERATION_PREFIX}{iteration}"
        report_path = candidate / "generation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"Could not find '{report_path}'. Pass a valid --iteration or a direct --metrics-json path."
            )
        return candidate.resolve(), f"{ITERATION_PREFIX}{iteration}"

    iteration_candidates = [
        (number, path)
        for number, path in list_iteration_dirs(base_dir)
        if (path / "generation_report.json").exists()
    ]
    if iteration_candidates:
        number, latest_dir = iteration_candidates[-1]
        return latest_dir.resolve(), f"{ITERATION_PREFIX}{number}"

    report_path = base_dir / "generation_report.json"
    if report_path.exists():
        return base_dir.resolve(), None

    raise FileNotFoundError(
        f"Could not find a generation_report.json under '{base_dir}'. "
        "Pass --metrics-json directly or point --generated-dir at a valid generated artifacts root."
    )


def resolve_metrics_json_path(
    metrics_json: str | None,
    generated_dir: str | Path,
    iteration: int | None,
) -> tuple[Path, str | None]:
    if metrics_json:
        json_path = resolve_existing_path(metrics_json) or Path(metrics_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Metrics JSON file not found: '{metrics_json}'.")
        parent_iteration = parse_iteration_dir(json_path.parent)
        iteration_label = f"{ITERATION_PREFIX}{parent_iteration}" if parent_iteration is not None else None
        return json_path.resolve(), iteration_label

    iteration_dir, iteration_label = resolve_generated_iteration_dir(generated_dir, iteration)
    return (iteration_dir / "generation_report.json").resolve(), iteration_label


def resolve_plot_output_dir(output_dir: str | Path | None, iteration_label: str | None) -> Path:
    root = resolve_output_path(output_dir or "plots")
    if iteration_label is None:
        return root
    if root.name == iteration_label or parse_iteration_dir(root) is not None:
        return root
    return root / iteration_label


def has_pretrained_rows(metric_dicts: Sequence[dict]) -> bool:
    for index, record in enumerate(metric_dicts):
        flattened = flatten_metric_record(record, index)
        model = normalize_model_label(flattened.get("model"))
        training = normalize_training_label(flattened.get("training") or flattened.get("scope"))
        if model == "Suno" or training == "pretrained":
            return True
    return False


def resolve_suno_metrics_path(
    suno_metrics_json: str | None,
    plots_root: str | Path,
    iteration_label: str | None,
) -> Path:
    if suno_metrics_json:
        json_path = resolve_existing_path(suno_metrics_json) or Path(suno_metrics_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Suno metrics JSON file not found: '{suno_metrics_json}'.")
        return json_path.resolve()

    if iteration_label is None:
        raise ValueError(
            "Could not infer a Suno metrics file automatically because no iteration could be resolved. "
            "Pass --iteration or provide --suno-metrics-json explicitly."
        )

    candidate = resolve_output_path(plots_root) / iteration_label / SUNO_OUTPUT_DIRNAME / SUNO_METRICS_REPORT_NAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find '{candidate}'. Run suno_baseline_comparison.py for {iteration_label} first, "
            "or pass --suno-metrics-json explicitly."
        )
    return candidate.resolve()


def flatten_metric_record(record: dict, index: int) -> dict:
    flat = dict(record)
    metrics = flat.pop("metrics", None)
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            flat.setdefault(key, value)
    flat.setdefault("row_id", index)
    flat["model"] = normalize_model_label(flat.get("model"))
    flat["training"] = normalize_training_label(flat.get("training") or flat.get("scope"))
    return flat


def coalesce_numeric(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for candidate in candidates:
        if candidate in df.columns:
            values = pd.to_numeric(df[candidate], errors="coerce")
            result = result.where(result.notna(), values)
    return result


def scale_percentage_like(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return values
    max_value = float(non_null.max())
    if max_value <= 1.0:
        return values * 100.0
    if max_value <= 10.0:
        return values * 10.0
    return values


def minmax_scale_to_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna()
    if valid.empty:
        return result
    minimum = float(valid.min())
    maximum = float(valid.max())
    if abs(maximum - minimum) < 1e-9:
        result.loc[valid.index] = 50.0
        return result
    result.loc[valid.index] = 100.0 * (valid - minimum) / (maximum - minimum)
    return result


def compute_overall_z_series(df: pd.DataFrame) -> pd.Series:
    z_components: dict[str, pd.Series] = {}
    for metric_name, candidates in COMPOSITE_SOURCE_METRICS.items():
        values = coalesce_numeric(df, candidates)
        valid = values.dropna()
        if valid.empty:
            continue
        std = float(valid.std())
        denominator = std if std > 1e-8 else 1.0
        z_components[metric_name] = (values - float(valid.mean())) / denominator

    if not z_components:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    z_frame = pd.DataFrame(z_components, index=df.index)
    return z_frame.mean(axis=1, skipna=True)


def build_metrics_dataframe(metric_dicts: Sequence[dict]) -> pd.DataFrame:
    normalized_records = [flatten_metric_record(record, index) for index, record in enumerate(metric_dicts)]
    df = pd.DataFrame(normalized_records)
    if df.empty:
        raise ValueError("No metric dictionaries were provided.")

    df = df[df["model"].notna() & df["training"].notna()].copy()
    if df.empty:
        raise ValueError("Metric dictionaries must include model and training/scope fields.")

    explicit_novelty = scale_percentage_like(coalesce_numeric(df, ["novelty"]))
    explicit_creativity = scale_percentage_like(coalesce_numeric(df, ["creativity", "creativity_composite", "creativity_score"]))

    df["intentionality"] = scale_percentage_like(coalesce_numeric(df, ["intentionality"]))
    df["aesthetic"] = scale_percentage_like(coalesce_numeric(df, ["aesthetic", "aesthetics"]))
    df["motif"] = scale_percentage_like(coalesce_numeric(df, ["motif", "motif_recurrence"]))
    df["grammar"] = scale_percentage_like(coalesce_numeric(df, ["grammar", "grammar_score"]))
    df["kl"] = coalesce_numeric(df, ["kl", "novelty_kl_divergence"])
    df["entropy"] = coalesce_numeric(df, ["entropy", "novelty_entropy_diff"])

    novelty_proxy = 0.5 * minmax_scale_to_100(df["entropy"]) + 0.5 * minmax_scale_to_100(df["kl"])
    df["novelty"] = explicit_novelty.where(explicit_novelty.notna(), novelty_proxy)

    computed_overall_z = compute_overall_z_series(df).round(4)
    existing_overall_z = (
        pd.to_numeric(df["overall_z"], errors="coerce")
        if "overall_z" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    if computed_overall_z.notna().any():
        df["overall_z"] = computed_overall_z.where(computed_overall_z.notna(), existing_overall_z)
    else:
        df["overall_z"] = existing_overall_z

    fallback_creativity = (
        minmax_scale_to_100(df["overall_z"])
        if df["overall_z"].notna().any()
        else df[CORE_METRICS].mean(axis=1, skipna=True)
    )
    df["creativity"] = explicit_creativity.where(explicit_creativity.notna(), fallback_creativity)

    df["training_label"] = df["training"].map(TRAINING_LABELS).fillna(df["training"])
    return df


def annotate_metric_records_with_composites(metric_dicts: Sequence[dict]) -> list[dict]:
    df = build_metrics_dataframe(metric_dicts)
    if df.empty:
        return [copy.deepcopy(record) for record in metric_dicts]

    row_metrics = df.set_index("row_id")
    annotated: list[dict] = []
    metric_field_map = {
        "overall_z": "overall_z",
        "creativity": "creativity",
        "novelty": "novelty",
        "intentionality": "intentionality",
        "aesthetic": "aesthetic",
        "motif": "motif",
        "grammar": "grammar",
        "kl": "kl",
        "entropy": "entropy",
    }

    for index, record in enumerate(metric_dicts):
        updated_record = copy.deepcopy(record)
        existing_metrics = updated_record.get("metrics", {})
        metrics = dict(existing_metrics) if isinstance(existing_metrics, dict) else {}
        if index in row_metrics.index:
            row = row_metrics.loc[index]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for source_key, target_key in metric_field_map.items():
                value = row.get(source_key)
                if pd.notna(value):
                    metrics[target_key] = round(float(value), 4)
        updated_record["metrics"] = metrics
        annotated.append(updated_record)
    return annotated


def aggregate_metrics(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(["model", "training", "training_label"], as_index=False)[list(metrics)]
        .mean(numeric_only=True)
        .sort_values(["model", "training"])
    )


def calculate_improvement(df: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_metrics(df, IMPROVEMENT_METRICS)
    rows: list[dict] = []
    for model in ordered_models(summary["model"].unique()):
        model_rows = summary[summary["model"] == model].set_index("training")
        if "global" not in model_rows.index or "per_raga" not in model_rows.index:
            continue
        delta = model_rows.loc["per_raga", IMPROVEMENT_METRICS] - model_rows.loc["global", IMPROVEMENT_METRICS]
        row = {"model": model}
        row.update(delta.to_dict())
        rows.append(row)
    improvement = pd.DataFrame(rows)
    if improvement.empty:
        return pd.DataFrame(columns=IMPROVEMENT_METRICS)
    return improvement.set_index("model")[IMPROVEMENT_METRICS]


def save_figure(fig: plt.Figure, output_path: str | Path, *, tight: bool = True) -> Path:
    path = resolve_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    ylabel: str,
    output_path: str | Path | None = None,
    *,
    ax: plt.Axes | None = None,
    palette: dict[str, str] | None = None,
    order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
    rotation: int = 0,
) -> plt.Axes:
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="")
    sns.despine(ax=ax)
    if created and output_path is not None:
        save_figure(fig, output_path)
    return ax


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path | None = None,
    *,
    ax: plt.Axes | None = None,
    palette: dict[str, str] | None = None,
    style: str | None = None,
    add_regression: bool = False,
) -> plt.Axes:
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 6))
    subset = data[[x, y, hue] + ([style] if style else [])].dropna()
    sns.scatterplot(data=subset, x=x, y=y, hue=hue, style=style, palette=palette, s=90, ax=ax)
    if add_regression and len(subset) >= 2:
        x_values = subset[x].to_numpy(dtype=float)
        y_values = subset[y].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        xs = np.linspace(x_values.min(), x_values.max(), 200)
        ax.plot(xs, slope * xs + intercept, linestyle="--", linewidth=2, color="black", label="Regression")
        ax.legend(title="")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    if created and output_path is not None:
        save_figure(fig, output_path)
    return ax


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "vlag",
    center: float | None = None,
    annot: bool = True,
    fmt: str = ".2f",
) -> plt.Axes:
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, cmap=cmap, center=center, annot=annot, fmt=fmt, linewidths=0.5, cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if created and output_path is not None:
        save_figure(fig, output_path)
    return ax


def plot_radar(
    data: pd.DataFrame,
    categories: Sequence[str],
    title: str,
    output_path: str | Path | None = None,
    *,
    ax: plt.Axes | None = None,
    label_column: str = "training_label",
    color_column: str = "training",
    palette: dict[str, str] | None = None,
) -> plt.Axes:
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    labels = [DISPLAY_METRICS.get(category, category.title()) for category in categories]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in data.iterrows():
        values = [float(row[category]) for category in categories]
        values += values[:1]
        label = row.get(label_column, "Series")
        color_key = row.get(color_column)
        color = palette.get(color_key, None) if palette else None
        ax.plot(angles, values, linewidth=2.5, label=label, color=color)
        ax.fill(angles, values, alpha=0.18, color=color)

    upper = float(np.nanmax(data[list(categories)].to_numpy(dtype=float))) if not data.empty else 100.0
    ax.set_ylim(0, max(100.0, upper * 1.1))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), title="")
    if created and output_path is not None:
        save_figure(fig, output_path)
    return ax


def make_distribution_samples(values: object) -> list[float]:
    if values is None:
        return []
    if isinstance(values, dict):
        samples: list[float] = []
        for note, weight in values.items():
            count = max(int(round(float(weight) * 200.0)), 1)
            sample_value = note_to_position(note)
            samples.extend([sample_value] * count)
        return samples
    if isinstance(values, list):
        if not values:
            return []
        if all(isinstance(item, str) for item in values):
            return [note_to_position(item) for item in values if note_to_position(item) is not None]
        if all(isinstance(item, (int, float)) for item in values):
            numeric = [float(item) for item in values]
            if len(numeric) == len(SWARA_ORDER) and abs(sum(numeric) - 1.0) < 0.25:
                samples = []
                for index, weight in enumerate(numeric):
                    count = max(int(round(weight * 200.0)), 1)
                    samples.extend([float(index)] * count)
                return samples
            return numeric
    return []


def note_to_position(note: object) -> float | None:
    if note is None:
        return None
    if isinstance(note, (int, float)):
        return float(note)
    text = str(note).strip()
    if text in SWARA_ORDER:
        return float(SWARA_ORDER.index(text))
    try:
        return float(text)
    except ValueError:
        return None


def extract_pitch_values(record: dict, prefix: str) -> list[float]:
    nested = record.get("pitch", {})
    for key in (
        f"{prefix}_pitch_values",
        f"{prefix}_pitches",
        f"{prefix}_pitch_distribution",
        f"{prefix}_distribution",
        f"{prefix}_notes",
    ):
        if key in record:
            return make_distribution_samples(record[key])
        if isinstance(nested, dict) and key in nested:
            return make_distribution_samples(nested[key])
    return []


def build_pitch_distribution_dataframe(metric_dicts: Sequence[dict], pitch_dicts: Sequence[dict] | None = None) -> pd.DataFrame:
    source_records = pitch_dicts if pitch_dicts else metric_dicts
    rows: list[dict] = []
    for index, raw_record in enumerate(source_records):
        record = flatten_metric_record(raw_record, index)
        model = record.get("model")
        training = record.get("training")
        if model is None or training is None:
            continue
        for prefix, source_label in (("training", "Training"), ("generated", "Generated")):
            values = extract_pitch_values(raw_record, prefix)
            for value in values:
                rows.append(
                    {
                        "model": model,
                        "training": training,
                        "training_label": TRAINING_LABELS.get(training, training),
                        "source": source_label,
                        "pitch_value": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_training_strategy_comparison_bars(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_metrics(df, CORE_METRICS)
    outputs: list[Path] = []
    for model in ordered_models(summary["model"].unique()):
        model_data = summary[summary["model"] == model].copy()
        if model_data.empty:
            continue
        melted = model_data.melt(
            id_vars=["training", "training_label"],
            value_vars=CORE_METRICS,
            var_name="metric",
            value_name="score",
        )
        melted["metric"] = melted["metric"].map(DISPLAY_METRICS)
        path = output_dir / f"{slugify(model)}_training_strategy_comparison.png"
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_bar(
            melted,
            x="metric",
            y="score",
            hue="training",
            title=f"{model}: Metric Comparison by Strategy",
            ylabel="Score",
            output_path=None,
            ax=ax,
            palette=TRAINING_PALETTE,
            order=[DISPLAY_METRICS[name] for name in CORE_METRICS],
            hue_order=TRAINING_ORDER,
        )
        outputs.append(save_figure(fig, path))
    return outputs


def plot_creativity_radar_comparison(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = aggregate_metrics(df, CORE_METRICS)
    outputs: list[Path] = []
    for model in ordered_models(summary["model"].unique()):
        model_data = summary[summary["model"] == model]
        if model_data.empty:
            continue
        path = output_dir / f"{slugify(model)}_creativity_radar.png"
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        plot_radar(
            model_data,
            CORE_METRICS,
            title=f"{model}: Creativity Profile by Strategy",
            output_path=None,
            ax=ax,
            palette=TRAINING_PALETTE,
        )
        outputs.append(save_figure(fig, path))
    return outputs


def plot_grouped_strategy_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    ax: plt.Axes | None = None,
) -> Path | plt.Axes:
    summary = aggregate_metrics(df, [metric])
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_bar(
        summary,
        x="model",
        y=metric,
        hue="training",
        title=title,
        ylabel=ylabel,
        output_path=None,
        ax=ax,
        palette=TRAINING_PALETTE,
        order=ordered_models(summary["model"].unique()),
        hue_order=TRAINING_ORDER,
    )
    if created:
        return save_figure(fig, output_path)
    return ax


def plot_kl_divergence_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    return plot_grouped_strategy_metric(
        df,
        metric="kl",
        title="KL Divergence by Model and Training Strategy",
        ylabel="KL Divergence",
        output_path=output_dir / "kl_divergence_comparison.png",
    )


def plot_kl_vs_grammar_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "kl_vs_grammar_scatter.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_scatter(
        df,
        x="kl",
        y="grammar",
        hue="model",
        style="training",
        title="KL Divergence vs Grammar Score",
        xlabel="KL Divergence",
        ylabel="Grammar Score",
        output_path=None,
        ax=ax,
        palette=MODEL_PALETTE,
        add_regression=True,
    )
    return save_figure(fig, path)


def plot_kl_vs_motif_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "kl_vs_motif_tradeoff.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_scatter(
        df,
        x="kl",
        y="motif",
        hue="model",
        style="training",
        title="KL Divergence vs Motif Recurrence",
        xlabel="KL Divergence",
        ylabel="Motif Recurrence",
        output_path=None,
        ax=ax,
        palette=MODEL_PALETTE,
        add_regression=True,
    )
    return save_figure(fig, path)


def plot_creativity_composite_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    return plot_grouped_strategy_metric(
        df,
        metric="creativity",
        title="Creativity Composite by Model and Training Strategy",
        ylabel="Creativity Composite",
        output_path=output_dir / "creativity_composite_comparison.png",
    )


def plot_improvement_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    improvement = calculate_improvement(df)
    if improvement.empty:
        raise ValueError("Improvement heatmap requires both global and per-raga rows for each model.")
    matrix = improvement.rename(columns=DISPLAY_METRICS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_heatmap(
        matrix,
        title="Per-raga Improvement over Global Training",
        output_path=None,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
    )
    return save_figure(fig, output_dir / "training_strategy_improvement_heatmap.png")


def plot_entropy_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    return plot_grouped_strategy_metric(
        df,
        metric="entropy",
        title="Entropy by Model and Training Strategy",
        ylabel="Entropy",
        output_path=output_dir / "entropy_comparison.png",
    )


def plot_distribution_overlap(dist_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if dist_df.empty:
        warnings.warn("No pitch distribution records were found. Skipping distribution overlap plots.", RuntimeWarning)
        return []
    outputs: list[Path] = []
    for model in ordered_models(dist_df["model"].unique()):
        model_df = dist_df[dist_df["model"] == model]
        trainings = [training for training in TRAINING_ORDER if training in set(model_df["training"])]
        if not trainings:
            continue
        fig, axes = plt.subplots(1, len(trainings), figsize=(7 * len(trainings), 5), sharey=True)
        if len(trainings) == 1:
            axes = [axes]
        for ax, training in zip(axes, trainings):
            subset = model_df[model_df["training"] == training]
            sns.kdeplot(
                data=subset,
                x="pitch_value",
                hue="source",
                fill=True,
                common_norm=False,
                bw_adjust=0.7,
                clip=(-0.5, len(SWARA_ORDER) - 0.5),
                ax=ax,
            )
            ax.set_title(f"{model} ({TRAINING_LABELS.get(training, training)})")
            ax.set_xlabel("Pitch / Swara Index")
            ax.set_ylabel("Density")
            ax.set_xticks(range(len(SWARA_ORDER)))
            ax.set_xticklabels(SWARA_ORDER)
        outputs.append(save_figure(fig, output_dir / f"{slugify(model)}_distribution_overlap.png"))
    return outputs


def plot_stability_analysis(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        df.groupby(["model", "training", "training_label"], as_index=False)["creativity"]
        .agg(mean="mean", std="std", count="count")
        .fillna({"std": 0.0})
    )
    summary["label"] = summary["model"] + "\n" + summary["training_label"]
    order = []
    for model in ordered_models(summary["model"].unique()):
        for training in TRAINING_ORDER:
            mask = (summary["model"] == model) & (summary["training"] == training)
            if mask.any():
                order.extend(summary.loc[mask, "label"].tolist())
    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary = summary.sort_values("label")

    fig, ax = plt.subplots(figsize=(11, 6))
    x_positions = np.arange(len(summary))
    colors = [TRAINING_PALETTE[training] for training in summary["training"]]
    ax.bar(
        x_positions,
        summary["mean"],
        yerr=summary["std"],
        capsize=6,
        color=colors,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["label"])
    ax.set_title("Stability Analysis: Creativity Mean with Standard Deviation")
    ax.set_ylabel("Creativity Composite")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRAINING_PALETTE[key], label=label)
        for key, label in TRAINING_LABELS.items()
        if key in set(summary["training"])
    ]
    ax.legend(handles=handles, title="")
    sns.despine(ax=ax)
    return save_figure(fig, output_dir / "stability_analysis.png")


def plot_metric_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> Path:
    corr = df[CORRELATION_METRICS].corr(numeric_only=True)
    matrix = corr.rename(index=DISPLAY_METRICS, columns=DISPLAY_METRICS)
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_heatmap(matrix, title="Metric Correlation Matrix", output_path=None, ax=ax, cmap="crest", center=0.0)
    return save_figure(fig, output_dir / "metric_correlation_matrix.png")


def plot_novelty_vs_intentionality_tradeoff(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "novelty_vs_intentionality_tradeoff.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_scatter(
        df,
        x="novelty",
        y="intentionality",
        hue="model",
        style="training",
        title="Novelty vs Intentionality Tradeoff",
        xlabel="Novelty",
        ylabel="Intentionality",
        output_path=None,
        ax=ax,
        palette=MODEL_PALETTE,
        add_regression=True,
    )
    return save_figure(fig, path)


def plot_model_creativity_profile(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = aggregate_metrics(df, CORE_METRICS + ["creativity"])
    summary["label"] = summary["model"] + "\n" + summary["training_label"]
    contributions = summary[CORE_METRICS].div(summary[CORE_METRICS].sum(axis=1), axis=0).fillna(0.0)
    contributions = contributions.mul(summary["creativity"], axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(summary))
    color_cycle = sns.color_palette("Set2", n_colors=len(CORE_METRICS))
    for metric, color in zip(CORE_METRICS, color_cycle):
        values = contributions[metric].to_numpy(dtype=float)
        ax.bar(summary["label"], values, bottom=bottom, label=DISPLAY_METRICS[metric], color=color, edgecolor="white")
        bottom += values
    ax.set_title("Model Creativity Profile")
    ax.set_ylabel("Contribution to Creativity Composite")
    ax.legend(title="", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    sns.despine(ax=ax)
    return save_figure(fig, output_dir / "model_creativity_profile.png")


def plot_training_strategy_summary_dashboard(df: pd.DataFrame, output_dir: Path) -> Path:
    summary = aggregate_metrics(df, CORE_METRICS + ["kl", "creativity"])
    radar_summary = summary.groupby(["training", "training_label"], as_index=False)[CORE_METRICS].mean(numeric_only=True)
    improvement = calculate_improvement(df).rename(columns=DISPLAY_METRICS)

    fig = plt.figure(figsize=(16, 12))
    ax_radar = fig.add_subplot(2, 2, 1, polar=True)
    plot_radar(
        radar_summary,
        CORE_METRICS,
        title="Average Creativity Radar",
        output_path=None,
        ax=ax_radar,
        palette=TRAINING_PALETTE,
    )

    ax_kl = fig.add_subplot(2, 2, 2)
    plot_bar(
        summary,
        x="model",
        y="kl",
        hue="training",
        title="KL Divergence Comparison",
        ylabel="KL Divergence",
        output_path=None,
        ax=ax_kl,
        palette=TRAINING_PALETTE,
        order=ordered_models(summary["model"].unique()),
        hue_order=TRAINING_ORDER,
    )

    ax_creativity = fig.add_subplot(2, 2, 3)
    plot_bar(
        summary,
        x="model",
        y="creativity",
        hue="training",
        title="Creativity Composite Comparison",
        ylabel="Creativity Composite",
        output_path=None,
        ax=ax_creativity,
        palette=TRAINING_PALETTE,
        order=ordered_models(summary["model"].unique()),
        hue_order=TRAINING_ORDER,
    )

    ax_heatmap = fig.add_subplot(2, 2, 4)
    if improvement.empty:
        ax_heatmap.axis("off")
        ax_heatmap.set_title("Improvement Heatmap\nNot enough paired data")
    else:
        plot_heatmap(
            improvement,
            title="Per-raga Minus Global",
            output_path=None,
            ax=ax_heatmap,
            cmap="coolwarm",
            center=0.0,
        )

    fig.suptitle("Training Strategy Summary Dashboard", fontsize=18, fontweight="bold", y=1.02)
    return save_figure(fig, output_dir / "training_strategy_summary_dashboard.png")


def generate_all_plots(
    metric_dicts: Sequence[dict],
    output_dir: str | Path,
    *,
    pitch_dicts: Sequence[dict] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    output_root = resolve_output_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df = build_metrics_dataframe(metric_dicts)
    pitch_df = build_pitch_distribution_dataframe(metric_dicts, pitch_dicts=pitch_dicts)

    outputs: list[Path] = []
    outputs.extend(plot_training_strategy_comparison_bars(df, output_root))
    outputs.extend(plot_creativity_radar_comparison(df, output_root))
    outputs.append(plot_kl_divergence_comparison(df, output_root))
    outputs.append(plot_kl_vs_grammar_scatter(df, output_root))
    outputs.append(plot_kl_vs_motif_scatter(df, output_root))
    outputs.append(plot_creativity_composite_comparison(df, output_root))
    outputs.append(plot_improvement_heatmap(df, output_root))
    outputs.append(plot_entropy_comparison(df, output_root))
    outputs.extend(plot_distribution_overlap(pitch_df, output_root))
    outputs.append(plot_stability_analysis(df, output_root))
    outputs.append(plot_metric_correlation_matrix(df, output_root))
    outputs.append(plot_novelty_vs_intentionality_tradeoff(df, output_root))
    outputs.append(plot_model_creativity_profile(df, output_root))
    outputs.append(plot_training_strategy_summary_dashboard(df, output_root))
    return df, outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate research-quality evaluation plots for music generation experiments.")
    parser.add_argument(
        "--metrics-json",
        help=(
            "Path to a JSON file containing metric dictionaries. "
            "If omitted, the script resolves generation_report.json from --generated-dir and --iteration."
        ),
    )
    parser.add_argument(
        "--generated-dir",
        default=str(DEFAULT_GENERATED_DIR),
        help=(
            "Generated artifacts root or a specific iteration directory. "
            "Used only when --metrics-json is omitted."
        ),
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help=(
            "Iteration number to load from --generated-dir, for example '--iteration 5'. "
            "If omitted, the latest available iteration is used."
        ),
    )
    parser.add_argument(
        "--include-suno",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include Suno metrics in the aggregate plot suite. "
            "When enabled, the script resolves the matching iteration's suno_metrics_report.json unless --suno-metrics-json is provided."
        ),
    )
    parser.add_argument(
        "--suno-metrics-json",
        help=(
            "Optional path to suno_metrics_report.json. "
            "Used only when --include-suno is enabled."
        ),
    )
    parser.add_argument(
        "--plots-root",
        default=str(DEFAULT_PLOTS_ROOT),
        help=(
            "Plots root used to auto-resolve Suno metrics when --include-suno is enabled and --suno-metrics-json is omitted."
        ),
    )
    parser.add_argument(
        "--pitch-json",
        help=(
            "Optional JSON file containing training/generated pitch values or pitch distributions. "
            "If omitted, the script tries to find pitch distribution payloads inside the metric dictionaries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory where the plot PNG files should be saved. "
            "When the metrics source maps to an iteration, plots are stored under an iteration subfolder by default."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path, iteration_label = resolve_metrics_json_path(args.metrics_json, args.generated_dir, args.iteration)
    metric_dicts = load_json_records(metrics_path)
    suno_metrics_path: Path | None = None
    if args.include_suno and not has_pretrained_rows(metric_dicts):
        suno_metrics_path = resolve_suno_metrics_path(args.suno_metrics_json, args.plots_root, iteration_label)
        metric_dicts = [*metric_dicts, *load_json_records(suno_metrics_path)]
    pitch_dicts = load_json_records(args.pitch_json) if args.pitch_json else None
    output_dir = resolve_plot_output_dir(args.output_dir, iteration_label)
    df, outputs = generate_all_plots(metric_dicts, output_dir, pitch_dicts=pitch_dicts)
    summary = {
        "rows": int(len(df)),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "trainings": sorted(df["training"].dropna().unique().tolist()),
        "metrics_source": str(metrics_path),
        "suno_metrics_source": str(suno_metrics_path) if suno_metrics_path is not None else None,
        "output_dir": str(output_dir),
        "iteration": iteration_label,
        "plots_saved": [str(path) for path in outputs],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
