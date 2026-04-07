from __future__ import annotations

import argparse
import json
import math
import random
import struct
import unicodedata
import warnings
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import mirdata
except ImportError:
    mirdata = None


TARGET_RAAGS = ("Bageshree", "Khamaj", "Bhoop")
SWARA_ORDER = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<SEP>"]
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
ITERATION_PREFIX = "iteration-"
GLOBAL_BUNDLE_DIRNAME = "all_raags"


@dataclass
class DataConfig:
    dataset_root: str
    output_dir: str
    data_source: str = "filesystem"
    mirdata_home: str | None = None
    mirdata_download: bool = False
    sequence_length: int = 64
    hop_length: int = 16
    min_phrase_tokens: int = 4
    train_split: float = 0.8
    seed: int = 42


@dataclass
class TrainConfig:
    prepared_dir: str
    output_dir: str
    epochs: int = 30
    batch_size: int = 16
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 3e-4
    transformer_heads: int = 4
    transformer_layers: int = 4
    grad_accum_steps: int = 2
    early_stopping_patience: int = 5
    train_scope: str = "both"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerateConfig:
    prepared_dir: str
    models_dir: str
    output_dir: str
    max_new_tokens: int = 192
    temperature: float = 0.95
    top_k: int = 8
    note_ticks: int = 480
    write_midi: bool = False
    write_audio: bool = False
    write_plots: bool = True
    audio_sample_rate: int = 22050
    seconds_per_note: float = 0.32
    drone_gain: float = 0.16
    grammar_bias_strength: float = 3.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


RAAG_GRAMMAR = {
    "Bageshree": {
        "allowed_notes": {"S", "R", "g", "M", "P", "D", "n"},
        "illegal_transitions": {("G", "R"), ("n", "R"), ("D", "R")},
        "vadi": "M",
        "samvadi": "S",
        "preferred_endings": ("M", "g", "S"),
    },
    "Khamaj": {
        "allowed_notes": {"S", "R", "G", "M", "P", "D", "n", "N"},
        "illegal_transitions": {("g", "P"), ("d", "S")},
        "vadi": "G",
        "samvadi": "N",
        "preferred_endings": ("G", "n", "S"),
    },
    "Bhoop": {
        "allowed_notes": {"S", "R", "G", "P", "D"},
        "illegal_transitions": {("G", "M"), ("D", "N"), ("P", "m"), ("R", "n")},
        "vadi": "G",
        "samvadi": "D",
        "preferred_endings": ("G", "R", "S"),
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_iteration_dir(path: Path) -> int | None:
    if not path.is_dir():
        return None
    if not path.name.startswith(ITERATION_PREFIX):
        return None
    suffix = path.name[len(ITERATION_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def list_iteration_dirs(base_dir: Path) -> List[Tuple[int, Path]]:
    if not base_dir.exists():
        return []
    iteration_dirs: List[Tuple[int, Path]] = []
    for child in base_dir.iterdir():
        iteration = parse_iteration_dir(child)
        if iteration is not None:
            iteration_dirs.append((iteration, child))
    return sorted(iteration_dirs, key=lambda item: item[0])


def create_run_output_dir(base_dir: str | Path) -> Path:
    output_root = Path(base_dir)
    explicit_iteration = parse_iteration_dir(output_root)
    if explicit_iteration is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    iterations = list_iteration_dirs(output_root)
    next_iteration = iterations[-1][0] + 1 if iterations else 1
    run_dir = output_root / f"{ITERATION_PREFIX}{next_iteration}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_bundle_dir(root_dir: Path, scope: str, raag: str | None = None) -> Path:
    if scope == "global":
        return root_dir / "global" / GLOBAL_BUNDLE_DIRNAME
    if raag is None:
        raise ValueError("A raag name is required for per_raag bundle directories.")
    return root_dir / "per_raag" / raag.lower()


def get_model_dir(bundle_dir: Path, model_name: str) -> Path:
    return bundle_dir / model_name


def get_bundle_file_prefix(bundle_dir: Path) -> str:
    if bundle_dir.name == "global" or bundle_dir.parent.name == "global":
        return "global"
    if bundle_dir.parent.name == "per_raag":
        return f"per_raag_{bundle_dir.name.lower()}"
    return bundle_dir.name.lower()


def get_training_model_path(bundle_dir: Path, model_name: str) -> Path:
    model_dir = get_model_dir(bundle_dir, model_name)
    prefix = get_bundle_file_prefix(bundle_dir)
    if model_name == "markov":
        return model_dir / f"{prefix}_{model_name}_model.json"
    return model_dir / f"{prefix}_{model_name}.pt"


def get_training_history_path(bundle_dir: Path, model_name: str) -> Path:
    prefix = get_bundle_file_prefix(bundle_dir)
    return get_model_dir(bundle_dir, model_name) / f"{prefix}_{model_name}_history.json"


def get_generated_model_dir(output_dir: Path, scope: str, raag: str, model_name: str) -> Path:
    return output_dir / scope / raag.lower() / model_name


def get_generated_file_prefix(scope: str, raag: str, model_name: str) -> str:
    return f"{scope}_{raag.lower()}_{model_name}"


def resolve_global_bundle_dir(models_dir: Path) -> Path | None:
    global_root = models_dir / "global"
    nested_bundle = global_root / GLOBAL_BUNDLE_DIRNAME
    if nested_bundle.exists():
        return nested_bundle
    if global_root.exists():
        return global_root
    return None


def resolve_training_model_path(bundle_dir: Path, model_name: str) -> Path:
    candidates: List[Path]
    if model_name == "markov":
        candidates = [
            get_training_model_path(bundle_dir, model_name),
            get_model_dir(bundle_dir, model_name) / "model.json",
            bundle_dir / "markov_model.json",
        ]
    elif model_name == "lstm":
        candidates = [
            get_training_model_path(bundle_dir, model_name),
            get_model_dir(bundle_dir, model_name) / "model.pt",
            bundle_dir / "lstm.pt",
        ]
    else:
        candidates = [
            get_training_model_path(bundle_dir, model_name),
            get_model_dir(bundle_dir, model_name) / "model.pt",
            bundle_dir / "music_transformer.pt",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find trained artifact for model '{model_name}' under '{bundle_dir}'.")


def resolve_prepared_dir(prepared_dir: str | Path) -> Path:
    base_dir = Path(prepared_dir)
    if (base_dir / "prepared_dataset.json").exists():
        return base_dir

    for _, candidate in reversed(list_iteration_dirs(base_dir)):
        if (candidate / "prepared_dataset.json").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find prepared_dataset.json under '{base_dir}'. "
        "Pass a prepared iteration directory or run the prepare step first."
    )


def resolve_models_dir(models_dir: str | Path) -> Path:
    base_dir = Path(models_dir)
    if (base_dir / "global").exists() or (base_dir / "per_raag").exists():
        return base_dir

    for _, candidate in reversed(list_iteration_dirs(base_dir)):
        if (candidate / "global").exists() or (candidate / "per_raag").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find trained model bundles under '{base_dir}'. "
        "Pass a models iteration directory or run the train-all step first."
    )


def normalize_raag_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    lookup = {
        "bageshri": "Bageshree",
        "bageshree": "Bageshree",
        "khamaj": "Khamaj",
        "bhoop": "Bhoop",
        "bhup": "Bhoop",
        "bhupali": "Bhoop",
        "bhoopali": "Bhoop",
    }
    key = ascii_name.strip().lower().replace("-", "").replace(" ", "")
    return lookup.get(key, ascii_name.strip() or name.strip())


def safe_read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                rows.append([part.strip() for part in line.split("\t")])
    except OSError:
        return []
    return rows


def strip_known_suffixes(name: str) -> str:
    suffixes = [
        ".ctonic",
        ".pitch",
        ".tempo-manual",
        ".sama-manual",
        ".sections-manual-p",
        ".phrases-manual",
        ".metadata",
    ]
    stem = Path(name).stem
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def infer_track_groups(dataset_root: Path) -> Dict[Tuple[Path, str], Dict[str, Path]]:
    groups: Dict[Tuple[Path, str], Dict[str, Path]] = {}
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        key = (path.parent, strip_known_suffixes(path.name))
        bucket = groups.setdefault(key, {})
        lower = path.name.lower()
        if lower.endswith(".json"):
            bucket["metadata"] = path
        elif ".phrases" in lower and lower.endswith(".txt"):
            bucket["phrases"] = path
        elif ".pitch" in lower and lower.endswith(".txt"):
            bucket["pitch"] = path
        elif ".ctonic" in lower and lower.endswith(".txt"):
            bucket["ctonic"] = path
    return groups


def extract_raags(metadata: dict) -> List[str]:
    results: List[str] = []
    for item in metadata.get("raags", []):
        if isinstance(item, str):
            results.append(normalize_raag_name(item))
        elif isinstance(item, dict):
            name = (
                item.get("common_name")
                or item.get("name")
                or item.get("raag")
                or item.get("title")
            )
            if name:
                results.append(normalize_raag_name(str(name)))
    if not results and metadata.get("title"):
        results.append(normalize_raag_name(str(metadata["title"])))
    return results


def get_raga_from_metadata_path(path: str | Path) -> str | None:
    metadata = safe_read_json(Path(path))
    if not metadata:
        return None
    raags = extract_raags(metadata)
    return raags[0] if raags else None


def extract_phrase_tokens(path: Path) -> List[str]:
    tokens: List[str] = []
    for row in read_rows(path):
        if len(row) < 4:
            continue
        phrase = [char for char in row[3] if char in SWARA_ORDER]
        if len(phrase) >= 2:
            tokens.extend(phrase)
            tokens.append(SEP_TOKEN)
    if tokens and tokens[-1] == SEP_TOKEN:
        tokens.pop()
    return tokens


def hz_to_cents(freq: float, tonic_hz: float) -> float:
    return 1200.0 * math.log2(max(freq, 1e-6) / max(tonic_hz, 1e-6))


def nearest_swara(cents: float) -> str:
    cent_map = {
        "S": 0.0,
        "r": 100.0,
        "R": 200.0,
        "g": 300.0,
        "G": 400.0,
        "m": 500.0,
        "M": 600.0,
        "P": 700.0,
        "d": 800.0,
        "D": 900.0,
        "n": 1000.0,
        "N": 1100.0,
    }
    normalized = cents % 1200.0
    return min(cent_map, key=lambda k: min(abs(cent_map[k] - normalized), 1200.0 - abs(cent_map[k] - normalized)))


def extract_pitch_tokens(pitch_path: Path, tonic_path: Path) -> List[str]:
    tonic_rows = read_rows(tonic_path)
    if not tonic_rows or not tonic_rows[0]:
        return []
    tonic_hz = float(tonic_rows[0][0])
    rows = read_rows(pitch_path)
    tokens: List[str] = []
    previous = None
    silence_run = 0
    for row in rows[::5]:
        if len(row) < 2:
            continue
        freq = float(row[1])
        if freq <= 0.0:
            silence_run += 1
            if silence_run >= 3 and (not tokens or tokens[-1] != SEP_TOKEN):
                tokens.append(SEP_TOKEN)
            continue
        silence_run = 0
        note = nearest_swara(hz_to_cents(freq, tonic_hz))
        if note != previous:
            tokens.append(note)
            previous = note
    return tokens


def load_pitch_hz_series(path: str | Path) -> np.ndarray | None:
    rows = read_rows(Path(path))
    freqs: List[float] = []
    for row in rows:
        if len(row) < 2:
            continue
        try:
            hz = float(row[1])
        except (TypeError, ValueError):
            continue
        if hz > 1.0:
            freqs.append(hz)
    if not freqs:
        return None
    return np.asarray(freqs, dtype=np.float32)


def hz_to_midi_value(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-6) / 440.0)


def extract_tokens_from_pitch_path(track: object) -> List[str]:
    pitch_path = getattr(track, "pitch_path", None)
    ctonic_path = getattr(track, "ctonic_path", None)
    if pitch_path and ctonic_path:
        tokens = extract_pitch_tokens(Path(pitch_path), Path(ctonic_path))
        if tokens:
            return tokens

    hz_series = load_pitch_hz_series(pitch_path) if pitch_path else None
    if hz_series is None:
        return []

    midi = np.round([hz_to_midi_value(hz) for hz in hz_series]).astype(int)
    if midi.size == 0:
        return []
    normalized = midi - int(np.min(midi))
    tokens: List[str] = []
    previous = None
    for value in normalized[::5]:
        token = SWARA_ORDER[int(value) % 12]
        if token != previous:
            tokens.append(token)
            previous = token
    return tokens


def encode_sequences(sequences: Iterable[List[str]], token_to_id: Dict[str, int]) -> List[List[int]]:
    return [[token_to_id[BOS_TOKEN], *[token_to_id[token] for token in sequence], token_to_id[EOS_TOKEN]] for sequence in sequences]


def sliding_windows(tokens: Sequence[int], seq_len: int, hop_len: int) -> List[List[int]]:
    if len(tokens) < seq_len + 1:
        return []
    return [list(tokens[i : i + seq_len + 1]) for i in range(0, len(tokens) - seq_len, hop_len)]


def build_vocab(sequences: Iterable[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    ordered = list(SPECIAL_TOKENS)
    seen = set(ordered)
    for seq in sequences:
        for token in seq:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def extract_raags_from_mirdata_track(track: object) -> List[str]:
    raag_items = getattr(track, "raags", None) or []
    results: List[str] = []
    for item in raag_items:
        if isinstance(item, str):
            results.append(normalize_raag_name(item))
            continue
        for key in ("common_name", "name", "raag", "title"):
            value = getattr(item, key, None)
            if value:
                results.append(normalize_raag_name(str(value)))
                break
        else:
            if isinstance(item, dict):
                value = item.get("common_name") or item.get("name") or item.get("raag") or item.get("title")
                if value:
                    results.append(normalize_raag_name(str(value)))
    if not results:
        title = getattr(track, "title", None)
        if title:
            results.append(normalize_raag_name(str(title)))
    return results


def extract_tokens_from_mirdata_track(track: object) -> List[str]:
    phrases_attr = getattr(track, "phrases", None)
    if phrases_attr:
        phrase_tokens: List[str] = []
        try:
            phrase_rows = list(phrases_attr)
        except TypeError:
            phrase_rows = []
        for row in phrase_rows:
            phrase_text = ""
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                phrase_text = str(row[3])
            elif isinstance(row, dict):
                phrase_text = str(row.get("label") or row.get("phrase") or row.get("text") or "")
            else:
                phrase_text = str(row)
            notes = [char for char in phrase_text if char in SWARA_ORDER]
            if len(notes) >= 2:
                phrase_tokens.extend(notes)
                phrase_tokens.append(SEP_TOKEN)
        if phrase_tokens and phrase_tokens[-1] == SEP_TOKEN:
            phrase_tokens.pop()
        if phrase_tokens:
            return phrase_tokens

    pitch_attr = getattr(track, "pitch", None)
    tonic_attr = getattr(track, "ctonic", None) or getattr(track, "tonic", None)
    if pitch_attr is None or tonic_attr is None:
        return []

    try:
        tonic_hz = float(tonic_attr if np.isscalar(tonic_attr) else np.asarray(tonic_attr).squeeze())
    except (TypeError, ValueError):
        return []

    frequency_values = getattr(pitch_attr, "frequencies", None)
    if frequency_values is not None:
        freqs = np.asarray(frequency_values, dtype=float).reshape(-1)
        voicing_values = getattr(pitch_attr, "voicing", None)
        if voicing_values is not None:
            voicing = np.asarray(voicing_values, dtype=float).reshape(-1)
            if voicing.shape == freqs.shape:
                freqs = np.where(voicing > 0.0, freqs, 0.0)
    else:
        pitch_array = np.asarray(pitch_attr)
        if pitch_array.size == 0:
            return []
        if pitch_array.ndim == 2 and pitch_array.shape[1] >= 2:
            freqs = pitch_array[:, 1]
        else:
            freqs = pitch_array.reshape(-1)
        try:
            freqs = np.asarray(freqs, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return []

    if freqs.size == 0:
        return []

    tokens: List[str] = []
    previous = None
    silence_run = 0
    for freq in freqs[::5]:
        freq = float(freq)
        if freq <= 0.0:
            silence_run += 1
            if silence_run >= 3 and (not tokens or tokens[-1] != SEP_TOKEN):
                tokens.append(SEP_TOKEN)
            continue
        silence_run = 0
        note = nearest_swara(hz_to_cents(freq, tonic_hz))
        if note != previous:
            tokens.append(note)
            previous = note
    return tokens


def validate_cli_path(path_value: str | None, arg_name: str) -> None:
    if not path_value:
        return
    normalized = str(Path(path_value))
    if normalized == "/path" or normalized.startswith("/path/"):
        raise ValueError(
            f"{arg_name} is still using the README placeholder path: {path_value}. "
            "Replace it with a real writable directory, for example "
            "./data/saraga_hindustani."
        )


def ensure_directory_writable(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise RuntimeError(
            f"{label} is not writable: {path}. Choose a directory you can write to, "
            "for example ./data/saraga_hindustani or another folder in your home directory."
        ) from exc


def get_mirdata_track_ids(dataset: object) -> List[str]:
    try:
        return list(getattr(dataset, "track_ids", []))
    except FileNotFoundError as exc:
        index_data = getattr(dataset, "_index_data", None)
        remote_index = getattr(index_data, "remote", None)
        dataset_name = getattr(dataset, "name", "dataset")
        if remote_index is None:
            raise
        try:
            dataset.download(partial_download=["index"])
            return list(getattr(dataset, "track_ids", []))
        except Exception as download_exc:
            raise RuntimeError(
                f"mirdata could not load the packaged index for {dataset_name}. "
                "The pipeline tried to download the missing index automatically, "
                "but that failed. Re-run with network access, install a mirdata "
                "build that includes the Saraga index, or switch to "
                "--data-source filesystem if the dataset is already on disk."
            ) from download_exc


def build_raga_dataset_from_mirdata(dataset: object) -> Dict[str, List[List[str]]]:
    raga_data: Dict[str, List[List[str]]] = {raag: [] for raag in TARGET_RAAGS}
    track_ids = get_mirdata_track_ids(dataset)
    for track_id in track_ids:
        track = dataset.track(track_id)
        metadata_path = getattr(track, "metadata_path", None)
        if metadata_path:
            primary_raag = get_raga_from_metadata_path(metadata_path)
            raags = {primary_raag} if primary_raag else set()
        else:
            raags = set()
        raags |= set(extract_raags_from_mirdata_track(track))
        matched_raags = [raag for raag in TARGET_RAAGS if raag in raags]
        if not matched_raags:
            continue
        tokens = extract_tokens_from_mirdata_track(track)
        if not tokens:
            tokens = extract_tokens_from_pitch_path(track)
        tokens = [token for token in tokens if token in SWARA_ORDER or token == SEP_TOKEN]
        if not tokens:
            continue
        for raag in matched_raags:
            raga_data[raag].append(tokens)
    return raga_data


def build_raw_records_from_mirdata(config: DataConfig) -> List[dict]:
    if mirdata is None:
        raise ImportError("mirdata is not installed. Add it to requirements and install dependencies to use --data-source mirdata.")
    data_home = config.mirdata_home or config.dataset_root
    validate_cli_path(config.dataset_root, "--dataset-root")
    validate_cli_path(config.mirdata_home, "--mirdata-home")
    if config.mirdata_download:
        ensure_directory_writable(Path(data_home), "--mirdata-home/--dataset-root")
    dataset = mirdata.initialize("saraga_hindustani", data_home=data_home)
    if config.mirdata_download:
        dataset.download()
    raga_data = build_raga_dataset_from_mirdata(dataset)
    raw_records: List[dict] = []
    for raag, sequences in raga_data.items():
        for index, tokens in enumerate(sequences):
            if len(tokens) >= config.min_phrase_tokens:
                raw_records.append({"track": f"{raag}_{index}", "raag": raag, "tokens": tokens})
    return raw_records


def prepare_dataset(config: DataConfig) -> Dict[str, str]:
    set_seed(config.seed)
    output_dir = create_run_output_dir(config.output_dir)

    if config.data_source == "mirdata":
        raw_records = build_raw_records_from_mirdata(config)
    else:
        dataset_root = Path(config.dataset_root)
        track_groups = infer_track_groups(dataset_root)
        raw_records = []
        for (_, track_name), group in sorted(track_groups.items(), key=lambda item: item[0][1]):
            metadata_path = group.get("metadata")
            if metadata_path is None:
                continue
            metadata = safe_read_json(metadata_path)
            if not metadata:
                continue
            raags = set(extract_raags(metadata))
            matched = [raag for raag in TARGET_RAAGS if raag in raags]
            if not matched:
                continue
            tokens = extract_phrase_tokens(group["phrases"]) if "phrases" in group else []
            if not tokens and "pitch" in group and "ctonic" in group:
                tokens = extract_pitch_tokens(group["pitch"], group["ctonic"])
            tokens = [token for token in tokens if token in SWARA_ORDER or token == SEP_TOKEN]
            if len(tokens) < config.min_phrase_tokens:
                continue
            for raag in matched:
                raw_records.append({"track": track_name, "raag": raag, "tokens": tokens})

    if not raw_records:
        raise RuntimeError("No matching Saraga tracks were found for Bageshree, Khamaj, or Bhoop.")

    token_to_id, id_to_token = build_vocab([record["tokens"] for record in raw_records])
    raag_to_id = {raag: idx for idx, raag in enumerate(TARGET_RAAGS)}

    encoded_tracks: List[dict] = []
    for record in raw_records:
        encoded = encode_sequences([record["tokens"]], token_to_id)[0]
        windows = sliding_windows(encoded, config.sequence_length, config.hop_length)
        if windows:
            encoded_tracks.append(
                {
                    "track": record["track"],
                    "raag": record["raag"],
                    "raag_id": raag_to_id[record["raag"]],
                    "token_ids": encoded,
                    "windows": windows,
                }
            )

    if not encoded_tracks:
        raise RuntimeError("Found Saraga tracks, but none were long enough after preprocessing.")

    random.shuffle(encoded_tracks)
    split = max(1, int(len(encoded_tracks) * config.train_split))
    split = min(split, len(encoded_tracks) - 1) if len(encoded_tracks) > 1 else len(encoded_tracks)
    train_tracks = encoded_tracks[:split]
    val_tracks = encoded_tracks[split:] if split < len(encoded_tracks) else encoded_tracks[-1:]

    payload_config = asdict(config)
    payload_config["output_dir"] = str(output_dir)
    payload = {
        "config": payload_config,
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "raag_to_id": raag_to_id,
        "id_to_raag": {str(v): k for k, v in raag_to_id.items()},
        "tracks": encoded_tracks,
        "train_examples": [
            {"raag_id": track["raag_id"], "tokens": window}
            for track in train_tracks
            for window in track["windows"]
        ],
        "val_examples": [
            {"raag_id": track["raag_id"], "tokens": window}
            for track in val_tracks
            for window in track["windows"]
        ],
    }

    prepared_path = output_dir / "prepared_dataset.json"
    prepared_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "prepared_path": str(prepared_path),
        "tracks_per_raag": {raag: sum(track["raag"] == raag for track in encoded_tracks) for raag in TARGET_RAAGS},
        "train_windows": len(payload["train_examples"]),
        "val_windows": len(payload["val_examples"]),
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "prepared_path": str(prepared_path), "summary_path": str(summary_path)}


class MelodyDataset(Dataset):
    def __init__(self, examples: List[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.examples[index]
        seq = torch.tensor(item["tokens"], dtype=torch.long)
        return seq[:-1], seq[1:], torch.tensor(item["raag_id"], dtype=torch.long)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, r = zip(*batch)
    return torch.stack(x), torch.stack(y), torch.stack(r)


class ConditionalLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_raags: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.raag_embedding = nn.Embedding(num_raags, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        raag_ids: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        token_emb = self.token_embedding(tokens)
        raag_emb = self.raag_embedding(raag_ids).unsqueeze(1).expand(-1, tokens.size(1), -1)
        x = torch.cat([token_emb, raag_emb], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return self.output(self.dropout(out)), hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_raags: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.raag_embedding = nn.Embedding(num_raags, embedding_dim)
        self.condition_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        self.position = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=embedding_dim * 2,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, raag_ids: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(tokens)
        raag_emb = self.raag_embedding(raag_ids).unsqueeze(1).expand(-1, tokens.size(1), -1)
        x = self.condition_projection(torch.cat([token_emb, raag_emb], dim=-1))
        x = self.position(x)
        seq_len = tokens.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        padding_mask = tokens == self.pad_idx
        encoded = self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        return self.output(encoded)


def load_prepared(prepared_dir: str) -> dict:
    path = resolve_prepared_dir(prepared_dir) / "prepared_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_loss(token_to_id: Dict[str, int], device: str) -> nn.CrossEntropyLoss:
    vocab_size = len(token_to_id)
    weights = torch.ones(vocab_size, dtype=torch.float32)
    weights[token_to_id[SEP_TOKEN]] = 0.5
    return nn.CrossEntropyLoss(ignore_index=token_to_id[PAD_TOKEN], weight=weights.to(device))


def filter_prepared_by_raag(prepared: dict, raag: str) -> dict:
    raag_id = prepared["raag_to_id"][raag]
    tracks = [track for track in prepared["tracks"] if track["raag"] == raag]
    train_examples = [example for example in prepared["train_examples"] if example["raag_id"] == raag_id]
    val_examples = [example for example in prepared["val_examples"] if example["raag_id"] == raag_id]
    subset = dict(prepared)
    subset["tracks"] = tracks
    subset["train_examples"] = train_examples
    subset["val_examples"] = val_examples
    return subset


def ensure_bundle_examples(prepared: dict, bundle_name: str) -> dict:
    if prepared["train_examples"] and prepared["val_examples"]:
        return prepared

    all_examples = [
        {"raag_id": track["raag_id"], "tokens": window}
        for track in prepared["tracks"]
        for window in track.get("windows", [])
    ]
    if not all_examples:
        return prepared

    train_split = float(prepared.get("config", {}).get("train_split", 0.8))
    seed = int(prepared.get("config", {}).get("seed", 42))
    shuffled_examples = list(all_examples)
    random.Random(seed).shuffle(shuffled_examples)

    if len(shuffled_examples) == 1:
        train_examples = shuffled_examples
        val_examples = shuffled_examples
    else:
        split = max(1, int(len(shuffled_examples) * train_split))
        split = min(split, len(shuffled_examples) - 1)
        train_examples = shuffled_examples[:split]
        val_examples = shuffled_examples[split:]

    warnings.warn(
        f"Bundle '{bundle_name}' had an empty train or validation split after track-level filtering. "
        "Falling back to a window-level split for this bundle.",
        RuntimeWarning,
    )
    subset = dict(prepared)
    subset["train_examples"] = train_examples
    subset["val_examples"] = val_examples
    return subset


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_epoch_lstm(
    model: ConditionalLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.CrossEntropyLoss,
    device: str,
    grad_accum_steps: int,
) -> float:
    model.train(optimizer is not None)
    losses: List[float] = []
    if optimizer is not None:
        optimizer.zero_grad()
    for step, (x, y, raag_ids) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        raag_ids = raag_ids.to(device)
        logits, _ = model(x, raag_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if optimizer is not None:
            (loss / grad_accum_steps).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def run_epoch_transformer(
    model: MusicTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.CrossEntropyLoss,
    device: str,
    grad_accum_steps: int,
) -> float:
    model.train(optimizer is not None)
    losses: List[float] = []
    if optimizer is not None:
        optimizer.zero_grad()
    for step, (x, y, raag_ids) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        raag_ids = raag_ids.to(device)
        logits = model(x, raag_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if optimizer is not None:
            (loss / grad_accum_steps).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_markov_model(prepared: dict, output_dir: Path) -> str:
    token_map = prepared["id_to_token"]
    model: Dict[str, Dict[str, Dict[str, int]]] = {}
    for raag in TARGET_RAAGS:
        transitions: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        starts: Counter[str] = Counter()
        for track in prepared["tracks"]:
            if track["raag"] != raag:
                continue
            tokens = [token_map[str(token_id)] for token_id in track["token_ids"] if token_map[str(token_id)] not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
            note_tokens = [token for token in tokens if token in SWARA_ORDER or token == SEP_TOKEN]
            if len(note_tokens) < 2:
                continue
            starts[note_tokens[0]] += 1
            for left, right in zip(note_tokens[:-1], note_tokens[1:]):
                transitions[left][right] += 1
        model[raag] = {
            "starts": dict(starts),
            "transitions": {state: dict(counter) for state, counter in transitions.items()},
        }
    path = get_training_model_path(output_dir, "markov")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return str(path)


def train_model_bundle(prepared: dict, config: TrainConfig, output_dir: Path, bundle_name: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = ensure_bundle_examples(prepared, bundle_name)
    if not prepared["train_examples"] or not prepared["val_examples"]:
        raise RuntimeError(f"No train/validation windows available for bundle '{bundle_name}'.")

    train_ds = MelodyDataset(prepared["train_examples"])
    val_ds = MelodyDataset(prepared["val_examples"])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    criterion = build_loss(prepared["token_to_id"], config.device)

    results: Dict[str, str] = {}
    results["markov"] = train_markov_model(prepared, output_dir)

    lstm = ConditionalLSTM(
        vocab_size=len(prepared["token_to_id"]),
        num_raags=len(prepared["raag_to_id"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)
    lstm_optim = torch.optim.Adam(lstm.parameters(), lr=config.learning_rate)
    lstm_history: List[dict] = []
    best_lstm = float("inf")
    stale_epochs = 0
    lstm_path = get_training_model_path(output_dir, "lstm")
    lstm_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch_lstm(lstm, train_loader, lstm_optim, criterion, config.device, config.grad_accum_steps)
        val_loss = run_epoch_lstm(lstm, val_loader, None, criterion, config.device, config.grad_accum_steps)
        lstm_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_lstm:
            best_lstm = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": lstm.state_dict(),
                    "config": asdict(config),
                    "bundle_name": bundle_name,
                    "vocab_size": len(prepared["token_to_id"]),
                    "num_raags": len(prepared["raag_to_id"]),
                    "pad_idx": prepared["token_to_id"][PAD_TOKEN],
                },
                lstm_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    save_json(get_training_history_path(output_dir, "lstm"), lstm_history)
    results["lstm"] = str(lstm_path)

    transformer = MusicTransformer(
        vocab_size=len(prepared["token_to_id"]),
        num_raags=len(prepared["raag_to_id"]),
        embedding_dim=config.embedding_dim,
        num_heads=config.transformer_heads,
        num_layers=config.transformer_layers,
        dropout=config.dropout,
        pad_idx=prepared["token_to_id"][PAD_TOKEN],
    ).to(config.device)
    transformer_optim = torch.optim.Adam(transformer.parameters(), lr=config.learning_rate)
    transformer_history: List[dict] = []
    best_transformer = float("inf")
    stale_epochs = 0
    transformer_path = get_training_model_path(output_dir, "music_transformer")
    transformer_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch_transformer(transformer, train_loader, transformer_optim, criterion, config.device, config.grad_accum_steps)
        val_loss = run_epoch_transformer(transformer, val_loader, None, criterion, config.device, config.grad_accum_steps)
        transformer_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_transformer:
            best_transformer = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": transformer.state_dict(),
                    "config": asdict(config),
                    "bundle_name": bundle_name,
                    "vocab_size": len(prepared["token_to_id"]),
                    "num_raags": len(prepared["raag_to_id"]),
                    "pad_idx": prepared["token_to_id"][PAD_TOKEN],
                },
                transformer_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    save_json(get_training_history_path(output_dir, "music_transformer"), transformer_history)
    results["music_transformer"] = str(transformer_path)
    return results


def train_all_models(config: TrainConfig) -> Dict[str, str]:
    set_seed(config.seed)
    prepared = load_prepared(config.prepared_dir)
    output_dir = create_run_output_dir(config.output_dir)

    results: Dict[str, str] = {"output_dir": str(output_dir)}
    if config.train_scope in ("global", "both"):
        global_dir = get_bundle_dir(output_dir, "global")
        for name, path in train_model_bundle(prepared, config, global_dir, "global").items():
            results[f"global_{name}"] = path

    if config.train_scope in ("per_raag", "both"):
        for raag in TARGET_RAAGS:
            subset = filter_prepared_by_raag(prepared, raag)
            per_raag_dir = get_bundle_dir(output_dir, "per_raag", raag)
            for name, path in train_model_bundle(subset, config, per_raag_dir, raag).items():
                results[f"{raag}_{name}"] = path

    return results


def decode_token_ids(token_ids: Sequence[int], id_to_token: Dict[int, str]) -> List[str]:
    out: List[str] = []
    for token_id in token_ids:
        token = id_to_token[token_id]
        if token in (PAD_TOKEN, BOS_TOKEN):
            continue
        if token == EOS_TOKEN:
            break
        out.append(token)
    return out


def last_swara(tokens: Sequence[str]) -> str | None:
    for token in reversed(tokens):
        if token in SWARA_ORDER:
            return token
    return None


def phrase_length(tokens: Sequence[str]) -> int:
    count = 0
    for token in reversed(tokens):
        if token == SEP_TOKEN:
            break
        if token in SWARA_ORDER:
            count += 1
    return count


def apply_raag_bias_to_logits(
    logits: torch.Tensor,
    id_to_token: Dict[int, str],
    generated_tokens: Sequence[str],
    raag: str,
    strength: float,
) -> torch.Tensor:
    adjusted = logits.clone()
    grammar = RAAG_GRAMMAR[raag]
    previous_note = last_swara(generated_tokens)
    current_phrase_len = phrase_length(generated_tokens)

    for index in range(adjusted.numel()):
        token = id_to_token[index]
        if token in SWARA_ORDER:
            if token not in grammar["allowed_notes"]:
                adjusted[index] -= 100.0
                continue
            if previous_note and (previous_note, token) in grammar["illegal_transitions"]:
                adjusted[index] -= 100.0
                continue
            if token == grammar["vadi"]:
                adjusted[index] += 0.45 * strength
            elif token == grammar["samvadi"]:
                adjusted[index] += 0.25 * strength
            if current_phrase_len >= 10 and token in grammar["preferred_endings"]:
                adjusted[index] += 0.35 * strength
        elif token == SEP_TOKEN:
            if current_phrase_len < 5:
                adjusted[index] -= 1.2 * strength
            elif current_phrase_len >= 12:
                adjusted[index] += 0.35 * strength
        elif token == EOS_TOKEN and len(generated_tokens) < 24:
            adjusted[index] -= 2.0 * strength
    return adjusted


def top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    scaled = logits / max(temperature, 1e-6)
    values, indices = torch.topk(scaled, k=min(top_k, scaled.numel()))
    probs = torch.softmax(values, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1)
    return int(indices[chosen].item())


def sample_markov(counter_map: Dict[str, int], allowed_tokens: Sequence[str] | None = None) -> str:
    keys = list(counter_map.keys())
    if allowed_tokens is not None:
        allowed = set(allowed_tokens)
        filtered = [(key, counter_map[key]) for key in keys if key in allowed]
        if filtered:
            keys = [key for key, _ in filtered]
            weights = [weight for _, weight in filtered]
            return random.choices(keys, weights=weights, k=1)[0]
    weights = [counter_map[key] for key in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def generate_with_markov(markov_model: dict, raag: str, max_new_tokens: int) -> List[str]:
    raag_model = markov_model[raag]
    if not raag_model["starts"]:
        return []
    grammar = RAAG_GRAMMAR[raag]
    allowed_starts = list(grammar["allowed_notes"]) + [SEP_TOKEN]
    current = sample_markov(raag_model["starts"], allowed_starts)
    generated = [current]
    for _ in range(max_new_tokens - 1):
        transitions = raag_model["transitions"].get(current)
        if not transitions:
            current = sample_markov(raag_model["starts"], allowed_starts)
        else:
            valid_transitions = {
                token: weight
                for token, weight in transitions.items()
                if token == SEP_TOKEN or (token in grammar["allowed_notes"] and (current, token) not in grammar["illegal_transitions"])
            }
            current = sample_markov(valid_transitions or transitions)
        generated.append(current)
    return refine_swara_sequence(generated, raag)


def generate_with_lstm(
    checkpoint_path: Path,
    prepared: dict,
    raag: str,
    device: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    grammar_bias_strength: float,
) -> List[str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConditionalLSTM(
        vocab_size=checkpoint["vocab_size"],
        num_raags=checkpoint["num_raags"],
        embedding_dim=checkpoint["config"]["embedding_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        num_layers=checkpoint["config"]["num_layers"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token_to_id = prepared["token_to_id"]
    id_to_token = {int(key): value for key, value in prepared["id_to_token"].items()}
    raag_id = prepared["raag_to_id"][raag]
    generated = [token_to_id[BOS_TOKEN]]
    generated_tokens: List[str] = []
    hidden = None
    x = torch.tensor([[token_to_id[BOS_TOKEN]]], dtype=torch.long, device=device)
    r = torch.tensor([raag_id], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, hidden = model(x, r, hidden)
            biased_logits = apply_raag_bias_to_logits(logits[0, -1], id_to_token, generated_tokens, raag, grammar_bias_strength)
            next_id = top_k_sample(biased_logits, top_k=top_k, temperature=temperature)
            generated.append(next_id)
            if next_id == token_to_id[EOS_TOKEN]:
                break
            next_token = id_to_token[next_id]
            if next_token not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                generated_tokens.append(next_token)
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return refine_swara_sequence(decode_token_ids(generated, id_to_token), raag)


def generate_with_transformer(
    checkpoint_path: Path,
    prepared: dict,
    raag: str,
    device: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    grammar_bias_strength: float,
) -> List[str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        num_raags=checkpoint["num_raags"],
        embedding_dim=checkpoint["config"]["embedding_dim"],
        num_heads=checkpoint["config"]["transformer_heads"],
        num_layers=checkpoint["config"]["transformer_layers"],
        dropout=checkpoint["config"]["dropout"],
        pad_idx=checkpoint["pad_idx"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token_to_id = prepared["token_to_id"]
    id_to_token = {int(key): value for key, value in prepared["id_to_token"].items()}
    raag_id = prepared["raag_to_id"][raag]
    generated = [token_to_id[BOS_TOKEN]]
    generated_tokens: List[str] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            r = torch.tensor([raag_id], dtype=torch.long, device=device)
            logits = model(x, r)
            biased_logits = apply_raag_bias_to_logits(logits[0, -1], id_to_token, generated_tokens, raag, grammar_bias_strength)
            next_id = top_k_sample(biased_logits, top_k=top_k, temperature=temperature)
            generated.append(next_id)
            if next_id == token_to_id[EOS_TOKEN]:
                break
            next_token = id_to_token[next_id]
            if next_token not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                generated_tokens.append(next_token)
    return refine_swara_sequence(decode_token_ids(generated, id_to_token), raag)


def ngram_set(tokens: Sequence[str], n: int) -> set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def note_sequence(tokens: Sequence[str]) -> List[str]:
    return [token for token in tokens if token in SWARA_ORDER]


def refine_swara_sequence(tokens: Sequence[str], raag: str) -> List[str]:
    grammar = RAAG_GRAMMAR[raag]
    refined: List[str] = []
    previous_note: str | None = None

    for token in tokens:
        if token == SEP_TOKEN:
            if refined and refined[-1] != SEP_TOKEN:
                refined.append(token)
            continue
        if token not in grammar["allowed_notes"]:
            continue
        if previous_note and (previous_note, token) in grammar["illegal_transitions"]:
            continue
        if refined and refined[-1] == token:
            continue
        refined.append(token)
        previous_note = token

    while refined and refined[-1] == SEP_TOKEN:
        refined.pop()

    if not refined:
        return [grammar["vadi"], grammar["samvadi"], grammar["preferred_endings"][-1]]

    if grammar["vadi"] not in refined:
        refined.insert(min(len(refined), 2), grammar["vadi"])
    if grammar["samvadi"] not in refined:
        refined.insert(min(len(refined), 4), grammar["samvadi"])
    if refined[-1] not in grammar["preferred_endings"]:
        if refined[-1] != SEP_TOKEN:
            refined.append(SEP_TOKEN)
        refined.extend([grammar["samvadi"], grammar["vadi"], grammar["preferred_endings"][-1]])

    return refined


def empirical_entropy(tokens: Sequence[str]) -> float:
    notes = note_sequence(tokens)
    if not notes:
        return 0.0
    counts = Counter(notes)
    total = len(notes)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(max(p, 1e-12))
    return entropy


def note_distribution(tokens: Sequence[str]) -> np.ndarray:
    notes = note_sequence(tokens)
    counts = np.ones(len(SWARA_ORDER), dtype=np.float64) * 1e-6
    if not notes:
        return counts / counts.sum()
    index = {note: idx for idx, note in enumerate(SWARA_ORDER)}
    for note in notes:
        counts[index[note]] += 1.0
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def novelty_score(candidate: Sequence[str], references: Sequence[Sequence[str]]) -> Tuple[float, float]:
    training_entropy = empirical_entropy([note for ref in references for note in ref])
    generated_entropy = empirical_entropy(candidate)
    training_distribution = note_distribution([note for ref in references for note in ref])
    generated_distribution = note_distribution(candidate)
    return abs(generated_entropy - training_entropy), kl_divergence(generated_distribution, training_distribution)


def grammar_violation_stats(candidate: Sequence[str], raag: str) -> Dict[str, float]:
    notes = note_sequence(candidate)
    grammar = RAAG_GRAMMAR[raag]
    if not notes:
        return {"outside_ratio": 1.0, "illegal_transition_ratio": 1.0, "vadi_missing": 1.0, "samvadi_missing": 1.0}
    outside = sum(note not in grammar["allowed_notes"] for note in notes) / len(notes)
    pairs = list(zip(notes[:-1], notes[1:]))
    illegal = sum(pair in grammar["illegal_transitions"] for pair in pairs) / max(len(pairs), 1)
    return {
        "outside_ratio": outside,
        "illegal_transition_ratio": illegal,
        "vadi_missing": 0.0 if grammar["vadi"] in notes else 1.0,
        "samvadi_missing": 0.0 if grammar["samvadi"] in notes else 1.0,
    }


def intentionality_score(candidate: Sequence[str], raag: str) -> float:
    stats = grammar_violation_stats(candidate, raag)
    score = 1.0 - (
        0.45 * stats["outside_ratio"]
        + 0.35 * stats["illegal_transition_ratio"]
        + 0.10 * stats["vadi_missing"]
        + 0.10 * stats["samvadi_missing"]
    )
    return round(100.0 * max(0.0, score), 2)


def aesthetics_score(candidate: Sequence[str]) -> float:
    notes = note_sequence(candidate)
    if len(notes) < 4:
        return 0.0
    indices = {note: idx for idx, note in enumerate(SWARA_ORDER)}
    steps = [abs(indices[right] - indices[left]) for left, right in zip(notes[:-1], notes[1:])]
    smoothness = sum(step <= 2 for step in steps) / max(len(steps), 1)
    phrasing = min(candidate.count(SEP_TOKEN) / max(len(candidate) / 24.0, 1.0), 1.0)
    return round(100.0 * (0.7 * smoothness + 0.3 * phrasing), 2)


def reflection_score(candidate: Sequence[str]) -> float:
    notes = note_sequence(candidate)
    if len(notes) < 6:
        return 0.0
    motifs = [tuple(notes[i : i + 4]) for i in range(len(notes) - 3)]
    counts = Counter(motifs)
    repeated_instances = sum(count for count in counts.values() if count > 1)
    return round(100.0 * repeated_instances / max(len(motifs), 1), 2)


def motif_recurrence_score(candidate: Sequence[str]) -> float:
    notes = note_sequence(candidate)
    motifs = [tuple(notes[i : i + 4]) for i in range(len(notes) - 3)]
    if not motifs:
        return 0.0
    counts = Counter(motifs)
    repeating = sum(count > 1 for count in counts.values())
    return round(100.0 * repeating / max(len(counts), 1), 2)


def compute_metrics(candidate: Sequence[str], references: Sequence[Sequence[str]], raag: str) -> Dict[str, float]:
    grammar = grammar_violation_stats(candidate, raag)
    entropy_diff, kl_score = novelty_score(candidate, references)
    return {
        "novelty_entropy_diff": round(entropy_diff, 4),
        "novelty_kl_divergence": round(kl_score, 4),
        "intentionality": intentionality_score(candidate, raag),
        "aesthetics": aesthetics_score(candidate),
        "reflection": reflection_score(candidate),
        "motif_recurrence": motif_recurrence_score(candidate),
        "grammar_score": round(100.0 * max(0.0, 1.0 - (grammar["outside_ratio"] + grammar["illegal_transition_ratio"] + grammar["vadi_missing"] + grammar["samvadi_missing"]) / 4.0), 2),
        "outside_ratio": round(grammar["outside_ratio"], 4),
        "illegal_transition_ratio": round(grammar["illegal_transition_ratio"], 4),
    }


def zscore_equal_weight_reports(reports: List[dict], metric_names: Sequence[str]) -> None:
    for metric in metric_names:
        values = np.array([report["metrics"][metric] for report in reports], dtype=np.float32)
        mean = float(values.mean())
        std = float(values.std()) if float(values.std()) > 1e-8 else 1.0
        for report, value in zip(reports, values):
            report["metrics"][f"{metric}_z"] = round((float(value) - mean) / std, 4)
    for report in reports:
        z_values = [report["metrics"][f"{metric}_z"] for metric in metric_names]
        report["metrics"]["overall_z"] = round(float(np.mean(z_values)), 4)


def swara_to_midi(note: str, tonic_midi: int = 60) -> int:
    semitone = {
        "S": 0,
        "r": 1,
        "R": 2,
        "g": 3,
        "G": 4,
        "m": 5,
        "M": 6,
        "P": 7,
        "d": 8,
        "D": 9,
        "n": 10,
        "N": 11,
    }
    return tonic_midi + semitone[note]


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def swara_to_hz(note: str, tonic_midi: int = 60) -> float:
    return midi_to_hz(swara_to_midi(note, tonic_midi))


def encode_varlen(value: int) -> bytes:
    buffer = value & 0x7F
    out = bytearray()
    while True:
        value >>= 7
        if value:
            buffer <<= 8
            buffer |= ((value & 0x7F) | 0x80)
        else:
            break
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def write_midi(tokens: Sequence[str], path: Path, note_ticks: int) -> None:
    events = bytearray()
    tempo = 500000
    events.extend(encode_varlen(0))
    events.extend(b"\xFF\x51\x03" + struct.pack(">I", tempo)[1:])
    delta = 0
    for token in tokens:
        if token == SEP_TOKEN:
            delta += note_ticks
            continue
        if token not in SWARA_ORDER:
            continue
        pitch = swara_to_midi(token)
        events.extend(encode_varlen(delta))
        events.extend(bytes([0x90, pitch, 80]))
        events.extend(encode_varlen(note_ticks))
        events.extend(bytes([0x80, pitch, 0]))
        delta = 0
    events.extend(encode_varlen(0))
    events.extend(b"\xFF\x2F\x00")

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
    track = b"MTrk" + struct.pack(">I", len(events)) + bytes(events)
    path.write_bytes(header + track)


def synth_lead_note(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    length = max(1, int(sample_rate * duration))
    t = np.linspace(0.0, duration, length, endpoint=False)
    attack = max(1, int(0.08 * length))
    release = max(1, int(0.18 * length))
    sustain = max(1, length - attack - release)

    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack, endpoint=False),
            np.full(sustain, 0.92, dtype=np.float32),
            np.linspace(0.92, 0.0, release, endpoint=True),
        ]
    )[:length]

    vibrato = 0.003 * np.sin(2.0 * np.pi * 5.0 * t)
    carrier = np.sin(2.0 * np.pi * freq * t * (1.0 + vibrato))
    harmonic_2 = 0.35 * np.sin(2.0 * np.pi * 2.0 * freq * t)
    harmonic_3 = 0.18 * np.sin(2.0 * np.pi * 3.0 * freq * t)
    air = 0.02 * np.sin(2.0 * np.pi * 0.7 * t)
    return (0.42 * (carrier + harmonic_2 + harmonic_3) * env + air * env).astype(np.float32)


def synth_drone(total_duration: float, sample_rate: int, tonic_midi: int, gain: float) -> np.ndarray:
    length = max(1, int(sample_rate * total_duration))
    t = np.linspace(0.0, total_duration, length, endpoint=False)
    sa = midi_to_hz(tonic_midi - 12)
    pa = midi_to_hz(tonic_midi - 5)
    upper_sa = midi_to_hz(tonic_midi)
    slow_pulse = 0.82 + 0.18 * np.sin(2.0 * np.pi * 0.22 * t)
    drone = (
        0.55 * np.sin(2.0 * np.pi * sa * t)
        + 0.28 * np.sin(2.0 * np.pi * pa * t)
        + 0.20 * np.sin(2.0 * np.pi * upper_sa * t)
    )
    return (gain * slow_pulse * drone).astype(np.float32)


def synthesize_swara_audio(
    tokens: Sequence[str],
    sample_rate: int,
    seconds_per_note: float,
    tonic_midi: int = 60,
    drone_gain: float = 0.16,
) -> np.ndarray:
    segments: List[np.ndarray] = []
    silence = np.zeros(max(1, int(sample_rate * seconds_per_note * 0.45)), dtype=np.float32)
    for token in tokens:
        if token == SEP_TOKEN:
            segments.append(silence.copy())
            continue
        if token not in SWARA_ORDER:
            continue
        freq = swara_to_hz(token, tonic_midi=tonic_midi)
        segments.append(synth_lead_note(freq, seconds_per_note, sample_rate))
    if not segments:
        return np.zeros(sample_rate, dtype=np.float32)
    lead = np.concatenate(segments)
    drone = synth_drone(len(lead) / sample_rate, sample_rate, tonic_midi=tonic_midi, gain=drone_gain)
    audio = lead + drone[: len(lead)]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        audio = 0.92 * audio / peak
    return audio.astype(np.float32)


def write_audio_wav(
    tokens: Sequence[str],
    path: Path,
    sample_rate: int,
    seconds_per_note: float,
    drone_gain: float,
) -> None:
    audio = synthesize_swara_audio(
        tokens,
        sample_rate=sample_rate,
        seconds_per_note=seconds_per_note,
        tonic_midi=60,
        drone_gain=drone_gain,
    )
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def split_phrases(tokens: Sequence[str]) -> List[List[str]]:
    phrases: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token == SEP_TOKEN:
            if current:
                phrases.append(current)
                current = []
            continue
        if token in SWARA_ORDER:
            current.append(token)
    if current:
        phrases.append(current)
    return phrases


def write_swara_file(path: Path, raag: str, model: str, scope: str, tokens: Sequence[str], metrics: Dict[str, float]) -> None:
    phrases = split_phrases(tokens)
    lines = [
        f"# Raga: {raag}",
        f"# Model: {model}",
        f"# Scope: {scope}",
        f"# Grammar score: {metrics['grammar_score']}",
        f"# Intentionality: {metrics['intentionality']}",
        "",
        "Full swara sequence:",
        " ".join(tokens),
        "",
        "Phrase-wise notation:",
    ]
    for index, phrase in enumerate(phrases, start=1):
        lines.append(f"Phrase {index}: {' '.join(phrase)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_swara_json(path: Path, raag: str, model: str, scope: str, tokens: Sequence[str], metrics: Dict[str, float]) -> None:
    payload = {
        "raag": raag,
        "model": model,
        "scope": scope,
        "tokens": list(tokens),
        "phrases": split_phrases(tokens),
        "metrics": metrics,
    }
    save_json(path, payload)


def plot_creativity_radar(path: Path, metrics: Dict[str, float], label: str) -> None:
    if plt is None:
        return
    categories = ["intentionality", "aesthetics", "reflection", "motif_recurrence", "grammar_score"]
    values = [metrics[name] for name in categories]
    angles = np.linspace(0.0, 2.0 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title(f"Creativity Radar\n{label}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pitch_distribution(path: Path, generated: Sequence[str], references: Sequence[Sequence[str]], label: str) -> None:
    if plt is None:
        return
    generated_dist = note_distribution(generated)
    training_dist = note_distribution([note for ref in references for note in ref])
    x = np.arange(len(SWARA_ORDER))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    ax.bar(x - width / 2, training_dist, width=width, label="training")
    ax.bar(x + width / 2, generated_dist, width=width, label="generated")
    ax.set_xticks(x)
    ax.set_xticklabels(SWARA_ORDER)
    ax.set_ylabel("Probability")
    ax.set_title(f"Training vs Generated Pitch Distribution\n{label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def phrase_similarity_heatmap(path: Path, generated: Sequence[str], references: Sequence[Sequence[str]], label: str) -> None:
    if plt is None:
        return
    generated_phrases = split_phrases(generated)
    reference_phrases: List[List[str]] = []
    for reference in references:
        reference_phrases.extend(split_phrases(reference))
    generated_phrases = generated_phrases[:8]
    reference_phrases = reference_phrases[:8]
    if not generated_phrases or not reference_phrases:
        return

    matrix = np.zeros((len(generated_phrases), len(reference_phrases)), dtype=np.float32)
    for i, generated_phrase in enumerate(generated_phrases):
        g_grams = ngram_set(generated_phrase, 3)
        for j, reference_phrase in enumerate(reference_phrases):
            r_grams = ngram_set(reference_phrase, 3)
            union = len(g_grams | r_grams)
            matrix[i, j] = 0.0 if union == 0 else len(g_grams & r_grams) / union

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Training Phrase Index")
    ax.set_ylabel("Generated Phrase Index")
    ax.set_title(f"Structure Similarity Heatmap\n{label}")
    fig.colorbar(image, ax=ax, label="3-gram Jaccard")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_bundle(
    prepared: dict,
    bundle_models_dir: Path,
    output_dir: Path,
    scope: str,
    config: GenerateConfig,
) -> List[dict]:
    id_to_token = {int(key): value for key, value in prepared["id_to_token"].items()}
    references = {
        raag: [
            decode_token_ids(track["token_ids"], id_to_token)
            for track in prepared["tracks"]
            if track["raag"] == raag
        ]
        for raag in TARGET_RAAGS
    }
    markov_model_path = resolve_training_model_path(bundle_models_dir, "markov")
    lstm_model_path = resolve_training_model_path(bundle_models_dir, "lstm")
    transformer_model_path = resolve_training_model_path(bundle_models_dir, "music_transformer")
    markov_model = json.loads(markov_model_path.read_text(encoding="utf-8"))
    bundle_reports: List[dict] = []
    for raag in TARGET_RAAGS:
        if not references[raag]:
            continue
        for model_name in ("markov", "lstm", "music_transformer"):
            model_output_dir = get_generated_model_dir(output_dir, scope, raag, model_name)
            model_output_dir.mkdir(parents=True, exist_ok=True)
            file_prefix = get_generated_file_prefix(scope, raag, model_name)
            if model_name == "markov":
                tokens = generate_with_markov(markov_model, raag, config.max_new_tokens)
            elif model_name == "lstm":
                tokens = generate_with_lstm(
                    lstm_model_path,
                    prepared,
                    raag,
                    config.device,
                    config.max_new_tokens,
                    config.top_k,
                    config.temperature,
                    config.grammar_bias_strength,
                )
            else:
                tokens = generate_with_transformer(
                    transformer_model_path,
                    prepared,
                    raag,
                    config.device,
                    config.max_new_tokens,
                    config.top_k,
                    config.temperature,
                    config.grammar_bias_strength,
                )
            swara_path = model_output_dir / f"{file_prefix}.swara.txt"
            swara_json_path = model_output_dir / f"{file_prefix}.swara.json"
            metrics = compute_metrics(tokens, references[raag], raag)
            write_swara_file(swara_path, raag, model_name, scope, tokens, metrics)
            write_swara_json(swara_json_path, raag, model_name, scope, tokens, metrics)
            midi_path = None
            audio_path = None
            radar_plot_path = None
            distribution_plot_path = None
            heatmap_plot_path = None
            if config.write_midi:
                midi_path = model_output_dir / f"{file_prefix}.mid"
                write_midi(tokens, midi_path, config.note_ticks)
            if config.write_audio:
                audio_path = model_output_dir / f"{file_prefix}.wav"
                write_audio_wav(
                    tokens,
                    audio_path,
                    sample_rate=config.audio_sample_rate,
                    seconds_per_note=config.seconds_per_note,
                    drone_gain=config.drone_gain,
                )
            if config.write_plots and plt is not None:
                label = f"{scope} | {raag} | {model_name}"
                radar_plot_path = model_output_dir / f"{file_prefix}_radar.png"
                distribution_plot_path = model_output_dir / f"{file_prefix}_pitch_distribution.png"
                heatmap_plot_path = model_output_dir / f"{file_prefix}_structure_heatmap.png"
                plot_creativity_radar(radar_plot_path, metrics, label)
                plot_pitch_distribution(distribution_plot_path, tokens, references[raag], label)
                phrase_similarity_heatmap(heatmap_plot_path, tokens, references[raag], label)
            bundle_reports.append(
                {
                    "scope": scope,
                    "raag": raag,
                    "model": model_name,
                    "artifact_dir": str(model_output_dir),
                    "swara_path": str(swara_path),
                    "swara_json_path": str(swara_json_path),
                    "midi_path": str(midi_path) if midi_path is not None else None,
                    "audio_path": str(audio_path) if audio_path is not None else None,
                    "radar_plot_path": str(radar_plot_path) if radar_plot_path is not None else None,
                    "pitch_distribution_plot_path": str(distribution_plot_path) if distribution_plot_path is not None else None,
                    "structure_heatmap_plot_path": str(heatmap_plot_path) if heatmap_plot_path is not None else None,
                    "tokens": tokens,
                    "metrics": metrics,
                }
            )
    return bundle_reports


def generate_all(config: GenerateConfig) -> Dict[str, str]:
    set_seed(config.seed)
    prepared = load_prepared(config.prepared_dir)
    models_dir = resolve_models_dir(config.models_dir)
    output_dir = create_run_output_dir(config.output_dir)
    outputs: Dict[str, str] = {"output_dir": str(output_dir), "models_dir": str(models_dir)}
    reports: List[dict] = []

    global_dir = resolve_global_bundle_dir(models_dir)
    if global_dir is not None:
        reports.extend(generate_bundle(prepared, global_dir, output_dir, "global", config))

    per_raag_root = models_dir / "per_raag"
    if per_raag_root.exists():
        for raag in TARGET_RAAGS:
            bundle_dir = per_raag_root / raag.lower()
            if not bundle_dir.exists():
                continue
            subset = filter_prepared_by_raag(prepared, raag)
            reports.extend(generate_bundle(subset, bundle_dir, output_dir, "per_raag", config))

    if not reports:
        raise RuntimeError("No trained model bundles were found under the models directory.")

    zscore_equal_weight_reports(
        reports,
        metric_names=("novelty_entropy_diff", "novelty_kl_divergence", "intentionality", "aesthetics", "reflection", "motif_recurrence", "grammar_score"),
    )

    for report in reports:
        key = f"{report['scope']}_{report['raag']}_{report['model']}"
        outputs[key] = report["swara_path"]
        if report["audio_path"] is not None:
            outputs[f"{key}_audio"] = report["audio_path"]
        if report["midi_path"] is not None:
            outputs[f"{key}_midi"] = report["midi_path"]
        if report["radar_plot_path"] is not None:
            outputs[f"{key}_radar"] = report["radar_plot_path"]
        if report["pitch_distribution_plot_path"] is not None:
            outputs[f"{key}_pitch_dist"] = report["pitch_distribution_plot_path"]
        if report["structure_heatmap_plot_path"] is not None:
            outputs[f"{key}_structure_heatmap"] = report["structure_heatmap_plot_path"]
    report_path = output_dir / "generation_report.json"
    save_json(report_path, reports)
    outputs["report"] = str(report_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, train, and generate Hindustani melody models for Saraga.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--data-source", choices=["filesystem", "mirdata"], default="filesystem")
    prepare.add_argument("--mirdata-home")
    prepare.add_argument("--mirdata-download", action="store_true")
    prepare.add_argument("--sequence-length", type=int, default=64)
    prepare.add_argument("--hop-length", type=int, default=16)
    prepare.add_argument("--min-phrase-tokens", type=int, default=4)
    prepare.add_argument("--train-split", type=float, default=0.8)
    prepare.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train-all")
    train.add_argument("--prepared-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--transformer-heads", type=int, default=4)
    train.add_argument("--transformer-layers", type=int, default=4)
    train.add_argument("--grad-accum-steps", type=int, default=2)
    train.add_argument("--early-stopping-patience", type=int, default=5)
    train.add_argument("--train-scope", choices=["global", "per_raag", "both"], default="both")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    generate = sub.add_parser("generate-all")
    generate.add_argument("--prepared-dir", required=True)
    generate.add_argument("--models-dir", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=192)
    generate.add_argument("--temperature", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=8)
    generate.add_argument("--note-ticks", type=int, default=480)
    generate.add_argument("--write-midi", action="store_true")
    generate.add_argument("--write-audio", action="store_true")
    generate.add_argument("--write-plots", action=argparse.BooleanOptionalAction, default=True)
    generate.add_argument("--audio-sample-rate", type=int, default=22050)
    generate.add_argument("--seconds-per-note", type=float, default=0.32)
    generate.add_argument("--drone-gain", type=float, default=0.16)
    generate.add_argument("--grammar-bias-strength", type=float, default=3.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_dataset(
            DataConfig(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                data_source=args.data_source,
                mirdata_home=args.mirdata_home,
                mirdata_download=args.mirdata_download,
                sequence_length=args.sequence_length,
                hop_length=args.hop_length,
                min_phrase_tokens=args.min_phrase_tokens,
                train_split=args.train_split,
                seed=args.seed,
            )
        )
    elif args.command == "train-all":
        result = train_all_models(
            TrainConfig(
                prepared_dir=args.prepared_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                transformer_heads=args.transformer_heads,
                transformer_layers=args.transformer_layers,
                grad_accum_steps=args.grad_accum_steps,
                early_stopping_patience=args.early_stopping_patience,
                train_scope=args.train_scope,
                seed=args.seed,
                device=args.device,
            )
        )
    else:
        result = generate_all(
            GenerateConfig(
                prepared_dir=args.prepared_dir,
                models_dir=args.models_dir,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                note_ticks=args.note_ticks,
                write_midi=args.write_midi,
                write_audio=args.write_audio,
                write_plots=args.write_plots,
                audio_sample_rate=args.audio_sample_rate,
                seconds_per_note=args.seconds_per_note,
                drone_gain=args.drone_gain,
                grammar_bias_strength=args.grammar_bias_strength,
                seed=args.seed,
                device=args.device,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
