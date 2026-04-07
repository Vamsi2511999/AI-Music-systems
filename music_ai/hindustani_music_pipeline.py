from __future__ import annotations

import argparse
import json
import math
import random
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGET_RAAGS = ("Bageshree", "Khamaj", "Bhoop")
SWARA_ORDER = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
PHRASE_TOKEN_PATTERN = re.compile(r"[SrRgGmMPdDnN]")
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<SEP>"]
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"


@dataclass
class DataConfig:
    dataset_root: str
    output_dir: str
    target_raags: Tuple[str, ...] = TARGET_RAAGS
    sequence_length: int = 64
    hop_length: int = 16
    min_phrase_tokens: int = 4
    train_split: float = 0.8
    random_seed: int = 42


@dataclass
class TrainConfig:
    prepared_dir: str
    output_dir: str
    epochs: int = 35
    batch_size: int = 32
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 3e-4
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerationConfig:
    prepared_dir: str
    checkpoint: str
    output_dir: str
    raag: str
    max_new_tokens: int = 192
    temperature: float = 0.95
    top_k: int = 8
    sample_count: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    render_seconds_per_note: float = 0.32
    sample_rate: int = 22050


@dataclass
class RaagProfile:
    allowed_notes: Tuple[str, ...]
    preferred_endings: Tuple[str, ...]
    pakad_motifs: Tuple[Tuple[str, ...], ...]
    vadi: str
    samvadi: str


RAAG_PROFILES: Dict[str, RaagProfile] = {
    "Bageshree": RaagProfile(
        allowed_notes=("S", "R", "g", "M", "P", "D", "n"),
        preferred_endings=("M", "g", "S"),
        pakad_motifs=(("g", "M", "D", "n"), ("n", "D", "M", "g"), ("M", "g", "R", "S")),
        vadi="M",
        samvadi="S",
    ),
    "Khamaj": RaagProfile(
        allowed_notes=("S", "R", "G", "M", "P", "D", "n", "N"),
        preferred_endings=("G", "n", "S"),
        pakad_motifs=(("G", "M", "P", "D"), ("n", "D", "P", "M"), ("G", "R", "S")),
        vadi="G",
        samvadi="N",
    ),
    "Bhoop": RaagProfile(
        allowed_notes=("S", "R", "G", "P", "D"),
        preferred_endings=("G", "R", "S"),
        pakad_motifs=(("S", "R", "G"), ("G", "P", "D"), ("D", "P", "G", "R")),
        vadi="G",
        samvadi="D",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_raag_name(name: str) -> str:
    text = name.strip().lower().replace("-", "").replace(" ", "")
    aliases = {
        "bageshri": "Bageshree",
        "bageshree": "Bageshree",
        "khamaj": "Khamaj",
        "bhoop": "Bhoop",
        "bhup": "Bhoop",
        "bhupali": "Bhoop",
        "bhoopali": "Bhoop",
    }
    return aliases.get(text, name.strip())


def safe_read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def read_tabular_lines(path: Path, delimiter: str = "\t") -> List[List[str]]:
    rows: List[List[str]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append([token.strip() for token in line.split(delimiter)])
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
        stem = strip_known_suffixes(path.name)
        key = (path.parent, stem)
        record = groups.setdefault(key, {})
        lower_name = path.name.lower()
        if lower_name.endswith(".json"):
            record["metadata"] = path
        elif ".phrases" in lower_name and lower_name.endswith(".txt"):
            record["phrases"] = path
        elif ".pitch" in lower_name and lower_name.endswith(".txt"):
            record["pitch"] = path
        elif ".ctonic" in lower_name and lower_name.endswith(".txt"):
            record["ctonic"] = path
        elif lower_name.endswith((".mp3", ".wav", ".flac")):
            record.setdefault("audio", path)
    return groups


def extract_raags(metadata: dict) -> List[str]:
    raags = []
    for item in metadata.get("raags", []):
        if isinstance(item, dict):
            candidate = item.get("name") or item.get("title") or item.get("raag")
            if candidate:
                raags.append(normalize_raag_name(str(candidate)))
        elif isinstance(item, str):
            raags.append(normalize_raag_name(item))
    return raags


def extract_phrase_tokens(phrase_path: Path) -> List[str]:
    rows = read_tabular_lines(phrase_path, delimiter="\t")
    phrase_sequences: List[List[str]] = []
    for row in rows:
        if len(row) < 4:
            continue
        phrase_text = row[3]
        notes = PHRASE_TOKEN_PATTERN.findall(phrase_text)
        if len(notes) >= 2:
            phrase_sequences.append(notes)
    tokens: List[str] = []
    for phrase in phrase_sequences:
        tokens.extend(phrase)
        tokens.append(SEP_TOKEN)
    return tokens[:-1] if tokens and tokens[-1] == SEP_TOKEN else tokens


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
    return min(cent_map, key=lambda swara: min(abs(normalized - cent_map[swara]), 1200.0 - abs(normalized - cent_map[swara])))


def extract_pitch_tokens(pitch_path: Path, tonic_path: Path) -> List[str]:
    tonic_rows = read_tabular_lines(tonic_path, delimiter="\t")
    if not tonic_rows or not tonic_rows[0]:
        return []
    tonic_hz = float(tonic_rows[0][0])
    pitch_rows = read_tabular_lines(pitch_path, delimiter="\t")
    freqs: List[float] = []
    for row in pitch_rows:
        if len(row) < 2:
            continue
        freq = float(row[1])
        if freq > 0.0:
            freqs.append(freq)
        else:
            freqs.append(0.0)
    if not freqs:
        return []

    tokens: List[str] = []
    previous = None
    rest_run = 0
    for freq in freqs[::5]:
        if freq == 0.0:
            rest_run += 1
            if rest_run >= 3 and (not tokens or tokens[-1] != SEP_TOKEN):
                tokens.append(SEP_TOKEN)
            continue
        rest_run = 0
        swara = nearest_swara(hz_to_cents(freq, tonic_hz))
        if swara != previous:
            tokens.append(swara)
            previous = swara
    return tokens


def load_track_tokens(group: Dict[str, Path]) -> List[str]:
    if "phrases" in group:
        tokens = extract_phrase_tokens(group["phrases"])
        if tokens:
            return tokens
    if "pitch" in group and "ctonic" in group:
        return extract_pitch_tokens(group["pitch"], group["ctonic"])
    return []


def sliding_windows(tokens: Sequence[int], seq_len: int, hop_len: int) -> List[List[int]]:
    if len(tokens) < seq_len + 1:
        return []
    return [list(tokens[start : start + seq_len + 1]) for start in range(0, len(tokens) - seq_len, hop_len)]


def build_vocabulary(sequences: Iterable[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab_tokens = list(SPECIAL_TOKENS)
    seen = set(vocab_tokens)
    for sequence in sequences:
        for token in sequence:
            if token not in seen:
                vocab_tokens.append(token)
                seen.add(token)
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_sequences(sequences: Iterable[List[str]], token_to_id: Dict[str, int]) -> List[List[int]]:
    return [[token_to_id[BOS_TOKEN], *[token_to_id[token] for token in sequence], token_to_id[EOS_TOKEN]] for sequence in sequences]


def prepare_dataset(config: DataConfig) -> Dict[str, Path]:
    set_seed(config.random_seed)
    dataset_root = Path(config.dataset_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = infer_track_groups(dataset_root)
    target_set = {normalize_raag_name(name) for name in config.target_raags}

    track_records = []
    for (_, track_name), group in groups.items():
        metadata_path = group.get("metadata")
        if metadata_path is None:
            continue
        metadata = safe_read_json(metadata_path)
        if not metadata:
            continue
        raags = extract_raags(metadata)
        matched_raags = sorted(target_set.intersection(raags))
        if not matched_raags:
            continue

        tokens = load_track_tokens(group)
        tokens = [token for token in tokens if token in SWARA_ORDER or token == SEP_TOKEN]
        if len(tokens) < config.min_phrase_tokens:
            continue

        for raag in matched_raags:
            track_records.append(
                {
                    "track": track_name,
                    "raag": raag,
                    "tokens": tokens,
                    "source_files": {key: str(value) for key, value in group.items()},
                }
            )

    if not track_records:
        raise RuntimeError("No matching tracks found. Check the dataset root and raag names.")

    track_records.sort(key=lambda item: (item["raag"], item["track"]))
    random.shuffle(track_records)

    sequences = [record["tokens"] for record in track_records]
    token_to_id, id_to_token = build_vocabulary(sequences)
    raag_to_id = {raag: idx for idx, raag in enumerate(sorted({record["raag"] for record in track_records}))}

    encoded_tracks = []
    for record in track_records:
        encoded = encode_sequences([record["tokens"]], token_to_id)[0]
        windows = sliding_windows(encoded, config.sequence_length, config.hop_length)
        if not windows:
            continue
        encoded_tracks.append(
            {
                "track": record["track"],
                "raag": record["raag"],
                "raag_id": raag_to_id[record["raag"]],
                "token_ids": encoded,
                "windows": windows,
                "source_files": record["source_files"],
            }
        )

    if not encoded_tracks:
        raise RuntimeError("Tracks were found, but none were long enough for training windows.")

    split_index = max(1, int(len(encoded_tracks) * config.train_split))
    split_index = min(split_index, len(encoded_tracks) - 1) if len(encoded_tracks) > 1 else len(encoded_tracks)
    train_tracks = encoded_tracks[:split_index]
    val_tracks = encoded_tracks[split_index:] if split_index < len(encoded_tracks) else encoded_tracks[-1:]

    train_examples = [
        {"raag_id": track["raag_id"], "tokens": window}
        for track in train_tracks
        for window in track["windows"]
    ]
    val_examples = [
        {"raag_id": track["raag_id"], "tokens": window}
        for track in val_tracks
        for window in track["windows"]
    ]

    payload = {
        "config": asdict(config),
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "raag_to_id": raag_to_id,
        "id_to_raag": {str(v): k for k, v in raag_to_id.items()},
        "train_examples": train_examples,
        "val_examples": val_examples,
        "tracks": encoded_tracks,
        "raag_profiles": {name: asdict(profile) for name, profile in RAAG_PROFILES.items()},
    }

    prepared_path = output_dir / "prepared_dataset.json"
    with prepared_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "prepared_path": str(prepared_path),
        "num_tracks": len(encoded_tracks),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "raags": {raag: sum(track["raag"] == raag for track in encoded_tracks) for raag in raag_to_id},
    }
    summary_path = output_dir / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {"prepared_path": prepared_path, "summary_path": summary_path}


class MelodyWindowDataset(Dataset):
    def __init__(self, examples: List[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.examples[index]
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        inputs = tokens[:-1]
        targets = tokens[1:]
        raag_ids = torch.tensor(item["raag_id"], dtype=torch.long)
        return inputs, targets, raag_ids


class ConditionalLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_raags: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.raag_embedding = nn.Embedding(num_raags, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
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
        outputs, hidden = self.lstm(x, hidden)
        logits = self.output(self.dropout(outputs))
        return logits, hidden


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets, raag_ids = zip(*batch)
    return torch.stack(inputs), torch.stack(targets), torch.stack(raag_ids)


def load_prepared(prepared_dir: str) -> dict:
    prepared_path = Path(prepared_dir) / "prepared_dataset.json"
    with prepared_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_epoch(
    model: ConditionalLSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    pad_idx: int,
) -> float:
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    for inputs, targets, raag_ids in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        raag_ids = raag_ids.to(device)
        logits, _ = model(inputs, raag_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(config: TrainConfig) -> Dict[str, Path]:
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_prepared(config.prepared_dir)
    token_to_id = prepared["token_to_id"]
    train_dataset = MelodyWindowDataset(prepared["train_examples"])
    val_dataset = MelodyWindowDataset(prepared["val_examples"])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = ConditionalLSTM(
        vocab_size=len(token_to_id),
        num_raags=len(prepared["raag_to_id"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_val = float("inf")
    checkpoint_path = output_dir / "hindustani_lstm.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, config.device, token_to_id[PAD_TOKEN])
        val_loss = run_epoch(model, val_loader, None, config.device, token_to_id[PAD_TOKEN])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "train_config": asdict(config),
                    "vocab_size": len(token_to_id),
                    "num_raags": len(prepared["raag_to_id"]),
                },
                checkpoint_path,
            )

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {"checkpoint_path": checkpoint_path, "history_path": history_path}


def top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        picked = indices[torch.multinomial(probs, num_samples=1)]
        return int(picked.item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_tokens(token_ids: Sequence[int], id_to_token: Dict[int, str]) -> List[str]:
    tokens = [id_to_token[idx] for idx in token_ids]
    cleaned: List[str] = []
    for token in tokens:
        if token in (BOS_TOKEN, PAD_TOKEN):
            continue
        if token == EOS_TOKEN:
            break
        cleaned.append(token)
    return cleaned


def ngram_set(tokens: Sequence[str], n: int) -> set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def longest_common_ngram(tokens_a: Sequence[str], tokens_b: Sequence[str], n: int = 4) -> float:
    set_a = ngram_set(tokens_a, n)
    set_b = ngram_set(tokens_b, n)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / max(len(union), 1)


def swara_only(tokens: Sequence[str]) -> List[str]:
    return [token for token in tokens if token in SWARA_ORDER]


def interval_series(tokens: Sequence[str]) -> List[int]:
    note_index = {note: idx for idx, note in enumerate(SWARA_ORDER)}
    notes = swara_only(tokens)
    return [note_index[notes[i + 1]] - note_index[notes[i]] for i in range(len(notes) - 1)]


def score_novelty(candidate: Sequence[str], reference_sequences: Sequence[Sequence[str]]) -> float:
    candidate_3grams = ngram_set(candidate, 3)
    union_ref_3grams = set()
    max_similarity = 0.0
    for reference in reference_sequences:
        union_ref_3grams |= ngram_set(reference, 3)
        max_similarity = max(max_similarity, longest_common_ngram(candidate, reference, n=4))
    new_3gram_ratio = len(candidate_3grams - union_ref_3grams) / max(len(candidate_3grams), 1)
    score = 100.0 * max(0.0, min(1.0, 0.55 * new_3gram_ratio + 0.45 * (1.0 - max_similarity)))
    return score


def score_intentionality(candidate: Sequence[str], raag: str) -> float:
    profile = RAAG_PROFILES[raag]
    notes = swara_only(candidate)
    if not notes:
        return 0.0
    allowed_ratio = sum(note in profile.allowed_notes for note in notes) / len(notes)
    motif_hits = sum(1 for motif in profile.pakad_motifs if " ".join(motif) in " ".join(notes))
    motif_score = motif_hits / max(len(profile.pakad_motifs), 1)
    ending_score = 1.0 if notes[-1] in profile.preferred_endings else 0.0
    vadi_score = min(notes.count(profile.vadi) / max(len(notes) * 0.12, 1), 1.0)
    return 100.0 * (0.45 * allowed_ratio + 0.25 * motif_score + 0.15 * ending_score + 0.15 * vadi_score)


def score_aesthetics(candidate: Sequence[str]) -> float:
    notes = swara_only(candidate)
    if len(notes) < 4:
        return 0.0
    intervals = interval_series(notes)
    stepwise_ratio = sum(abs(interval) <= 2 for interval in intervals) / max(len(intervals), 1)
    large_leap_penalty = sum(abs(interval) > 5 for interval in intervals) / max(len(intervals), 1)
    repetition_ratio = len(ngram_set(notes, 3)) / max(len(notes) - 2, 1)
    phrase_breaks = candidate.count(SEP_TOKEN)
    phrase_balance = min(phrase_breaks / max(len(candidate) / 24.0, 1.0), 1.0)
    score = 100.0 * (
        0.35 * stepwise_ratio
        + 0.25 * (1.0 - large_leap_penalty)
        + 0.20 * min(repetition_ratio * 1.5, 1.0)
        + 0.20 * phrase_balance
    )
    return score


def score_reflection(candidate: Sequence[str], reference_sequences: Sequence[Sequence[str]], raag: str) -> float:
    notes = swara_only(candidate)
    if len(notes) < 6:
        return 0.0
    internal_reuse = len(ngram_set(notes, 4)) / max(len(notes) - 3, 1)
    novelty_anchor = score_novelty(candidate, reference_sequences) / 100.0
    intentional_anchor = score_intentionality(candidate, raag) / 100.0
    balance = 1.0 - abs(internal_reuse - 0.45)
    return 100.0 * max(0.0, min(1.0, 0.30 * internal_reuse + 0.35 * novelty_anchor + 0.35 * intentional_anchor + 0.20 * balance - 0.20))


def creativity_report(candidate: Sequence[str], reference_sequences: Sequence[Sequence[str]], raag: str) -> dict:
    novelty = score_novelty(candidate, reference_sequences)
    intentionality = score_intentionality(candidate, raag)
    aesthetics = score_aesthetics(candidate)
    reflection = score_reflection(candidate, reference_sequences, raag)
    overall = 0.30 * novelty + 0.30 * intentionality + 0.25 * aesthetics + 0.15 * reflection
    return {
        "novelty": round(novelty, 2),
        "intentionality": round(intentionality, 2),
        "aesthetics": round(aesthetics, 2),
        "reflection": round(reflection, 2),
        "overall": round(overall, 2),
    }


def generate_candidate(
    model: ConditionalLSTM,
    raag_id: int,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    device: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
) -> List[str]:
    model.eval()
    generated = [token_to_id[BOS_TOKEN]]
    hidden = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            current = torch.tensor([generated[-1:]], dtype=torch.long, device=device)
            raag_tensor = torch.tensor([raag_id], dtype=torch.long, device=device)
            logits, hidden = model(current, raag_tensor, hidden)
            next_token = top_k_sample(logits[0, -1], top_k=top_k, temperature=temperature)
            generated.append(next_token)
            if next_token == token_to_id[EOS_TOKEN]:
                break
    return decode_tokens(generated, id_to_token)


def swara_to_frequency(note: str, tonic_hz: float = 261.63) -> float:
    semitone_map = {
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
    return tonic_hz * (2.0 ** (semitone_map[note] / 12.0))


def synthesize_wav(tokens: Sequence[str], output_path: Path, seconds_per_note: float, sample_rate: int) -> None:
    audio: List[np.ndarray] = []
    for token in tokens:
        if token == SEP_TOKEN:
            silence = np.zeros(int(sample_rate * seconds_per_note * 0.5), dtype=np.float32)
            audio.append(silence)
            continue
        if token not in SWARA_ORDER:
            continue
        freq = swara_to_frequency(token)
        duration = int(sample_rate * seconds_per_note)
        t = np.linspace(0, seconds_per_note, duration, endpoint=False)
        envelope = np.minimum(1.0, np.linspace(0.0, 1.0, duration)) * np.minimum(1.0, np.linspace(1.0, 0.8, duration))
        wave_data = 0.2 * np.sin(2.0 * np.pi * freq * t) * envelope
        audio.append(wave_data.astype(np.float32))
    if not audio:
        raise RuntimeError("Generated sequence was empty after decoding.")
    signal = np.concatenate(audio)
    pcm = np.int16(np.clip(signal, -1.0, 1.0) * 32767)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def generate_music(config: GenerationConfig) -> Dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared(config.prepared_dir)

    raag = normalize_raag_name(config.raag)
    if raag not in prepared["raag_to_id"]:
        raise ValueError(f"Raag '{config.raag}' not found in prepared dataset.")

    id_to_token = {int(key): value for key, value in prepared["id_to_token"].items()}
    token_to_id = prepared["token_to_id"]
    checkpoint = torch.load(config.checkpoint, map_location=config.device)
    model = ConditionalLSTM(
        vocab_size=checkpoint["vocab_size"],
        num_raags=checkpoint["num_raags"],
        embedding_dim=checkpoint["train_config"]["embedding_dim"],
        hidden_dim=checkpoint["train_config"]["hidden_dim"],
        num_layers=checkpoint["train_config"]["num_layers"],
        dropout=checkpoint["train_config"]["dropout"],
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state"])

    reference_sequences = [
        decode_tokens(track["token_ids"], id_to_token)
        for track in prepared["tracks"]
        if track["raag"] == raag
    ]
    raag_id = prepared["raag_to_id"][raag]

    candidates = []
    for _ in range(config.sample_count):
        candidate = generate_candidate(
            model=model,
            raag_id=raag_id,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            top_k=config.top_k,
            temperature=config.temperature,
        )
        report = creativity_report(candidate, reference_sequences, raag)
        candidates.append({"tokens": candidate, "report": report})

    best = max(candidates, key=lambda item: item["report"]["overall"])
    wav_path = output_dir / f"{raag.lower()}_generated.wav"
    synthesize_wav(best["tokens"], wav_path, config.render_seconds_per_note, config.sample_rate)

    report_path = output_dir / f"{raag.lower()}_creativity_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "raag": raag,
                "generation_config": asdict(config),
                "best_candidate": best,
                "all_candidates": candidates,
            },
            handle,
            indent=2,
        )

    sequence_path = output_dir / f"{raag.lower()}_generated_tokens.txt"
    with sequence_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(best["tokens"]))

    return {"wav_path": wav_path, "report_path": report_path, "sequence_path": sequence_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a Hindustani classical melody generator on Saraga Hindustani.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Preprocess Saraga Hindustani into model-ready windows.")
    prepare_parser.add_argument("--dataset-root", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--sequence-length", type=int, default=64)
    prepare_parser.add_argument("--hop-length", type=int, default=16)
    prepare_parser.add_argument("--min-phrase-tokens", type=int, default=4)
    prepare_parser.add_argument("--train-split", type=float, default=0.8)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Train the conditional LSTM generator.")
    train_parser.add_argument("--prepared-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=35)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--embedding-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    generate_parser = subparsers.add_parser("generate", help="Generate and score a melody for one raag.")
    generate_parser.add_argument("--prepared-dir", required=True)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.add_argument("--raag", required=True, choices=list(TARGET_RAAGS))
    generate_parser.add_argument("--max-new-tokens", type=int, default=192)
    generate_parser.add_argument("--temperature", type=float, default=0.95)
    generate_parser.add_argument("--top-k", type=int, default=8)
    generate_parser.add_argument("--sample-count", type=int, default=5)
    generate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate_parser.add_argument("--render-seconds-per-note", type=float, default=0.32)
    generate_parser.add_argument("--sample-rate", type=int, default=22050)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_dataset(
            DataConfig(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                sequence_length=args.sequence_length,
                hop_length=args.hop_length,
                min_phrase_tokens=args.min_phrase_tokens,
                train_split=args.train_split,
                random_seed=args.seed,
            )
        )
    elif args.command == "train":
        result = train_model(
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
                random_seed=args.seed,
                device=args.device,
            )
        )
    else:
        result = generate_music(
            GenerationConfig(
                prepared_dir=args.prepared_dir,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                raag=args.raag,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                sample_count=args.sample_count,
                device=args.device,
                render_seconds_per_note=args.render_seconds_per_note,
                sample_rate=args.sample_rate,
            )
        )

    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
