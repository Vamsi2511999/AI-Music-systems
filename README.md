# Hindustani Multi-Model Music Generation

This project prepares Saraga Hindustani data and generates raaga-focused swara notation for:

- Bageshree
- Khamaj
- Bhoop

The main script is `hindustani_multi_model_pipeline.py` and supports:

- Markov, LSTM, and Music Transformer models
- global and per-raaga training
- swara-based output files as the primary final artifact
- optional MIDI sidecar export
- optional WAV audio artifacts for human evaluation
- evaluation plots for model comparison
- filesystem-based dataset scanning
- mirdata-based dataset initialization and download
- automatic `iteration-N` run folders under each artifact output root

The repo also includes `musicgen_baseline_comparison.py` for prompt-based Hugging Face MusicGen generation, per-raaga evaluation artifacts, and aggregate comparison plots against the trained baseline models.

## Install

```bash
pip install -r requirements.txt
```

The MusicGen workflow uses Hugging Face `transformers`, `sentencepiece`, and `scipy` in addition to the baseline pipeline dependencies.

## Prepare Using mirdata

```bash
python hindustani_multi_model_pipeline.py prepare \
  --dataset-root ./data/saraga_hindustani \
  --mirdata-home ./data/saraga_hindustani \
  --data-source mirdata \
  --mirdata-download \
  --output-dir artifacts/prepared
```

Use `--mirdata-download` on the first run so `mirdata` can fetch the Saraga dataset and its index.
If the dataset is already present locally, the pipeline will still try to fetch a missing `mirdata` index automatically when needed.
Do not keep `/path/...` literally from an example command. Use a real writable directory on your machine.

## Prepare Using Filesystem Scan

```bash
python hindustani_multi_model_pipeline.py prepare \
  --dataset-root /path/to/saraga_hindustani \
  --data-source filesystem \
  --output-dir artifacts/prepared
```

Each run creates a fresh folder such as `artifacts/prepared/iteration-1`, `artifacts/prepared/iteration-2`, and so on.

## Train Models

```bash
python hindustani_multi_model_pipeline.py train-all \
  --prepared-dir artifacts/prepared \
  --output-dir artifacts/models \
  --train-scope both \
  --epochs 30 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --early-stopping-patience 5
```

If `--prepared-dir` points at the root (`artifacts/prepared`), the script automatically uses the latest prepared iteration.
Training outputs go to a new folder such as `artifacts/models/iteration-1`.

Training scaffold:

```text
artifacts/models/iteration-1/
├── global/
│   └── all_raags/
│       ├── markov/global_markov_model.json
│       ├── lstm/global_lstm.pt
│       ├── lstm/global_lstm_history.json
│       ├── music_transformer/global_music_transformer.pt
│       └── music_transformer/global_music_transformer_history.json
└── per_raag/
    ├── bageshree/
    │   ├── markov/per_raag_bageshree_markov_model.json
    │   ├── lstm/per_raag_bageshree_lstm.pt
    │   ├── lstm/per_raag_bageshree_lstm_history.json
    │   ├── music_transformer/per_raag_bageshree_music_transformer.pt
    │   └── music_transformer/per_raag_bageshree_music_transformer_history.json
    ├── khamaj/
    └── bhoop/
```

## Generate Swara, Audio, MIDI, and Plots

```bash
python hindustani_multi_model_pipeline.py generate-all \
  --prepared-dir artifacts/prepared \
  --models-dir artifacts/models \
  --output-dir artifacts/generated \
  --grammar-bias-strength 3.0 \
  --write-audio \
  --write-midi
```

If `--prepared-dir` or `--models-dir` points at the root artifact directory, the script automatically resolves the latest iteration.
Generation outputs go to a new folder such as `artifacts/generated/iteration-1`.

Generation scaffold:

```text
artifacts/generated/iteration-1/
├── global/
│   ├── bageshree/
│   │   ├── markov/
│   │   │   ├── global_bageshree_markov.swara.txt
│   │   │   └── global_bageshree_markov.swara.json
│   │   ├── lstm/
│   │   │   ├── global_bageshree_lstm.swara.txt
│   │   │   └── global_bageshree_lstm.swara.json
│   │   └── music_transformer/
│   │       ├── global_bageshree_music_transformer.swara.txt
│   │       └── global_bageshree_music_transformer.swara.json
│   ├── khamaj/
│   └── bhoop/
├── per_raag/
│   ├── bageshree/
│   │   ├── markov/
│   │   │   ├── per_raag_bageshree_markov.swara.txt
│   │   │   └── per_raag_bageshree_markov.swara.json
│   │   ├── lstm/
│   │   │   ├── per_raag_bageshree_lstm.swara.txt
│   │   │   └── per_raag_bageshree_lstm.swara.json
│   │   └── music_transformer/
│   │       ├── per_raag_bageshree_music_transformer.swara.txt
│   │       └── per_raag_bageshree_music_transformer.swara.json
│   ├── khamaj/
│   └── bhoop/
└── generation_report.json
```

Each generated model folder contains files such as `global_bageshree_lstm.swara.txt`, `global_bageshree_lstm.swara.json`, `global_bageshree_lstm.wav`, `global_bageshree_lstm.mid`, `global_bageshree_lstm_radar.png`, `global_bageshree_lstm_pitch_distribution.png`, and `global_bageshree_lstm_structure_heatmap.png` depending on the scope, raag, model, and flags you enable.

When `--write-audio` is enabled, the renderer now keeps the model's generated swara sequence intact but voices it with a sitar-leaning plucked lead, note-to-note glide on phrase-internal transitions, and a more tanpura-like drone bed.

Useful generation flags:

- `--seconds-per-note 0.32`
- `--audio-sample-rate 22050`
- `--drone-gain 0.16`
- `--glide-ratio 0.24`
- `--sympathetic-gain 0.12`
- `--no-write-plots` if you want to skip plot generation

## Generate Aggregate Evaluation Plots by Iteration

Use `evaluation_plot_suite.py` to regenerate the aggregate evaluation figures from a baseline `generation_report.json`.

Examples:

```bash
python evaluation_plot_suite.py --iteration 5
```

This resolves `artifacts/generated/iteration-5/generation_report.json` and saves the plot set under `plots/iteration-5`.

```bash
python evaluation_plot_suite.py --generated-dir artifacts/generated --iteration 4 --output-dir plots
```

This still writes into an iteration-specific folder: `plots/iteration-4`.

If you omit `--iteration`, the script uses the latest available generated iteration automatically:

```bash
python evaluation_plot_suite.py
```

To include Suno metrics in the same aggregate research plots, add `--include-suno`:

```bash
python evaluation_plot_suite.py --iteration 5 --include-suno
```

This resolves baseline metrics from `artifacts/generated/iteration-5/generation_report.json`, loads Suno metrics from `plots/iteration-5/suno_vs_baselines/suno_metrics_report.json`, and renders a unified plot set under `plots/iteration-5`.

You can still point directly at any report file if you prefer:

```bash
python evaluation_plot_suite.py \
  --metrics-json artifacts/generated/iteration-5/generation_report.json
```

## Compare Suno Against Baselines by Iteration

Use `suno_baseline_comparison.py` to compare Suno artifacts against the baseline models from a specific generated iteration.

Examples:

```bash
python suno_baseline_comparison.py \
  --prepared-dir artifacts/prepared \
  --iteration 5 \
  --suno-dir artifacts/external_model_artifacts
```

This resolves `artifacts/generated/iteration-5/generation_report.json` automatically and writes the comparison reports and plots under `plots/iteration-5/suno_vs_baselines`.

If you omit `--iteration`, the script uses the latest available generated iteration automatically:

```bash
python suno_baseline_comparison.py \
  --prepared-dir artifacts/prepared \
  --suno-dir artifacts/external_model_artifacts
```

You can still point directly at a baseline report when needed:

```bash
python suno_baseline_comparison.py \
  --prepared-dir artifacts/prepared \
  --baseline-report artifacts/generated/iteration-5/generation_report.json \
  --suno-dir artifacts/external_model_artifacts
```

## Compare Hugging Face MusicGen Against Baselines

The MusicGen flow is separate from the baseline training script because it:

- generates prompt-based audio with `facebook/musicgen-small` (or another MusicGen checkpoint)
- saves raw audio plus per-raaga swara and evaluation artifacts under `artifacts/external_model_artifacts/musicgen_generation`
- saves aggregate comparison reports and plots under `plots/musicgen_vs_baselines`

Example:

```bash
python musicgen_baseline_comparison.py \
  --prepared-dir artifacts/prepared \
  --baseline-report artifacts/generated/iteration-1/generation_report.json \
  --artifact-dir artifacts/external_model_artifacts/musicgen_generation \
  --output-dir plots/musicgen_vs_baselines \
  --model-id facebook/musicgen-small \
  --max-new-tokens 1024 \
  --guidance-scale 3.0
```

Default MusicGen prompts follow this structure:

```text
Indian classical [instrument], raga [name], [time/mood], slow alaap, melodic improvisation, tanpura drone, expressive ornamentation, no percussion
```

You can override those prompts with `--prompt-template` or pass `--prompt-manifest` with per-raaga prompt overrides.

MusicGen artifact scaffold:

```text
artifacts/external_model_artifacts/musicgen_generation/
├── bageshree/
│   ├── musicgen_bageshree.wav
│   ├── musicgen_bageshree_prompt.json
│   ├── musicgen_bageshree_prompt.txt
│   ├── musicgen_bageshree.swara.txt
│   ├── musicgen_bageshree.swara.json
│   ├── musicgen_bageshree_radar.png
│   ├── musicgen_bageshree_pitch_distribution.png
│   └── musicgen_bageshree_structure_heatmap.png
├── khamaj/
├── bhoop/
└── musicgen_generation_report.json
```

Comparison output scaffold:

```text
plots/musicgen_vs_baselines/
├── musicgen_metrics_report.json
├── combined_comparison_records.json
├── combined_metrics_table.csv
├── creativity_leaderboard.png
├── system_metric_heatmap.png
├── kl_vs_grammar_comparison.png
└── musicgen_vs_baseline_dashboard.png
```

## Primary Outputs

For each generated sample, the script writes:

- `<scope>_<raag>_<model>.swara.txt` as the main readable notation file
- `<scope>_<raag>_<model>.swara.json` as the structured machine-readable version
- `<scope>_<raag>_<model>.wav` if `--write-audio` is enabled
- `<scope>_<raag>_<model>.mid` if `--write-midi` is enabled
- `<scope>_<raag>_<model>_radar.png`
- `<scope>_<raag>_<model>_pitch_distribution.png`
- `<scope>_<raag>_<model>_structure_heatmap.png`
- `generation_report.json`

## Included Evaluation Visuals

- creativity radar plot
- training vs generated pitch distribution
- structure similarity heatmap

## Modeling Notes

- LSTM generation now runs incrementally with hidden-state carryover.
- Transformer conditioning uses token and raaga embedding concatenation followed by projection.
- Novelty includes entropy difference and KL divergence from the training note distribution.
- Reflection is defined using repeated motif reuse rather than unique motif count.
- Swara outputs remain the canonical final artifact and are biased toward stronger raaga identity.

## Notes

- The swara files are the canonical final output and are designed to emphasize the identity of the raaga as strongly as the model allows.
- The raaga grammar rules are still heuristic research defaults and can be refined further for stricter musicological fidelity.
