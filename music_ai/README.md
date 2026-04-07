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

## Install

```bash
pip install -r requirements.txt
```

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

Useful generation flags:

- `--seconds-per-note 0.32`
- `--audio-sample-rate 22050`
- `--drone-gain 0.16`
- `--no-write-plots` if you want to skip plot generation

## Primary Outputs

For each generated sample, the script writes:

- `*.swara.txt` as the main readable notation file
- `*.swara.json` as the structured machine-readable version
- `*.wav` if `--write-audio` is enabled
- `*.mid` if `--write-midi` is enabled
- `*_radar.png`
- `*_pitch_distribution.png`
- `*_structure_heatmap.png`
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
