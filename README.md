# StrokeID-NER

An annotated dataset and benchmark for stroke named entity recognition in Indonesian patient health queries.

StrokeID-NER contains **1,742 informal Indonesian patient queries** about stroke, collected from alodokter.com, annotated with **6,765 entity spans** of four types in BIO format:

| Type | Meaning | Spans |
|---|---|---|
| SYM | Symptom | 2,643 |
| DGN | Diagnosis | 2,189 |
| TMP | Temporal expression | 1,094 |
| RF | Risk factor | 839 |

This repository contains the dataset, the annotation guideline, the code, the configurations and seeds, and the test-set predictions of every run reported in the paper.

Archived version (DOI): https://doi.org/10.5281/zenodo.23910283

---

## Repository structure

```
data/          dataset files (train / val / test split)
guideline/     annotation guideline (19 rules)
code/          training, evaluation and analysis scripts + Colab notebooks
prompts/       GPT-5.4 system prompt
predictions/   test-set predictions and settings of every run
results/       numbers behind every table in the paper
LICENSE        MIT (code)
LICENSE-DATA   CC BY 4.0 (data, guideline, predictions)
```

### `data/`

| File | Content |
|---|---|
| `train.tsv` | 1,211 queries (69.5%) |
| `val.tsv` | 256 queries (14.7%) |
| `test.tsv` | 275 queries (15.8%) |

Format: one `token<TAB>label` per line, with a blank line between queries. The label set is `O, B-SYM, I-SYM, B-DGN, I-DGN, B-TMP, I-TMP, B-RF, I-RF`. The split was made with stratified sampling (seed 42).

### `predictions/`

One folder per system and one sub-folder per seed (`seed42`, `seed123`, `seed456`). Each contains:

- `test_predictions.jsonl`: gold and predicted labels for every test query
- `metrics.json`: the exact settings, seed, GPU, library versions, best epoch, run time and test scores of that run

| Folder | System in the paper |
|---|---|
| `m1` / `m2` | IndoBERT-base + linear (M1) / + CRF (M2) |
| `m4` / `m4crf` | IndoBERT-large + linear (M4) / + CRF |
| `m5` / `m5crf` | XLM-R-large + linear (M5) / + CRF |
| `m3` / `m3crf` | XLM-R-large-ID-NER + linear (M3) / + CRF |
| `m3_bin-DGN`, `m3_bin-RF`, `m3_bin-SYM`, `m3_bin-TMP` | Binary (single-entity) models, Supplementary Table S5 |
| `gpt0` | GPT-5.4 zero-shot (single run) |
| `gpt5` | GPT-5.4 five-shot (three demonstration sets; `metrics.json` lists the demonstration query IDs) |

Encoders: `indobenchmark/indobert-base-p2`, `indobenchmark/indobert-large-p2`, `FacebookAI/xlm-roberta-large` (XLM-R-large), `cahya/xlm-roberta-large-indonesian-NER` (XLM-R-large-ID-NER).

### `results/`

| File | Supports in the paper |
|---|---|
| `bootstrap_summary.json` | Table 3 (all scores and 95% CIs), Table 4 (paired differences), Supplementary Tables S4 and S6 |
| `results_table.md` | Table 3 (readable version) |
| `table6.md` | Supplementary Table S5 (joint vs. binary training) |
| `fn_overlap_seed42.txt` | Cross-model error consistency (spans missed by all four models) |
| `iaa/iaa.json` | Inter-annotator agreement, Supplementary Table S2 |
| `scope/scope_summary.json` | Annotation scope, Supplementary Table S3 |

---

## Configurations and seeds

- Hyperparameters for every model are defined at the top of `code/train.py`. CRF variants use the settings of their linear counterpart; XLM-R-large (M5) uses the settings of XLM-R-large-ID-NER (M3).
- Each run's exact settings, seed, GPU and library versions are stored in `predictions/<model>/seed<N>/metrics.json`.
- Seeds: 42, 123 and 456 for every fine-tuned system; the five-shot demonstration sets were drawn with the same three seeds; the data split used seed 42.
- Runs used an NVIDIA L4 GPU (24 GB), Python 3.13.18, PyTorch 2.11.0 and Transformers 5.20.2.

---

## Reproducing the results

Requirements: `torch`, `transformers`, `numpy`, `matplotlib`; `openai` only for the GPT-5.4 runs. Span-level scoring is implemented in `code/common.py` and matches seqeval's macro average.

The notebooks in `code/` were run on Google Colab (L4 GPU). Change the Google Drive paths at the top of each notebook to your own folder.

| Notebook | Produces |
|---|---|
| `plain_XLM-R` | XLM-R-large (M5) runs |
| `seeds_main_models` | M1, M2, M3, M4 with three seeds |
| `CRF_on_all_encoders` | CRF variants of IndoBERT-large and both XLM-R models |
| `few-shot` | GPT-5.4 five-shot runs |
| `table6_binary_NER` | Binary models (Supplementary Table S5) |
| `annotation_analyses` | Inter-annotator agreement and annotation scope |
| `results_and_tables` | Tables 3, 4 and 5, error overlap |

Single run from the command line:

```bash
python code/train.py --model m3crf --seed 42 --data_dir data --out_dir predictions
```

Confidence intervals (10,000 query-level bootstrap resamples):

```bash
python code/bootstrap_ci.py --runs predictions \
  --models gpt0 gpt5 m1 m2 m4 m4crf m5 m5crf m3 m3crf --B 10000 --out results
```

Table 5:

```bash
python code/span_overlap.py --a predictions/m3crf/seed42 --b predictions/gpt0/seed42
python code/span_overlap.py --a predictions/m3crf/seed42 --b predictions/gpt5/seed42
```

The GPT-5.4 runs need an OpenAI API key, set as the environment variable `OPENAI_API_KEY` (or a Colab secret). Never write the key into a notebook or script.

---

## License

- **Code:** MIT (see `LICENSE`).
- **Dataset, annotation guideline and predictions:** CC BY 4.0 (see `LICENSE-DATA`).

The source queries come from the publicly available indonesia-medical-qna dataset (Abdillah, 2026; https://doi.org/10.57967/hf/8356), which does not carry an explicit licence. They are provided for academic research in line with the platform's terms of use. The dataset contains no personally identifiable information; ages are kept as risk-factor annotations.

---

## Citation

The paper is currently under review. Until it is published, please cite the archived dataset:

```
Cordiaz S, M., Yaseen, M., Mozumder, M. A. I., & Kim, H.-C. (2026).
StrokeID-NER: dataset, annotation guideline, code and predictions (v2.0) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.23910283
```

## Contact

Hee-Cheol Kim (corresponding author), Inje University, heeki@inje.ac.kr
