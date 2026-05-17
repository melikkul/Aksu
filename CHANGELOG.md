# Changelog

All notable changes to the Aksu project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0a0] - 2026-05-17

### Added
- **NeuralBackend GPU acceleration stack**: device auto-detection (CUDA → MPS → XPU → CPU), `torch.amp.autocast` bf16 mixed precision on GPU, `torch.compile(mode="reduce-overhead", dynamic=True)` with graceful fallback, `torch.inference_mode()` inference guard, warm-up passes to pre-allocate GPU memory. New kwargs `device`, `precision`, `compile_mode`, `batch_size` (all defaulted — fully backwards-compatible).
- **`NeuralBackend.predict_batch(surfaces, batch_size)`**: batched GPU-accelerated inference API with sort-by-length padding and caller-order preservation. `analyze(word)` now delegates to `predict_batch`.
- **`reconstruct_line_breaks(text, use_lm=True)`**: LM-aware PDF hyphenation decoder using a three-signal decision matrix (lexicon hit → CP prefix → vowel harmony → fall-through JOIN). Requires `regex` for Unicode `\p{L}` patterns.
- **`fix_pdf_artifacts(text, aggressive=False, repair_diacritics=False)`**: multi-stage PDF cleaning pipeline — base NFC normalisation, ftfy mojibake/ligature repair, NFKC, zero-width character removal, repeated-char collapse, space-injection repair, optional header/footer removal, optional diacritic restoration stub.
- **`is_morphologically_valid(word)`**: heuristic vowel-harmony sanity check with 30% loanword tolerance.
- **`TextCleaner.fix_line_breaks(text)`** and **`TextCleaner.fix_artifacts(text)`**: delegating wrappers on the new normalize functions.
- **Bundled Zemberek Turkish wordlist** (`aksu.ariturk.data.turkish_wordlist.txt.gz`, ~100K entries, Apache 2.0) used by the hyphenation decoder.
- **`aksu[full]` optional extra**: `kenlm>=0.2.0` and `pyhyphen>=4.0.3` for LM-scored hyphenation (v1.1 reserved — kenlm scoring is no-op without install).
- **New public exports** from `aksu.ariturk`: `reconstruct_line_breaks`, `fix_pdf_artifacts`, `is_morphologically_valid`.
- **New tests** (71 new cases): `tests/test_neural_backend_gpu.py` (14), `tests/test_normalize_pdf.py` (22), `tests/test_normalize_artifacts.py` (21), `tests/integration/test_pdf_pipeline.py` (5).
- **Benchmark script**: `scripts/benchmark_neural_backend.py` — measures baseline (fp32/cpu/no-compile) first, then v1.1 configs; synthetic corpus fallback if `data/intermediate/` absent.
- **Phase 0 expectations fixture**: `tests/fixtures/captured_expectations.json` — Atomizer zeyrek output on smoke surfaces.

### Changed
- `NeuralBackend.__init__` now loads checkpoint directly onto the target device (previously always `map_location="cpu"`).

### Dependencies
- Added required: `ftfy>=6.1`, `regex>=2023.0` (total new mandatory footprint <1.5 MB).
- Added optional extra `[full]`: `kenlm>=0.2.0`, `pyhyphen>=4.0.3`.

### Notes
- bf16 on GPU produces slightly different numerical outputs than fp32. All neural-output comparisons should use `atol=1e-4` tolerance.
- Repeated-char collapse uses a "→2 chars" conservative rule; lexicon-aware snap (snap to canonical surface) is deferred to v1.2.

## [1.0.0a0] - 2026-05-16

### Changed
- **Package renamed `kokturk` → `aksu`** (Workstream A). Source tree consolidated under `src/aksu/`.
  Sub-packages `aksu.kokturk` and `aksu.ariturk` preserved with full history via `git mv`.
- Top-level `kokturk` and `ariturk` packages kept as deprecated compatibility shims (DeprecationWarning; removed in v2.0).
- Version centralised in `aksu/_version.py`; `pyproject.toml` reads it dynamically.
- License field corrected: `Apache-2.0` → `MIT` (code). Dataset published under CC BY 4.0.
- Console script entry point: `aksu analyze "evlerinden"` (was `python -m kokturk.cli.main`).

## [0.5.0] - 2026-04-09

### Added
- **Category A** — Diagnostic infrastructure: tag frequency analysis, FocalLoss/SymmetricCE/LabelSmoothing losses, noise audit (cleanlab seq2seq adapter), paradigm augmentation, domain importers (BounTi, Trendyol, Bilkent), contrastive root head, polysemy evaluation
- **Category B** — Linguistic coverage: fused LVC decomposition, special token preprocessing (abbreviations, numerics, reduplication), morphotactic FSA constraint mask, deep chain & compound evaluation
- **Category C** — Training optimization: R-Drop regularization, variational dropout (locked masks), character augmentation (keyboard/diacritic/stemcorrupt), EMA weights, Optuna HPO with TPE+Hyperband, AdamW optimizer support
- **Category D** — Evaluation & benchmarking: error analysis pipeline with diacritic-aware Levenshtein, weighted-EM metric, robustness perturbation suite, CheckList behavioral tests, speed benchmarking, standard benchmarks (TrMor2018/UD), minimal pairs challenge set, paired bootstrap + Holm-Bonferroni significance tests
- **Category E** — Engineering quality: pyproject.toml with ruff/mypy/pytest config, reproducibility utilities (seed_everything, environment capture), golden regression tests (52 entries), GitHub Actions CI pipeline, MLflow model registry, CHANGELOG

### Changed
- Project renamed from morpho-tr to kokturk
- Label smoothing corrected from epsilon=0.1 to epsilon=0.01 for morphological seq2seq
- Root vocab expanded 3,871 to 63,198 (OOV: 8.2% to 0.6%)
- Reproducibility: seed_everything now sets PYTHONHASHSEED, cudnn.deterministic, cudnn.benchmark

### Fixed
- atomizer_v2 "82%" was val EM — true test EM is 72.68% (9.6pp overfitting gap)
- Training seed handling now covers all random generators (was missing PYTHONHASHSEED and cuDNN flags)
