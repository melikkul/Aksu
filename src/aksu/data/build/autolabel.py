"""Auto-labeling pipeline for TR-Gold-Morph unique tokens.

For each unique token from data/intermediate/unique_tokens.jsonl:
  1. Run Zeyrek → candidates list
  2. If 0 candidates → Zeyrek-OOV; log to oov.jsonl and skip from main
  3. If 1 candidate → accept, confidence=1.0, unanimous=True, seed_agreement=5
  4. If >1 candidates → run v6 BERTurkDisambiguator 5-seed ensemble

Ensemble tie-breaking (§H+ A):
  - Majority vote (argmax per seed) across 5 seeds
  - Tie (e.g. 2-2-1): average softmax across seeds, argmax among tied candidates
  - <3 seeds usable (NaN/error): mark method=drop_low_seeds

Outputs:
  data/intermediate/autolabeled_shard_N.jsonl  (per shard)
  data/tr_gold_morph/v2/oov.jsonl              (zero-candidate tokens, appended)

Usage (from project root):
    python -m aksu.data.build.autolabel --shard-start 0 --shard-end 50000
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import zeyrek

logger = logging.getLogger(__name__)

ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]
SEED_DIRS = {
    42: "models/v6/disambiguator",
    123: "models/v6/disambiguator_s123",
    456: "models/v6/disambiguator_s456",
    789: "models/v6/disambiguator_s789",
    1337: "models/v6/disambiguator_s1337",
}

_MAX_CANDIDATES = 10
_MAX_PARSE_LEN = 15


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def _load_tag_vocab(vocab_path: str = "models/vocabs/tag_vocab.json") -> list[str]:
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def _encode_candidate(
    candidate_str: str,
    token2idx: dict[str, int],
    unk_idx: int = 3,
    max_len: int = _MAX_PARSE_LEN,
) -> list[int]:
    """Tokenize a canonical parse string to tag-vocab indices."""
    parts = candidate_str.split()
    ids: list[int] = []
    for part in parts[:max_len]:
        ids.append(token2idx.get(part, unk_idx))
    return ids


def _candidates_to_tensor(
    candidates: list[str],
    token2idx: dict[str, int],
    max_cands: int = _MAX_CANDIDATES,
    max_len: int = _MAX_PARSE_LEN,
) -> tuple:
    """Encode candidates into (max_cands, max_len) ids + (max_cands,) mask."""
    import torch
    ids = torch.zeros(max_cands, max_len, dtype=torch.long)
    mask = torch.zeros(max_cands, dtype=torch.bool)
    for i, cand in enumerate(candidates[:max_cands]):
        encoded = _encode_candidate(cand, token2idx)
        for j, idx in enumerate(encoded):
            ids[i, j] = idx
        mask[i] = True
    return ids, mask


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_disambiguator(seed: int, tag_vocab_size: int) -> object:
    """Load a single BERTurkDisambiguator checkpoint."""
    from aksu.kokturk.models.disambiguator import BERTurkDisambiguator

    model_dir = SEED_DIRS[seed]
    ckpt_path = f"{model_dir}/best_model.pt"

    config_path = f"{model_dir}/config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = BERTurkDisambiguator(
        tag_vocab_size=tag_vocab_size,
        **{k: v for k, v in config.items() if k != "tag_vocab_size"},
    )
    import torch
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _load_ensemble(tag_vocab_size: int) -> tuple[list, int]:
    """Load all 5 BERTurkDisambiguator seeds. Returns (models, n_usable)."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        try:
            m = _load_disambiguator(seed, tag_vocab_size)
            models.append(m)
            logger.info("Loaded seed %d", seed)
        except Exception as e:
            logger.warning("Failed to load seed %d: %s", seed, e)
    return models, len(models)


# ---------------------------------------------------------------------------
# Ensemble scoring (§H+ A)
# ---------------------------------------------------------------------------

def _ensemble_score(
    models: list,
    candidates: list[str],
    token2idx: dict[str, int],
    context_sentence: str,
    target_position: int,
) -> tuple[int, float, int, list[int]]:
    """Score candidates via 5-seed ensemble.

    Returns:
        best_idx        — winning candidate index
        disambig_score  — max softmax score for winner
        seed_agreement  — number of seeds voting for winner (or n_usable if <3)
        seed_votes      — per-seed argmax indices (length = n_models)
    """
    import math
    import torch

    n_cands = min(len(candidates), _MAX_CANDIDATES)
    cand_ids, cand_mask = _candidates_to_tensor(candidates, token2idx)

    seed_argmaxes: list[int] = []
    seed_softmaxes: list[list[float]] = []  # per-seed full softmax over candidates

    for model in models:
        try:
            with torch.no_grad():
                logits, _ = model(
                    sentence_texts=[context_sentence],
                    target_positions=[target_position],
                    candidate_ids=cand_ids.unsqueeze(0),
                    candidate_mask=cand_mask.unsqueeze(0),
                )
            row = logits[0, :n_cands]
            if torch.any(torch.isnan(row)):
                continue
            probs = torch.softmax(row, dim=-1).tolist()
            argmax = int(row.argmax().item())
            seed_argmaxes.append(argmax)
            seed_softmaxes.append(probs)
        except Exception as e:
            logger.debug("Seed error: %s", e)
            continue

    n_usable = len(seed_argmaxes)
    if n_usable < 3:
        # Not enough seeds — caller routes to drop
        return 0, 0.0, n_usable, seed_argmaxes

    # Majority vote
    from collections import Counter as _Counter
    vote_counts = _Counter(seed_argmaxes)
    max_votes = vote_counts.most_common(1)[0][1]
    tied = [idx for idx, cnt in vote_counts.items() if cnt == max_votes]

    if len(tied) == 1:
        best_idx = tied[0]
    else:
        # Tie-break: average softmax over all seeds, argmax among tied
        avg_softmax = [0.0] * n_cands
        for probs in seed_softmaxes:
            for k, p in enumerate(probs[:n_cands]):
                avg_softmax[k] += p
        avg_softmax = [s / n_usable for s in avg_softmax]
        best_idx = max(tied, key=lambda k: avg_softmax[k])

    seed_agreement = vote_counts[best_idx]
    # disambig_score = average softmax score for winner across seeds
    if seed_softmaxes:
        disambig_score = sum(p[best_idx] for p in seed_softmaxes if best_idx < len(p)) / n_usable
    else:
        disambig_score = 0.0

    return best_idx, disambig_score, seed_agreement, seed_argmaxes


# ---------------------------------------------------------------------------
# Sentence index helpers
# ---------------------------------------------------------------------------

def _load_sentence_index(token_sents_path: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    if not token_sents_path.exists():
        return index
    with token_sents_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index[row["token"]] = row["sentence_ids"]
    return index


def _load_sentence_texts(
    sentences_path: Path, limit: int | None = None
) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not sentences_path.exists():
        return texts
    with sentences_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts[row["sentence_id"]] = row["text"]
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _filter_tokens_by_sentences(
    all_tokens: list[dict],
    sentence_index: dict[str, list[str]],
    valid_sentence_ids: frozenset[str],
) -> list[dict]:
    """Keep only tokens that appear in at least one valid sentence."""
    kept = []
    for tok in all_tokens:
        token_str = tok.get("token", tok.get("surface", ""))
        sids = sentence_index.get(token_str, [])
        if any(sid in valid_sentence_ids for sid in sids):
            kept.append(tok)
    return kept


# ---------------------------------------------------------------------------
# Core autolabeling loop
# ---------------------------------------------------------------------------

def autolabel_tokens(
    tokens: list[dict],
    *,
    sentence_index: dict[str, list[str]],
    sentence_texts: dict[str, str],
    output_path: Path,
    oov_path: Path,
    models: list,
    token2idx: dict[str, int],
    rng_seed: int = 42,
) -> dict[str, int]:
    """Autolabel a list of unique token records.

    Returns stats dict with counts per outcome.
    """
    import random
    rng = random.Random(rng_seed)
    analyzer = zeyrek.MorphAnalyzer()
    stats: Counter = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    oov_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as out, \
         oov_path.open("a", encoding="utf-8") as oov_out:

        for row in tokens:
            token = row.get("token", row.get("surface", ""))
            source_id = row.get("source", row.get("source_id", "unknown"))
            tier_from_row = row.get("tier", "")

            try:
                results = analyzer.analyze(token)
            except Exception as e:
                logger.warning("Zeyrek error for %r: %s", token, e)
                results = []

            # Flatten Zeyrek results to canonical strings
            candidates: list[str] = []
            for r in results:
                try:
                    parses = r if isinstance(r, list) else [r]
                    for parse in parses:
                        lemma = getattr(parse, "lemma", "") or ""
                        morphemes = getattr(parse, "morphemes", []) or []
                        if morphemes:
                            tags = " ".join(f"+{m}" for m in morphemes if m)
                            candidates.append(f"{lemma} {tags}".strip())
                        elif lemma:
                            candidates.append(lemma)
                except Exception:
                    continue
            # Deduplicate while preserving order
            seen: set[str] = set()
            uniq_cands: list[str] = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    uniq_cands.append(c)
            candidates = uniq_cands

            if not candidates:
                # Zeyrek-OOV: log and skip
                oov_record = {
                    "token": token,
                    "source_id": source_id,
                    "method": "zeyrek_oov",
                }
                oov_out.write(json.dumps(oov_record, ensure_ascii=False) + "\n")
                stats["zeyrek_oov"] += 1
                continue

            if len(candidates) == 1:
                record = {
                    "token": token,
                    "canonical": candidates[0],
                    "candidates": candidates,
                    "candidate_count": 1,
                    "confidence": 1.0,
                    "seed_agreement": len(ENSEMBLE_SEEDS),
                    "seed_votes": [0] * len(ENSEMBLE_SEEDS),
                    "disambig_score": 1.0,
                    "source_id": source_id,
                    "method": "unambiguous",
                    "tier": tier_from_row,
                }
                stats["unambiguous"] += 1
            else:
                # Multi-candidate: use ensemble
                sids = sentence_index.get(token, [])
                context = ""
                target_pos = 0
                if sids:
                    chosen_sid = rng.choice(sids)
                    context = sentence_texts.get(chosen_sid, "")
                    if context:
                        # Approximate token position
                        words = context.split()
                        try:
                            target_pos = words.index(token)
                        except ValueError:
                            target_pos = 0

                if models:
                    best_idx, disambig_score, seed_agreement, seed_votes = _ensemble_score(
                        models, candidates, token2idx,
                        context_sentence=context,
                        target_position=target_pos,
                    )
                    if seed_agreement < 3:
                        method = "drop_low_seeds"
                    else:
                        method = "ensemble"
                else:
                    # No ensemble loaded (--no-ensemble mode)
                    best_idx, disambig_score, seed_agreement, seed_votes = 0, 0.80, 0, []
                    method = "zeyrek_first"

                record = {
                    "token": token,
                    "canonical": candidates[best_idx] if candidates else None,
                    "candidates": candidates[:_MAX_CANDIDATES],
                    "candidate_count": len(candidates),
                    "confidence": disambig_score,
                    "seed_agreement": seed_agreement,
                    "seed_votes": seed_votes,
                    "disambig_score": disambig_score,
                    "source_id": source_id,
                    "context_sentence": context[:200] if context else None,
                    "method": method,
                    "tier": tier_from_row,
                }
                stats["ensemble" if method == "ensemble" else method] += 1

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return dict(stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--unique-tokens", default="data/intermediate/unique_tokens.jsonl")
    ap.add_argument("--token-sentences", default="data/intermediate/token_sentences.jsonl")
    ap.add_argument("--sentences", default="data/intermediate/sentences.jsonl")
    ap.add_argument("--output", default="data/intermediate/autolabeled.jsonl")
    ap.add_argument("--oov-output", default="data/tr_gold_morph/v2/oov.jsonl")
    ap.add_argument("--tag-vocab", default="models/vocabs/tag_vocab.json")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=None)
    ap.add_argument(
        "--sentence-limit",
        type=int,
        default=None,
        help="Restrict autolabel to tokens appearing in the first N sentences "
             "(pilot mode — §H+ B). Overrides --shard-start/--shard-end.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip BERTurk ensemble loading (for testing/debugging only; "
             "produces zeyrek_first outputs)",
    )
    args = ap.parse_args()

    tokens_path = Path(args.unique_tokens)
    if not tokens_path.exists():
        ap.error(f"unique_tokens file not found: {tokens_path}")

    with tokens_path.open(encoding="utf-8") as f:
        all_tokens = [json.loads(line) for line in f if line.strip()]

    tag_vocab_list = _load_tag_vocab(args.tag_vocab)
    token2idx = {tok: i for i, tok in enumerate(tag_vocab_list)}
    tag_vocab_size = len(tag_vocab_list)
    logger.info("Tag vocab size: %d", tag_vocab_size)

    models: list = []
    if not args.no_ensemble:
        logger.info("Loading BERTurkDisambiguator ensemble (5 seeds)...")
        models, n_usable = _load_ensemble(tag_vocab_size)
        logger.info("Loaded %d / %d seeds", n_usable, len(ENSEMBLE_SEEDS))
        if n_usable < 3:
            ap.error(
                f"Only {n_usable} seeds loaded — need ≥3 for ensemble. "
                "Check models/v6/disambiguator*/best_model.pt. Use --no-ensemble for debug."
            )

    logger.info("Loading sentence index...")
    sentence_index = _load_sentence_index(Path(args.token_sentences))
    sentence_texts = _load_sentence_texts(Path(args.sentences), limit=args.sentence_limit)
    logger.info("Index: %d tokens, %d sentences", len(sentence_index), len(sentence_texts))

    if args.sentence_limit is not None:
        valid_sids = frozenset(sentence_texts.keys())
        shard = _filter_tokens_by_sentences(all_tokens, sentence_index, valid_sids)
        logger.info(
            "Sentence-limit pilot: %d sentences → %d unique tokens",
            len(valid_sids), len(shard),
        )
    else:
        shard = all_tokens[args.shard_start: args.shard_end]
        logger.info(
            "Autolabeling %d tokens (shard %d:%s)",
            len(shard), args.shard_start, args.shard_end,
        )

    stats = autolabel_tokens(
        shard,
        sentence_index=sentence_index,
        sentence_texts=sentence_texts,
        output_path=Path(args.output),
        oov_path=Path(args.oov_output),
        models=models,
        token2idx=token2idx,
        rng_seed=args.seed,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
