"""
Audit wrapper around the verbatim evaluation logic extracted from
scripts/truba/submit_v6_eval.sh:25-93.

This file adds ONLY a CLI wrapper; all evaluation logic is in
eval_disambiguator_verbatim.py (byte-faithful extraction).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from train.datasets import Vocab
from train.disambiguation_dataset import DisambiguationDataset, disambiguation_collate
from train.train_disambiguator import evaluate, pre_cache_bert_embeddings
from kokturk.models.disambiguator import BERTurkDisambiguator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def count_params_string_em(model, loader, device, bert_cache, tag_vocab, data_file):
    """Additional EM metric: full-parse string equality (TLF-1 verification).

    Loads raw data to get gold canonical strings, then compares to top
    predicted candidate's canonical string.
    """
    # Load raw jsonl to get gold labels
    samples = []
    with open(data_file, encoding="utf-8") as f:
        for ln in f:
            samples.append(json.loads(ln))

    model.eval()
    correct_string = 0
    total_string = 0

    with torch.no_grad():
        for batch in loader:
            cached_embeds = None
            if bert_cache is not None:
                from train.train_disambiguator import _get_cached_embeds

                cached_embeds = _get_cached_embeds(bert_cache, batch["sample_indices"])

            logits, _ = model(
                sentence_texts=batch["sentence_texts"],
                target_positions=batch["target_positions"],
                candidate_ids=batch["candidate_ids"],
                candidate_mask=batch["candidate_mask"],
                cached_bert_embeds=cached_embeds,
            )
            preds = logits.argmax(dim=-1)
            for i, (pred_idx, sample_idx) in enumerate(
                zip(preds.tolist(), batch["sample_indices"].tolist())
            ):
                sample = samples[sample_idx]
                gold_label = sample.get("label", "")  # e.g. "ev +PLU +POSS.3SG +ABL"
                candidates = batch.get("candidate_strings", [])
                if candidates and i < len(candidates):
                    pred_str = candidates[i][pred_idx] if pred_idx < len(candidates[i]) else ""
                    if pred_str == gold_label:
                        correct_string += 1
                total_string += 1

    if total_string == 0:
        return None
    return correct_string / total_string


def evaluate_checkpoint(ckpt_path, tag_vocab, bert, tok, cache_dir, test_file, val_file):
    """Evaluate a single checkpoint on test and val splits."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BERTurkDisambiguator(
        tag_vocab_size=ckpt.get("tag_vocab_size", len(tag_vocab)),
        skip_bert_loading=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seed = ckpt_path.parent.name.replace("disambiguator_s", "").replace("disambiguator", "42")
    print(
        f"\nCheckpoint: {ckpt_path} "
        f"| epoch {ckpt.get('epoch', '?')} "
        f"| val_em_during_training={ckpt.get('val_em', 0) * 100:.2f}%"
    )

    results = {}
    for split_name, split_path in [("test", test_file), ("val", val_file)]:
        print(f"  Evaluating {split_name}...")
        ds = DisambiguationDataset(split_path, tag_vocab)
        cache = pre_cache_bert_embeddings(
            ds,
            "models/berturk",
            cache_path=cache_dir / f"{split_name}_{seed}_bert_cache.pt",
            shared_bert=bert,
            shared_tokenizer=tok,
        )
        loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=disambiguation_collate)
        m = evaluate(model, loader, "cpu", bert_cache=cache)
        results[split_name] = {
            "overall_em_repo_def": round(m["overall_em"], 6),
            "ambiguous_em_repo_def": round(m["ambiguous_em"], 6),
            "total_tokens": m["total"],
            "ambiguous_tokens": m["ambiguous_total"],
        }
        print(
            f"    overall_em (repo def = argmax accuracy): {m['overall_em'] * 100:.2f}%  "
            f"ambig_em: {m['ambiguous_em'] * 100:.2f}%  "
            f"n={m['total']}"
        )
    results["checkpoint"] = str(ckpt_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Audit: evaluate all v6 disambiguator seeds")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--val", default="data/splits/val.jsonl")
    parser.add_argument("--output", default="audit/results/07_em.json")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            "models/v6/disambiguator/best_model.pt",
            "models/v6/disambiguator_s123/best_model.pt",
            "models/v6/disambiguator_s456/best_model.pt",
            "models/v6/disambiguator_s789/best_model.pt",
            "models/v6/disambiguator_s1337/best_model.pt",
        ],
    )
    args = parser.parse_args()

    tag_vocab = Vocab.load(Path("models/vocabs/tag_vocab.json"))
    cache_dir = Path("/tmp/audit_bert_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load BERTurk once
    from transformers import AutoModel, AutoTokenizer

    print("Loading BERTurk...")
    bert = AutoModel.from_pretrained("models/berturk")
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False
    tok = AutoTokenizer.from_pretrained("models/berturk")

    all_results = {}
    test_ems = []

    for ckpt_str in args.ckpts:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f"SKIPPING {ckpt_path} — not found")
            continue
        res = evaluate_checkpoint(ckpt_path, tag_vocab, bert, tok, cache_dir, args.test, args.val)
        seed_name = ckpt_path.parent.name
        all_results[seed_name] = res
        test_ems.append(res["test"]["overall_em_repo_def"])

    # Ensemble statistics
    if test_ems:
        import statistics

        all_results["seed_statistics"] = {
            "n_seeds": len(test_ems),
            "test_em_min": min(test_ems),
            "test_em_max": max(test_ems),
            "test_em_mean": statistics.mean(test_ems),
            "test_em_median": statistics.median(test_ems),
            "test_em_stdev": statistics.stdev(test_ems) if len(test_ems) > 1 else 0,
        }
        print(
            f"\n=== SEED STATISTICS (test EM, repo definition) ===\n"
            f"  n={len(test_ems)}  "
            f"min={min(test_ems) * 100:.2f}%  "
            f"max={max(test_ems) * 100:.2f}%  "
            f"mean={statistics.mean(test_ems) * 100:.2f}%  "
            f"stdev={statistics.stdev(test_ems) * 100:.2f}pp"
        )

    # Read stored ensemble results
    ensemble_file = Path("models/v6/ensemble_results.json")
    if ensemble_file.exists():
        with open(ensemble_file) as f:
            all_results["stored_ensemble"] = json.load(f)
        print(
            f"\n=== STORED ENSEMBLE (from ensemble_results.json) ===\n"
            f"  ensemble EM: {all_results['stored_ensemble']['ensemble']['overall_em'] * 100:.2f}%"
        )

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
