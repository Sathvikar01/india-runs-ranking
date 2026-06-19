"""Local CPU fine-tune for the BGE-reranker-base (Agent 4).

Fine-tunes ``BAAI/bge-reranker-base`` (278M params) on the binary
"tier-3+" labels using ``data/reranker/train.jsonl`` and ``val.jsonl``.

This is the in-sandbox alternative to the Modal-based fine-tune
(``modal/reranker_finetune.py``). It's slower (~6-8 h on 16 GB CPU) but
it works offline and produces a model that fits inside the 5 GB artifact
cap. Output: ``artifacts/ce_bge_finetuned/``.

Usage:
    python scripts/finetune_ce_bge.py \\
        --base BAAI/bge-reranker-base \\
        --out artifacts/ce_bge_finetuned \\
        --epochs 1 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("ft_ce_bge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="BAAI/bge-reranker-base")
    p.add_argument("--train", default="data/reranker/train.jsonl")
    p.add_argument("--val", default="data/reranker/val.jsonl")
    p.add_argument("--out", default="artifacts/ce_bge_finetuned")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    if not Path(args.train).exists():
        log.error("Training file %s not found. Run `modal/gen_reranker_data.py` first.", args.train)
        return 1
    if not Path(args.val).exists():
        log.error("Validation file %s not found. Run `modal/gen_reranker_data.py` first.", args.val)
        return 1

    import torch
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator,
    )
    from sentence_transformers.readers import InputExample

    log.info("Loading training examples from %s …", args.train)
    train_examples = []
    with open(args.train, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            train_examples.append(InputExample(
                texts=[d["query"], d["doc"]], label=float(d["label"]),
            ))
    log.info("  %d train examples", len(train_examples))

    val_examples = []
    with open(args.val, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            val_examples.append(InputExample(
                texts=[d["query"], d["doc"]], label=float(d["label"]),
            ))
    log.info("  %d val examples", len(val_examples))

    log.info("Loading base model %s …", args.base)
    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length)

    log.info("Fine-tuning …")
    t0 = time.perf_counter()
    train_dataloader = torch.utils.data.DataLoader(
        train_examples, batch_size=args.batch_size, shuffle=True,
    )
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_examples, name="tier3-eval",
    )
    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        output_path=str(args.out),
        save_best_model=True,
        evaluator=evaluator,
        evaluation_steps=500,
    )
    log.info("Fine-tune done in %.1fs. Model saved to %s",
             time.perf_counter() - t0, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
