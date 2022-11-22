# -*- coding:utf-8 -*-
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.data_utils import load_and_cache_examples, tag_to_id, get_chunks
from flashtool import Logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
# )
# logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
# logging_fh.setLevel(logging.DEBUG)
# logger.addHandler(logging_fh)
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#     args.local_rank,
#     device,
#     args.n_gpu,
#     bool(args.local_rank != -1),
#     args.fp16,
# )
def evaluate(args, type_model, tokenizer, id_to_label_span, id_to_label_type, \
    pad_token_label_id, best, mode, logger, gold_outs, prefix="", verbose=True):
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    span_to_id = {id_to_label_span[id_]:id_ for id_ in id_to_label_span}
    non_entity_id = span_to_id["O"]

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    span_ident = None
    type_model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1]} # span_labels: batch[2], type_labels: batch[3]
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "mobilebert"] else None
                # )  # XLM and RoBERTa don"t use segment_ids
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "lm_mask": batch[2], "labels": batch[3], "target": True}
            outputs = type_model(**inputs) # B*L, type_num, span_num
            logits = outputs[1] # B, C
            eval_loss += outputs[0]

        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach() # B, C
            span_ident = batch[4] # B, 3
        else:
            preds = torch.cat((preds, logits.detach()), dim=0)
            span_ident = torch.cat((span_ident, batch[4]), dim=0)

    pred_res = torch.argmax(preds, dim=-1) # N
    pred_outs = []
    cnt = 0
    for ident, p in zip(span_ident, pred_res):
        if id_to_label_type[p.item()] != "O":
            pred_outs.append((ident[0].item(), ident[1].item(), ident[2].item(), p.item())) # (sen_id, start, end, type_id)
        else:
            cnt += 1

    # gold_outs [(sen_id, start, end, type_id)]
    gold_outs = set(gold_outs)
    pred_outs = set(pred_outs)
    correct_preds = len(gold_outs & pred_outs)
    total_preds   = len(pred_outs)
    total_correct = len(gold_outs)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True

    results = {
       "loss": eval_loss.item()/nb_eval_steps,
       "precision": p,
       "recall": r,
       "f1": new_F,
       "best_precision": best[0],
       "best_recall": best[1],
       "best_f1": best[-1],
       "#O": cnt
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # logger.info("***** Meta Eval results %s *****", prefix)
    # for key in sorted(meta_results.keys()):
    #     logger.info("  %s = %s", key, str(meta_results[key]))

    return best, is_updated
