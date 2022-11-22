# -*- coding:utf-8 -*-
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.data_utils import load_and_cache_examples, tag_to_id, get_chunks, get_chunks_tb, get_chunks_se
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


def evaluate(args, span_model, tokenizer, id_to_label_span, \
    pad_token_label_id, best, mode, logger, prefix="", verbose=True):
    
    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)
    # input_ids, input_mask, span_label_ids = eval_dataset
    # eval_dataset = TensorDataset(input_ids, input_mask, span_label_ids)
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
    preds_bio = None
    preds_se_s = None
    preds_se_e = None
    preds_tb = None
    out_labels = None
    sen_ids = None
    span_model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1]} # span_labels: batch[2], type_labels: batch[3]
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "mobilebert"] else None
                # )  # XLM and RoBERTa don"t use segment_ids
            inputs = {"input_ids": batch[0], 
                      "attention_mask": batch[1], 
                      "labels_bio": batch[2],
                      "start_labels": batch[3],
                      "end_labels": batch[4],
                      "tb_labels": batch[5]
                    }
            outputs = span_model(**inputs) # B*L, type_num, span_num
            logits_bio = outputs[6] # B, L, C
            logits_se_s = outputs[3]
            logits_se_e = outputs[4]
            logits_tb = outputs[1]
            eval_loss += outputs[0]+outputs[2]+outputs[5]

        nb_eval_steps += 1
        
        if preds_bio is None:
            preds_bio = logits_bio.detach() # B, L, C
            preds_se_s = logits_se_s.detach()
            preds_se_e = logits_se_e.detach()
            preds_tb = logits_tb.detach()
            out_labels = batch[2] # B, L
            # sen_ids = batch[5] # B
        else:
            preds_bio = torch.cat((preds_bio, logits_bio.detach()), dim=0)
            preds_se_s = torch.cat((preds_se_s, logits_se_s.detach()), dim=0)
            preds_se_e = torch.cat((preds_se_e, logits_se_e.detach()), dim=0)
            preds_tb = torch.cat((preds_tb, logits_tb.detach()), dim=0)
            out_labels = torch.cat((out_labels, batch[2]), dim=0)
            # sen_ids = torch.cat((sen_ids, batch[5]), dim=0)

    preds_bio = torch.argmax(preds_bio, dim=-1) # N, L
    preds_tb = torch.argmax(preds_tb, dim=-1) # N, L
    preds_se_s = torch.argmax(preds_se_s, dim=-1) # N, L
    preds_se_e = torch.argmax(preds_se_e, dim=-1) # N, L

    # out_id_list_type = [[] for _ in range(out_labels.shape[0])]
    # preds_id_list_type = [[] for _ in range(out_labels.shape[0])]

    out_id_list_bio = [[] for _ in range(out_labels.shape[0])]
    preds_id_list_bio = [[] for _ in range(out_labels.shape[0])]
    preds_id_list_tb = [[] for _ in range(out_labels.shape[0])]
    preds_id_list_se_s = [[] for _ in range(out_labels.shape[0])]
    preds_id_list_se_e = [[] for _ in range(out_labels.shape[0])]

    for i in range(out_labels.shape[0]):
        for j in range(out_labels.shape[1]):
            if out_labels[i, j] != pad_token_label_id:
                out_id_list_bio[i].append(out_labels[i][j])
                preds_id_list_bio[i].append(preds_bio[i][j])
                preds_id_list_tb[i].append(preds_tb[i][j])
                preds_id_list_se_s[i].append(preds_se_s[i][j])
                preds_id_list_se_e[i].append(preds_se_e[i][j])

    correct_preds_bio, total_correct_bio, total_preds_bio = 0., 0., 0. # i variables
    tags_tb = {"Tie": 0, "Break": 1}
    tags_se = {"No-SE": 0, "SE": 1}
    # print("EVAL:")
    for ground_truth_id_bio, predicted_id_bio, predicted_id_tb, predicted_id_s, predicted_id_e in zip(out_id_list_bio, preds_id_list_bio, preds_id_list_tb, preds_id_list_se_s, preds_id_list_se_e):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks_bio = get_chunks(ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("ground_truth:")
        # print(lab_chunks_bio)
        # print(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)

        lab_pred_chunks_bio = get_chunks(predicted_id_bio, tag_to_id(args.data_dir, args.dataset))
        lab_pred_chunks_tb = get_chunks_tb(predicted_id_tb, tags_tb)
        lab_pred_chunks_se = get_chunks_se(predicted_id_s, predicted_id_e, tags_se)
        # print("pred:")
        # print(lab_pred_chunks_bio)
        # print(lab_pred_chunks)
        lab_pred_chunks_bio = set(lab_pred_chunks_bio+lab_pred_chunks_se+lab_pred_chunks_tb)
        # lab_pred_chunks_bio = set(lab_pred_chunks_bio)


        # Updating the i variables
        correct_preds_bio += len(lab_chunks_bio & lab_pred_chunks_bio)
        total_preds_bio   += len(lab_pred_chunks_bio)
        total_correct_bio += len(lab_chunks_bio)

    p_ens   = correct_preds_bio / total_preds_bio if correct_preds_bio > 0 else 0
    r_ens   = correct_preds_bio / total_correct_bio if correct_preds_bio > 0 else 0
    new_F_ens  = 2 * p_ens * r_ens / (p_ens + r_ens) if correct_preds_bio > 0 else 0

    # is_updated = False
    if new_F_ens > best["Ens"][-1]: # best: {"BIO":[p,r,f], "Start-End":[p,r,f], "Tie-Break":[p,r,f]}
        best["Ens"] = [p_ens, r_ens, new_F_ens]
        # if new_F_ens > 0.8 and new_F_ens < 0.85:
        #     is_updated = True

    # results = {
    #    "loss": eval_loss.item()/nb_eval_steps,
    #    "ens-precision": p_bio,
    #    "ens-recall": r_bio,
    #    "ens-f1": new_F_bio,
    #    "ens-best_precision": best["BIO"][0],
    #    "ens-best_recall": best["BIO"][1],
    #    "ens-best_f1": best["BIO"][-1],
    # }


    # correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    correct_preds_bio, total_correct_bio, total_preds_bio = 0., 0., 0. # i variables
    # print("EVAL:")
    for ground_truth_id_bio, predicted_id_bio in zip(out_id_list_bio, preds_id_list_bio):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks_bio = get_chunks(ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("ground_truth:")
        # print(lab_chunks_bio)
        # print(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)
        lab_pred_chunks_bio = get_chunks(predicted_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("pred:")
        # print(lab_pred_chunks_bio)
        # print(lab_pred_chunks)
        lab_pred_chunks_bio = set(lab_pred_chunks_bio)

        # Updating the i variables
        correct_preds_bio += len(lab_chunks_bio & lab_pred_chunks_bio)
        total_preds_bio   += len(lab_pred_chunks_bio)
        total_correct_bio += len(lab_chunks_bio)
    # pred_outs = []
    # for ident, p in zip(span_ident, pred_res):
    #     if id_to_label_type[p.item()] != "O":
    #         pred_outs.append((ident[0].item(), ident[1].item(), ident[2].item(), p.item())) # (sen_id, start, end, type_id)
    # # gold_outs [(sen_id, start, end, type_id)]
    # gold_outs = set(gold_outs)
    # pred_outs = set(pred_outs)
    # correct_preds = len(gold_outs & pred_outs)
    # total_preds   = len(pred_outs)
    # total_correct = len(gold_outs)

    p_bio   = correct_preds_bio / total_preds_bio if correct_preds_bio > 0 else 0
    r_bio   = correct_preds_bio / total_correct_bio if correct_preds_bio > 0 else 0
    new_F_bio  = 2 * p_bio * r_bio / (p_bio + r_bio) if correct_preds_bio > 0 else 0

    is_updated = False
    if new_F_bio > best["BIO"][-1]: # best: {"BIO":[p,r,f], "Start-End":[p,r,f], "Tie-Break":[p,r,f]}
        best["BIO"] = [p_bio, r_bio, new_F_bio]
        is_updated = True
        # if new_F_bio > 0.8 and new_F_bio < 0.85:
        #     is_updated = True

    correct_preds_tb, total_correct_tb, total_preds_tb = 0., 0., 0. # i variables
    # print("EVAL:")
    tags_tb = {"Tie": 0, "Break": 1}
    for ground_truth_id_bio, predicted_id_tb in zip(out_id_list_bio, preds_id_list_tb):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks_bio = get_chunks(ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("ground_truth:")
        # print(lab_chunks_bio)
        # print(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)
        # print(lab_chunks_bio)
        lab_pred_chunks_tb = get_chunks_tb(predicted_id_tb, tags_tb)
        # lab_pred_chunks_bio = get_chunks(predicted_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("pred:")
        # print(lab_pred_chunks_bio)
        # print(lab_pred_chunks)
        lab_pred_chunks_tb = set(lab_pred_chunks_tb)
        # print(lab_pred_chunks_tb)

        # Updating the i variables
        correct_preds_tb += len(lab_chunks_bio & lab_pred_chunks_tb)
        total_preds_tb   += len(lab_pred_chunks_tb)
        total_correct_tb += len(lab_chunks_bio)

    p_tb   = correct_preds_tb / total_preds_tb if correct_preds_tb > 0 else 0
    r_tb   = correct_preds_tb / total_correct_tb if correct_preds_tb > 0 else 0
    new_F_tb  = 2 * p_tb * r_tb / (p_tb + r_tb) if correct_preds_tb > 0 else 0

    # is_updated = False
    if new_F_tb > best["Tie-Break"][-1]: # best: {"BIO":[p,r,f], "Start-End":[p,r,f], "Tie-Break":[p,r,f]}
        best["Tie-Break"] = [p_tb, r_tb, new_F_tb]
        # is_updated = True

    correct_preds_se, total_correct_se, total_preds_se = 0., 0., 0. # i variables
    # print("EVAL:")
    tags_se = {"No-SE": 0, "SE": 1}
    for ground_truth_id_bio, predicted_id_s, predicted_id_e in zip(out_id_list_bio, preds_id_list_se_s, preds_id_list_se_e):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks_bio = get_chunks(ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("ground_truth:")
        # print(lab_chunks_bio)
        # print(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)
        lab_pred_chunks_se = get_chunks_se(predicted_id_s, predicted_id_e, tags_se)
        # lab_pred_chunks_bio = get_chunks(predicted_id_bio, tag_to_id(args.data_dir, args.dataset))
        # print("pred:")
        # print(lab_pred_chunks_bio)
        # print(lab_pred_chunks)
        lab_pred_chunks_se = set(lab_pred_chunks_se)

        # Updating the i variables
        correct_preds_se += len(lab_chunks_bio & lab_pred_chunks_se)
        total_preds_se   += len(lab_pred_chunks_se)
        total_correct_se += len(lab_chunks_bio)

    p_se   = correct_preds_se / total_preds_se if correct_preds_se > 0 else 0
    r_se   = correct_preds_se / total_correct_se if correct_preds_se > 0 else 0
    new_F_se  = 2 * p_se * r_se / (p_se + r_se) if correct_preds_se > 0 else 0

    # is_updated = False
    if new_F_se > best["Start-End"][-1]: # best: {"BIO":[p,r,f], "Start-End":[p,r,f], "Tie-Break":[p,r,f]}
        best["Start-End"] = [p_se, r_se, new_F_se]

    results = {
       "loss": eval_loss.item()/nb_eval_steps,
       "BIO-precision": p_bio,
       "BIO-recall": r_bio,
       "BIO-f1": new_F_bio,
       "BIO-best_precision": best["BIO"][0],
       "BIO-best_recall": best["BIO"][1],
       "BIO-best_f1": best["BIO"][-1],
       "SE-precision": p_se,
       "SE-recall": r_se,
       "SE-f1": new_F_se,
       "SE-best_precision": best["Start-End"][0],
       "SE-best_recall": best["Start-End"][1],
       "SE-best_f1": best["Start-End"][-1],
       "TB-precision": p_tb,
       "TB-recall": r_tb,
       "TB-f1": new_F_tb,
       "TB-best_precision": best["Tie-Break"][0],
       "TB-best_recall": best["Tie-Break"][1],
       "TB-best_f1": best["Tie-Break"][-1],
       "Ens-precision": p_ens,
       "Ens-recall": r_ens,
       "Ens-f1": new_F_ens,
       "Ens-best_precision": best["Ens"][0],
       "Ens-best_recall": best["Ens"][1],
       "Ens-best_f1": best["Ens"][-1],
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # logger.info("***** Meta Eval results %s *****", prefix)
    # for key in sorted(meta_results.keys()):
    #     logger.info("  %s = %s", key, str(meta_results[key]))

    return best, is_updated
