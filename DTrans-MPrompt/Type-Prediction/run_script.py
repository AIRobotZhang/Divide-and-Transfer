# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig
)

# from models.modeling_span import Span_Learner
from models.modeling_type import Type_Learner
from utils.data_utils import load_and_cache_examples, get_labels, get_gold_labels, tag_to_id
# from utils.model_utils import mask_tokens, soft_frequency, opt_grad, get_hard_label, _update_mean_model_variables
from utils.eval import evaluate
from utils.config import config
# from utils.loss_utils import CycleConsistencyLoss

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # "span": (Span_Learner, BertConfig, BertTokenizer),
    "type": (Type_Learner, BertConfig, BertTokenizer),
    # "finetune": (TokenClassification, BertConfig, BertTokenizer)
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, t_total, label_to_vocab_src, label_to_vocab_tgt, span_num_labels, type_num_labels):
    model_class, config_class, _ = MODEL_CLASSES["type"]

    if args.continu:
        type_model = model_class(args.tokenizer_name_or_path, label_to_vocab_src, label_to_vocab_tgt, args.device)
        type_model.load_state_dict(torch.load(args.type_model_name_or_path))

    else:
        type_model = model_class(args.type_model_name_or_path, label_to_vocab_src, label_to_vocab_tgt, args.device)
    type_model.to(args.device)
    # total = sum([param.nelement() for param in type_model.parameters()])
    # print(total)
    # exit()

    # model_class, config_class, _ = MODEL_CLASSES["type"]
    # # config_class, model_class, _ = MODEL_CLASSES["student1"]
    # # config_s1 = config_class.from_pretrained(
    # #     args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
    # #     num_labels=num_labels,
    # #     cache_dir=args.cache_dir if args.cache_dir else None,
    # # )
    # config = config_class.from_pretrained(
    #     args.type_model_name_or_path,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    # type_model = model_class.from_pretrained(
    #     args.type_model_name_or_path,
    #     config=config,
    #     type_num_labels=type_num_labels, 
    #     device=args.device,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    # type_model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    # optimizer_grouped_parameters_span = [
    #     {
    #         "params": [p for n, p in span_model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {"params": [p for n, p in span_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]
    # # print(optimizer_grouped_parameters_meta)
    # # for n, p in meta_model.named_parameters():
    # #     print(n)
    # # exit()
    # optimizer_span = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
    #         eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    # scheduler_span = get_linear_schedule_with_warmup(
    #     optimizer_span, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )

    optimizer_grouped_parameters_type = [
        {
            "params": [p for n, p in type_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in type_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_type = AdamW(optimizer_grouped_parameters_type, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_type = get_linear_schedule_with_warmup(
        optimizer_type, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     [span_model, type_model], [optimizer_span, optimizer_type] = amp.initialize(
    #                  [span_model, type_model], [optimizer_span, optimizer_type], opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # span_model = torch.nn.DataParallel(span_model)
        type_model = torch.nn.DataParallel(type_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # span_model = torch.nn.parallel.DistributedDataParallel(
        #     span_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        # )
        type_model = torch.nn.parallel.DistributedDataParallel(
            type_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # span_model.zero_grad()
    type_model.zero_grad()

    return type_model, optimizer_type, scheduler_type

def validation(args, type_model, tokenizer, id_to_label_span, id_to_label_type, pad_token_label_id, best_dev, best_test, \
         global_step, t_total, epoch, dev_golds, test_golds):
    best_dev, is_updated_dev = evaluate(args, type_model, tokenizer, \
        id_to_label_span, id_to_label_type, pad_token_label_id, best_dev, mode="dev", logger=logger, gold_outs=dev_golds,\
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    best_test, is_updated_test = evaluate(args, type_model, tokenizer, \
        id_to_label_span, id_to_label_type, pad_token_label_id, best_test, mode="test", logger=logger, gold_outs=test_golds,\
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    # output_dirs = []
    if args.local_rank in [-1, 0] and is_updated_dev:
        # updated_self_training_teacher = True
        path = os.path.join(args.output_dir, "checkpoint-best")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(type_model.state_dict(), path+"/pytorch_model.bin")
        # model_to_save = (
        #         span_model.module if hasattr(span_model, "module") else span_model
        # )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(path)
        # tokenizer.save_pretrained(path)
    # # output_dirs = []
    # if args.local_rank in [-1, 0] and is_updated2:
    #     # updated_self_training_teacher = True
    #     path = os.path.join(args.output_dir+tors, "checkpoint-best-2")
    #     logger.info("Saving model checkpoint to %s", path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     model_to_save = (
    #             model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(path)
    #     tokenizer.save_pretrained(path)

    return best_dev, best_test, is_updated_dev

def get_label_ind(tokenizer, tag_to_id):
    types = tag_to_id["type"]
    templates = tag_to_id["template"]
    label_to_map = torch.zeros(len(types))
    for t in types:
        ind = types[t]
        vocab_ind = tokenizer.convert_tokens_to_ids([templates[t][1]])[0]
        label_to_map[ind] = vocab_ind

    label_to_map = label_to_map.long()

    return label_to_map

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(args, train_dataset, train_dataset_tgt, tag_to_id_src, tag_to_id, tokenizer, pad_token_label_id):
    """ Train the model """
    # num_labels = len(labels)
    dev_golds = get_gold_labels(args, args.data_dir, tag_to_id, mode="dev")
    test_golds = get_gold_labels(args, args.data_dir, tag_to_id, mode="test")
    span_to_id = tag_to_id["span"]
    type_to_id = tag_to_id["type"]
    id_to_label_span = {span_to_id[s]:s for s in span_to_id}
    id_to_label_type = {type_to_id[t]:t for t in type_to_id}
    span_num_labels = len(id_to_label_span)
    type_num_labels = len(id_to_label_type)-1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.train_batch_size_tgt = args.per_gpu_train_batch_size_tgt * max(1, args.n_gpu)
    train_sampler_tgt = RandomSampler(train_dataset_tgt) if args.local_rank==-1 else DistributedSampler(train_dataset_tgt)
    train_dataloader_tgt = DataLoader(train_dataset_tgt, sampler=train_sampler_tgt, batch_size=args.train_batch_size_tgt)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs
    label_to_vocab_src = get_label_ind(tokenizer, tag_to_id_src)
    label_to_vocab_tgt = get_label_ind(tokenizer, tag_to_id)
    type_model, optimizer_type, scheduler_type = initialize(args, t_total, label_to_vocab_src, label_to_vocab_tgt, span_num_labels, type_num_labels)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]

    # begin_global_step = len(train_dataloader)*args.begin_epoch//args.gradient_accumulation_steps
    iterator_tgt = iter(cycle(train_dataloader_tgt))
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            type_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            batch_tgt = next(iterator_tgt)
            batch_tgt = tuple(t.to(args.device) for t in batch_tgt)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "lm_mask": batch[2], "labels": batch[3], "target": False} # all_input_ids, all_input_mask, all_lm_mask, all_label_id, all_span_id, all_ids
            # span_logits = span_model(**inputs) # B*L, type_num, span_num
            outputs = type_model(**inputs) # B, L, type_num+1

            inputs = {"input_ids": batch_tgt[0], "attention_mask": batch_tgt[1], "lm_mask": batch_tgt[2], "labels": batch_tgt[3], "target": True}
            # span_logits_meta = span_model(**inputs) # B*L, type_num, span_num
            outputs_tgt = type_model(**inputs) # B, L, type_num+1

            loss = outputs[0] + outputs_tgt[0]
            # loss = outputs_tgt[0]

            # loss = Loss(epoch, batch[4], batch_meta[4], span_logits, type_logits, span_logits_meta, type_logits_meta, batch_meta[2], batch_meta[3])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer_span) as scaled_loss:
            #         scaled_loss.backward()
            #     with amp.scale_loss(loss, optimizer_type) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_span), args.max_grad_norm)
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_type), args.max_grad_norm)
                # else:
                #     torch.nn.utils.clip_grad_norm_(span_model.parameters(), args.max_grad_norm)
                #     torch.nn.utils.clip_grad_norm_(type_model.parameters(), args.max_grad_norm)

                # optimizer_span.step()
                optimizer_type.step()
                # scheduler_span.step()  # Update learning rate schedule
                scheduler_type.step()
                # span_model.zero_grad()
                type_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f *****", loss.item())
                        best_dev, best_test, _ = validation(args, type_model, tokenizer, \
                            id_to_label_span, id_to_label_type, pad_token_label_id, best_dev, best_test, \
                            global_step, t_total, epoch, dev_golds, test_golds)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return results

def main():
    args = config()
    args.do_train = args.do_train.lower()
    args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    tag_to_id_src = tag_to_id(args.data_dir, args.dataset_src)
    tag_to_id_tgt = tag_to_id(args.data_dir, args.dataset)
    # id_to_label_span, id_to_label_type, non_entity_id = get_labels(args.data_dir, args.dataset)
    # num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = MODEL_CLASSES["type"][2].from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Loss = CycleConsistencyLoss(non_entity_id, args.device)

    # Training
    if args.do_train=="true":
        train_dataset, train_dataset_tgt = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        # train(args, train_dataset, train_dataset_meta, tag_to_id, tokenizer, pad_token_label_id)
        best_results = train(args, train_dataset, train_dataset_tgt, tag_to_id_src,\
            tag_to_id_tgt, tokenizer, pad_token_label_id)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # # Testing
    if args.do_test=="true" and args.local_rank in [-1, 0]:
        predict(args, pad_token_label_id, tag_to_id_src, tag_to_id_tgt, mode=args.eval.lower())

def predict(args, pad_token_label_id, tag_to_id_src, tag_to_id_tgt, mode="test"):
    path = os.path.join(args.output_dir, "checkpoint-best/pytorch_model.bin")
    tokenizer = MODEL_CLASSES["type"][2].from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    label_to_vocab_src = get_label_ind(tokenizer, tag_to_id_src)
    label_to_vocab_tgt = get_label_ind(tokenizer, tag_to_id_tgt)
    type_model = MODEL_CLASSES["type"][0](args.type_model_name_or_path, label_to_vocab_src, label_to_vocab_tgt, args.device)
    type_model.load_state_dict(torch.load(path))
    type_model.to(args.device)

    _, eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    gold_outs = get_gold_labels(args, args.data_dir, tag_to_id_tgt, mode=mode)
    # span_to_id = tag_to_id["span"]
    type_to_id = tag_to_id_tgt["type"]
    id_to_label_type = {type_to_id[t]:t for t in type_to_id}

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", mode)
    # if verbose:
    #     logger.info("  Num examples = %d", len(eval_dataset))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    span_ident = None
    type_model.eval()
    # from datetime import datetime
    # s = datetime.now()
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
    # e = datetime.now()
    # print(e-s)
    # exit()

    preds = torch.softmax(preds,dim=-1)
    pred_prob, pred_res = torch.max(preds, dim=-1) # N
    pred_outs = []
    for ident, p, v in zip(span_ident, pred_res, pred_prob):
        if id_to_label_type[p.item()] != "O" and v > args.delta:
            pred_outs.append((ident[0].item(), ident[1].item(), ident[2].item(), p.item())) # (sen_id, start, end, type_id)
    # gold_outs [(sen_id, start, end, type_id)]
    gold_outs = set(gold_outs)
    pred_outs = set(pred_outs)
    correct_preds = len(gold_outs & pred_outs)
    total_preds   = len(pred_outs)
    total_correct = len(gold_outs)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    # is_updated = False
    # if new_F > best[-1]:
    #     best = [p, r, new_F]
    #     is_updated = True

    results = {
       "loss": eval_loss.item()/nb_eval_steps,
       "precision": p,
       "recall": r,
       "f1": new_F
    }

    logger.info("***** Eval results %s *****", mode)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # # if not best_test:
   
    # # result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
    # result, _, best_test, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
    #                                                     logger=logger, verbose=False)
    # # Save results
    # output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    # with open(output_test_results_file, "w") as writer:
    #     for key in sorted(result.keys()):
    #         writer.write("{} = {}\n".format(key, str(result[key])))

    # return best_test
    # Save predictions
    # output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    # with open(output_test_predictions_file, "w") as writer:
    #     with open(os.path.join(args.data_dir, args.dataset+"_test.json"), "r") as f:
    #         example_id = 0
    #         data = json.load(f)
    #         for item in data: # original tag_ro_id must be {XXX:0, xxx:1, ...}
    #             tags = item["tags"]
    #             golden_labels = [labels[tag] for tag in tags]
    #             output_line = str(item["str_words"]) + "\n" + str(golden_labels)+"\n"+str(predictions[example_id]) + "\n"
    #             writer.write(output_line)
    #             example_id += 1

if __name__ == "__main__":
    main()
