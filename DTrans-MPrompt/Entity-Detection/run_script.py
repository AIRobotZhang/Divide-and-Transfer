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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
import pdb
import copy

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig,
    BertForMaskedLM
)

from models.modeling_span import Span_Detector
from utils.data_utils import load_and_cache_examples, get_labels, tag_to_id, get_chunks, get_chunks_tb, get_chunks_se
from utils.model_utils import _update_momentum_model, soft_frequency, mask_tokens, aug, update_parameters, functional_bert
from utils.eval import evaluate
from utils.config import config
from utils.loss_utils import get_current_consistency_weight

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "span": (Span_Detector, BertConfig, BertTokenizer),
    # "type": (Type_Learner, BertConfig, BertTokenizer),
    # "finetune": (TokenClassification, BertConfig, BertTokenizer)
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, t_total, span_num_labels):
    model_class, config_class, _ = MODEL_CLASSES["span"]

    config = config_class.from_pretrained(
        args.span_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model = model_class.from_pretrained(
        args.span_model_name_or_path,
        config=config,
        span_num_labels=span_num_labels, 
        se_soft_label=args.soft_label,
        loss_type=args.loss_type,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model.to(args.device)
    # total = sum([param.nelement() for param in span_model.parameters()])
    # print(total)

    span_model_momentum = model_class.from_pretrained(
        args.span_model_name_or_path,
        config=config,
        span_num_labels=span_num_labels, 
        se_soft_label=args.soft_label,
        loss_type=args.loss_type,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model_momentum.to(args.device)
    # total = sum([param.nelement() for param in span_model_momentum.parameters()])
    # print(total)
    # exit()

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer_span = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span = get_linear_schedule_with_warmup(
        optimizer_span, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
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
        span_model = torch.nn.DataParallel(span_model)
        span_model_momentum = torch.nn.DataParallel(span_model_momentum)

    for param in span_model_momentum.parameters():
        param.detach_()

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # span_model = torch.nn.parallel.DistributedDataParallel(
        #     span_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        # )
        span_model = torch.nn.parallel.DistributedDataParallel(
            span_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        span_model_momentum = torch.nn.parallel.DistributedDataParallel(
            span_model_momentum, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    span_model.zero_grad()
    span_model_momentum.zero_grad()

    return span_model, optimizer_span, scheduler_span, span_model_momentum, config

def validation(args, span_model, tokenizer, id_to_label_span, pad_token_label_id, best_dev, best_test, \
         global_step, t_total, epoch):
    best_dev, is_updated_dev = evaluate(args, span_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_dev, mode="dev", logger=logger,\
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    best_test, is_updated_test = evaluate(args, span_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_test, mode="test", logger=logger,\
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    # output_dirs = []
    if args.local_rank in [-1, 0] and is_updated_dev:
        # updated_self_training_teacher = True
        path = os.path.join(args.output_dir, "checkpoint-best")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        # torch.save(type_model.state_dict(), path+"/pytorch_model.bin")
        model_to_save = (
                span_model.module if hasattr(span_model, "module") else span_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
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

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def multi_level_train(args, train_dataset, train_dataset_src, tag_to_id, tokenizer, pad_token_label_id):
    span_to_id = tag_to_id["span"]
    id_to_label_span = {span_to_id[s]:s for s in span_to_id}
    span_num_labels = len(id_to_label_span)

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # aug_input_ids = aug(args, mlm_model, train_dataset, mask_id, inv_vocab, mask_prob=0.15)
    # train_dataset_ = TensorDataset(input_ids, input_mask, span_label_ids, aug_input_ids)
    # train_sampler = RandomSampler(train_dataset_) if args.local_rank==-1 else DistributedSampler(train_dataset_)
    # train_dataloader = DataLoader(train_dataset_, sampler=train_sampler, batch_size=args.train_batch_size)
    args.train_batch_size_src = args.per_gpu_train_batch_size_src * max(1, args.n_gpu)
    train_sampler_src = RandomSampler(train_dataset_src) if args.local_rank==-1 else DistributedSampler(train_dataset_src)
    train_dataloader_src = DataLoader(train_dataset_src, sampler=train_sampler_src, batch_size=args.train_batch_size_src)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    data_size = len(train_dataloader_src)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(data_size//args.gradient_accumulation_steps)+1
    else:
        t_total = data_size//args.gradient_accumulation_steps*args.num_train_epochs

    span_model, optimizer, scheduler, span_model_momentum, _ = initialize(
        args, t_total, span_num_labels)
    logger.info("***** Running training*****")
    logger.info("  Num examples = %d", len(train_dataset_src))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size_src)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size_src
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev = {"BIO":[0.0, 0.0, 0.0], "Start-End":[0.0, 0.0, 0.0], "Tie-Break":[0.0, 0.0, 0.0], "Ens":[0.0, 0.0, 0.0]}
    best_test = {"BIO":[0.0, 0.0, 0.0], "Start-End":[0.0, 0.0, 0.0], "Tie-Break":[0.0, 0.0, 0.0], "Ens":[0.0, 0.0, 0.0]}

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    iterator = iter(cycle(train_dataloader))
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader_src, desc="Iteration", disable=args.local_rank not in [-1, 0])
        consistency_weight = get_current_consistency_weight(args, epoch)
        for step, batch_src in enumerate(epoch_iterator):
            span_model.train()
            span_model_momentum.train()
            batch_src = tuple(t.to(args.device) for t in batch_src)

            batch = next(iterator)
            batch = tuple(t.to(args.device) for t in batch)

            ####################################################
            inputs_aug = {"input_ids": batch[0], 
                      "attention_mask": batch[1], 
                      # "input_ids_aug": batch[2], 
                      # "labels_bio": batch[3],
                } #
            outputs = span_model_momentum(**inputs_aug)
            logits_aug_bio = outputs[3].detach() # B, L, C
            logits_aug_start = outputs[1].detach() # B, L, C
            logits_aug_end = outputs[2].detach() # B, L, C
            logits_aug_tb = outputs[0].detach() # B, L, C

            inputs = {"input_ids": batch[0], 
                      "attention_mask": batch[1], 
                      # "input_ids_aug": batch[2], 
                      "labels_bio": batch[2],
                      "soft_label_bio": logits_aug_bio,
                      "soft_label_start": logits_aug_start,
                      "soft_label_end": logits_aug_end,
                      "soft_label_tb": logits_aug_tb,
                      "loss_weight": consistency_weight,
                      "start_labels": batch[3],
                      "end_labels": batch[4],
                      "tb_labels": batch[5]
                } #

            outputs = span_model(**inputs)
            loss_bio = outputs[5]
            loss_se = outputs[2]
            loss_tb = outputs[0]

            inputs_src = {"input_ids": batch_src[0], 
                      "attention_mask": batch_src[1], 
                      # "input_ids_aug": batch[2], 
                      "labels_bio": batch_src[2],
                      # "soft_label": logits_aug,
                      # "loss_weight": consistency_weight,
                      "start_labels": batch_src[3],
                      "end_labels": batch_src[4],
                      "tb_labels": batch_src[5]
                }

            outputs_src = span_model(**inputs_src)
            loss_bio_src = outputs_src[5]
            loss_se_src = outputs_src[2]
            loss_tb_src = outputs_src[0]


            loss = loss_bio+loss_se+loss_tb+loss_bio_src+loss_se_src+loss_tb_src
            # loss = loss_bio+loss_bio_src
            # loss = loss_tb+loss_tb_src
            # loss = loss_bio+loss_se+loss_tb

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

                optimizer.step()
                # Update learning rate schedule
                scheduler.step()
                span_model.zero_grad()
                global_step += 1
                _update_momentum_model(span_model, span_model_momentum, args.mean_alpha, global_step)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f *****", loss.item())
                        best_dev, best_test, _ = validation(args, span_model, tokenizer, \
                            id_to_label_span, pad_token_label_id, best_dev, best_test, \
                            global_step, t_total, epoch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return results


def train(args, train_dataset, train_dataset_src, tag_to_id, tokenizer, pad_token_label_id):
    best_results = multi_level_train(
        args, 
        train_dataset, 
        train_dataset_src, 
        tag_to_id, 
        tokenizer, 
        pad_token_label_id
    )

    return best_results

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

    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(
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
        train_dataset_src, train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train(args, train_dataset, train_dataset_src, tag_to_id_tgt, tokenizer, pad_token_label_id)
    # Testing
    if args.do_test=="true" and args.local_rank in [-1, 0]:
        predict(args, pad_token_label_id, len(tag_to_id_tgt["span"]), mode=args.eval.lower())

def predict(args, pad_token_label_id, span_num_labels, mode="test"):
    file_path = os.path.join(args.data_dir, "{}_{}.json".format(args.dataset, mode))
    with open(file_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    path = os.path.join(args.output_dir, "checkpoint-best")
    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    span_model = MODEL_CLASSES["span"][0].from_pretrained(path, span_num_labels=span_num_labels, 
        se_soft_label=args.soft_label,
        loss_type=args.loss_type,device=args.device)
    span_model.to(args.device)
    id_to_label_span = {0:"B", 1:"I", 2:"O"}

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

    logger.info("***** Running evaluation %s *****", mode)
    # if verbose:
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
            sen_ids = batch[-1] # B
        else:
            preds_bio = torch.cat((preds_bio, logits_bio.detach()), dim=0)
            preds_se_s = torch.cat((preds_se_s, logits_se_s.detach()), dim=0)
            preds_se_e = torch.cat((preds_se_e, logits_se_e.detach()), dim=0)
            preds_tb = torch.cat((preds_tb, logits_tb.detach()), dim=0)
            out_labels = torch.cat((out_labels, batch[2]), dim=0)
            sen_ids = torch.cat((sen_ids, batch[-1]), dim=0)

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
    res = []
    for ground_truth_id_bio, predicted_id_bio, predicted_id_tb, predicted_id_s, predicted_id_e, sid in zip(out_id_list_bio, preds_id_list_bio, preds_id_list_tb, preds_id_list_se_s, preds_id_list_se_e, sen_ids):
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
        # lab_pred_chunks_bio = set(lab_pred_chunks_bio+lab_pred_chunks_tb+lab_pred_chunks_se)
        lab_pred_chunks_bio = set(lab_pred_chunks_bio)
        # lab_pred_chunks_bio = set(lab_pred_chunks_se)
        res.append({"id":sid.item(), "spans": list(lab_pred_chunks_bio)})


        # Updating the i variables
        correct_preds_bio += len(lab_chunks_bio & lab_pred_chunks_bio)
        total_preds_bio   += len(lab_pred_chunks_bio)
        total_correct_bio += len(lab_chunks_bio)
    # print(res)
    with open(os.path.join(args.output_dir, mode+"_pred_spans.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    p_bio   = correct_preds_bio / total_preds_bio if correct_preds_bio > 0 else 0
    r_bio   = correct_preds_bio / total_correct_bio if correct_preds_bio > 0 else 0
    new_F_bio  = 2 * p_bio * r_bio / (p_bio + r_bio) if correct_preds_bio > 0 else 0
    best = {"BIO":[0.0, 0.0, 0.0], "Start-End":[0.0, 0.0, 0.0], "Tie-Break":[0.0, 0.0, 0.0], "Ens":[0.0, 0.0, 0.0]}


    is_updated = False
    if new_F_bio > best["BIO"][1]: # best: {"BIO":[p,r,f], "Start-End":[p,r,f], "Tie-Break":[p,r,f]}
        best["BIO"] = [p_bio, r_bio, new_F_bio]
        is_updated = True

    results = {
       "loss": eval_loss.item()/nb_eval_steps,
       "ens-precision": p_bio,
       "ens-recall": r_bio,
       "ens-f1": new_F_bio,
       "ens-best_precision": best["BIO"][0],
       "ens-best_recall": best["BIO"][1],
       "ens-best_f1": best["BIO"][-1],
    }

    logger.info("***** Eval results %s *****", mode)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

if __name__ == "__main__":
    main()
