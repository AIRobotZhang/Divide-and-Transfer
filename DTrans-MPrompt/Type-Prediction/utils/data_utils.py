# -*- coding:utf-8 -*
import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, entity, type_label, span):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.entity = entity
        self.type_label = type_label
        self.span = span

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, lm_mask, label_id, span_id):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.lm_mask = lm_mask
        self.label_id = label_id
        self.span_id = span_id

def read_examples_from_file(args, data_dir, mode):
    file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            sid = item["id"]
            words = item["str_words"]
            labels_ner = item["tags_ner_pred"] # [(start, end, type), ...]
            # labels_esi = item["tags_esi"]
            # labels_net = item["tags_net"]
            for ner in labels_ner:
                entity = words[ner[0]:ner[1]]
                span = [sid, ner[0], ner[1]]
                examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, entity=entity, type_label=ner[2], span=span))
                guid_index += 1
    examples_src = []
    if mode == "train":
        file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset_src, mode))
        guid_index = 1
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                sid = item["id"]
                words = item["str_words"]
                labels_ner = item["tags_ner_pred"]
                # labels_esi = item["tags_esi"]
                # labels_net = item["tags_net"]
                for ner in labels_ner:
                    entity = words[ner[0]:ner[1]]
                    span = [sid, ner[0], ner[1]]
                    examples_src.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, entity=entity, type_label=ner[2], span=span))
                    guid_index += 1
    
    return examples, examples_src

def convert_examples_to_features(
    tag_to_id,
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = -1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    span_non_id = tag_to_id["span"]["O"]
    type_non_id = tag_to_id["type"]["O"]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        span_label_ids = []
        type_label_ids = []
        # print(len(example.words), len(example.labels))
        for word in example.words:
            # print(word, label)
            # label_id = tag_to_id["span"][span_label]
            # type_label = tag_to_id["type"][type_label]
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # if len(word_tokens) > 0:
            #     # span_label_ids.extend([span_label] + [span_non_id] * (len(word_tokens) - 1))
            #     # type_label_ids.extend([type_label] + [type_non_id] * (len(word_tokens) - 1))
            #     label_mask.extend([-1]*len(word_tokens))
            # full_label_ids.extend([label] * len(word_tokens))

        # print(len(tokens), len(label_ids), len(full_label_ids))
        type_label = example.type_label
        label_id = tag_to_id["type"][type_label]
        tpls = tag_to_id["template"][type_label]
        entity = example.entity
        template_tokens = []
        lm_mask = []
        for word in entity:
            word_tokens = tokenizer.tokenize(word)
            template_tokens.extend(word_tokens)
            lm_mask.extend([-1]*len(word_tokens))
        for word in tpls[0].split():
            word_tokens = tokenizer.tokenize(word)
            template_tokens.extend(word_tokens)
            lm_mask.extend([-1]*len(word_tokens))
        template_tokens.append(tokenizer.mask_token)
        lm_mask.append(1)
        for word in tpls[2].split():
            word_tokens = tokenizer.tokenize(word)
            template_tokens.extend(word_tokens)
            lm_mask.extend([-1]*len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        if len(tokens) + len(template_tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count - len(template_tokens))]
            # span_label_ids = span_label_ids[: (max_seq_length - special_tokens_count)]
            # type_label_ids = type_label_ids[: (max_seq_length - special_tokens_count)]
            # label_mask = label_mask[: (max_seq_length - special_tokens_count - len(template_tokens))]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        lm_mask = [-1]*len(tokens) + lm_mask
        tokens += template_tokens
        tokens += [sep_token]
        lm_mask.append(-1)
        # span_label_ids += [span_non_id]
        # type_label_ids += [type_non_id]
        # label_mask += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            lm_mask.append(-1)
            # span_label_ids += [span_non_id]
            # type_label_ids += [type_non_id]
            # label_mask += [0]
        # segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            # span_label_ids += [span_non_id]
            # type_label_ids += [type_non_id]
            # segment_ids += [cls_token_segment_id]
            lm_mask.append(-1)
        else:
            tokens = [cls_token] + tokens
            # span_label_ids = [span_non_id] + span_label_ids
            # type_label_ids = [type_non_id] + type_label_ids
            # segment_ids = [cls_token_segment_id] + segment_ids
            lm_mask = [-1] + lm_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            # segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # span_label_ids = ([span_non_id] * padding_length) + span_label_ids
            # type_label_ids = ([type_non_id] * padding_length) + type_label_ids
            lm_mask = ([-1] * padding_length) + lm_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            # segment_ids += [pad_token_segment_id] * padding_length
            # span_label_ids += [span_non_id] * padding_length
            # type_label_ids += [type_non_id] * padding_length
            lm_mask += [-1] * padding_length
        
        # print(len(input_ids))
        # print(len(label_ids))
        # print(max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        # assert len(span_label_ids) == max_seq_length
        # assert len(type_label_ids) == max_seq_length
        assert len(lm_mask) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            # logger.info("span_label_ids: %s", " ".join([str(x) for x in span_label_ids]))
            # logger.info("type_label_ids: %s", " ".join([str(x) for x in type_label_ids]))
            logger.info("lm_mask: %s", " ".join([str(x) for x in lm_mask]))
        # input_ids, input_mask, segment_ids, label_ids, full_label_ids, span_label_ids, type_label_ids
        span_id = example.span
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, lm_mask=lm_mask, label_id=label_id, span_id=span_id)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    
    return features

def load_and_cache_examples(args, tokenizer, pad_token_label_id, mode):

    tags_to_id = tag_to_id(args.data_dir, args.dataset)
    tags_to_id_src = tag_to_id(args.data_dir, args.dataset_src)
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "{}_{}.pt".format(
            args.dataset, mode
        ),
    )

    cached_features_file_src = None

    if mode == "train":
        cached_features_file_src = os.path.join(
            args.data_dir,
            "{}_{}.pt".format(
                args.dataset_src, mode
            ),
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if mode == "train":
            logger.info("Loading source features from cached file %s", cached_features_file_src)
            features_src = torch.load(cached_features_file_src)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples, examples_src = read_examples_from_file(args, args.data_dir, mode)
        features = convert_examples_to_features(
            tags_to_id,
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
            sep_token = tokenizer.sep_token,
            sep_token_extra = bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left = bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id = pad_token_label_id,
        )

        features_src = convert_examples_to_features(
            tags_to_id_src,
            examples_src,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end = bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token = tokenizer.cls_token,
            cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0,
            sep_token = tokenizer.sep_token,
            sep_token_extra = bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left = bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id = pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            if mode == "train":
                logger.info("Saving source features into cached file %s", cached_features_file_src)
                torch.save(features_src, cached_features_file_src)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    # all_span_label_ids = torch.tensor([f.span_label_ids for f in features], dtype=torch.long)
    # all_type_label_ids = torch.tensor([f.type_label_ids for f in features], dtype=torch.long)
    all_lm_mask = torch.tensor([f.lm_mask for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_span_id = torch.tensor([f.span_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset_meta = TensorDataset(all_input_ids, all_input_mask, all_lm_mask, all_label_id, all_span_id, all_ids)
    
    dataset = None
    if mode == "train":
        # Convert to Tensors and build dataset
        all_input_ids_src = torch.tensor([f.input_ids for f in features_src], dtype=torch.long)
        all_input_mask_src = torch.tensor([f.input_mask for f in features_src], dtype=torch.long)
        # all_segment_ids_u = torch.tensor([f.segment_ids for f in features_u], dtype=torch.long)
        # all_span_label_ids_u = torch.tensor([f.span_label_ids for f in features_u], dtype=torch.long)
        # all_type_label_ids_u = torch.tensor([f.type_label_ids for f in features_u], dtype=torch.long)
        all_lm_mask_src = torch.tensor([f.lm_mask for f in features_src], dtype=torch.long)
        all_label_id_src = torch.tensor([f.label_id for f in features_src], dtype=torch.long)
        all_span_id_src = torch.tensor([f.span_id for f in features_src], dtype=torch.long)
        all_ids_src = torch.tensor([f for f in range(len(features_src))], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_src, all_input_mask_src, all_lm_mask_src, all_label_id_src, all_span_id_src, all_ids_src)
    
    return dataset, dataset_meta

def get_gold_labels(args, data_dir, tag_to_id, mode):
    file_path = os.path.join(data_dir, "{}_{}.json".format(args.dataset, mode))
    # guid_index = 1
    gold_res = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            sid = item["id"]
            words = item["str_words"]
            labels_ner = item["tags_ner_gold"] # [(start, end, type), ...]
            # labels_esi = item["tags_esi"]
            # labels_net = item["tags_net"]
            for ner in labels_ner:
                gold_res.append((sid, ner[0], ner[1], tag_to_id["type"][ner[2]]))
                # entity = words[ner[0]:ner[1]]
                # span = [sid, ner[0], ner[1]]
                # examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, entity=entity, type_label=ner[2], span=span))
                # guid_index += 1

    return gold_res

def get_labels(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        labels_ner = {}
        labels_span = {}
        labels_type = {}
        non_entity_id = None
        with open(path+dataset+"_tag_to_id.json", "r") as f:
            data = json.load(f)
            spans = data["span"]
            for l, idx in spans.items():
                labels_span[idx] = l
            types = data["type"]
            for l, idx in types.items():
                labels_type[idx] = l

        # if "O" not in labels:
        #     labels = ["O"] + labels
        return labels_span, labels_type, spans["O"]
    else:
        return None, None, None

def tag_to_id(path=None, dataset=None):
    if path and os.path.exists(path+dataset+"_tag_to_id.json"):
        with open(path+dataset+"_tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data # {"ner":{}, "span":{}, "type":{}}
    else:
        return None

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    tags = tags["span"]
    default = tags["O"]
    bgn = tags["B"]
    inner = tags["I"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_start is not None:
            chunk = (chunk_start, i)
            chunks.append(chunk)
            chunk_start = None

        elif tok == bgn:
            if chunk_start is not None:
                chunk = (chunk_start, i)
                chunks.append(chunk)
                chunk_start = None
            chunk_start = i
        # elif tok == inner:
        #     if chunk_start is None:
        #         chunk_start = i

            # tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            # if chunk_start is None:
            #     if tok_chunk_class != "I":
            #         chunk_start = i
            #     else:
            #         pass
            # elif tok_chunk_type != chunk_type:
            #     chunk = (chunk_type, chunk_start, i)
            #     chunks.append(chunk)
            #     if tok_chunk_class != "I":
            #         chunk_type, chunk_start = tok_chunk_type, i
            #     else:
            #         chunk_type, chunk_start = None, None

        else:
            pass

    if chunk_start is not None:
        chunk = (chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def get_chunks_token(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # if tok == default and chunk_type is not None:
        #     chunk = (chunk_type, chunk_start, i)
        #     chunks.append(chunk)
        #     chunk_type, chunk_start = None, None
        if tok != default:
            chunk = (idx_to_tag[tok], i)      
            chunks.append(chunk)
        else:
            pass

    return chunks


if __name__ == '__main__':
    save(args)