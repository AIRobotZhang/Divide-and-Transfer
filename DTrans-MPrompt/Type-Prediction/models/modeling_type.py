# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertForMaskedLM
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss

class Type_Learner(nn.Module):
    def __init__(self, model_name_or_path, label_to_vocab_src, label_to_vocab_tgt, device):
        super().__init__()

        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.ind_map_src = label_to_vocab_src.to(device)
        self.ind_map_tgt = label_to_vocab_tgt.to(device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        lm_mask=None,
        target=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        logits = outputs.logits # B, L, V
        # print(logits.size())
        # print(self.ind_map.size())
        if target:
            ind_map = self.ind_map_tgt
        else:
            ind_map = self.ind_map_src
        type_logits = logits[lm_mask>0, :][:, ind_map] # B, C
        res = (type_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(type_logits, labels)
            res = (loss,) + res

        return res
