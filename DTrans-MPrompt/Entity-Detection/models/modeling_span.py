# -*- coding:utf-8 -*-
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, BCELoss
from utils.loss_utils import FocalLoss, LabelSmoothingCrossEntropy, softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss, get_current_consistency_weight
torch.backends.cudnn.enabled = False
class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        # print("hidden_size")
        # print(hidden_size)
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        # print(hidden_states.size())
        # print(start_positions.size())
        # print(hidden_states.device)
        # print(start_positions.device)
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class Span_Detector(BertPreTrainedModel):
    def __init__(self, config, span_num_labels, se_soft_label, loss_type, device):
        super().__init__(config)

        self.device_ = device # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.span_num_labels = span_num_labels
        # self.type_num_labels_src = type_num_labels_src+1
        # self.type_num_labels_tgt = type_num_labels_tgt+1
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        # self.span = nn.Parameter(torch.randn(type_num_labels, config.hidden_size, span_num_labels))
        # self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels)
        self.classifier_bio = nn.Linear(config.hidden_size, span_num_labels) # BIO

        self.soft_label = se_soft_label
        self.num_labels_se = 2
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels_se) # Start-End No: 0, Yes: 1
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels_se, self.num_labels_se)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels_se)
        self.loss_type = loss_type

        self.num_labels_tb = 2

        self.classifier_tb = nn.Linear(config.hidden_size, self.num_labels_tb) # Tie-Break

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_bio=None,
        label_mask=None,
        soft_label_bio=None,
        soft_label_start=None,
        soft_label_end=None,
        soft_label_tb=None,
        loss_weight=0.0,
        reduction=True,
        start_labels=None,
        end_labels=None,
        tb_labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        final_embedding = outputs[0] # B, L, D

        #################BIO###################################
        sequence_output = self.dropout(final_embedding)
        logits_bio = self.classifier_bio(sequence_output) # B, L, C
        
        outputs = (logits_bio, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels_bio is not None:
            # logits = self.logsoftmax(logits)
            # Only keep active parts of the loss
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
            if label_mask is not None:
                active_loss = active_loss&(label_mask.view(-1)==1)
            
            active_logits = logits_bio.view(-1, self.span_num_labels)[active_loss]

            if labels_bio.size() != logits_bio.size():
                loss_fct = CrossEntropyLoss(reduce=reduction)
                if attention_mask is not None:
                    active_labels = labels_bio.view(-1)[active_loss]
                    loss_bio = loss_fct(active_logits, active_labels)
                else:
                    loss_bio = loss_fct(logits_bio.view(-1, self.span_num_labels), labels_bio.view(-1))
            else:
                loss_bio = softmax_mse_loss(active_logits, 
                                labels_bio.view(-1, self.span_num_labels)[active_loss])

            soft_loss = softmax_mse_loss(active_logits, 
                                soft_label_bio.view(-1, self.span_num_labels)[active_loss]) if soft_label_bio is not None else 0
            # soft_loss = softmax_kl_loss(active_logits, 
            #                                 soft_label.view(-1, self.span_num_labels)[active_loss]) if soft_label is not None else 0

            outputs = (loss_bio+loss_weight*soft_loss,) + outputs
            # print(loss_bio, soft_loss)
            # outputs = (loss_bio,) + outputs

        #########################Start-End##################################
        sequence_output = self.dropout(final_embedding) # B, L, D
        start_logits = self.start_fc(sequence_output)
        if start_labels is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels_se).to(self.device_)
                label_logits.zero_()
                label_logits = label_logits
                label_logits.scatter_(2, start_labels.unsqueeze(2), 1)
            else:
                label_logits = start_labels.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs

        if start_labels is not None and end_labels is not None:
            # print(self.loss_type)
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels_se)
            end_logits = end_logits.view(-1, self.num_labels_se)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_labels.view(-1)[active_loss]
            active_end_labels = end_labels.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs

            soft_loss_start = softmax_mse_loss(active_start_logits, 
                                soft_label_start.view(-1, self.num_labels_se)[active_loss]) if soft_label_start is not None else 0
            soft_loss_end = softmax_mse_loss(active_end_logits, 
                                soft_label_end.view(-1, self.num_labels_se)[active_loss]) if soft_label_end is not None else 0
            outputs = (total_loss+loss_weight*(soft_loss_start+soft_loss_end),) + outputs


        ######################Tie-Break#########################
        sequence_output = self.dropout(final_embedding) # B, L, D
        b,l,d = sequence_output.size()
        pad = torch.zeros(b,1,d).to(self.device_)
        a = torch.cat((sequence_output, pad), dim=1) # B, L+1, D
        b = torch.cat((pad, sequence_output), dim=1) # B, L+1, D
        h_rep = a+b
        logits_tb = self.classifier_tb(h_rep[:,1:,:])

        outputs = (logits_tb,) + outputs

        if tb_labels is not None:
            loss_fct = CrossEntropyLoss()
            logits_tb = logits_tb.view(-1, self.num_labels_tb)
            active_loss = attention_mask.view(-1) == 1
            active_logits_tb = logits_tb[active_loss]
            active_tb_labels = tb_labels.view(-1)[active_loss]

            tb_loss = loss_fct(active_logits_tb, active_tb_labels)

            soft_loss = softmax_mse_loss(active_logits_tb, 
                                soft_label_tb.view(-1, self.num_labels_tb)[active_loss]) if soft_label_tb is not None else 0
            # soft_loss = softmax_kl_loss(active_logits, 
            #                                 soft_label.view(-1, self.span_num_labels)[active_loss]) if soft_label is not None else 0

            outputs = (tb_loss+loss_weight*soft_loss,) + outputs

            # outputs = (tb_loss,) + outputs


        return outputs