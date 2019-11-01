from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from kobert.pytorch_kobert import get_pytorch_kobert_model
from pytorch_pretrained_bert import BertModel, BertConfig
import gluonnlp as nlp

bert_config = {'attention_probs_dropout_prob': 0.1,
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.1,
 'hidden_size': 768,
 'initializer_range': 0.02,
 'intermediate_size': 3072,
 'max_position_embeddings': 512,
 'num_attention_heads': 12,
 'num_hidden_layers': 12,
 'type_vocab_size': 2,
 'vocab_size': 8002}

class KobertSequenceFeatureExtractor(nn.Module):
    """ koBERT alone """
    def __init__(self, config, num_classes) -> None:
        super(KobertSequenceFeatureExtractor, self).__init__()
        self.bert, self.vocab = get_pytorch_kobert_model()
        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = all_encoder_layers[-1]
        last_encoder_layer = self.dropout(last_encoder_layer)
        logits = self.position_wise_ff(last_encoder_layer)
        return logits

from torchcrf import CRF
class KobertCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRF, self).__init__()

        if vocab is None: # pretraining model 사용
            self.bert, self.vocab = get_pytorch_kobert_model()
        else: # finetuning model 사용
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab

        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = all_encoder_layers[-1]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags