from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertModel, BertConfig
# from pytorch_pretrained_bert import BertModel, BertConfig
from torchcrf import CRF

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

class KobertCRF(nn.Module):
    """ KoBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRF, self).__init__()

        if vocab is None:
            self.bert, self.vocab = get_pytorch_kobert_model()
        else:
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab

        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()

        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

class KobertCRFViz(nn.Module):
    """ koBERT with CRF 시각화 가능하도록 BERT의 outputs도 반환해주게 수정  """        
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertCRFViz, self).__init__()
        # attention weight는 transformers 패키지에서만 지원됨
        from transformers import BertModel, BertConfig
        
        # 모델 로딩 전에 True 값으로 설정해야함
        bert_config['output_attentions']=True
        self.bert = BertModel(config=BertConfig.from_dict(bert_config))
        self.vocab = vocab

        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        
        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)        
        
        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags, outputs
        
class KobertBiLSTMCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertBiLSTMCRF, self).__init__()

        if vocab is None: # pretraining model 사용
            self.bert, self.vocab = get_pytorch_kobert_model()
        else: # finetuning model 사용           
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]

        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=config.dropout, batch_first=True, bidirectional=True)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=True):

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        attention_mask = input_ids.ne(self._pad_id).float()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            pack_padded_last_encoder_layer = pack_padded_sequence(last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, hc = self.bilstm(pack_padded_last_encoder_layer)
            outputs = pad_packed_sequence(outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ff(outputs)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

class KobertBiGRUCRF(nn.Module):
    """ koBERT with CRF """
    def __init__(self, config, num_classes, vocab=None) -> None:
        super(KobertBiGRUCRF, self).__init__()

        if vocab is None: # pretraining model 사용
            self.bert, self.vocab = get_pytorch_kobert_model()
        else: # finetuning model 사용
            self.bert = BertModel(config=BertConfig.from_dict(bert_config))
            self.vocab = vocab
        self._pad_id = self.vocab.token_to_idx[self.vocab.padding_token]

        self.dropout = nn.Dropout(config.dropout)
        self.bigru = nn.GRU(config.hidden_size, (config.hidden_size) // 2, dropout=config.dropout, batch_first=True, bidirectional=True)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, tags=None, using_pack_sequence=True):

        seq_length = input_ids.ne(self._pad_id).sum(dim=1)
        attention_mask = input_ids.ne(self._pad_id).float()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        if using_pack_sequence is True:
            pack_padded_last_encoder_layer = pack_padded_sequence(last_encoder_layer, seq_length, batch_first=True, enforce_sorted=False)
            outputs, hc = self.bigru(pack_padded_last_encoder_layer)
            outputs = pad_packed_sequence(outputs, batch_first=True, padding_value=self._pad_id)[0]
        else:
            outputs, hc = self.bigru(last_encoder_layer)
        emissions = self.position_wise_ff(outputs)

        if tags is not None: # crf training
            log_likelihood, sequence_of_tags = self.crf(emissions, tags), self.crf.decode(emissions)
            return log_likelihood, sequence_of_tags
        else: # tag inference
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

class KobertSequenceFeatureExtractor(nn.Module):
    """ koBERT alone """
    def __init__(self, config, num_classes) -> None:
        super(KobertSequenceFeatureExtractor, self).__init__()
        self.bert, self.vocab = get_pytorch_kobert_model()
        self.dropout = nn.Dropout(config.dropout)
        self.position_wise_ff = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None):
        attention_mask = input_ids.ne(self.vocab.token_to_idx[self.vocab.padding_token]).float()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        logits = self.position_wise_ff(last_encoder_layer)
        return logits
