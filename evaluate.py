# from tqdm import tqdm
# import torch
# from metric import correct_sum
# from chatspace import ChatSpace

# spacer = ChatSpace()
#
# def evaluate(model, data_loader, metrics, device, tokenizer=None):
#     if model.training:
#         model.eval()
#
#     summary = {metric: 0 for metric in metrics}
#     num_correct_elms = 0
#
#     for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
#         enc_input, dec_input, dec_output = map(lambda elm: elm.to(device), mb)
#
#         with torch.no_grad():
#             y_pred = model(enc_input, dec_input)
#
#             if step % 1000 == 0:
#                 decoding_from_result(enc_input, y_pred, dec_output, tokenizer)
#
#             y_pred = y_pred.reshape(-1, y_pred.size(-1))
#             dec_output = dec_output.view(-1).long()
#
#             for metric in metrics:
#                 if metric is 'acc':
#                     _correct_sum, _num_correct_elms = correct_sum(y_pred, dec_output)
#                     summary[metric] += _correct_sum
#                     num_correct_elms += _num_correct_elms
#                 else:
#                     summary[metric] += metrics[metric](y_pred, dec_output).item() #* dec_output.size()[0]
#
#     for metric in metrics:
#         if metric is 'acc':
#             summary[metric] /= num_correct_elms
#         else:
#             summary[metric] /= len(data_loader.dataset)
#
#     return summary

# from pathlib import Path
# def decoding_from_result(x_input, y_pred, y_real=None, tokenizer=None, model_dir=Path('experiments/base_model')):
#     list_of_input_ids = x_input.tolist()
#     list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()
#     input_token = tokenizer.decode_token_ids(list_of_input_ids)
#
#     import json
#     with open(model_dir / 'ner_to_index.json') as io:
#         ner_to_index = json.load(io)
#         index_to_ner = {v:k for k,v in ner_to_index.items()}
#
#
#     pred_token = [index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]] # test로 첫번째만
#
#     print("input: ", input_token)
#     print("pred: ", pred_token)
#     if y_real is not None:
#         real_token =  tokenizer.decode_token_ids(y_real.tolist())
#         print("real: ", real_token)
#         print("")
#         return None
#     else:
#         # 핑퐁의 띄어쓰기 교정기 적용
#         pred_str = ''.join([token.split('/')[0] for token in pred_token[0][:-1]])
#         # pred_str = spacer.space(pred_str)
#         print("pred_str: ", pred_str)
#         print("")
#         return pred_str