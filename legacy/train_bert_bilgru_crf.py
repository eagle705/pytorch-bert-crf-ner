from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import logging
import random
import pickle
import json
import os
from pathlib import Path

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm, trange
from data_utils.utils import CheckpointManager, SummaryManager
from model.net import KobertBiGRUCRF
from model.utils import Config

from data_utils.ner_dataset import NamedEntityRecognitionDataset, NamedEntityRecognitionFormatter
from data_utils.vocab_tokenizer import Vocabulary, Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from gluonnlp.data import SentencepieceTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from metric import clf_acc

logger = logging.getLogger(__name__)

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main(parser):
    # Config
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    tok_path = get_tokenizer() # ./tokenizer_78b3253a26.model
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    _, vocab_of_gluonnlp = get_pytorch_kobert_model()
    token_to_idx = vocab_of_gluonnlp.token_to_idx

    model_config.vocab_size = len(token_to_idx)
    vocab = Vocabulary(token_to_idx=token_to_idx)

    print("len(token_to_idx): ", len(token_to_idx))
    with open(model_dir / "token2idx_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(token_to_idx, f, ensure_ascii=False, indent=4)

    # save vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    ner_formatter = NamedEntityRecognitionFormatter(vocab=vocab, tokenizer=tokenizer, maxlen=model_config.maxlen, model_dir=model_dir)

    # Train & Val Datasets
    cwd = Path.cwd()
    data_in = cwd / "data_in"
    train_data_dir = data_in / "NER-master" / "말뭉치 - 형태소_개체명"
    print("model_config.batch_size: ", model_config.batch_size)
    tr_clf_ds = NamedEntityRecognitionDataset(train_data_dir=train_data_dir, model_dir=model_dir)
    tr_clf_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn, transform_target_fn=ner_formatter.transform_target_fn)
    tr_clf_dl = DataLoader(tr_clf_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=4, drop_last=False)

    # Model
    model = KobertBiGRUCRF(config=model_config, num_classes=len(tr_clf_ds.ner_to_index))
    model.train()

    # optim
    train_examples_len = len(tr_clf_ds)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
    t_total = len(tr_clf_dl) // model_config.gradient_accumulation_steps * model_config.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.learning_rate, eps=model_config.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=model_config.warmup_steps, t_total=t_total)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)

    # save
    tb_writer = SummaryWriter('{}/runs'.format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10
    best_train_acc = 0

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(tr_clf_ds))
    logger.info("  Num Epochs = %d", model_config.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", model_config.batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", model_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_steps = 0
    model.zero_grad()
    set_seed()  # Added here for reproductibility (even between python 2 and 3)

    # Train
    train_iterator = trange(int(model_config.epochs), desc="Epoch")
    for _epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(tr_clf_dl, desc="Iteration") # , disable=args.local_rank not in [-1, 0]
        epoch = _epoch
        for step, batch in enumerate(epoch_iterator):
            model.train()
            x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
            log_likelihood, sequence_of_tags = model(x_input, token_type_ids, y_real)

            # loss: negative log-likelihood
            loss = -1 * log_likelihood

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if model_config.gradient_accumulation_steps > 1:
                loss = loss / model_config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % model_config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                with torch.no_grad():
                    sequence_of_tags = torch.tensor(sequence_of_tags)
                    print("sequence_of_tags: ", sequence_of_tags)
                    print("y_real: ", y_real)
                    print("loss: ", loss)
                    print("(sequence_of_tags == y_real): ", (sequence_of_tags == y_real))

                    mb_acc = (sequence_of_tags == y_real).float()[y_real != vocab.PAD_ID].mean()

                tr_acc = mb_acc.item()
                tr_loss_avg = tr_loss / global_step
                tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

                # if step % 50 == 0:
                print('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, global_step,
                                                                                             tr_summary['loss'],
                                                                                             tr_summary['acc']))

                if model_config.logging_steps > 0 and global_step % model_config.logging_steps == 0:
                    # Log metrics
                    if model_config.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        pass
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / model_config.logging_steps, global_step)
                    logger.info("Average loss: %s at global step: %s",
                                str((tr_loss - logging_loss) / model_config.logging_steps), str(global_step))
                    logging_loss = tr_loss

                if model_config.save_steps > 0 and global_step % model_config.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(model_config.output_dir, 'epoch-{}'.format(epoch + 1))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    state = {'global_step': global_step + 1,
                             'model_state_dict': model.state_dict(),
                             'opt_state_dict': optimizer.state_dict()}
                    summary = {'train': tr_summary}
                    summary_manager.update(summary)
                    summary_manager.save('summary.json')

                    is_best = tr_acc >= best_train_acc  # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)
                    # Save
                    if is_best:
                        best_train_acc = tr_acc
                        checkpoint_manager.save_checkpoint(state,
                                                           'best-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                                         global_step,
                                                                                                         tr_acc))
                    else:
                        torch.save(state, os.path.join(output_dir,
                                                       'model-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                                      global_step,
                                                                                                      tr_acc)))

    tb_writer.close()
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

    return global_step, tr_loss / global_step, best_steps



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model_with_bigru_crf', help="Directory containing config.json of model")

    main(parser)