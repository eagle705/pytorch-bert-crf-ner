import codecs
import pickle
import pandas as pd

import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
import gluonnlp as nlp
from pathlib import Path
from collections import Counter
import os



def load_data_from_txt(file_full_name):
    with codecs.open(file_full_name, "r", "utf-8" ) as io:
        lines = io.readlines()

        # parsing에 문제가 있어서 아래 3개 변수 도입!
        prev_line = ""
        save_flag = False
        count = 0
        sharp_lines = []

        for line in lines:
            if prev_line == "\n" or prev_line == "":
                save_flag = True
            if line[:3] == "## " and save_flag is True:
                count += 1
                sharp_lines.append(line[3:])
            if count == 3:
                count = 0
                save_flag = False

            prev_line = line
        list_of_source_no, list_of_source_str, list_of_target_str = sharp_lines[0::3], sharp_lines[1::3], sharp_lines[2::3]
    return list_of_source_no, list_of_source_str, list_of_target_str


def main():
    cwd = Path.cwd()
    data_in = cwd / "data_in"
    train_data_in = data_in / "NER-master" / "말뭉치 - 형태소_개체명"
    list_of_file_name = [file_name for file_name in os.listdir(train_data_in) if '.txt' in file_name]
    list_of_full_file_path = [train_data_in / file_name for file_name in list_of_file_name]
    print("num of files: ", len(list_of_full_file_path))

    list_of_total_source_no, list_of_total_source_str, list_of_total_target_str = [], [], []
    for i, full_file_path in enumerate(list_of_full_file_path):
        list_of_source_no, list_of_source_str, list_of_target_str = load_data_from_txt(file_full_name=full_file_path)
        list_of_total_source_str.extend(list_of_source_str)
        list_of_total_target_str.extend(list_of_target_str)

    print("list_of_total_source_str: ", list_of_total_source_str[0])
    print("list_of_total_target_str: ", list_of_total_target_str[0])
    print("list_of_total_source_str: ", list_of_total_source_str[-10:])
    print("list_of_total_target_str: ", list_of_total_target_str[-10:])
    print("len(list_of_total_source_str): ", len(list_of_total_source_str))
    print("len(list_of_total_target_str): ", len(list_of_total_target_str))
    assert len(list_of_total_source_str) == len(list_of_total_target_str)


    corpus_full_path = '/var/tmp/corpus.txt'
    print("corpus_full_path:" , corpus_full_path)


    with open(corpus_full_path, 'w', encoding='utf-8') as io:
        for line in list_of_source_str:
            io.write(line)



    # kobert load 해서 쓸거기 때문에 이 vocab을 쓸수는 없음
    # https://github.com/google/sentencepiece/issues/4
    # what is hard_vocab_limit?
    import sentencepiece as spm
    templates = '--input={} --model_prefix={} --vocab_size={} --hard_vocab_limit=false --user_defined_symbols=[CLS],[SEP],[MASK] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3'

    prefix = 'sentencePiece'

    vocab_size = 8000
    cmd = templates.format(corpus_full_path, prefix, vocab_size)


    spm.SentencePieceTrainer.Train(cmd)

    # Load model
    sp = spm.SentencePieceProcessor()
    sp_model_path = '{}.model'.format(prefix)
    sp.Load(sp_model_path)

    print(sp.pad_id())  # 결과: 0
    print(sp.bos_id())  # 결과: 1
    print(sp.eos_id())  # 결과: 2
    print(sp.unk_id())  # 결과: 3

    tokenizer = nlp.data.SentencepieceTokenizer(path=sp_model_path)
    detokenizer = nlp.data.SentencepieceDetokenizer(path=sp_model_path)
    print(tokenizer)
    print(tokenizer("안녕하세요 ㅋㅋ"))
    print(detokenizer(tokenizer("안녕하세요 ㅋㅋ")))


    list_of_source_tokens = [tokenizer(source_str) for source_str in list_of_total_source_str]

    count_tokens = Counter(itertools.chain.from_iterable(list_of_source_tokens))
    print("list_of_tokens:", list_of_source_tokens)
    print("count_tokens: ", count_tokens)

    reserved_tokens = ['[CLS]','[SEP]','[MASK]']
    vocab = nlp.Vocab(counter=count_tokens, bos_token=None, eos_token=None, reserved_tokens=reserved_tokens)
    print(vocab.unknown_token)
    print(vocab.padding_token)
    print(vocab.token_to_idx)

    import json
    import pickle
    with open(data_in / 'token_to_index.json', 'w', encoding='utf-8') as io:
        json.dump(vocab.token_to_idx, io, ensure_ascii=False, indent=4)

    with open(data_in / 'vocab.pkl', mode='wb') as io:
        pickle.dump(vocab, io)

    with open(data_in / 'list_of_source_tokens.pkl', mode='wb') as io:
        pickle.dump(list_of_source_tokens, io)

    # with open(data_in / 'list_of_label.pkl', mode='wb') as io:
    #     pickle.dump(list_of_label, io)


if __name__ == '__main__':
    main()