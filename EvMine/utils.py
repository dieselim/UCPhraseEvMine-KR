from datetime import datetime
import copy
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cos
from collections import defaultdict
import os
from pathlib import Path
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import re
import string
from concurrent.futures import ProcessPoolExecutor

def load_doc_time(file):
    doc2time = []
    with open(file) as f:
        for d2t in f:
            doc_id, t = d2t.strip().split('\t')
            t = datetime.strptime(t, '%Y-%m-%d')  # 'YYYY-MM-DD' 형식으로 파싱
            t = t.replace(day=1)
            doc2time.append(t)
    min_t = min(t for t in doc2time)
    max_t = max(t for t in doc2time)
    # 월 단위로 계산
    num_t = (max_t.year - min_t.year) * 12 + max_t.month - min_t.month + 1
    all_t = [min_t + relativedelta(months=i) for i in range(num_t)]
    return doc2time, min_t, num_t, all_t


def load_ucphrase(file):
    with open(file) as f:
        data = json.load(f)
    doc_sents = []
    docs = []

    # doc_id를 데이터의 키로 사용
    for doc_id in data.keys():
        doc = data[doc_id]
        char = doc[0]['tokens'][0][0]
        sents = []
        for sent in doc:
            tokens = copy.deepcopy(sent['tokens'])
            for s, e, p in sent['spans']:
                rep = ['' for _ in range(s, e + 1)]
                rep[0] = ' ' + p.replace(' ', '_')
                tokens[s:e + 1] = rep
            sents.append(''.join(tokens).replace(char, ' ').strip())
        doc_sents.append(sents)
        docs.append(' '.join(sents))
    return docs, doc_sents

# for memory issue

# def get_phrase_emb_sim(args):
#     (p_emb, i2p, p2i) = pickle.load(open(os.path.join('data', args.data, args.phrase_emb+'.pkl'), 'rb'))

#     return cos(p_emb), i2p, p2i

def batch_cosine_similarity_to_disk(p_emb, batch_size=25000, output_file='phrase_emb_sim.dat'):
    n = p_emb.shape[0]
    similarity_matrix = np.memmap(output_file, dtype='float64', mode='w+', shape=(n, n))

    for i in tqdm(range(0, n, batch_size), desc="Processing rows", dynamic_ncols=True):
        end_i = min(i + batch_size, n)
        # 현재 행 배치와 전체 데이터 간의 유사도 계산
        batch_similarity = cos(p_emb[i:end_i], p_emb)
        similarity_matrix[i:end_i, :] = batch_similarity

    similarity_matrix.flush()
    del similarity_matrix

def get_phrase_emb_sim(args):
    with open(os.path.join('data', args.data, args.phrase_emb + '.pkl'), 'rb') as file:
        (p_emb, i2p, p2i) = pickle.load(file)

    output_file = os.path.join('data', args.data, 'phrase_emb_sim.dat')

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Loading the results.")
    else:
        batch_cosine_similarity_to_disk(p_emb, output_file=output_file)
    
    # 결과를 다시 불러와서 사용할 수 있음
    phrase_emb_sim = np.memmap(output_file, dtype='float64', mode='r', shape=(p_emb.shape[0], p_emb.shape[0]))
    return phrase_emb_sim, i2p, p2i


def find_all(s, sub):
    start = 0
    s = ' ' + s + ' '
    sub = ' ' + sub + ' '
    while True:
        start = s.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def word_counting(w, docs):
    ret = {}
    for did, doc in enumerate(docs):
        num_matches = len(list(find_all(doc, w)))
        if num_matches > 0:
            ret[did] = num_matches
    return ret


def tf_itf(w, t, w2tc, num_t, window_size=5):
    tf = 0.
    for i in range(window_size):
        new_t = t + relativedelta(months=i)
        new_t_str = new_t.strftime('%Y-%m')
        found = False
        for date, count in w2tc[w].items():
            if date.strftime('%Y-%m') == new_t_str and count != 0:
                tf += count * ((window_size-i)/float(window_size))
                found = True
                break
        if not found:
            if i == 0:
                return 0, 0, 0
    itf = float(num_t) / len(w2tc[w])
    return tf/window_size * np.log(itf), tf, itf



class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError
    

class TextFile(IO):
    @staticmethod
    def load(path):
        with open(path, encoding='utf-8') as rf:
            text = rf.read()
        return text

    @staticmethod
    def readlines(path, skip_empty_line=False):
        with open(path, encoding='utf-8') as rf:
            lines = rf.read().splitlines()
        if skip_empty_line:
            return [l for l in lines if l]
        return lines

    @staticmethod
    def dump(text, path):
        with open(path, 'w', encoding='utf-8') as wf:
            wf.write(text)

    @staticmethod
    def dumplist(target_list, path):
        with open(path, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join([str(o) for o in target_list]) + '\n')