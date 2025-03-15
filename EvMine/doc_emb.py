import json
from tqdm import tqdm
import numpy as np
import datetime
import copy
import argparse
from utils import *
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os

# 특정 GPU 지정 (예: GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CUDA 장치가 올바르게 설정되었는지 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
    
    
def get_doc_emb(args, model_name = 'bespin-global/klue-sroberta-base-continue-learning-by-mnr'):
    doc2time, _, _, _ = load_doc_time(os.path.join('data', args.data, args.doc_time))
    _, doc_sents = load_ucphrase(os.path.join('data', args.data, args.ucphrase_res))

    model = SentenceTransformer(model_name)
    model.to(device)

    doc_emb = []
    for sents in tqdm(doc_sents, dynamic_ncols=True):
        doc_emb.append(np.mean(model.encode(sents[:3]), axis=0))
    np.save(os.path.join('data', args.data, args.out), np.array(doc_emb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='news')
    parser.add_argument("--ucphrase_res", type=str, default='doc2sents-0.75-tokenized.id.final.json')
    parser.add_argument("--doc_time", type=str, default='doc2time.final.txt')
    parser.add_argument("--out", type=str, default='doc_emb')
    args = parser.parse_args()

    get_doc_emb(args)