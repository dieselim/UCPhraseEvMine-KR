import os
import sys
import argparse
from utils import *
from collections import defaultdict
import numpy as np
import igraph as ig
#from sklearn.svm import SVC
import re
import json
from tqdm import tqdm
from itertools import product, combinations

import inflect
infect_engine = inflect.engine()

#from nltk.corpus import stopwords
import string
stop_words = set(TextFile.readlines(os.path.join('data', 'stopwords.txt'))) 
import datefinder
import dateutil
MONTHS = dateutil.parser.parserinfo.MONTHS

from dateutil.relativedelta import relativedelta

import itertools

import cupy as cp
from cuml.svm import SVC as cuSVC
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch



class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)
    
def main(args, config):
    doc2time, min_t, num_t, all_t = load_doc_time(os.path.join('data', args.data, args.doc_time))
    docs, doc_sents = load_ucphrase(os.path.join('data', args.data, args.ucphrase_res))
    doc_emb = np.load(os.path.join('data', args.data, args.doc_emb))
    phrase_emb_sim, i2p, p2i = get_phrase_emb_sim(args)

    # Construct initial vocab
    word_count = defaultdict(int)
    for doc in docs:
        words = doc.split(' ')
        for word in words:
            word_count[word] += 1
    vocabulary = [w for w in word_count if word_count[w] >= 10 and w not in stop_words and w not in string.punctuation]

    print('Pre-processing')
    # word to document count
    w2dc = {w: word_counting(w, docs) for w in tqdm(vocabulary, dynamic_ncols=True)}

    # output_file = os.path.join('data', args.data, 'w2dc_results.pkl')
    # if os.path.exists(output_file):
    #     print(f"Output file {output_file} already exists. Loading the results.")
    #     with open(output_file, 'rb') as f:
    #         w2dc = pickle.load(f)
    # else:
    #     w2dc = {w: word_counting(w, docs) for w in tqdm(vocabulary, dynamic_ncols=True)}
    #     with open(output_file, 'wb') as f:
    #         pickle.dump(w2dc, f)
   

    # filter with lower tf-idf
    w2tfidf = {}
    for w, dc in w2dc.items():
        if len(dc) == 0:
            w2tfidf[w] = 0
        else:
            w2tfidf[w] = np.log(np.sum(list(dc.values()))+1) * np.log(float(len(docs)) / len(dc))

    w_tfidf_num = int(len(w2tfidf) * 0.3)
    w_tfidf_thres = np.partition(list(w2tfidf.values()), kth=w_tfidf_num)[w_tfidf_num]

    w2tc = {w:{} for w in vocabulary}
    for w, dc in w2dc.items():
        for did, c in dc.items():
            if doc2time[did] not in w2tc[w]:
                w2tc[w][doc2time[did]] = []
            w2tc[w][doc2time[did]].append(c)
    w2tc = {w:{t:(np.sum(c) if len(c) > config['phrase_single_day_freq'] and w2tfidf[w] > w_tfidf_thres else 0) for t,c in tc.items()} for w, tc in w2tc.items()}
    ##
    # Print a sample of w2tc
    # print("Sample of w2tc:")
    # for key, value in itertools.islice(w2tc.items(), 5):  # 첫 5개 항목 출력
    #     print(f"{key}: {value}")
    ##

    # Event-Related Peak Phrase Detection
    print('Event-related peak phrase detection')
    peak_phrases_file = os.path.join('data', args.data, 'peak_phrases_results.pkl')

    wt2score = {}
    for w, t in tqdm(product(vocabulary, all_t), total=len(vocabulary) * len(all_t), dynamic_ncols=True):
        wt2score[(w, t)] = tf_itf(w, t, w2tc, num_t, window_size=3)[0]

    peak_phrases = []
    for pt, s in sorted(wt2score.items(), key=lambda x: x[1], reverse=True):
        if '_' in pt[0]:
            peak_phrases.append(pt)
            if s <= 0 or len(peak_phrases) >= len(doc2time) / 3:
                break

    # if os.path.exists(peak_phrases_file):
    #     print(f"Peak phrases file {peak_phrases_file} already exists. Loading the results.")
    #     with open(peak_phrases_file, 'rb') as f:
    #         peak_phrases = pickle.load(f)
    # else:
    #     wt2score = {}
    #     for w, t in tqdm(product(vocabulary, all_t), total=len(vocabulary) * len(all_t), dynamic_ncols=True):
    #         wt2score[(w, t)] = tf_itf(w, t, w2tc, num_t, window_size=3)[0]

    #     peak_phrases = []
    #     for pt, s in sorted(wt2score.items(), key=lambda x: x[1], reverse=True):
    #         if '_' in pt[0]:
    #             peak_phrases.append(pt)
    #             if s <= 0 or len(peak_phrases) >= len(doc2time) / 3:
    #                 break

    #     with open(peak_phrases_file, 'wb') as f:
    #         pickle.dump(peak_phrases, f)

    ##
    print(len(peak_phrases))
    ##

    # Event-Indicative Peak Phrase Clustering
    print('Event-indicative peak phrase clustering')
    top_times = set([pt[1] for pt in peak_phrases])
    prev = set()
    prev_t = None
    nodes = set()
    edge2weight = {}
    
    # 문서 시간을 기반으로 문서 ID를 매핑
    doc_time_mapping = defaultdict(list)
    for doc_id, time in enumerate(doc2time):
        doc_time_mapping[time].append(doc_id)
    
    for t in tqdm(sorted(top_times), dynamic_ncols=True):
        if prev_t and (t.year != prev_t.year or t.month != prev_t.month):
            if (t - relativedelta(months=1)) == prev_t:
                prev = set()
        pt_on_t = [pt for pt in peak_phrases if pt[1]==t]
        nodes.update(pt_on_t)
        
        #total = len([tt for tt in doc2time if tt == t])
        total = len(doc_time_mapping[t])
        
        for pt0, pt1 in combinations(pt_on_t, 2):
            docs1 = set([d for d in w2dc[pt0[0]] if doc2time[d] == t])
            docs2 = set([d for d in w2dc[pt1[0]] if doc2time[d] == t])
            inter = len(docs1.intersection(docs2)) + 1e-5
            if len(docs1)==0 or len(docs2)==0:
                npmi = 0
            else:
                npmi = -np.log(inter * float(total) / len(docs1) / len(docs2)) / np.log(inter / float(total))
            ## 왜 이럴까요
            emb_sim = 0
            ## 해결해 보아요
            if pt0[0] in p2i and pt1[0] in p2i:
                emb_sim = phrase_emb_sim[p2i[pt0[0]], p2i[pt1[0]]]
            edge2weight[(pt0, pt1)] = np.sqrt(max(0, npmi) * max(0, emb_sim))
        for p, t in pt_on_t:
            if p in prev:
                edge2weight[((p,t), (p, t - relativedelta(months=1)))] = 3
        prev = set([p for p,t in pt_on_t])
        prev_t = t

    g = ig.Graph()
    nodes = list(nodes)
    n2i = {n:i for i,n in enumerate(nodes)}
    g.add_vertices(len(nodes))
    edges = [(n2i[i], n2i[j]) for i,j in edge2weight.keys()]
    weights = [edge2weight[(nodes[i], nodes[j])] for i,j in edges]
    g.add_edges(edges)
    levels = g.community_multilevel(weights=weights, return_levels=True)

    events = []
    for ci, c in enumerate(levels[-1]):
        c = [nodes[i] for i in c]
        if len(c) < 2:
            continue
        c_t2p = defaultdict(list)
        for pt in c:
            c_t2p[pt[1]].append(pt[0])
        cluster = set()
        sorted_times = sorted(c_t2p.keys())
        for t in sorted_times:
            for pp in c_t2p[t]:
                cluster.add(pp)
        tis = [(t.year - min_t.year) * 12 + (t.month - min_t.month) for t in sorted_times]
        start = min_t + relativedelta(months=min(tis))
        end = min_t + relativedelta(months=max(tis))
        events.append([list(cluster), start, end, [[],[]]])

    ##
    print(len(events))
    for i in range(len(events)):
        if events[i][1] != events[i][2]:
            print(events[i][0][:5], events[i][1], events[i][2])
    ##
    
    # Key Event Document Selection
    print('Iterative key event doc selection')
    
    
    for ite in tqdm(range(config['num_iterations']), desc="Iterations", dynamic_ncols=True):
        print(f'Iteration {ite}')
        events = sorted(events, key=lambda x: x[1])
        doc2event_matching = [{} for _ in range(len(docs))]
        for ev_i, (event_phrases, start, end, _) in enumerate(events):
            enriched_phrases = set(event_phrases)
            for p in event_phrases:
                if p in p2i:
                    for i, s in enumerate(phrase_emb_sim[p2i[p]]):
                        if s > 0.95 and i2p[i] not in enriched_phrases and all((i2p[i] not in ep) for ep,s2,e2,_ in events if e2 >= start and s2 <= end):
                            enriched_phrases.add(i2p[i])
            time_docs = [did for did in range(len(docs)) if doc2time[did] >= start and doc2time[did] <= end]
            for did in time_docs:
                doc2event_matching[did][ev_i] = (len([w2dc[w][did] for w in enriched_phrases if w in w2dc and did in w2dc[w]]),
                                                 sum([w2dc[w][did] for w in enriched_phrases if w in w2dc and did in w2dc[w]]))

        event_docs_coverage = [{} for _ in events]
        for did, ev_matching in enumerate(doc2event_matching):
            if len(ev_matching) == 0:
                continue
            ev_id = sorted(ev_matching.keys(), key=lambda x: ev_matching[x], reverse=True)[0]
            event_docs_coverage[ev_id][did] = ev_matching[ev_id]

        event2doc_id = {}
        final_eid = 0
        for eid, (e_docs, (ep,_,_,(doc_ids, negs))) in enumerate(zip(event_docs_coverage, events)):
            pseudo_doc_ids = [di for di in doc_ids if di not in negs]
            for (did, s) in [ds for ds in sorted(e_docs.items(), key=lambda x: x[1], reverse=True) if ds[0] not in negs][:10]:
                if s[0] == 0: 
                    break
                pseudo_doc_ids.append(did)
            pseudo_doc_ids = list(set(pseudo_doc_ids))
            if len(pseudo_doc_ids) >= config['min_pseudo_labels']:
                event2doc_id[final_eid] = {'doc_ids':pseudo_doc_ids, 'start':events[eid][1].strftime('%Y-%m-%d'), 'end':events[eid][2].strftime('%Y-%m-%d'), 'prev_id':eid}
                final_eid += 1

        ratio = 2
        repeat = 50
        doc2ev = []
        for evid, ev in tqdm(event2doc_id.items(), desc="Event2Doc ID", leave=False, dynamic_ncols=True):
            doc_ids = ev['doc_ids']
            all_pred = []
            for _ in range(repeat):
                
                pos = np.array(doc_ids)
                
                negs = np.random.choice(len(docs), len(pos) * (ratio + 1), replace=False)
                negs = np.array([i for i in negs if i not in pos])[:ratio * len(pos)]
                
                combined_idx = np.concatenate((pos, negs))
                X = doc_emb[combined_idx]
                y = np.array([1] * len(pos) + [0] * len(negs))
                
                # SVM 모델 초기화 및 학습
                svc = cuSVC()
                svc.fit(X, y)
                
                pred = svc.decision_function(doc_emb)
                all_pred.append(pred)
            all_pred = np.mean(all_pred, axis=0)
            doc2ev.append(all_pred)

        doc2ev = np.array(doc2ev).T
        ev2all_doc_pos = [{} for _ in event2doc_id]
        for doc_id, doc_scores in enumerate(doc2ev):
            classified = np.argmax(doc_scores)
            if doc_scores[classified] > 0:
                ev2all_doc_pos[classified][doc_id] = doc_scores[classified]

        # temporal filtering & feedback
        key_event_docs = []
        to_add_event = []
        for evid, ev_docs_with_score in enumerate(ev2all_doc_pos):
            if len(ev_docs_with_score) == 0: continue
            start = datetime.strptime(event2doc_id[evid]['start'], '%Y-%m-%d')
            end = datetime.strptime(event2doc_id[evid]['end'], '%Y-%m-%d')
            
            time_sorted = sorted(ev_docs_with_score.keys(), key=lambda x: doc2time[x])
            sub_clusters = []
            clus = []
            prev_t = min_t
            for di in time_sorted:
                if relativedelta(doc2time[di], prev_t).months > 1 or relativedelta(doc2time[di], prev_t).years > 0:
                    if len(clus) > 0:
                        sub_clusters.append(clus)
                    clus = [di]
                else:
                    clus.append(di)
                prev_t = doc2time[di]
            sub_clusters.append(clus)
            final_res = []
            for sci, sc in enumerate(sub_clusters):
                cluster_times = set([doc2time[di] for di in sc])
                if min(cluster_times) <= end and max(cluster_times) >= start:
                    final_res.extend(sc)
                    continue
                # try:
                #     extracted_time = [t for t in datefinder.find_dates(' '.join(doc_sents[di][:3]), base_date=doc2time[sc[0]])]
                # except:
                #     continue
                # if len(sc) == 1 and\
                #     any(t >= start - relativedelta(months=1) and t <= end + relativedelta(months=1) for t in extracted_time)\
                #         and doc2time[sc[0]] >= start:
                #     final_res.extend(sc)
                # elif len(sc) > 1:
                #     to_add_event.append((sc, min(cluster_times), max(cluster_times)))
                if len(sc) > 1:
                    to_add_event.append((sc, min(cluster_times), max(cluster_times)))
            key_event = sorted(final_res, key=lambda x: ev_docs_with_score[x], reverse=True)
            key_event_docs.append(key_event)
            events[event2doc_id[evid]['prev_id']][3] = [key_event[:5], [di for di in range(len(docs)) if doc2ev[di, evid] < 0.1]]

        if ite == config['num_iterations'] - 1:
            break

        for new_docs, start, end in to_add_event:
            
            if any((start <= e and end >= s) for _, s, e, _ in events):
                continue

            kp_candidates = set()
            for di in new_docs:
                kp_candidates |= set([w for w in docs[di].split(' ') if '_' in w])
            scores = {}
            for kp in kp_candidates:
                if kp not in w2dc or w2tfidf[kp] < w_tfidf_thres: 
                    continue
                tf = sum((w2dc[kp][di] if di in w2dc[kp] else 0) for di in new_docs)
                idf = float(len(docs)) / len(w2dc[kp]) * sum([1 for di in new_docs if di in w2dc[kp]])
                scores[kp] = tf * np.log(idf)
            new_keyphrases = set()
            for kp in sorted(scores.keys(), key=lambda x: scores[x], reverse=True):
                if scores[kp] <= 0:
                    break
                if any((kp in ep) for ep,s2,e2,_ in events if e2 >= start and s2 <= end):
                    continue
                new_keyphrases.add(kp)
                if len(new_keyphrases) >= 5: 
                    break
            events.append([list(new_keyphrases), start, end, [new_docs, []]])
    
    # Save results
    with open(os.path.join('data', args.data, args.out), 'w') as f:
        json.dump(key_event_docs, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='news')
    parser.add_argument("--ucphrase_res", type=str, default='doc2sents-0.75-tokenized.id.final.json')
    parser.add_argument("--doc_time", type=str, default='doc2time.final.txt')
    parser.add_argument("--doc_emb", type=str, default='doc_emb.npy')
    parser.add_argument("--phrase_emb", type=str, default='phrase_emb')
    parser.add_argument("--out", type=str, default='output.json')
    args = parser.parse_args()

    config = {'phrase_single_day_freq':0, 'min_pseudo_labels':3}
    
    config['num_iterations'] = 3

    main(args, config)