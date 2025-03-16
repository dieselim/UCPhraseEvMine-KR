import utils
import consts
import string
import functools
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from preprocess.preprocess import Preprocessor
from preprocess.annotator_base import BaseAnnotator


MINCOUNT = 2
MINGRAMS = 2
MAXGRAMS = consts.MAX_WORD_GRAM


PUNCS_SET = set(string.punctuation) - {'-'}
STPWD_SET = set(utils.TextFile.readlines('../data/stopwords.txt'))


@functools.lru_cache(maxsize=100000)
def is_valid_ngram(ngram: tuple, pos_list: tuple):
    # 한글 유니코드 범위 설정
    def is_hangul(token):
        return all(
            0xAC00 <= ord(char) <= 0xD7A3 or  # 한글 음절
            0x1100 <= ord(char) <= 0x11FF or  # 한글 자모
            0x3131 <= ord(char) <= 0x318E     # 호환용 한글 자모
            for char in token
        )
    for token in ngram:
        if not token or token in STPWD_SET or token.isdigit():
            return False
        if not is_hangul(token):
            return False
    # POS 조건 추가
    if not pos_list or not pos_list[0] in ['NNG', 'NNP']:
        return False
    return True


class CoreAnnotator(BaseAnnotator):
    def __init__(self, preprocessor: Preprocessor, use_cache):
        super().__init__(preprocessor, use_cache=use_cache)

    @staticmethod
    def _par_mine_doc_phrases(doc_tuple):
        tokenized_doc, tokenized_id_doc, tokenized_pos_doc = doc_tuple
        assert tokenized_doc['_id_'] == tokenized_id_doc['_id_'] == tokenized_pos_doc['_id_']
        assert len(tokenized_doc['sents']) == len(tokenized_id_doc['sents']) == len(tokenized_pos_doc['sents'])

        phrase2cnt = Counter()
        phrase2instances = defaultdict(list)
        for i_sent, (sent, sent_dict, pos_tags) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'], tokenized_pos_doc['sents'])):
            tokens = sent.split()
            widxs = sent_dict['widxs']
            num_words = len(widxs)
            widxs.append(len(tokens))  # for convenience
            for n in range(MINGRAMS, MAXGRAMS + 2):
                for i_word in range(num_words - n + 1):
                    l_idx = widxs[i_word]
                    r_idx = widxs[i_word + n] - 1
                    ngram = tuple(tokens[l_idx: r_idx + 1])
                    
                    combined = []
                    for token in ngram:
                        if token.startswith(consts.GPT_TOKEN):
                            combined[-1] += token[2:]
                        else:
                            combined.append(token)
                    ngram = tuple(combined)
                    
                    pos_ngram = tuple(pos_tags[l_idx: r_idx + 1])
                    if is_valid_ngram(ngram, pos_ngram):
                        phrase = ' '.join(ngram)
                        phrase2cnt[phrase] += 1
                        phrase2instances[phrase].append([i_sent, l_idx, r_idx])
        phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
        phrases = sorted(phrases, key=lambda p: len(p), reverse=True)
        cleaned_phrases = set()
        for p in phrases:
            has_longer_pattern = False
            for cp in cleaned_phrases:
                if p in cp:
                    has_longer_pattern = True
                    break
            if not has_longer_pattern and len(p.split()) <= MAXGRAMS:
                cleaned_phrases.add(p)
        phrase2instances = {p: phrase2instances[p] for p in cleaned_phrases}

        return phrase2instances

    def _mark_corpus(self):
        tokenized_docs = utils.JsonLine.load(self.path_tokenized_corpus)
        tokenized_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        tokenized_pos_docs = utils.JsonLine.load(self.path_tokenized_pos_corpus)
        phrase2instances_list = utils.Process.par(
            func=CoreAnnotator._par_mine_doc_phrases,
            iterables=list(zip(tokenized_docs, tokenized_id_docs, tokenized_pos_docs)),
            num_processes=consts.NUM_CORES,
            desc='[CoreAnno] Mine phrases'
        )
        doc2phrases = dict()
        for i_doc, doc in tqdm(list(enumerate(tokenized_id_docs)), ncols=100, desc='[CoreAnno] Tag docs', dynamic_ncols=True):
            for s in doc['sents']:
                s['phrases'] = []
            phrase2instances = phrase2instances_list[i_doc]
            doc2phrases[doc['_id_']] = list(phrase2instances.keys())
            for phrase, instances in phrase2instances.items():
                for i_sent, l_idx, r_idx in instances:
                    doc['sents'][i_sent]['phrases'].append([[l_idx, r_idx], phrase])
        utils.Json.dump(doc2phrases, self.dir_output / f'doc2phrases.{self.path_tokenized_corpus.stem}.json')

        return tokenized_id_docs
