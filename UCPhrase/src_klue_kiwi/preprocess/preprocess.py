import utils
import consts
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

from kiwipiepy import Kiwi
import logging

kiwi = Kiwi()
kiwi.load_user_dictionary('/data/dieselim/KISDI/UCPhrase/data/user_dic.txt')


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Preprocessor:

    def __init__(
            self,
            path_corpus,
            num_cores=8,
            use_cache=True):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        self.path_corpus = Path(path_corpus)
        self.dir_corpus = self.path_corpus.parent
        self.dir_preprocess = self.dir_corpus / f'preprocess-{consts.LM_NAME_SUFFIX}'
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.path_tokenized_corpus = self.dir_preprocess / f'tokenized.{self.path_corpus.name}'
        self.path_tokenized_id_corpus = self.dir_preprocess / f'tokenized.id.{self.path_corpus.name}'
        self.path_tokenized_pos_corpus = self.dir_preprocess / f'tokenized.pos.{self.path_corpus.name}'

    @staticmethod
    def _par_tokenize_doc(doc):
        docid = doc['_id_']
        sents = doc['sents']

        tokenized_sents = []
        tokenized_pos = []
        for s in sents:
            morphs = kiwi.tokenize(s)
            selected_tokens = [token.form for token in morphs if token.tag in ['NNG', 'NNP', 'MAG', 'XPN', 'VV', 'VA', 'XR', 'XSN']]
            selected_pos = [token.tag for token in morphs if token.tag in ['NNG', 'NNP', 'MAG', 'XPN', 'VV', 'VA', 'XR', 'XSN']]
            sent_subtokens = []
            sent_pos = []
            s_idx = 0
            for i, word in enumerate(selected_tokens):
                pos = selected_pos[i]
                subtokens = consts.LM_TOKENIZER.tokenize(word)
                if len(sent_subtokens) + len(subtokens) < 512:
                    sent_subtokens.extend(subtokens)
                    sent_pos.extend([pos]*len(subtokens))
                else:
                    tokenized_sents.append(sent_subtokens)
                    tokenized_pos.append(sent_pos)
                    sent_subtokens = subtokens
                    sent_pos = [pos] * len(subtokens)
                    s_idx = i
            if sent_subtokens:
                tokenized_sents.append(sent_subtokens)
                tokenized_pos.append(sent_pos)

        tokenized_doc = {'_id_': docid, 'sents': [' '.join(tokens) for tokens in tokenized_sents]}
        tokenized_pos_doc = {'_id_': docid, 'sents': tokenized_pos}

        tokenized_id_doc = {'_id_': doc['_id_'], 'sents': []}
        for tokens in tokenized_sents:
            widxs = [i for i, token in enumerate(tokens) if not token.startswith(consts.GPT_TOKEN)]  # the indices of start of words
            ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})

        return tokenized_doc, tokenized_id_doc, tokenized_pos_doc

    

    def tokenize_corpus(self):
        if self.use_cache and utils.IO.is_valid_file(self.path_tokenized_corpus) and utils.IO.is_valid_file(self.path_tokenized_id_corpus):
            print(f'[Preprocessor] Use cache: {self.path_tokenized_corpus}')
            return
        docs = utils.JsonLine.load(self.path_corpus)
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
        doc_tuples = list(tqdm(pool_func, total=len(docs), ncols=100, desc=f'[Tokenize] {self.path_corpus}', dynamic_ncols=True))
        tokenized_docs = [doc for doc, iddoc, posdoc in doc_tuples]
        tokenized_id_docs = [iddoc for doc, iddoc, posdoc in doc_tuples]
        tokenized_pos_docs = [posdoc for doc, iddoc, posdoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
        utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)
        utils.JsonLine.dump(tokenized_pos_docs, self.path_tokenized_pos_corpus)
