import torch
import utils
import consts
import model_att
import model_base
from tqdm import tqdm
from consts import ARGS
from pathlib import Path
from preprocess import Preprocessor
from preprocess import BaseAnnotator
from preprocess import CoreAnnotator


class Experiment:
    rootdir = Path('../experiments')
    rootdir.mkdir(exist_ok=True)

    def __init__(self):
        self.data_config = consts.DATA_CONFIG
        self.path_model_config = consts.PATH_MODEL_CONFIG
        self.config = utils.Json.load(self.path_model_config)
        self.config.update(self.data_config.todict())

        # establish experiment folder
        self.exp_name = f'{consts.DIR_DATA.stem}-{consts.LM_NAME_SUFFIX}-{self.path_model_config.stem}'
        if ARGS.exp_prefix:
            self.exp_name += f'.{ARGS.exp_prefix}'
        self.dir_exp = self.rootdir / self.exp_name
        self.dir_exp.mkdir(exist_ok=True)
        utils.Json.dump(self.config, self.dir_exp / 'config.json')
        print(f'Experiment outputs will be saved to {self.dir_exp}')

        # preprocessor
        self.train_preprocessor = Preprocessor(
            path_corpus=self.data_config.path_train,
            num_cores=consts.NUM_CORES,
            use_cache=True
        )

        # annotator (supervision)
        self.train_annotator: BaseAnnotator = {
            'core': CoreAnnotator(
                use_cache=True,
                preprocessor=self.train_preprocessor
            )
        }['core']

        # model
        model_prefix = '.' + ARGS.model_prefix if ARGS.model_prefix else ''
        model_dir = self.dir_exp / f'model{model_prefix}'
        model = model_att.AttmapModel(
            model_dir=model_dir,
            max_num_subwords=consts.MAX_SUBWORD_GRAM,
            num_BERT_layers=self.config['num_lm_layers'])
        self.trainer = model_att.AttmapTrainer(
            model=model)
    
    def train(self, num_epochs=15):
        self.train_preprocessor.tokenize_corpus()
        self.train_annotator.mark_corpus()
        path_sampled_train_data = self.train_annotator.sample_train_data()
        self.trainer.train(path_sampled_train_data=path_sampled_train_data, num_epochs=num_epochs)
        
    def select_best_epoch(self):
        paths_ckpt = [p for p in self.trainer.output_dir.iterdir() if p.suffix == '.ckpt']
        best_epoch = None
        best_valid_f1 = 0.0
        for p in paths_ckpt:
            ckpt = torch.load(p, map_location='cpu')
            if ckpt['valid_f1'] > best_valid_f1:
                best_valid_f1 = ckpt['valid_f1']
                best_epoch = ckpt['epoch']
        utils.Log.info(f'Best epoch: {best_epoch}. F1: {best_valid_f1}')
        return best_epoch

    def predict_final(self, epoch):
        final_preprocessor = Preprocessor(
            path_corpus=self.data_config.path_train,  # 사용하고자 하는 데이터 경로로 변경
            num_cores=consts.NUM_CORES,
            use_cache=True)

        final_preprocessor.tokenize_corpus()
        final_annotator: BaseAnnotator = {
            'core': CoreAnnotator(
                use_cache=True,
                preprocessor=final_preprocessor
            )
        }['core']
        final_annotator.mark_corpus()


        ''' Model Predict '''
        dir_final_predict = self.trainer.output_dir / f'final.predict.epoch-{epoch}'
        path_ckpt = self.trainer.output_dir / f'epoch-{epoch}.ckpt'
        model: model_base.BaseModel = model_base.BaseModel.load_ckpt(path_ckpt).eval().to(consts.DEVICE)
        path_predicted_docs = model.predict(
            path_tokenized_id_corpus=final_preprocessor.path_tokenized_id_corpus, 
            dir_output=dir_final_predict,
            batch_size=1024, 
            use_cache=True)

        ''' Model Decode '''
        dir_final_decoded = self.trainer.output_dir / f'final.decoded.epoch-{epoch}'
        dir_final_decoded.mkdir(exist_ok=True)


        path_decoded_doc2sents = model_base.BaseModel.decode(
            path_predicted_docs=path_predicted_docs,
            path_pos_docs = final_preprocessor.path_tokenized_pos_corpus,
            output_dir=dir_final_decoded,
            threshold=self.config['threshold'],
            use_cache=True,
            use_tqdm=True
        )
        print(f'Final Decoded Output: {path_decoded_doc2sents}')

if __name__ == '__main__':
    exp = Experiment()

    # Check if the model directory exists and has checkpoints
    if exp.trainer.output_dir.exists() and any(exp.trainer.output_dir.glob('*.ckpt')):
        print("Model checkpoints found. Skipping training.")
        best_epoch = exp.select_best_epoch()
    else:
        print("No model checkpoints found. Starting training.")
        exp.train()
        best_epoch = exp.select_best_epoch()
    
    exp.predict_final(epoch=best_epoch)