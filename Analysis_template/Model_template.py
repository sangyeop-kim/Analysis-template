import os
import pickle
import warnings
import numpy as np
import pandas as pd
from glob import glob
from time import sleep
from datetime import datetime
from matplotlib import pyplot as plt
warnings.filterwarnings(action = 'ignore')

import json
from itertools import groupby
from easydict import EasyDict
from nbconvert import HTMLExporter
from IPython.display import Javascript
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader




class Model_template(pl.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        self.__set_hyperparametrs(hyperparameters)
        sleep(0.3)
        '''
        model.state_dict() # 모델 weight 확인
        model.hparams # 모델 하이퍼파라미터 확인

        # 모델 학습, folder : 저장 경로
        model.fit(train_dataloader, train_dataloader, folder) 

        # metric은 nn.Loss나 새로운 형태의 loss를 정의해도 됨, default는 학습 때 사용한 metric과 같음. 

        def accuracy (y_hat, y) : # manual metric example
            return torch.sum(torch.max(y_hat, axis = 1)[1] == y).item()/len(y_hat)
        
        # 모델 테스트
        model.test(test_dataloader, 'metric 이름', metric)
        
        # 저장된 ckpt 불러오기
        model = model.load_model(ckpt_path)
        
        # tensorboard
        현재 dir에서
        tensorboard --logdir=./best_model
        
        # 모델 폴더 안에
        hparams.yaml : 하이퍼파라미터 및 loss 저장
        output_file.html : 실행 당시 ipynb 파일
        '''
        
        
        

        ###################### model layer ######################

        # self.loss = nn.CrossEntropyLoss()
        
        # self.conv1 = nn.Conv2d(3, 32, 3)
        # self.max_pool = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32, 16, 3)
        # self.linear = nn.Linear(16 * 6 * 6, 10)

        
    ################# specific model structure #################
    def forward(self, x):
        pass
        # x = self.conv1(x)
        # x = F.relu(self.max_pool(x))
        # x = self.conv2(x)
        # x = F.relu(self.max_pool(x))
        # x = F.relu(self.linear(x.view(x.size(0), -1)))
        
        # return x
    
    ################## optimizer & scheduler ##################
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
    ###############################################################################
    ################################ Do not change ################################
    ###############################################################################
    def __set_hyperparametrs(self, hyperparameters) :
        self.init_hyperparameters = hyperparameters
        

        self.checkpoint_callback = None
        
        if 'validation_loss' in hyperparameters.keys() :
            self.hparams.training_loss = hyperparameters['training_loss']
            self.hparams.validation_loss = hyperparameters['validation_loss']
        else : 
            self.hparams.training_loss = {}
            self.hparams.validation_loss = {}
            
        if 'now' in hyperparameters.keys() :
            self.hparams.now = hyperparameters['now']
        else :
            self.hparams.now = None
        if 'n_cv' in hyperparameters.keys() :
            self.hparams.n_cv = hyperparameters['n_cv']
        else :
            self.hparams.n_cv = None
        if 'patience' in hyperparameters.keys() :
            self.hparams.patience = hyperparameters['patience']
        else :
            self.hparams.patience = 5
            
        self.hparams.lr = hyperparameters['lr']
        self.hparams.step_size = hyperparameters['step_size'] # epoch 단위로 계산됨.
        self.hparams.gamma = hyperparameters['gamma']
        self.hparams.batch_size = hyperparameters['batch_size']
        self.hparams.test_batch_size = hyperparameters['test_batch_size']
        self.hparams.max_epochs = hyperparameters['max_epochs']
        self.hparams.gpus = hyperparameters['gpus']
        self.hparams.auto_lr_find = hyperparameters['auto_lr_find']
        self.hparams.save_top_k = hyperparameters['save_top_k']
        self.hparams.num_workers = hyperparameters['num_workers']
        self.hparams.folder = hyperparameters['folder']
        self.hparams.early_stopping = hyperparameters['early_stopping']
        self.hparams.now = 'Not_trained'

        # 새로운 hyperparameter 추가
        for kv in [[key, value] for key, value in self.init_hyperparameters.items() if key not in self.hparams.keys()]:
            self.hparams[kv[0]] = kv[1]


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        
        return result
    
    def training_epoch_end(self, outputs) :
        
        for key, value in outputs.items() :
            try :
                outputs[key] = torch.mean(outputs[key])
            except :
                continue

        self.hparams.training_loss['epoch : %s' % self.current_epoch] = \
            outputs[list(outputs.keys())[-1]].item()
        
        train_loss = self.hparams.training_loss['epoch : %s' % self.current_epoch]
        validation_loss = self.hparams.validation_loss['epoch : %s' % self.current_epoch]
        
        print('epoch : %s, training loss : %.4f, validation loss : %.4f, ckpt_path = %s/%s' % \
              (self.current_epoch, train_loss, validation_loss, (self.hparams.now + \
                                                                 self.fold_folder),
               ('epoch=%s_val_loss=%.4f' % (self.current_epoch, validation_loss))
              )) 

        return outputs
    

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) 
        loss = self.loss(y_hat, y)
        
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        
        return {'val_loss' : loss}

    
    def validation_epoch_end(self, val_outputs):
        self.hparams.validation_loss['epoch : %s' % self.current_epoch] = \
            float(np.mean([i['val_loss'].item() for i in val_outputs]))

        loss = torch.tensor(np.mean([i['val_loss'].item() for i in val_outputs]))
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        
        return result
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) 
        loss = self.test_loss(y_hat, y) * len(batch)
        return {'loss' : loss, 'len' : len(batch)}
        
    
    def test_epoch_end(self, outputs) :
        sum_loss = 0
        sum_len = 0
        
        for i in outputs :
            sum_loss += i['loss']
            sum_len += i['len']
            
        return {self.test_metric : sum_loss/sum_len}


    def fit(self, train_data, test_data, check_time=True, num=None, train_shuffle=True, scaler=None,
             oversampling=None, random_state=0) :
        
        self = self.train()
        
        if oversampling is not None:
            
            if type(oversampling) == float:
                train_data = self.resample_data_binary(train_data, oversampling, random_state)
                
            else:
                raise Exception('oversampling ratio should be float')
        
        train_data = self.train_data_change(train_data) 
        test_data = self.test_data_change(test_data)
        
        
        if scaler =='time-series':
            train_data = list(map(lambda x : (self.__timeseries_normalize(x[0]), x[1]), train_data))
            test_data = list(map(lambda x : (self.__timeseries_normalize(x[0]), x[1]), test_data))

        else:
            if scaler == 'standard':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = None
            
            if scaler is not None:
                scaler.fit(train_data)
                train_data = scaler.transform(train_data)
                test_data = scaler.transform(test_data)

        
        
        train_dataloader = self.__make_dataloader(train_data, self.hparams.batch_size, train_shuffle)
        test_dataloader = self.__make_dataloader(test_data, self.hparams.test_batch_size, False)

        if check_time:
            self.hparams.now = datetime.now().strftime("%y%m%d_%H:%M:%S")
        
        checkpoint_callback, tb_logger = self.__call_logger(num)
        stop = False
        
        self.early_stop_callback = pl.callbacks.EarlyStopping(min_delta=0.00, patience=self.hparams.patience, verbose=True)
        if self.hparams.early_stopping:
            stop = self.early_stop_callback
        
        self.trainer = pl.Trainer(max_epochs=self.hparams.max_epochs, gpus=self.hparams.gpus, auto_lr_find=self.hparams.auto_lr_find,
                                  logger = tb_logger, checkpoint_callback=checkpoint_callback, early_stop_callback=stop)
        
        self.__save_notebook()
        sleep(1.0)

        name = 'scaler'
        
        if scaler == 'time-series':
            name = 'time-series'

        if num is None:
            path = "./%s/%s/%s.pkl" % (self.hparams.folder, self.hparams.now, name)
        else:
            path = "./%s/%s/%s_fold/%s.pkl" % (self.hparams.folder, self.hparams.now, num, name)
        
        if scaler is not None:                                                    
            with open(path, "wb") as pkl_file:
                pickle.dump(scaler, pkl_file)

        current_file = self.__get_notebook_name()
        sleep(0.5)
        self.__output_HTML(current_file, './%s/%s/' % (self.hparams.folder, self.hparams.now))
        self.trainer.fit(self, train_dataloader, test_dataloader)        
        
        
    def fit_cross_validation(self, data, n_splits, random_state, train_shuffle = True, scaler = None,
                             oversampling=None) :
             
        check_time = True
        n_cv = KFold(n_splits=n_splits, random_state=random_state)
        
        
        
        for num, (train, test) in enumerate(n_cv.split(data)) :
            num += 1
            
            split_index = {'train' : train.tolist(), 'test' : test.tolist()}
            
            try :
                train_set = data[train]
                test_set = data[test]

            except :
                train_set = [data[i] for i in train]
                test_set = [data[i] for i in test]

                self.__init__(self.init_hyperparameters)
                if not check_time :
                    self.hparams.now = now

                self.fit(train_set, test_set, check_time, num, train_shuffle, scaler, oversampling, random_state)
                
                with open("./%s/%s/%s_fold/train_test_split_index.json" % (self.hparams.folder,
                                                                     self.hparams.now, num), "w") as json_file:
                    json.dump(split_index, json_file)
                
                if check_time :
                    now = self.hparams.now

                check_time = False



    def test(self, data, metric_name, loss = None, fold = None) :
        scaler = None
        self = self.eval()
        if fold is None:
            scaler_pkl = glob('%s/%s/*.pkl'% (self.hparams.folder, self.hparams.now))

        else:
            scaler_pkl = glob('%s/%sfold/*.pkl'% (self.hparams.folder, fold))
        

        if len(scaler_pkl) > 0:
            if 'time-series' in scaler_pkl[0]:
                data = list(map(lambda x : (self.__timeseries_normalize(x[0]), x[1]), data))
            else:
                with open(scaler_pkl[0], 'rb') as pkl_file:
                    scaler = pickle.load(pkl_file) 
                    
                data = scaler.transform(data)

        dataloader = self.__make_dataloader(data, self.hparams.test_batch_size, False)

        checkpoint_callback, tb_logger = self.__call_logger()

        trainer = pl.Trainer(max_epochs=self.hparams.max_epochs, gpus = self.hparams.gpus, 
                                  auto_lr_find=False,
                             checkpoint_callback=checkpoint_callback, logger = tb_logger)
        
        if loss is not None :
            self.test_loss = loss
            
        else :
            self.test_loss = self.loss
            
        self.test_metric = metric_name
        trainer.test(self, dataloader)

    
    def test_cross_validation(self, data, metric_name, ckpt_path = None, loss = None) :
        
        if ckpt_path is not None :
            ckpt_list = sorted(self.__best_model_find_algorithm(ckpt_path))
            
        else :

            if os.path.isdir('%s/%s'% (self.hparams.folder, self.hparams.now)) and \
            self.hparams.now != 'Not_trained':
                ckpt_list = sorted(self.__best_model_find_algorithm(self.hparams.now))
                
            else :
                raise Exception('n_fold 학습을 먼저하거나 save file을 불러오세요.(ckpt_path)')
        
        for ckpt in ckpt_list :
            self = self.load_model(ckpt)
            self.hparams.now = ckpt.split('/')[0]
            json_path = "%s/%sfold/train_test_split_index.json" % (self.hparams.folder,
                                                                   ckpt.split('fold')[0])
            with open(json_path, "r") as json_file:
                test_index = json.load(json_file)['test']    


            try :
                test_set = data[test_index]

            except :
                test_set = [data[i] for i in test_index]

            self.test(test_set, metric_name, loss, ckpt.split('fold')[0])


    def load_model(self, ckpt_path) :
        ckpt_path = self.__best_model_find_algorithm(ckpt_path)
        
        if len(ckpt_path) > 1 :
            raise Exception('n_fold를 포함한 데이터를 load 하였습니다. \
            test_cross_validation()을 이용하세요.')
        else : 
            ckpt_path = ckpt_path[0]
           
        loaded_model = self.load_from_checkpoint('./%s/%s.ckpt' % (self.hparams.folder, 
                                                                   ckpt_path), 
                                         hparams_file = './%s/%s/hparams.yaml' % \
                                           (self.hparams.folder, 
                                            ('/').join(ckpt_path.split('/')[:-1])))
        print('succefully load : %s' % ckpt_path)
        
        loaded_model.hparams.now = ckpt_path.split('/')[0]
        return loaded_model


    def __make_dataloader(self, data, batch_size, is_train) :
        if batch_size == -1 :
            batch_size = len(data)
    
        return DataLoader(data, batch_size=batch_size, shuffle = is_train, 
                          num_workers=self.hparams.num_workers)


    def __timeseries_normalize(self, data):
        # (시간, 변수)
        return ((data) - np.mean(data, axis = 0))/ np.std(data, axis = 0)

    def __get_notebook_name(self) :

        time = [os.path.getmtime(i) for i in glob('*.ipynb')]
        return glob('*.ipynb')[np.where(np.max(time) == time)[0][0]]


    def __save_notebook(self):
        display(
            Javascript("IPython.notebook.save_notebook()"),
            include=['application/javascript']
        )

    def __output_HTML(self, current_file, path):
        import codecs
        import nbformat
        exporter = HTMLExporter()
        
        output_notebook = nbformat.read(current_file, as_version=4)
        output, resources = exporter.from_notebook_node(output_notebook)
        codecs.open(path +'saved_ipynb_file.html', 'w', encoding='utf-8').write(output)    
    
    
    def __call_logger(self, num = None) :
        filepath=os.getcwd() + '/%s/%s/{epoch:d}_{val_loss:.4f}' % (self.hparams.folder, 
                                                                    self.hparams.now)
        if num is not None :
            num = str(num) + '_fold'
            filepath = os.getcwd() + '/%s/%s/%s/{epoch:d}_{val_loss:.4f}' % (self.hparams.folder, 
                                                                    self.hparams.now, num) 
            self.fold_folder = '/' + num
            tb_logger = pl.loggers.TensorBoardLogger(save_dir=self.hparams.folder, 
                                                 name = self.hparams.now, version = num)
        else :
            tb_logger = pl.loggers.TensorBoardLogger(save_dir=self.hparams.folder, 
                                                 name = num, version = self.hparams.now)
            self.fold_folder = ''
            
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath = filepath,
            save_top_k=self.hparams.save_top_k,
            monitor='val_loss',
            mode='min')
        
        return checkpoint_callback, tb_logger
    
    
    def __best_model_find_algorithm(self, ckpt_path) :
        ckpt_file = glob(self.hparams.folder +'/'+ ckpt_path +'.ckpt')
        
        if len(ckpt_file) > 0 :
            return [ckpt_file[0].split(self.hparams.folder)[-1][1:-5]]
        
        else :
            ckpt_file = glob(self.hparams.folder +'/' + ckpt_path + '/**/*.ckpt', recursive=True)
            
            group = groupby(ckpt_file, lambda x: x[x.find('fold')-2:x.find('fold')+4])

            best_model_list = []
            for key, items in group : 
                items = [i for i in items]
                loss = [float(i.split('val_loss=')[-1].split('.ckpt')[0]) for i in items]
                file = items[np.where(np.min(loss) == loss)[0][0]]
                file = file.split(self.hparams.folder)[-1][1:-5]
                best_model_list.append(file)
                
            return best_model_list

    def resample_data_binary(self, dataset, pos_ratio=0.2, random_state=0):
        # i : 길이
        y = list(map(lambda x : x[1], dataset))
        
        # pos_idx : 아웃라이어
        pos_idx = np.where(np.array(y)==1)[0]
        neg_idx = np.where(np.array(y)==0)[0]
        
        # 정상 * (정상:비정상 비)
        n_new_pos = int(len(neg_idx) * pos_ratio / (1 - pos_ratio))
        
        np.random.seed(random_state)
        
        # 오버샘플링
        new_pos_idx = np.random.choice(pos_idx, n_new_pos)
        
        total_index = np.concatenate((new_pos_idx, neg_idx))
        np.random.shuffle(total_index)
        
        dataset = [dataset[i] for i in total_index]
        
        return dataset

    def train_data_change(self, train_data):
        return train_data
    
    def test_data_change(self, test_data):
        return test_data