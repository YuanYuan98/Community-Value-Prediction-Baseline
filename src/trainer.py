#torch-1.0-py3
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import KFold
from Data_loader import Data_loader
from config import Config
from sklearn.ensemble import RandomForestRegressor
#import lightgbm as lgb
from DeepWalk import DeepWalk
from dataset_processing import *
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import time
import torch.nn.functional as F
import torch
import os
import torch.nn as nn


class Train:
    def __init__(self,args):
        # Dataloader = Data_loader()
        # Data = Dataloader.laod()
        # self.deepwalk_embedding = Dataloader.embed_process()
        # self.X = Data[0]
        # self.y = Data[1]
        # self.feature_matrix = Data[2]
        # self.adj_matrix = Data[3]
        # self.edge_attr = Data[4] 
        # self.label_matrix = Data[5]
        # self.community_partition_index = Data[6]

        self.args = args
        self.config = Config
        self.kf = KFold(n_splits=10, random_state=np.random.RandomState(630), shuffle=True)
        self.res = []
        self.mse_folds = []
        self.rmse_folds = []
        # self.model = DeepWalk().cuda()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def init_model(self):
        # model
        model = DeepWalk(self.args).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.args.patience, min_lr=1e-5)
        return (model, optimizer, scheduler)

    def train(self, model,optimizer,scheduler,num_train, data_x,data,train_loaders, valid_id):
        data_x = data_x.to(self.args.device)
        model.train()
        min_loss = 1e5
        early_stop_num = 0
        for epoch in range(self.args.epoch_num):
            print('Epoch {}:'.format(epoch))
            train_loss = 0.0
            num_iters = len(train_loaders)
            for batch_idx, train_ids in enumerate(train_loaders):
                optimizer.zero_grad()
                out = model(data_x)
                y_true = data.y[train_ids].unsqueeze(dim=1).to(self.args.device)
                loss = F.l1_loss(out[train_ids], y_true, reduction='sum')
                #loss = F.mse_loss(out[train_ids], y_true, reduction='sum')
                train_loss += loss.item()
                #train_loss += F.l1_loss(out[train_ids], y_true, reduction='sum').item()
                with torch.autograd.detect_anomaly():
                    loss.backward()
                optimizer.step()
            train_loss /= num_train
            val_loss,_,_ = self.test(model, data_x, data, valid_id)
            scheduler.step(val_loss)
            print('train',train_loss)
            print('test',val_loss)
            if val_loss < min_loss:
                torch.save(model.state_dict(),
                            '/home/yuanyuan/workplace/influence/model/DWmodel_{}_{}_{}'.format(self.args.lr, self.args.weight_decay,
                                                            self.args.batch_size))
                print("!!!!!!!!!! Model Saved !!!!!!!!!!")
                min_loss = val_loss
                early_stop_num = 0
            else:
                early_stop_num += 1
            if early_stop_num > self.args.early_stop_num:
                break

    def test(self,model, data_x, data, test_ids):
        model.eval()
        data_x = data_x.to(self.args.device)
        out = model(data_x)
        max_y = max(out[test_ids]).item()
        min_y = min(out[test_ids]).item()
        mae = F.l1_loss(out[test_ids], data.y[test_ids].unsqueeze(dim=1).to(self.args.device), reduction='mean')
        mse = F.mse_loss(out[test_ids], data.y[test_ids].unsqueeze(dim=1).to(self.args.device), reduction='mean')
        nrmse = np.sqrt(mse.cpu().detach().numpy())/(max_y-min_y)
        return mae.item(),mse.item(),nrmse
    
    def trainstep(self):
        acc_folds = []
        mse_folds = []
        rmse_folds = []
        print('------------------------- Loading data -------------------------')
        dataset2 = create_dataset_DW(self.args.dataset_path2)
        dataset = create_dataset_global(self.args.dataset_path)
        self.args.num_communities = int(dataset.community.max().item() + 1)
        train_ids, test_ids = split_train_test(self.args.num_communities, self.args.n_folds, self.args.rnd_state)

        for fold in range(self.args.n_folds):
            
            # train_loaders, num_train = make_batch(train_ids[fold], self.args.batch_size, self.args.rnd_state)

            # print('\nFOLD {}, train {}, test {}'.format(fold, num_train, len(test_ids[fold])))

            print('\n------------------------- Initialize Model -------------------------')
            model, optimizer, scheduler = self.init_model()

            # print('\n------------------------- Training -------------------------')
            # self.train(model,optimizer,scheduler,num_train,dataset2,dataset, train_loaders, test_ids[fold])
            

            print('\n------------------------- Testing -------------------------')
            model.load_state_dict(torch.load('/home/yuanyuan/workplace/influence/model/DWmodel_{}_{}_{}'.format(self.args.lr, 
                self.args.weight_decay, self.args.batch_size)))

            test_loss,mse,rmse = self.test(model, dataset2, dataset, test_ids[fold])
            acc_folds.append(test_loss)
            mse_folds.append(mse)
            rmse_folds.append(rmse)

            #np.save('/home/yuanyuan/workplace/influence/data/save/acc_folds.npy',np.array(acc_folds))
            
            print('---------------------------------------')
            print('test_loss: {}'.format(test_loss))

        print(acc_folds)
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(self.args.n_folds, np.mean(acc_folds), np.std(acc_folds)))
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(self.args.n_folds, np.mean(mse_folds), np.std(mse_folds)))
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(self.args.n_folds, np.mean(rmse_folds), np.std(rmse_folds)))
        print('batch_size-{} lr-{} weight_decay-{}'.format(self.args.batch_size,self.args.lr,self.args.weight_decay))
    
    def svr_fit(self):
        regr = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
            tol=0.001, C=2000, epsilon=0.1, shrinking=True, cache_size=200, 
            verbose=False, max_iter=-1)

        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            regr.fit(X_train,y_train)
            y_pred = regr.predict(X_test)
            max_y = max(y_test)
            min_y = min(y_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test,y_pred)
            rmse = np.sqrt(mse)/(max_y-min_y)
            self.res.append(mae)
            self.mse_folds.append(mse)
            self.rmse_folds.append(rmse)
            print('Mean Absolute Error:', mae)
            print('##############################################')
            print('Mean Squared Error:', mse)
            print('##############################################')
            print('R Mean Squared Error:', rmse)
            print('##############################################')
        print('------------------------------')
        print(self.res)
        print(np.mean(self.res), np.std(self.res))
        print(np.mean(self.mse_folds), np.std(self.mse_folds))
        print(np.mean(self.rmse_folds), np.std(self.rmse_folds))

    def randomforest(self):
        regr = RandomForestRegressor(oob_score=True, random_state=0, 
                n_estimators=170,
                max_depth=13,
                min_samples_split=30,
                min_samples_leaf=10,
                max_features=11)
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            regr.fit(X_train,y_train)
            y_pred = regr.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            self.res.append(mae)
            print('Mean Absolute Error:', mae)
            print('##############################################')
        print('------------------------------')
        print(self.res)
        print(np.mean(self.res), np.std(self.res))

    def lgb_fit(self):
        """
        Args:
            config: xgb  {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
            X_train:array like, shape = n_sample * n_feature
            y_train:  shape = n_sample * 1
        """
        params = self.config.params
        max_round = self.config.max_round
        early_stop_round = self.config.early_stop_round

        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            if self.config.categorical_feature is not None:
                dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=self.config.categorical_feature)
                dvalid = lgb.Dataset(X_test, label=y_test, categorical_feature=self.config.categorical_feature)
            else:
                dtrain = lgb.Dataset(X_train, label=y_train)
                dvalid = lgb.Dataset(X_test, label=y_test)
            regr = lgb.train(params, dtrain, max_round, valid_sets = dvalid,early_stopping_rounds=early_stop_round)
            y_pred = regr.predict(X_test)
            max_y = max(y_test)
            min_y = min(y_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test,y_pred)
            rmse = np.sqrt(mse)/(max_y-min_y)
            self.res.append(mae)
            self.mse_folds.append(mse)
            self.rmse_folds.append(rmse)
            print('Mean Absolute Error:', mae)
            print('##############################################')
            print('Mean Squared Error:', mse)
            print('##############################################')
            print('R Mean Squared Error:', rmse)
            print('##############################################')
        print('------------------------------')
        print(self.res)
        print(np.mean(self.res), np.std(self.res))
        print(np.mean(self.mse_folds), np.std(self.mse_folds))
        print(np.mean(self.rmse_folds), np.std(self.rmse_folds))

    def train_classic(self,m='lgb'):
        if m=='svr':
            self.svr_fit()
        elif m=='lgb':
            self.lgb_fit()
        elif m=='rfst':
            self.randomforest()


