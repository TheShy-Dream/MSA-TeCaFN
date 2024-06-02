import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import MMIM
import csv
torch.autograd.set_detect_anomaly(True)

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None,
                 pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.gamma=hp.gamma
        self.theta=hp.theta
        self.record=hp.record
        self.losses=[]
        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = MMIM(hp)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")


        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)

                for p in (mmilb_param + main_param):
                    if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(
            mmilb_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5,
                                                 verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5,
                                                verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        # mem_size = 256
        mem_size = self.hp.mem_size
        tanh=torch.nn.Tanh()
        relu=torch.nn.ReLU()


        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            cross_infonce_loss=0.0
            ap_loss_sum=0.0
            infonce_loss=0.0
            nce_loss = 0.0
            ba_loss = 0.0
            task_loss=0.0
            start_time = time.time()

            left_batch = self.update_batch

            mem_pos_tv = []
            mem_neg_tv = []
            mem_pos_ta = []
            mem_neg_ta = []
            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []
            _loss = 0
            _loss_a = 0
            _loss_v = 0
            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids,v_mask,a_mask = batch_data
                # (0,32) (161,32,20)  tensor:32 (134,32,5) tensor:32 32 32 (32,50) (32,50) 32 32

                # for mosei we only use 50% dataset in stage 1
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                model.zero_grad()
                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask,v_mask,a_mask = \
                        text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                        bert_sent_type.cuda(), bert_sent_mask.cuda(),v_mask.cuda(),a_mask.cuda()
                    if self.hp.dataset == "ur_funny":
                        y = y.squeeze()

                batch_size = y.size(0)

                if stage == 0:  # Neg-lld  默认stage=1
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:  # TASK+BA+CPC  memory
                    mem = {'tv': {'pos': mem_pos_tv, 'neg': mem_neg_tv},
                           'ta': {'pos': mem_pos_ta, 'neg': mem_neg_ta},
                           'va': {'pos': mem_pos_va, 'neg': mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}

                # 训练模型，得到（极大似然loss，平均绝对误差loss，预测值，，）H:0.0
                lld, nce, preds, pn_dic, H,infonce,cross_infonce,ap_loss = model(text, visual, audio, vlens, alens,
                                                   bert_sent, bert_sent_type, bert_sent_mask, y, mem,v_mask,a_mask)

                if stage == 1:  # TASK+BA+CPC
                    y_loss = criterion(preds, y)  # mosei/mosi:L1Loss

                    # update memory
                    if len(mem_pos_tv) < mem_size:
                        mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                        mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                        mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                        mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                        if self.hp.add_va:
                            mem_pos_va.append(pn_dic['va']['pos'].detach())
                            mem_neg_va.append(pn_dic['va']['neg'].detach())
                    else:  # memory is full! replace the oldest with the newest data
                        oldest = i_batch % mem_size
                        mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                        mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                        mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                        mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()

                        if self.hp.add_va:
                            mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                            mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                    if self.hp.contrast:
                        loss = y_loss + self.alpha * nce - self.beta * lld +self.gamma*(infonce+cross_infonce) +self.theta*ap_loss# 公式 14
                    else:
                        loss = y_loss
                    if i_batch > mem_size:
                        #ur-funny
                        pass
                        #loss -= self.beta * H
                    loss.backward()
                elif stage == 0:
                    # maximize likelihood equals minimize neg-likelihood
                    loss = -lld
                    loss.backward()
                else:
                    raise ValueError('stage index can either be 0 or 1')

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size  # CPC loss
                ba_loss += (-H - lld) * batch_size  # BA loss
                infonce_loss += infonce.item() * batch_size
                cross_infonce_loss += cross_infonce.item() * batch_size
                ap_loss_sum+=ap_loss.item()*batch_size
                task_loss += y_loss.item() * batch_size if stage==1 else 0  # Task loss
                if i_batch % self.hp.log_interval == 0 and i_batch > 0 and stage==1:#
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size  #
                    avg_ba = ba_loss / proc_size
                    avg_infonce=infonce_loss/proc_size
                    avg_cross_infonce=cross_infonce_loss/proc_size
                    avg_ap_loss=ap_loss_sum/proc_size
                    print(
                        'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f} | INFONCE {:.3f} | CrossInfoNCE {:.4f} | APloss{:.4f}'.
                            format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                                   'TASK+All' if stage == 1 else 'Neg-lld',
                                   avg_loss, avg_nce, avg_ba,avg_infonce,avg_cross_infonce,avg_ap_loss))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    infonce_loss=0.0
                    cross_infonce_loss=0.0
                    ba_loss = 0.0
                    task_loss = 0
                    ap_loss_sum=0.0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0

            results = []
            truths = []

            # 验证模型
            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids,v_mask,a_mask = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        v_mask, a_mask=v_mask.cuda(),a_mask.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset == 'iemocap':
                            y = y.long()

                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    _, _, preds, _, _,_,_,_= model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type,
                                              bert_sent_mask,v_mask=v_mask,a_mask=a_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()  # 任务误差，MAE
                    elif self.hp.dataset == "ur_funny" and test:
                        criterion = nn.CrossEntropyLoss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)  # torch.Size([229, 1])
            truths = torch.cat(truths)  # torch.Size([229, 1])


            #test
            # print(truths)
            # McNemar_test = {
            #     'results': results.cpu().squeeze(1),
            #     'truths': truths.cpu().squeeze(1)
            # }
            # data_frame = pd.DataFrame(data=McNemar_test)
            # data_frame.to_csv('TeFNA_mosi_MTest.csv')

            return avg_loss, results, truths

        best_valid = 1e8
        best_loss = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs + 1):
            start = time.time()

            self.epoch = epoch

            # maximize likelihood
            if self.hp.contrast:
                pass
                #train_loss = train(model, optimizer_mmilb, criterion, 0)

            # minimize all losses left
            train_loss = train(model, optimizer_main, criterion, 1)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)

            end = time.time()
            duration = end - start
            scheduler_main.step(val_loss)  # Decay learning rate by validation loss

            # validation F1
            # print("-" * 50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration,
                                                                                                   val_loss, test_loss))
            # print("-" * 50)

            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny" and test_loss < best_loss:
                    eval_humor(results, truths, True)
                    best_epoch = epoch
                    best_loss = test_loss
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    self.hp.modelname = self.hp.dataset + now_time + '_loss' + str(best_loss)
                    print(f"Saved model at pre_trained_best_models_mosei/" + self.hp.modelname + ".pt!")
                    save_model(self.hp, model)
                    best_results = results
                    best_truths = truths
                elif not self.hp.dataset == "ur_funny" and test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    # 验证模型
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        res = eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        res = eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)

                    # print(res['to_exl'])

                    best_results = results
                    best_truths = truths
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    mae = str(best_mae).split('.')[0] + '.' + str(best_mae).split('.')[1][:5]
                    self.hp.modelname = self.hp.dataset + now_time + '_eam' + mae
                    print(f"Saved model at pre_trained_best_models_mosei/" + self.hp.modelname + ".pt!")
                    save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break
        print(f'Best epoch: {best_epoch}')
        # 验证模型：
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            self.best_dict = eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)
        elif self.hp.dataset == 'ur_funny':
            self.best_dict = eval_humor(best_results, best_truths, True)
            return self.best_dict, self.hp.modelname

        # -------------保存识别结果----------
        print(best_results)
        print(best_truths)
        McNemar_test = {
            'results': best_results.cpu().squeeze(1),
            'truths': best_truths.cpu().squeeze(1)
        }
        data_frame = pd.DataFrame(data=McNemar_test)
        data_frame.to_csv('TeFNA_mosei_MTest.csv')

        return self.best_dict['to_exl'], self.hp.modelname
        # self.to_exl['mae'].append()

        # to_exl = self.best_dict['to_exl']
        # data_frame = pd.DataFrame(
        #     data={'mae': to_exl[0], 'coor': to_exl[1], 'acc7':to_exl[2], 'acc2_1':to_exl[3],
        #           'acc2_2': to_exl[4], 'f1_1': to_exl[5], 'f1_2': to_exl[6]}, index=[1])
        # data_frame.to_csv('pre_trained_best_models/'+self.hp.modelname+'.csv')

        sys.stdout.flush()
