import torch
import os
import time
from torch import optim
from torch.optim import lr_scheduler 
import numpy as np
import matplotlib.pyplot as plt

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.loss import get_loss

from models import Transformer, Informer, Autoformer, Crossformer, iTransformer, PatchTST, \
    Linear, CrossGNN, WITRAN, BiMamba4TS
# , DMamba, TimeMachine

models_dict={
    'Transformer': Transformer,
    'Informer': Informer,
    'Autoformer': Autoformer,
    'DLinear': Linear,
    'Crossformer': Crossformer,
    'PatchTST': PatchTST, 
    'iTransformer': iTransformer,
    'CrossGNN': CrossGNN,
    'WITRAN': WITRAN,
    # 'TimeMachine': TimeMachine,
    'BiMamba4TS': BiMamba4TS,
    # 'DMamba': DMamba
}

optimizer_catagory = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}

loss_funcs = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
    "huber": torch.nn.HuberLoss(reduction='mean', delta=1.0)
}

class LTF_Trainer():
    def __init__(self, args, task, setting, corr=None) -> None:
        self.args = args
        self.setting = setting
        self.c_path = os.path.join(args.checkpoints, str(args.seed), task, setting)
        self.r_path = os.path.join(args.results, str(args.seed), task, setting)
        self.p_path = os.path.join(args.predictions, str(args.seed), task, args.model)
        self.device = torch.device(args.device)
        if hasattr(args, 'use_gcn') and args.use_gcn:
            model = models_dict[self.args.model].Model(self.args, corr=corr).to(self.device)
        else:
            model = models_dict[self.args.model].Model(self.args).to(self.device)
        if args.use_multi_gpu:
            self.model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            self.model = model

    def train(self, data):
        print(f'>>>>>> The model checkpoint path will be: {self.c_path}')
        print(f'>>>>>> The traing result path will be: {self.r_path}')
        if not os.path.exists(self.c_path):
            os.makedirs(self.c_path)
        if not os.path.exists(self.r_path):
            os.makedirs(self.r_path)

        if self.args.is_training == 1:
            print('>>>>>> start training : {} >>>>>>'.format(self.setting))
        elif self.args.is_training == 2:
            print('>>>>>> keep  training : {} >>>>>>'.format(self.setting))
            self.model.load_state_dict(torch.load(os.path.join(self.c_path, "checkpoint.pth")))
        else:
            print('>>>>>> current training settings are not supported! <<<<<<')
            exit()

        train_loader = data['train_loader']
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = optimizer_catagory[self.args.opt](self.model.parameters(), lr=self.args.learning_rate)
        loss_func = loss_funcs[self.args.loss]

        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        train_loss_list = []
        vali_loss_list = []
        epoch_list = []
        
        for epoch in range(self.args.train_epochs):
            epoch_list.append(epoch+1)
            epoch_time = time.time()
            interval = 100
            train_loss = []
            
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i==0:
                    current_time = time.time()
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                # For those with Decoder, set Decoder input of them
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # Graph A
                if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                    outputs, res, A = outputs
                else:
                    outputs, res = outputs

                f_dim = -1 if self.args.task == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)
                loss = loss_func(outputs, batch_y)
                # deform patch loss
                if self.args.deform_patch:
                    diff_loss, attns = res
                    loss += diff_loss
                else:
                    attns = res

                train_loss.append(loss.item())

                if (i + 1) % interval == 0:
                    speed = (time.time() - current_time) / interval
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tepoch_{} iter_{} | loss: {:.6f} | speed: {:.3f} (s/iter) | left time: {:.3f}s".format(epoch + 1, 
                                                                                                               i + 1, 
                                                                                                               loss.item(), 
                                                                                                               speed,
                                                                                                               left_time))
                    current_time = time.time()
                
                loss.backward()
                optimizer.step()
            
                if self.args.lradj == 'type4':
                    adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                index='0'
                for a in [A]:
                    a = a[:self.args.enc_in]
                    a = a.detach().cpu().numpy()
                    plt.figure(figsize=(7.2, 7.2))
                    plt.imshow(a, cmap='coolwarm', interpolation='nearest')
                    plt.colorbar()
                    # 设置坐标刻度
                    xticks = np.arange(0, a.shape[1], 100)
                    yticks = np.arange(0, a.shape[0], 100)
                    plt.xticks(xticks)
                    plt.yticks(yticks)
                    plt.title(f'Original Embedding A{index} Heatmap')
                    plt.savefig(self.r_path+'/A{}_{}.png'.format(index, self.args.dataset_name))
                    plt.clf()
                    index=''
            
            train_loss = np.average(train_loss)
            train_loss_list.append(train_loss)
            vali_loss = self.validate(data['val_loader'], loss_func)
            vali_loss_list.append(vali_loss)
            # test_loss = self.validate(data['test_loader'], loss_func)
            print('\tspeed: {:.3f}s/iter; left time: {:.3f}s'.format(speed, left_time))
            print("epoch {} finished. cost time {:.3f}s".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, self.c_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj != 'type4':
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {:6f}'.format(scheduler.get_last_lr()[0]))

        _, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(epoch_list, train_loss_list, 'r', marker='x', label='train_loss')
        ax.plot(epoch_list, vali_loss_list, 'b', marker='*', label='vali_loss')
        ax.legend()
        ax.set_title('loss func value')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        plt.savefig(self.r_path+'/loss.png', dpi=100)
        plt.clf()

    def validate(self, vali_loader, loss_func):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                    outputs, res, _ = outputs
                else:
                    outputs, res = outputs

                f_dim = -1 if self.args.task == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = loss_func(pred, true)
                # deform patch loss
                if self.args.deform_patch:
                    diff_loss, _ = res
                    loss += diff_loss.detach().cpu()
                
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test_loader):
        print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        self.model.load_state_dict(torch.load(os.path.join(self.c_path, "checkpoint.pth")))
        self.model.eval()
        preds = []
        trues = []
        mse_list, rmse_list, mae_list, mape_list = [], [], [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                    outputs, _, _ = outputs
                else:
                    outputs, _ = outputs

                f_dim = -1 if self.args.task == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                preds = outputs.detach().cpu()
                trues = batch_y.detach().cpu()
                mse, rmse, mae, mape = get_loss(preds, trues)
                mse_list.append(float(mse))
                rmse_list.append(float(rmse))
                mae_list.append(float(mae))
                mape_list.append(float(mape))
        
        mse, rmse, mae, mape = np.average(mse_list), \
                            np.average(rmse_list), \
                            np.average(mae_list), \
                            np.average(mape_list)

        result_path = os.path.join(self.r_path, "result.txt")
        print('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}, mape:{:.6f}'.format(mse, mae, rmse, mape))
        f = open(result_path, 'a')
        f.write(self.setting + "\n")
        f.write('mse:{:.6f}, mae:{:.6f}, rse:{:.6f}, mape:{:.6f} ==== lr={} seed={}\n\n'.format(
            mse, mae, rmse, mape, self.args.learning_rate, self.args.seed
            ))
        f.close()

        return round(float(mse), 6)

    def predict(self, pred_loader):
        print('>>>>>>>start predicting : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        self.model.load_state_dict(torch.load(os.path.join(self.c_path, "checkpoint.pth"), map_location="cuda:0"))

        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                    outputs, _, _ = outputs
                else:
                    outputs, _ = outputs
                    
                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                input = batch_x[:, -self.args.seq_len:, :].detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputs.append(input)

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        folder_path = self.p_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + f'/{self.args.model}_pred.npy', preds)
        np.save(folder_path + f'/{self.args.model}_true.npy', trues)
        np.save(folder_path + f'/{self.args.model}_in.npy', inputs)
        
        print('Prediction done...')
