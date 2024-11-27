import sys
import pathlib as P
sys.path.append('..')
prj_root = str(P.Path(__file__).parent.parent)
if prj_root not in sys.path:
    sys.path.append(prj_root)
from util.utils import EarlyStopping
# from models.GCN import *
from models.GCN_hg import *
# from dataset.gcn_dataset import *
from dataset.hg_dataset import *
import dgl
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, auc
from torch.optim.lr_scheduler import OneCycleLR
import warnings
from sklearn.exceptions import UndefinedMetricWarning
# 忽略 UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

class GCN_Trainer(object):

    def __init__(self, args):
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.device = args['device']
        # self.dataset = GCN_Dataset('data', args['dataset_name'], args['batch_size'], self.device)
        self.dataset = DBLPDataset('data', args['dataset_name'], 'GCN', args['batch_size'], self.device)
        self.g = self.dataset.g.to(self.device)
        self.g = dgl.node_subgraph(self.g, {'protein':torch.arange(self.dataset.protein_num)})
        # self.model = GCN(self.dataset.feature_dim, labels_num=self.dataset.go_num, hidden_size=args['hidden_size'])
        self.ppi_etype = ['interacts_0', '_interacts_0']
        self.model = RGCN(in_dim=self.dataset.feature_dim, hidden_dim=128, out_dim=self.dataset.go_num, etypes=self.ppi_etype, num_bases=2)
        self.model = self.model.to(self.device)
        self.model_path = args['model_path']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.loss_func = nn.BCELoss()
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.best_fmax = 0
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.patience = args['patience']
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.005, total_steps=self.epoch*len(self.dataset.train_loader))
        self.train_idx, self.valid_idx, self.test_idx = self.dataset.train_idx, self.dataset.valid_idx, self.dataset.test_idx
        
    def train(self):
        model = self.model
        stopper = EarlyStopping(self.patience, self.model_path)

        for epoch in range(self.epoch):
            # print('Epoch: {:04d}'.format(epoch+1), end='')
            train_loss = self.train_step(self)
            
            mode = 'valid'
            metric_dict, losses = self.test_step(mode=mode)
            val_loss = losses['valid']
            # self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
            #                            + self.logger.metric2str(metric_dict))
            print("Epoch: {}, Train loss: {}, Valid loss: {} ".format(epoch, train_loss, val_loss)
                                       + str(metric_dict))
        #     early_stop = stopper.loss_step(val_loss, self.model)
        #     if early_stop:
        #         # self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
        #         print(('Early Stop!\tEpoch:' + str(epoch)))
        #         # break
        # stopper.load_model(self.model)
        mode = 'test'
        metric_dict, losses = self.test_step(mode=mode)
        for key in metric_dict:
            print(metric_dict[key] + losses[key].cpu())
            
    def train_step(self, logits=None):
            # return self._full_train_step(
            self.model.train()
            loss_all = 0.0
            loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
            for i, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
                pred = self.model(blocks, blocks[0].srcdata['h'])
                # pred = self.model(blocks, blocks[0].srcdata['h'])
                labels = self.dataset.get_label(output_nodes.tolist())
                labels = torch.tensor(labels).to(self.device)
                loss = self.loss_func(pred, labels.float())
                loss_all += loss
                # loss.backward()
                # self.optimizer.zero_grad()
                # self.optimizer.step()
                loss.backward()
                self.optimizer.step
                self.optimizer.zero_grad()
                
                self.scheduler.step()

            loss_all = loss_all / (i+1)
            return loss_all
        
    def test_step(self, mode):
        self.model.eval()

        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            if mode == 'train':
                loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
            elif mode == 'valid':
                loader_tqdm = tqdm(self.dataset.valid_loader, ncols=120)
            elif mode == 'test':
                self.load_model()
                loader_tqdm = tqdm(self.dataset.test_loader, ncols=120)
            y_trues = []
            y_predicts = []

            for i, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
                labels = self.dataset.get_label(output_nodes.tolist())
                labels = torch.tensor(labels).to(self.device)
                with autocast():
                    pred = self.model(blocks, blocks[0].srcdata['h'])
                    loss = self.loss_func(pred, labels.float())
                    loss_all += loss
                    pred = F.sigmoid(pred)
                    y_trues.append(labels.to(torch.float16).detach().cpu())
                    y_predicts.append(pred.to(torch.float16).detach().cpu())
            y_trues = torch.cat(y_trues, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
            # evaluator = self.task.get_evaluator(name='f1')
            # metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
            # fmax, aupr = self.calculate_metrics(y_trues, y_predicts)
            fmax, aupr = self.get_fmax_aupr(y_trues, y_predicts)
            if mode == 'valid':
                if fmax > self.best_fmax:
                    self.best_fmax = fmax
                    self.save_model()
            elif mode == 'test':
                index = self.dataset.test_idx.unsqueeze(1)
                results = torch.cat((index.detach().cpu(), y_predicts), dim=1).numpy()
                np.savetxt(self.dataset.result_path, results)
            print('fmax:{}, aupr:{}'.format(fmax, aupr))
            # metric_dict[mode] = self.calculate_metrics(y_trues, y_predicts.argmax(dim=1).to('cpu'))
            metric_dict[mode] = (fmax, aupr)
            loss_dict[mode] = loss
        return metric_dict, loss_dict

    def calculate_metrics(self, y_true, y_pred):
        num_classes = y_true.shape[1]
        fmax_list = []
        aupr_list = []
        
        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            
            aupr = auc(recall, precision)
            aupr_list.append(aupr)
            
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
            
            fmax = np.nanmax(f1_scores)
            fmax_list.append(fmax)
        
        final_fmax = np.mean(fmax_list)
        final_aupr = np.mean(aupr_list)
        
        return final_fmax, final_aupr
    
    def get_fmax_aupr(self, y_true, y_pred):

        # return self.fmax_tensor(y_true,y_pred), self.aupr_tensor(y_true, y_pred)
        # y_true = np.array(y_true.view(-1))
        # y_pred = np.array(y_pred.view(-1))
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        fmax = np.max(f1_scores)
        aupr = auc(recall, precision)
        return fmax, aupr
    
    def fmax_tensor(self, targets: torch.Tensor, scores: torch.Tensor):
        fmax_ = 0.0, 0.0
        for cut in (c / 100 for c in range(101)):
            cut_sc = (scores >= cut).to(torch.int32)  # 使用PyTorch的方法来转换类型
            correct = torch.sum(cut_sc * targets, dim=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = correct / torch.sum(cut_sc, dim=1)
                r = correct / torch.sum(targets, dim=1)
                p = torch.nanmean(p)
                r = torch.nanmean(r)
            if torch.isnan(p):
                continue
            try:
                fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
            except ZeroDivisionError:
                pass
        return fmax_[0].item()

    def aupr_tensor(self, targets: torch.Tensor, scores: torch.Tensor, top=200):
        # _, indices = torch.topk(scores, k=scores.shape[1] - top, dim=1)
        # scores[:, indices[:, -top:]] = -1e100

        targets = targets.flatten()
        scores = scores.flatten()
        precision, recall, _ = precision_recall_curve(targets, scores)
        aupr = auc(recall, precision)
        
        return aupr
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
    
    
args = {'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_path':prj_root + '/models/gcn_bp',
        'dataset_name': 'bp',
        'hidden_size': 128,
        'epoch':10,
        'batch_size':32,
        'patience':10,
        'lr':0.001,
        'weight_decay':5e-4}

GCN_Trainer = GCN_Trainer(args=args)
GCN_Trainer.train()