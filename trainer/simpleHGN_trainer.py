import sys
import pathlib as P
sys.path.append('..')
prj_root = str(P.Path(__file__).parent.parent)
if prj_root not in sys.path:
    sys.path.append(prj_root)
from util.utils import EarlyStopping
from models.SimpleHGN import *
from dataset.hg_dataset import *
from time import time
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import dgl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, auc
import pathlib as P
import concurrent.futures

import warnings
from warnings import catch_warnings, simplefilter
from sklearn.exceptions import UndefinedMetricWarning
# 忽略 UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)



class GIN_Trainer(object):

    def __init__(self, args):
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.device = args['device']
        self.dataset = DBLPDataset('data', args['dataset_name'], 'gin', args['batch_size'], self.device)
        self.g = self.dataset.g.to(self.device)
        # def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
        #         num_layers, heads, feat_drop, negative_slope,
        #         residual, beta, ntypes):
        self.model = SimpleHGN(edge_dim=4, num_etypes=6, in_dim=self.dataset.feature_dim, hidden_dim=128, num_classes=self.dataset.go_num,
                               num_layers=2, heads=[4,4], feat_drop=0.5, negative_slope=0.2, residual=True,
                               beta=0.5, ntypes=self.dataset.node_type)
        self.model = self.model.to(self.device)
        self.model_path = args['model_path']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.best_fmax = 0
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.patience = args['patience']
        self.scheduler = OneCycleLR(self.optimizer,max_lr=0.001,total_steps=self.epoch*len(self.dataset.train_loader))
        self.train_idx, self.valid_idx, self.test_idx = self.dataset.train_idx, self.dataset.valid_idx, self.dataset.test_idx
        # self.label = self.dataset.get_label()
        # self.input_feature = self.dataset.input_feature.to(self.device)

        self.category = 'protein'

        self.meta_paths_dict = {
                'mp_0': [('protein', 'interacts_0', 'go_annotation')],
                'mp_1': [('protein', 'interacts_1', 'protein')],
                'mp_2': [('go_annotation', 'interacts_2', 'go_annotation')]
            }

    def train(self):
        # train_dataset = DBLPDataset(features_list, self.device)
        # dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        model = self.model
        stopper = EarlyStopping(self.patience, self.model_path)

        for epoch in range(self.epoch):
            # print('Epoch: {:04d}'.format(epoch+1), end='')
            time_0 = time()
            train_loss = self.train_step(self)
            
            modes = ['valid']
            metric_dict, losses = self.test_step(modes=modes)
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
        modes = ['test']
        metric_dict, losses = self.test_step(modes=modes)
        for key in metric_dict:
            print(metric_dict[key] + losses[key].cpu())
            
                
    def train_step(self,logits=None):
            # return self._full_train_step()
            self.model.train()
            loss_all = 0.0
            loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
            for i, (ntype_mp_name_input_nodes_dict, seed,ntype_mp_name_block_dict) in enumerate(loader_tqdm):
            # for i, (seed, subgraph_protein_nodes, subgraph) in enumerate(loader_tqdm): 
                subgraph = dgl.node_subgraph(self.g, ntype_mp_name_input_nodes_dict)
                # subgraph = blocks_to_hetero_graph(ntype_mp_name_block_dict)
                # h_dict = {}
                # mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]

                # for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                #     h_dict[meta_path_name] = self.input_feature.forward_nodes({self.category: input_nodes.to(self.device)})

                # label = self.label[seed[self.category]].to(self.device)
                if 'protein' not in seed.keys():
                    break
                seed = seed['protein']
                label = self.dataset.get_label(seed.cpu().numpy().tolist())
                # pred = self.model(merged_graph, h_dict={self.category:h_dict})
                pred = self.model(subgraph, subgraph.ndata['h'])

                # seed_indices = [torch.where(subgraph_protein_nodes == x)[0].item() for x in seed]
                
                seed_indices = [torch.where(ntype_mp_name_input_nodes_dict['protein'] == x)[0].item() for x in seed]
                pred = pred['protein'][seed_indices]
                # label = th.argmax(th.tensor(label), dim=1).to(self.device)
                label = th.tensor(label).to(self.device)
                # train_loss = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
                train_loss = self.loss_func(pred, label.float())
                loss_all += train_loss
                # self.optimizer.zero_grad()
                # train_loss.backward()
                # self.optimizer.step()
                accumulation_steps = 4  # 每个大批次分成的小批次数量
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.scheduler.step()
                # 每 accumulation_steps 个小批次进行一次优化更新
                # if (i + 1) % accumulation_steps == 0:
                #     self.optimizer.step()
            
            return loss_all / (i + 1)

    def test_step(self, modes):
        self.model.eval()

        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.dataset.valid_loader, ncols=120)
                elif mode == 'test':
                    self.load_model()
                    loader_tqdm = tqdm(self.dataset.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (ntype_mp_name_input_nodes_dict, seed,ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                # for i, (seed, subgraph_protein_nodes, subgraph) in enumerate(loader_tqdm):
                    seed = seed['protein']
                    subgraph = dgl.node_subgraph(self.g, ntype_mp_name_input_nodes_dict)
                    # subgraph = blocks_to_hetero_graph(ntype_mp_name_block_dict)
                    label = self.dataset.get_label(seed.cpu().numpy().tolist())
                    pred = self.model(subgraph, subgraph.ndata['h'])

                    # seed_indices = [torch.where(subgraph_protein_nodes == x)[0].item() for x in seed]
                    seed_indices = [torch.where(ntype_mp_name_input_nodes_dict['protein'] == x)[0].item() for x in seed]
                    pred = pred['protein'][seed_indices]
                    # label = th.argmax(th.tensor(label), dim=1).to(self.device)
                    label = th.tensor(label).to(self.device)
                    # loss = F.nll_loss(pred, label.long())
                    loss = self.loss_func(pred, label.float())
                    loss_all += loss
                    pred = F.sigmoid(pred)
                    y_trues.append(label.detach().cpu())
                    y_predicts.append(pred.detach().cpu())
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                # evaluator = self.task.get_evaluator(name='f1')
                # metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                # fmax, aupr = self.calculate_metrics(y_trues, y_predicts)
                fmax, aupr = self.get_fmax_aupr(y_trues, y_predicts)
                print('fmax:{}, aupr:{}'.format(fmax, aupr))
                # metric_dict[mode] = self.calculate_metrics(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                if mode == 'valid':
                    if fmax > self.best_fmax:
                        self.best_fmax = fmax
                        self.save_model()
                elif mode == 'test':
                    index = self.dataset.test_idx.unsqueeze(1)
                    results = torch.cat((index.detach().cpu(), y_predicts), dim=1).numpy()
                    np.savetxt(self.dataset.result_path, results)
                metric_dict[mode] = (fmax, aupr)
                loss_dict[mode] = loss
        return metric_dict, loss_dict
                
                # h_dict = self.input_feature
                # h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
                # pred = self.model(self.g, h_dict)
                # masks = {}
                # if mode == "valid":
                #     pred = pred[self.valid_idx]
                # elif mode == "test":
                #     pred = pred[self.test_idx]
                
                # loss = self.loss_func(pred, label)
                # return loss


            # loss_all = 0.0
            # loader_tqdm = tqdm(self.train_loader, ncols=120)
            # for i, (ntype_mp_name_input_nodes_dict, seed,g) in enumerate(loader_tqdm):
            #     # mmd_loss=0
            #     # l1_loss=0
            #     # cls_lossS_MP=0
            #     h_dict = {}
            #     mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
            #     for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
            #         h_dict[meta_path_name] = self.model.input_featureS.forward_nodes({self.category: input_nodes})
            #     label = self.label[seed[self.category]].to(self.device)
            #     pred = self.model(g, h_dict={self.category:h_dict})

            #     loss = self.loss_func(pred, label)
            #     loss_all += loss
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            # return loss_all / (i + 1)

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
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        fmax = np.max(f1_scores)
        aupr = auc(recall, precision)
        return fmax, aupr

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

def blocks_to_hetero_graph(blocks):
    graph_list = []
    
    node_types = ['protein', 'go_annotation']
    edge_types = ['interacts_0', 'interacts_1', 'interacts_2', '_interacts_0', '_interacts_1', '_interacts_2']
    edge_types_full = {'interacts_0':('protein', 'interacts_0', 'go_annotation'),
                        '_interacts_0':('go_annotation', '_interacts_0', 'protein'),
                        'interacts_1':('protein', 'interacts_1', 'protein'),
                        '_interacts_1':('protein', '_interacts_1', 'protein'),
                        'interacts_2':('go_annotation', 'interacts_2', 'go_annotation'),
                        '_interacts_2':('go_annotation', '_interacts_2', 'go_annotation')}

    for block in blocks:
        graph_data = {}
        node_data = {}
        for ntype in node_types:
            node_data[ntype] = block.nodes[ntype].data['h']
        for etype in edge_types:
            # node_l = block.edges(etype=etype)[0]
            # node_r = block.edges(etype=etype)[1]
            
            graph_data[edge_types_full[etype]] = block.edges(etype=etype)
        g = dgl.heterograph(graph_data)
        for ntype in node_types:
            g.nodes[ntype].data['h'] = node_data[ntype]
        graph_list.append(g)
    
    return graph_list

args = {'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_path':prj_root + '/models/simpleHGN_cc',
        'dataset_name': 'cc',
        'epoch':50,
        'batch_size':16,
        'patience':10,
        'lr':0.05,
        'weight_decay':5e-4}

GIN_Trainer = GIN_Trainer(args=args)
GIN_Trainer.train()