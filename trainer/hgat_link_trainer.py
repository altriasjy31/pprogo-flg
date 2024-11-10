from models.hgat_link import *
from dataset.hg_dataset import *
from time import time
import torch.optim as optim
import dgl
from dgl.sampling import RandomWalkNeighborSampler
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc


class HGAT_Trainer(object):

    def __init__(self, args):
        self.dataset = DBLPDataset()
        self.g = self.dataset.g
        self.model = HGATLinkPrediction(self.dataset.node_type)
        self.model = self.model.to(self.device)
        self.model_path = args['model_path']
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_func = nn.BCELoss()
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.patience = args['patience']
        self.device = args['device']
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.train_idx, self.valid_idx = self.dataset.get_split()
        self.label = self.dataset.get_label()

        # self.meta_paths_dict = {
        #         'UPU1': [('user', 'user-buy-poi', 'poi'), ('poi', 'user-buy-poi-rev', 'user')],
        #         'UPU2': [('user', 'user-click-poi', 'poi'), ('poi', 'user-click-poi-rev', 'user')],
        #         'USU': [('user', 'user-buy-spu', 'spu'), ('spu', 'user-buy-spu-rev', 'user')],
        #         'UPSPU1': [('user', 'user-buy-poi', 'poi'), ('poi', 'poi-contain-spu', 'spu'),
        #                    ('spu', 'poi-contain-spu-rev', 'poi'), ('poi', 'user-buy-poi-rev', 'user')
        #                    ],
        #         'UPSPU2': [
        #             ('user', 'user-click-poi', 'poi'), ('poi', 'poi-contain-spu', 'spu'),
        #             ('spu', 'poi-contain-spu-rev', 'poi'), ('poi', 'user-click-poi-rev', 'user')
        #         ]
        #     }
        self.meta_paths_dict = {
                'mp_0': [('protein', 'interacts_0', 'go_annotation')],
                'mp_1': [('protein', 'interacts_1', 'protein')],
                'mp_2': [('go_annotation', 'interacts_2', 'go_annotation')]
            }

    def train(self):
        # train_dataset = DBLPDataset(features_list, self.device)
        # dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        sampler = dgl.dataloading.NeighborSampler([20])
        dataloader = dgl.dataloading.DataLoader(
            hg,
            nids = {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
            sampler  = sampler,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False,
            num_worker = 4
        )
        
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        for epoch in range(self.epoch):
            # print('Epoch: {:04d}'.format(epoch+1), end='')
            time_0 = time()
            
            for input_nodes, output_nodes, blocks in dataloader:
                h_dict = {ntype: blocks[0].srcdata['h'][ntype] for ntype in blocks[0].ntypes}
                train_loss = self.train_step(self)
                pred = model(blocks, h_dict)
                labels = blocks[-1].dstdata['label']

                loss = self.loss_func(pred, labels)
            model.train()
            optimizer.zero_grad()
            val_loss = self.test_step(self, 'valid')
            fmax, aupr = self.cal_fmax_aupr(pred, labels)
            self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}, Fmax: {fmax:.4f}, AUPR:{aupr:.4f}. ")
            
                
    def train_step(self,logits=None):
            # return self._full_train_step()
            self.model.train()
            loss_all = 0.0
            loader_tqdm = tqdm(self.train_loader, ncols=120)
            for i, (ntype_mp_name_input_nodes_dict, seed,g) in enumerate(loader_tqdm):
                # mmd_loss=0
                # l1_loss=0
                # cls_lossS_MP=0
                h_dict = {}
                mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]

                for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                    h_dict[meta_path_name] = self.model.input_featureS.forward_nodes({self.category: input_nodes})

                label = self.label[seed[self.category]].to(self.device)
                pred = self.model(self.g, h_dict={self.category:h_dict})
                # #TODO mmd_loss:homo_out,S_label，clabel_predT1
                # for i in range(len(clabel_predSs)):
                #     mmd_loss += self.lmmd(list(homo_outS.values())[i], list(homo_outT.values())[i], S_label, torch.nn.functional.softmax(clabel_predTs[i], dim=1))
                # #TODO l1_loss:clabel_pred
                # for i in range(len(clabel_predTs)):
                #     for j in range(i+1,len(clabel_predTs)):
                #         l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(clabel_predTs[j], dim=1)
                #                                             - torch.nn.functional.softmax(clabel_predTs[i], dim=1)) )
                # #TODO cls_lossS
                # for i in range(len(clabel_predSs)):
                #     cls_lossS_MP += F.nll_loss(F.log_softmax(clabel_predSs[i], dim=1), S_label.long())
                # cls_lossS = F.nll_loss(F.log_softmax(clabel_predS, dim=1), S_label.long())
                # cls_lossT = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

                # loss = cls_lossS + self.gamma * (mmd_loss + l1_loss) + cls_lossT
                loss = self.loss_func(pred, label)
                loss_all += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss_all / (i + 1)

    def test_step(self, mode):
            self.model.eval()

            with torch.no_grad():
                loss_all = 0.0
                h_dict = self.input_feature
                h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
                pred = self.model(self.g, h_dict)
                masks = {}
                if mode == "valid":
                    pred = pred[self.valid_idx]
                elif mode == "test":
                    pred = pred[self.test_idx]
                
                loss = self.loss_func(pred, label)
                return loss


            loss_all = 0.0
            loader_tqdm = tqdm(self.train_loader, ncols=120)
            for i, (ntype_mp_name_input_nodes_dict, seed,g) in enumerate(loader_tqdm):
                # mmd_loss=0
                # l1_loss=0
                # cls_lossS_MP=0
                h_dict = {}
                mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
                for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                    h_dict[meta_path_name] = self.model.input_featureS.forward_nodes({self.category: input_nodes})
                label = self.label[seed[self.category]].to(self.device)
                pred = self.model(g, h_dict={self.category:h_dict})

                loss = self.loss_func(pred, label)
                loss_all += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss_all / (i + 1)

    def cal_loss(logits, label, mode):

        return

    def cal_fmax_aupr(pred, labels):
        # 计算 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(labels, pred)
        
        # 计算每个阈值对应的 F1-score
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        
        # 找到最大的 F1-score 作为 Fmax
        fmax = np.max(f1_scores)
        
        # 计算 AUPR
        aupr = auc(recall, precision)
        
        return fmax, aupr
            

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))




class HANSampler(dgl.dataloading.Sampler):
    """HANSampler.
    Sample blocks by node types and meta paths.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    ntypes : list[str]
        List of center node types.
    meta_paths_dict: dict[str, list[etype]]
        Dict from meta path name to meta path.
    num_neighbors: int
        Number of neighbors to sample.
    """

    def __init__(self, g, seed_ntypes, meta_paths_dict, num_neighbors):
        self.output_device = None  # as_edge_prediction_sampler requires this attribute

        self.ntype_mp_name_sampler_dict = {}
        self.seed_ntypes = seed_ntypes
        self.ntype_meta_paths_dict = {}

        # build ntype_meta_paths_dict
        for ntype in self.seed_ntypes:
            self.ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in meta_paths_dict.items():
                # a meta path starts with this node type
                if meta_path[0][0] == ntype:
                    self.ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in self.ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                self.ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, g.canonical_etypes)

        for ntype, meta_paths_dict in self.ntype_meta_paths_dict.items():
            self.ntype_mp_name_sampler_dict[ntype] = {}
            for meta_path_name, meta_path in meta_paths_dict.items():
                # note: random walk may get same route(same edge), which will be removed in the sampled graph.
                # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
                self.ntype_mp_name_sampler_dict[ntype][meta_path_name] = RandomWalkNeighborSampler(G=g,
                                                                                                   num_traversals=1,
                                                                                                   termination_prob=0,
                                                                                                   num_random_walks=num_neighbors,
                                                                                                   num_neighbors=num_neighbors,
                                                                                                   metapath=meta_path)

    def sample(self, g, seeds, exclude_eids=None):  # exclude_eids is for compatibility with link prediction
        """sample method.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            Input node ids. Dict from node type to dict from meta path name to node ids.
        dict[str, Tensor]
            Seeds. Dict from node type to node ids
        dict[str, dict[str, DGLBlock]]
            Sampled blocks. Dict from node type to dict from meta path name to sampled blocks.
        """
        input_nodes_dict = {}
        ntype_mp_name_block_dict = {}
        for ntype, nid in seeds.items():
            if len(nid) == 0:
                continue
            input_nodes_dict[ntype] = {}
            ntype_mp_name_block_dict[ntype] = {}
            for meta_path_name, sampler in self.ntype_mp_name_sampler_dict[ntype].items():
                frontier = sampler(nid)
                frontier = dgl.remove_self_loop(frontier)
                frontier.add_edges(nid.clone().detach(), nid.clone().detach())
                block = dgl.to_block(frontier, nid)
                ntype_mp_name_block_dict[ntype][meta_path_name] = block
                input_nodes_dict[ntype][meta_path_name] = block.srcdata[dgl.NID]
        return input_nodes_dict, seeds, ntype_mp_name_block_dict


def extract_metapaths(category, canonical_etypes, self_loop=False):
    meta_paths_dict = {}
    for etype in canonical_etypes:
        if etype[0] in category:
            for dst_e in canonical_etypes:
                if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                    if self_loop:
                        mp_name = 'mp' + str(len(meta_paths_dict))
                        meta_paths_dict[mp_name] = [etype, dst_e]
                    else:
                        if etype[0] != etype[2]:
                            mp_name = 'mp' + str(len(meta_paths_dict))
                            meta_paths_dict[mp_name] = [etype, dst_e]
    return meta_paths_dict