import torch
from torch import nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .PoinTr import Fold, fps
from .dbad import (
    GlobalDiscriminator,
    LocalDiscriminator,
    grad_reverse
)
from .build import MODELS

@MODELS.register_module()
class CausalCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans=3, embed_dim=self.trans_dim, depth=[6,8], drop_rate=0., num_query=self.num_query, knn_layer=self.knn_layer)
        self.global_disc = GlobalDiscriminator(self.trans_dim)
        self.local_disc = LocalDiscriminator(self.trans_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.foldingnet = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim,1024,1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024,1024,1)
        )
        self.reduce_map = nn.Linear(self.trans_dim+1027,self.trans_dim)
        self.lambda_adv = config.lambda_adv
        self.lambda_feat_c = config.lambda_feat_c
        self.lambda_geo_c = config.lambda_geo_c
        self.build_loss_func()
    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
    def cgc_loss(self,global_feature,output):
        B = global_feature.shape[0]
        assert B % 3 == 0
        K = 3
        num = B // K
        feat_loss = 0.
        geo_loss = 0.
        global_feature = global_feature.reshape(num,K,-1)
        output = output.reshape(num,K,*output.shape[1:])
        pairs = 0
        for i in range(K):
            for j in range(i+1,K):
                pairs += 1
                feat_loss += F.mse_loss(
                    global_feature[:,i],
                    global_feature[:,j]
                )
                geo_loss += self.loss_func(
                    output[:,i],
                    output[:,j]
                )
        feat_loss = feat_loss / pairs
        geo_loss = geo_loss / pairs
        return feat_loss,geo_loss
    def get_loss(self,ret,gt,ug,ul,epoch=0,max_epoch=5):
        (
            coarse,
            fine,
            pred_points,
            global_feature,
            ug_pred,
            ul_pred
        ) = ret
        loss_coarse = self.loss_func(
            coarse,
            gt
        )
        loss_fine = self.loss_func(
            fine,
            gt
        )
        ug_loss = self.bce_loss(
            ug_pred.squeeze(-1),
            ug.float()
        )
        ul_loss = self.bce_loss(
            ul_pred.squeeze(-1),
            ul.float()
        )
        feat_cgc,geo_cgc = self.cgc_loss(
            global_feature,
            pred_points
        )
        warm_up = min(epoch/max_epoch,1.0)
        loss = (
            loss_coarse
            + loss_fine
            + self.lambda_adv*warm_up*(ug_loss+ul_loss)
            + self.lambda_feat_c*feat_cgc
            + self.lambda_geo_c*geo_cgc
        )
        return loss
    def forward(self,xyz):
        z,coarse_point_cloud = self.base_model(xyz)
        B,M,C = z.shape
        global_feature = torch.max(z,dim=1)[0]
        z_adv = grad_reverse(z,1.0)
        global_adv = grad_reverse(global_feature,1.0)
        ug_pred = self.global_disc(global_adv)
        ul_pred = self.local_disc(z_adv)
        decoder_global = self.increase_dim(z.transpose(1,2)).transpose(1,2)
        decoder_global = torch.max(decoder_global, dim=1)[0]
        rebuild_feature = torch.cat(
            [
                decoder_global.unsqueeze(-2).expand(-1,M,-1),
                z,
                coarse_point_cloud
            ], dim=-1
        )
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M,-1))
        relative_xyz = self.foldingnet(
            rebuild_feature
        )
        relative_xyz = relative_xyz.reshape(B, M, 3, -1)
        pred_points = (
            relative_xyz
            +
            coarse_point_cloud.unsqueeze(-1)
        )
        pred_points = (
            pred_points.transpose(2,3)
            .reshape(B,-1,3)
        )
        inp_sparse = fps(
            xyz,
            self.num_query
        )
        coarse_out = torch.cat(
            [
                coarse_point_cloud,
                inp_sparse
            ],
            dim=1
        ).contiguous()
        fine_out = torch.cat(
            [
                pred_points,
                xyz
            ],
            dim=1
        ).contiguous()
        ret = (
            coarse_out,
            fine_out,
            pred_points,
            global_feature,
            ug_pred,
            ul_pred
        )
        return ret
