# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import random
from math import sqrt

from block import fusions


def sum_attention(nnet, query, value, mask=None, dropout=None, mode='1D'):
    if mode == '2D':
        batch, dim = query.size(0), query.size(1)
        query = query.permute(0, 2, 3, 1).view(batch, -1, dim)
        value = value.permute(0, 2, 3, 1).view(batch, -1, dim)
        mask = mask.view(batch, 1, -1)

    scores = nnet(query).transpose(-2, -1)
    if mask is not None:
        scores.data.masked_fill_(mask.eq(0), -65504.0)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    weighted = torch.matmul(p_attn, value)

    return weighted, p_attn


class SummaryAttn(nn.Module):

    def __init__(self, dim, num_attn, dropout, is_multi_head=False, mode='1D'):
        super(SummaryAttn, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_attn),
        )
        self.h = num_attn
        self.is_multi_head = is_multi_head
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.mode = mode

    def forward(self, query, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch = query.size(0)

        weighted, self.attn = sum_attention(
            self.linear, query, value, mask=mask, dropout=self.dropout, mode=self.mode)
        weighted = weighted.view(
            batch, -1) if self.is_multi_head else weighted.mean(dim=-2)

        return weighted


class PredictLayer(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_ans, dropout, dropattn=0):
        super(PredictLayer, self).__init__()
        self.summaries = nn.ModuleList([
            SummaryAttn(dim1, num_attn, dropattn, is_multi_head=False),
            SummaryAttn(dim2, num_attn, dropattn, is_multi_head=False),
        ])

        self.predict = nn.Sequential(
            nn.Linear(dim1 + dim2, (dim1 + dim2) // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear((dim1 + dim2) // 2, num_ans),
        )

    def forward(self, data1, data2, mask1, mask2):
        weighted1 = self.summaries[0](data1, data1, mask1)
        weighted2 = self.summaries[1](data2, data2, mask2)
        weighted = torch.cat([weighted1, weighted2], dim=1)

        mm = fusions.Block([2048, 1024], 3072).cuda()
        proj_feat = mm([weighted1, weighted2])
        # return proj_feat
        feat = self.predict(proj_feat)
        return feat
        


def qkv_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores.data.masked_fill_(mask.eq(0), -65504.0)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class DenseCoAttn(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_none, dropout, is_multi_head=False):
        super(DenseCoAttn, self).__init__()
        dim = min(dim1, dim2)
        self.linears = nn.ModuleList([nn.Linear(dim1, dim, bias=False),
                                      nn.Linear(dim2, dim, bias=False)])
        self.nones = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none, dim1))),
                                       nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none, dim2)))])
        self.d_k = dim // num_attn
        self.h = num_attn
        self.num_none = num_none
        self.is_multi_head = is_multi_head
        self.attn = None
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout) for _ in range(2)])

    def forward(self, value1, value2, mask1=None, mask2=None):
        batch = value1.size(0)
        dim1, dim2 = value1.size(-1), value2.size(-1)
        value1 = torch.cat([self.nones[0].unsqueeze(0).expand(
            batch, self.num_none, dim1), value1], dim=1)
        value2 = torch.cat([self.nones[1].unsqueeze(0).expand(
            batch, self.num_none, dim2), value2], dim=1)
        none_mask = value1.new_ones((batch, self.num_none))

        if mask1 is not None:
            mask1 = torch.cat([none_mask, mask1], dim=1)
            mask1 = mask1.unsqueeze(1).unsqueeze(2)
        if mask2 is not None:
            mask2 = torch.cat([none_mask, mask2], dim=1)
            mask2 = mask2.unsqueeze(1).unsqueeze(2)

        query1, query2 = [l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2)
                          for l, x in zip(self.linears, (value1, value2))]

        if self.is_multi_head:
            weighted1, attn1 = qkv_attention(
                query2, query1, query1, mask=mask1, dropout=self.dropouts[0])
            weighted1 = weighted1.transpose(1, 2).contiguous()[
                :, self.num_none:, :]
            weighted2, attn2 = qkv_attention(
                query1, query2, query2, mask=mask2, dropout=self.dropouts[1])
            weighted2 = weighted2.transpose(1, 2).contiguous()[
                :, self.num_none:, :]
        else:
            weighted1, attn1 = qkv_attention(query2, query1, value1.unsqueeze(1), mask=mask1,
                                             dropout=self.dropouts[0])
            weighted1 = weighted1.mean(dim=1)[:, self.num_none:, :]
            weighted2, attn2 = qkv_attention(query1, query2, value2.unsqueeze(1), mask=mask2,
                                             dropout=self.dropouts[1])
            weighted2 = weighted2.mean(dim=1)[:, self.num_none:, :]
        self.attn = [attn1[:, :, self.num_none:, self.num_none:],
                     attn2[:, :, self.num_none:, self.num_none:]]

        return weighted1, weighted2


class NormalSubLayer(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_none, dropout, dropattn=0):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(
            dim1, dim2, num_attn, num_none, dropattn)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2, mask1, mask2):
        weighted1, weighted2 = self.dense_coattn(data1, data2, mask1, mask2)
        data1 = data1 + self.linears[0](torch.cat([data1, weighted2], dim=2))
        data2 = data2 + self.linears[1](torch.cat([data2, weighted1], dim=2))

        return data1, data2


class DCNLayer(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_none, num_seq, dropout, dropattn=0):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer(dim1, dim2, num_attn, num_none,
                                                        dropout, dropattn) for _ in range(num_seq)])

    def forward(self, data1, data2, mask1, mask2):
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2, mask1, mask2)

        return data1, data2


class Initializer(object):

    @staticmethod
    def manual_seed(seed):
        """
        Set all of random seed to seed.
        --------------------
        Arguments:
                seed (int): seed number.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def xavier_normal(module, lstm_forget_bias_init=2):
        """
        Xavier Gaussian initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_normal_(
                module.weight.data) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(
                    0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(
                        param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[hidden_size//4:hidden_size //
                                   2] = lstm_forget_bias_init

    @staticmethod
    def xavier_uniform(module, lstm_forget_bias_init=2):
        """
        Xavier Uniform initialization.
        """
        lstm_forget_bias_init = float(lstm_forget_bias_init) / 2
        normal_classes = (nn.Conv2d, nn.Linear, nn.Embedding)
        recurrent_classes = (nn.RNN, nn.LSTM, nn.GRU)
        if any([isinstance(module, cl) for cl in normal_classes]):
            nn.init.xavier_uniform_(
                module.weight.data) if module.weight.requires_grad else None
            try:
                module.bias.data.fill_(
                    0) if module.bias.requires_grad else None
            except AttributeError:
                pass
        elif any([isinstance(module, cl) for cl in recurrent_classes]):
            for name, param in module.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(
                        param.data) if param.requires_grad else None
                elif name.startswith("bias"):
                    if param.requires_grad:
                        hidden_size = param.size(0)
                        param.data.fill_(0)
                        param.data[hidden_size//4:hidden_size //
                                   2] = lstm_forget_bias_init


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    # mlp_model = None
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        # mlp_model = self.mlp

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        copy_data = __C
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            2048
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(1024)
        self.proj = nn.Linear(1024, answer_size)

        self.dense_coattn = DCNLayer(2048, 1024, 4, 3, 5, 0.3)
        self.predict = PredictLayer(2048, 1024, 4, 3129, 0.3)

        # self.apply(Initializer.xavier_normal)

    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        img_feat, lang_feat = self.dense_coattn(
            img_feat, lang_feat, None, None)

        proj_feat = self.predict(img_feat, lang_feat, None, None)

        # img_att = self.mlp(img_feat)

        # lang_att = self.mlp(lang_feat)

        # img_feat = img_feat.reshape((-1,img_feat.size(1)*img_feat.size(2)))
        # lang_feat = lang_feat.res hape((-1,lang_feat.size(1)*lang_feat.size(2)))

        # Backbone Framework
        # lang_feat, img_feat = self.backbone(
        #     lang_feat,
        #     img_feat,
        #     lang_feat_mask,
        #     img_feat_mask
        # )

        # lang_feat = self.attflat_lang(
        #     lang_feat,
        #     lang_feat_mask
        # )

        # img_feat = self.attflat_img(
        #     img_feat,
        #     img_feat_mask
        # )

        # NUM_LAYERS = 3
        # conv_layer_1 = nn.Linear(1024,1024).cuda()
        # conv_layer_2 = nn.ModuleList([
        #     nn.Linear(1024, 1024)
        #     for i in range(NUM_LAYERS)]).cuda()

        # img_feat = conv_layer_1(img_feat)
        # lang_feat = conv_layer_1(lang_feat)
        # feat1 = nn.Dropout(0.25)(feat1)
        # feat2 = nn.Dropout(0.25)(feat2)

        # x_mm = []

        # for i in range(NUM_LAYERS):
        #     x1 = conv_layer_2[i](img_feat)
        #     # x1 = nn.Tanh()(x1)

        #     x2 = conv_layer_2[i](lang_feat)
        #     # x2 = nn.Tanh()(x2)

        #     x_mm.append(torch.mul(x1,x2))

        # x_mm = torch.stack(x_mm,dim=1)
        # batch_size = x_mm.size(0)
        # nc,w,h = x_mm.shape[2],x_mm.shape[3],x_mm.shape[4]
        # proj_feat = torch.sum(x_mm,dim=1)

        # mm = fusions.LinearSum([1024,1024],3129).cuda()
        # proj_feat = mm([img_feat,lang_feat])

        # mul_feat = lang_feat * img_feat
        # add_feat = lang_feat + img_feat
        # proj_feat = mul_feat + add_feat

        # proj_feat = lang_feat + img_feat

        # proj_feat = F.softmax(proj_feat, dim=1)

        # proj_feat = self.proj_norm(proj_feat)
        # proj_feat = torch.sigmoid(proj_feat)
        # proj_feat = torch.sigmoid(self.proj(proj_feat))
        # proj_feat = self.proj(proj_feat)
        # return proj_feat
        return torch.sigmoid(proj_feat)

    # Masking

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
