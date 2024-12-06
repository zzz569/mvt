from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)

#从预训练模型中获取嵌入
class LookupEmbeddingPretrain(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, src_u_emb_path, src_i_emb_path):
        super().__init__()
        u_df1 = pd.DataFrame({'uid': np.arange(uid_all)})
        u_df2 = pd.read_csv(src_u_emb_path, names=['uid', 'user_emb', 'user_rela_emb', 'user_path_emb'], header=None)
        u_df3 = u_df1.merge(u_df2, on='uid', how='left')
        u_df3 = u_df3.fillna(value=','.join([str(0.0) for _ in range(emb_dim)]))

        u_user_emb_weight = torch.FloatTensor(u_df3.user_emb.str.split(',', expand=True).values.astype(np.float32))
        u_user_rela_emb_weight = torch.FloatTensor(u_df3.user_rela_emb.str.split(',', expand=True).values.astype(np.float32))
        u_user_path_emb_weight = torch.FloatTensor(u_df3.user_path_emb.str.split(',', expand=True).values.astype(np.float32))
        # u_pre_weight = torch.FloatTensor(u_df3.u_emb.str.split(',', expand=True).values.astype(np.float32))
        self.user_emb_embedding = torch.nn.Embedding.from_pretrained(u_user_emb_weight)
        self.user_rela_emb_embedding = torch.nn.Embedding.from_pretrained(u_user_rela_emb_weight)
        self.user_path_emb_embedding = torch.nn.Embedding.from_pretrained(u_user_path_emb_weight)
        # self.uid_embedding = torch.nn.Embedding.from_pretrained(u_pre_weight)


        # u_uiu_weight = torch.FloatTensor(u_df3.uiu_emb.str.split(',', expand=True).values.astype(np.float32))
        # u_uiciu_weight = torch.FloatTensor(u_df3.uiciu_emb.str.split(',', expand=True).values.astype(np.float32))
        # u_uibiu_weight = torch.FloatTensor(u_df3.uibiu_emb.str.split(',', expand=True).values.astype(np.float32))
        # u_pre_weight = torch.FloatTensor(u_df3.u_emb.str.split(',', expand=True).values.astype(np.float32))
        # self.uiu_embedding = torch.nn.Embedding.from_pretrained(u_uiu_weight)
        # self.uiciu_embedding = torch.nn.Embedding.from_pretrained(u_uiciu_weight)
        # self.uibiu_embedding = torch.nn.Embedding.from_pretrained(u_uibiu_weight)
        # self.uid_embedding = torch.nn.Embedding.from_pretrained(u_pre_weight)

        i_df1 = pd.DataFrame({'iid': np.arange(iid_all)})
        i_df2 = pd.read_csv(src_i_emb_path, names=['iid', 'item_emb', 'item_rela_emb', 'item_path_emb'], header=None)
        i_df3 = i_df1.merge(i_df2, on='iid', how='left')
        i_df3 = i_df3.fillna(value=','.join([str(0.0) for _ in range(emb_dim)]))

        i_item_emb_weight = torch.FloatTensor(i_df3.item_emb.str.split(',', expand=True).values.astype(np.float32))
        i_item_rela_emb_weight = torch.FloatTensor(i_df3.item_rela_emb.str.split(',', expand=True).values.astype(np.float32))
        i_item_path_emb_weight = torch.FloatTensor(i_df3.item_path_emb.str.split(',', expand=True).values.astype(np.float32))
        # i_pre_weight = torch.FloatTensor(i_df3.i_emb.str.split(',', expand=True).values.astype(np.float32))
        self.item_emb_embedding = torch.nn.Embedding.from_pretrained(i_item_emb_weight)
        self.item_rela_emb_embedding = torch.nn.Embedding.from_pretrained(i_item_rela_emb_weight)
        self.item_path_emb_embedding = torch.nn.Embedding.from_pretrained(i_item_path_emb_weight)
        # self.iid_embedding = torch.nn.Embedding.from_pretrained(i_pre_weight)

        # i_iui_weight = torch.FloatTensor(i_df3.iui_emb.str.split(',', expand=True).values.astype(np.float32))
        # i_ici_weight = torch.FloatTensor(i_df3.ici_emb.str.split(',', expand=True).values.astype(np.float32))
        # i_ibi_weight = torch.FloatTensor(i_df3.ibi_emb.str.split(',', expand=True).values.astype(np.float32))
        # i_pre_weight = torch.FloatTensor(i_df3.i_emb.str.split(',', expand=True).values.astype(np.float32))
        # self.iui_embedding = torch.nn.Embedding.from_pretrained(i_iui_weight)
        # self.ici_embedding = torch.nn.Embedding.from_pretrained(i_ici_weight)
        # self.ibi_embedding = torch.nn.Embedding.from_pretrained(i_ibi_weight)
        # self.iid_embedding = torch.nn.Embedding.from_pretrained(i_pre_weight)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, inputs):
        # inputs 为三个输入张量列表，每个张量维度为 [256, 200]
        encoder_outputs = torch.stack(inputs, dim=1)  # 将三个输入张量堆叠在一起，得到维度为 [256, 3, 200]

        batch_size, seq_len, _ = encoder_outputs.size()

        # 计算注意力权重
        attn_scores = torch.matmul(encoder_outputs, self.attn_weights)  # [256, 3, 1]
        attn_scores = attn_scores.squeeze(-1)  # [256, 3]

        # 对注意力权重进行softmax操作
        attn_weights = torch.softmax(attn_scores, dim=1)  # [256, 3]

        # 根据注意力权重加权求和得到注意力向量
        attn_vectors = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [256, 200]

        return attn_vectors, attn_weights


class CDAE(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, meta_dim, is_sparse=False):
        super(CDAE, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)   #(5775,200)
        # self.user_y_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)   #(5775,200)
        self.item_x_embeddings = nn.Embedding(self.NUM_MOVIE,EMBED_SIZE,sparse= is_sparse)  #(16964,200)
        self.item_y_embeddings = nn.Embedding(self.NUM_BOOK,EMBED_SIZE,sparse= is_sparse)  #(2897,200)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.item_x_embeddings.weight.data = torch.from_numpy(np.random.normal(0,0.01,size=[self.NUM_MOVIE,EMBED_SIZE])).float()
        self.item_y_embeddings.weight.data = torch.from_numpy(np.random.normal(0,0.01,size=[self.NUM_BOOK,EMBED_SIZE])).float()
        self.user_rela_x_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_rela_y_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_path_x_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_path_y_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_rela_x_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.user_rela_y_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.user_path_x_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.user_path_y_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()
        self.src_pretrain_model = LookupEmbeddingPretrain(NUM_USER, NUM_MOVIE, EMBED_SIZE, "user_embeddings_CDs_and_Vinyl.csv", "item_embeddings_CDs_and_Vinyl.csv")
        self.tgt_pretrain_model = LookupEmbeddingPretrain(NUM_USER, NUM_BOOK, EMBED_SIZE, "user_embeddings_Movies_and_TV.csv", "item_embeddings_Movies_and_TV.csv")

        self.encoder_x = nn.Sequential(
            nn.Linear(self.NUM_MOVIE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_x = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_MOVIE)
            )
        self.encoder_y = nn.Sequential(
            nn.Linear(self.NUM_BOOK, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_y = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_BOOK)
            )
        self.meta_net = MetaNet(EMBED_SIZE, meta_dim)
        self.attention = Attention(EMBED_SIZE)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU

    def forward(self, batch_user, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))   #256 * num_movie
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)

        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)   #[256,200]
        z_y = F.relu(feature_y)   #[256,200]

        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)

        h_user_x_rela = self.src_pretrain_model.user_rela_emb_embedding(batch_user)
        h_user_y_rela = self.tgt_pretrain_model.user_rela_emb_embedding(batch_user)

        h_user_x_path = self.src_pretrain_model.user_path_emb_embedding(batch_user)
        h_user_y_path = self.tgt_pretrain_model.user_path_emb_embedding(batch_user)

        mapped_z_x = [feature_x, h_user_x_rela, h_user_x_path]
        mapped_z_y = [feature_y, h_user_y_rela, h_user_y_path]

        preds_x2y = self.decoder_y(mapped_z_x)
        preds_y2x = self.decoder_x(mapped_z_y)

        z_x_reg_loss = torch.norm(z_x - mapped_z_x, p=1, dim=1)
        z_y_reg_loss = torch.norm(z_y - mapped_z_y, p=1, dim=1)

        return preds_x, preds_y, preds_x2y, preds_y2x, feature_x, feature_y , z_x_reg_loss, z_y_reg_loss

    def get_user_embedding(self, batch_user_x, batch_user_y):
        # this is for SIGIR's experiment
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        return h_user_x, h_user_y

    def get_latent_z(self, batch_user, batch_user_x, batch_user_y):
        # this is for clustering visualization
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)
        
        return z_x, z_y


class Discriminator(nn.Module):
    def __init__(self, n_fts, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.training = True

        self.disc = nn.Sequential(
            nn.Linear(n_fts, int(n_fts/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_fts/2), 1))

    def forward(self, x):
        # make mlp for discriminator
        h = self.disc(x)
        return h

def save_embedding_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_user_embedding(v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)

def save_z_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_latent_z(v_user, v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)

def get_index(batch_user):
    batch_size = batch_user.size(0)  # 获取批量大小
    max_interactions = 20  # 假设每个用户最多有20个交互的物品
    indices_list = [torch.nonzero(batch_user[i, :] == 1).squeeze() for i in range(batch_user.size(0))]
    result = torch.full((batch_size, max_interactions), 0)  # 创建全为0的张量
    for i in range(len(indices_list)):
        indices = indices_list[i]
        if indices.numel() > 20:
            indices = indices[:20]
        result[i, :indices.numel()] = indices
    # print(result)
    # print(result.shape)
    return result


# def get_item_index(batch_user):
#     batch_size = batch_user.size(0)  # 获取批量大小
#     max_interactions = 20  # 假设每个用户最多有20个交互的物品
#     indices_list = [torch.nonzero(batch_user[i, :] == 1).squeeze() for i in range(batch_user.size(0))]
#     result = torch.full((batch_size, max_interactions), 0)  # 创建全为0的张量
#     for i in range(len(indices_list)):
#         indices = indices_list[i]
#         if indices.numel() > 20:
#             indices = indices[:20]
#         result[i, :indices.numel()] = indices
#     # print(result)
#     # print(result.shape)
#     return result