from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import time
import random
import os
import math
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from Dataset_CDAE import Dataset
from model_mvt import CDAE, Discriminator, save_embedding_process, save_z_process
import time
import itertools
from my_utils import *
from scipy.sparse import csr_matrix

method_name = 'my_my_5_variant2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
topK_list = [5, 10]
# (amazon means "Movie & Book", amazon2 means "Movie & Music" and amazon3 means "Music & Book").
#
# -->on Movie & Book dataset: CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon --reg 5.0
#
# -->on Movie & Music dataset:
# CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon2 --reg 0.5
#
# -->on Music & Book dataset:
# CUDA_VISIBLE_DEVICES=gpu_num python main_my.py --dataset amazon3 --reg 1.0

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='Gaussian', help='Gaussian, MVGaussian，Laplace，Uniform')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=256, help='batch size.')
parser.add_argument('--emb_size', type=int, default=200, help='embed size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')
parser.add_argument('--pos-weight', type=float, default=1.0, help='weight for positive samples')

parser.add_argument('--reg', type=float, default=1, help='lambda reg')

parser.add_argument('--self', type=float, default=1.0, help='lambda rec' )
parser.add_argument('--cross', type=float, default=1.0, help='lambda rec')
parser.add_argument('--gan', type=float, default=1.0, help='lambda gan')
parser.add_argument('--d_epoch', type=int, default=2, help='d epoch')
parser.add_argument('--t_percent', type=float, default=1.0, help='target percent')
parser.add_argument('--s_percent', type=float, default=1.0, help='source percent')
parser.add_argument('--dataset', type=str, default='amazon', help='amazon')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    log = os.path.join(args.log, '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.emb_size, args.weight_decay, args.self,
                                                     args.cross, args.gan, args.reg, args.t_percent, args.s_percent))
    if os.path.isdir(log):
        print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
        time.sleep(5)
        os.system('rm -rf %s/' % log)

    os.makedirs(log)
    print("made the log directory", log)

    print('preparing data...')
    dataset = Dataset(args.batch, dataset=args.dataset)

    NUM_USER = dataset.num_user   #16367
    NUM_MOVIE = dataset.num_movie   #18467
    NUM_BOOK = dataset.num_book   #23988

    print('Preparing the training data......')
    # prepare data for X
    row, col = dataset.get_part_train_indices('movie', args.s_percent)  # row:ndarray(203175,)  col:ndarray(203175,)
    values = np.ones(row.shape[0]) # ndarray(203175,)
    user_x = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray() #ndarray(16367,18467) 一个矩阵，有交互记为1  X域交互矩阵

    # prepare data for Y
    row, col = dataset.get_part_train_indices('book', args.t_percent)  #row:ndarray(262633,)  col:ndarray(262633,)
    values = np.ones(row.shape[0])  #ndarray(262633,)
    user_y = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()  #ndarray(16367,23988)  Y域交互矩阵

    print('Preparing the training data over......')
    # for shared user one-hot representation
    user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])  #ndarray(16367,1)

    user_x = torch.FloatTensor(user_x)  #Tensor(16367,18467) X域交互矩阵
    user_y = torch.FloatTensor(user_y)  #Tensor(16367,23988) Y域交互矩阵

    train_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),  #Dataloader:64
                                                     batch_size=args.batch,
                                                     shuffle=True)
    save_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                               batch_size=args.batch,
                                               shuffle=False)

    pos_weight = torch.FloatTensor([args.pos_weight])

    if args.cuda:
        pos_weight = pos_weight.cuda()

    model = CDAE(NUM_USER=NUM_USER, NUM_MOVIE=NUM_MOVIE, NUM_BOOK=NUM_BOOK,
                 EMBED_SIZE=args.emb_size, dropout=args.dropout,meta_dim = 50)
    '''
    CDAE(
  (user_embeddings): Embedding(16367, 200)
  (encoder_x): Sequential(
    (0): Linear(in_features=18467, out_features=200, bias=True)
    (1): ReLU() 
    (2): Linear(in_features=200, out_features=200, bias=True)
  )
  (decoder_x): Sequential(
    (0): Linear(in_features=200, out_features=200, bias=True)
    (1): ReLU()
    (2): Linear(in_features=200, out_features=18467, bias=True)
  )
  (encoder_y): Sequential(
    (0): Linear(in_features=23988, out_features=200, bias=True)
    (1): ReLU()
    (2): Linear(in_features=200, out_features=200, bias=True)
  )
  (decoder_y): Sequential(
    (0): Linear(in_features=200, out_features=200, bias=True)
    (1): ReLU()
    (2): Linear(in_features=200, out_features=23988, bias=True)
  )
  (dropout): Dropout(p=0.5, inplace=False)
)
    '''
    disc1 = Discriminator(args.emb_size, args.dropout)  #鉴别器
    '''
    Discriminator(
  (disc): Sequential(
    (0): Linear(in_features=200, out_features=100, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=100, out_features=1, bias=True)
  )
)
    '''
    disc2 = Discriminator(args.emb_size, args.dropout)  #鉴别器 输出的结果经过sigmoid函数处理后，
                                                        # 就可以作为二分类问题的预测概率值，并与真实标签计算交叉熵损失。输出一个用户对物品的预测概率
    '''
    Discriminator(
  (disc): Sequential(
    (0): Linear(in_features=200, out_features=100, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=100, out_features=1, bias=True)
  )
)
    '''
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = optim.SGD(itertools.chain(disc1.parameters(), disc2.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    BCEWL = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)  #二分类交叉熵损失函数 pos_weight是正类的权重，用于解决类别不平衡问题。
    BCEL = torch.nn.BCEWithLogitsLoss(reduction='none')

    if args.cuda:
        model = model.cuda()
        disc1 = disc1.cuda()
        disc2 = disc2.cuda()

    # prepare data for test process
    movie_vali, movie_test, movie_nega = dataset.movie_vali, dataset.movie_test, dataset.movie_nega
    book_vali, book_test, book_nega = dataset.book_vali, dataset.book_test, dataset.book_nega
    feed_data = {}
    feed_data['fts1'] = user_x
    feed_data['fts2'] = user_y
    feed_data['movie_vali'] = movie_vali
    feed_data['book_vali'] = book_vali
    feed_data['movie_test'] = movie_test
    feed_data['book_test'] = book_test
    feed_data['movie_nega'] = movie_nega
    feed_data['book_nega'] = book_nega

    best_hr1, best_ndcg1, best_mrr1 = 0.0, 0.0, 0.0
    best_hr2, best_ndcg2, best_mrr2 = 0.0, 0.0, 0.0
    val_hr1_list, val_ndcg1_list, val_mrr1_list = [], [], []
    val_hr2_list, val_ndcg2_list, val_mrr2_list = [], [], []

    G_loss_list = []
    D_loss_list = []
    reg_loss_list = []
    loss_list = []
    self_loss_list = []
    cross_loss_list = []
    JRL_loss_list = []

    epoch_time_list = []
    for epoch in range(args.epochs):
        model.train()
        batch_G_loss_list = []
        batch_D_loss_list = []
        batch_loss_list = []
        batch_reg_loss_list = []
        batch_self_loss_list = []
        batch_cross_loss_list = []
        batch_JRL_loss_list = []

        epoch_time = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.reshape([-1]).type(torch.long)

            if (batch_idx+1) % (args.d_epoch + 1) == 0:
                optimizer_d.zero_grad()
                # standard Gaussian distribution
                prior = torch.from_numpy(np.random.normal(0, 1.0, size=[data.shape[0], args.emb_size])).float()  #先验正则化损失
                # laplace distribution
                # prior = torch.from_numpy(np.random.laplace(0, 1.0, size=[data.shape[0], args.emb_size])).float()
                # uniform distribution
                # prior = torch.from_numpy(np.random.uniform(0, 1.0, size=[data.shape[0], args.emb_size])).float()
                # multi-variate Gaussian distribution
                # prior = 0.5*torch.from_numpy(np.random.normal(0, 1.0, size=[data.shape[0], args.emb_size])+
                #                          np.random.normal(3, 1.0, size=[data.shape[0], args.emb_size])).float()
                if args.cuda:
                    prior = prior.cuda()

                if args.cuda:
                    batch_user = data.cuda()
                    batch_user_x = user_x[data].cuda()
                    batch_user_y = user_y[data].cuda()
                else:
                    batch_user = data
                    batch_user_x = user_x[data]
                    batch_user_y = user_y[data]

                time1 = time.time()
                _, _, _, _, z_x, z_y, z_x_reg_loss, z_y_reg_loss = model.forward(batch_user, batch_user_x, batch_user_y)
                time2 = time.time()
                epoch_time += time2-time1

                true1 = disc1(prior).reshape([-1])
                true2 = disc2(prior).reshape([-1])
                fake1 = disc1(z_x).reshape([-1])
                fake2 = disc2(z_y).reshape([-1])

                dis_loss1 = BCEL(torch.cat([true1, fake1], 0),
                                torch.cat([torch.ones_like(true1), torch.zeros_like(fake1)], 0)).sum()
                dis_loss2 = BCEL(torch.cat([true2, fake2], 0),
                                torch.cat([torch.ones_like(true2), torch.zeros_like(fake2)], 0)).sum()

                D_loss = dis_loss1 + dis_loss2

                D_loss.backward()
                optimizer_d.step()
                batch_D_loss_list.append(D_loss.item())

            else:
                optimizer_g.zero_grad()

                if args.cuda:
                    batch_user = data.cuda()
                    batch_user_x = user_x[data].cuda()
                    batch_user_y = user_y[data].cuda()
                else:
                    batch_user = data  #用户id  [256]
                    batch_user_x = user_x[data]  #该用户在x域的交互物品   [256,16964]
                    batch_user_y = user_y[data]   #该用户在y域的交互物品  [256,2897]

                time1 = time.time()
                pred_x, pred_y, pred_x2y, pred_y2x, z_x, z_y, z_x_reg_loss, z_y_reg_loss = model.forward(batch_user, batch_user_x, batch_user_y)
                time2 = time.time()
                epoch_time += time2 - time1

                loss_x = BCEWL(pred_x, batch_user_x).sum()
                loss_y = BCEWL(pred_y, batch_user_y).sum()
                loss_x2y = BCEWL(pred_x2y, batch_user_y).sum()
                loss_y2x = BCEWL(pred_y2x, batch_user_x).sum()

                reg_loss = z_x_reg_loss.sum() + z_y_reg_loss.sum()

                fake_x = disc1(z_x).reshape([-1])
                fake_y = disc2(z_y).reshape([-1])

                G_loss1 = BCEL(fake_x, torch.ones_like(fake_x)).sum()
                G_loss2 = BCEL(fake_y, torch.ones_like(fake_y)).sum()
                G_loss = G_loss1+G_loss2

                # get plot JRL loss
                JRL_loss = args.self * (loss_x + loss_y) + args.cross * (loss_x2y + loss_y2x) + args.reg*reg_loss   #联合重建损失
                batch_JRL_loss_list.append(JRL_loss.item()/args.batch)

                # get the whole loss
                loss = G_loss + args.self * (loss_x + loss_y) + args.cross * (loss_x2y + loss_y2x) + args.reg*reg_loss
                loss.backward()
                optimizer_g.step()

                batch_G_loss_list.append(G_loss.item())
                batch_self_loss_list.append(args.self * (loss_x + loss_y).item())
                batch_cross_loss_list.append(args.cross * (loss_x2y + loss_y2x).item())
                batch_loss_list.append(loss.item())
                batch_reg_loss_list.append(args.reg*reg_loss.item())

        epoch_time_list.append(epoch_time)
        print('epoch time:{:.4f}'.format(epoch_time))
        with open(log + '/tmp.txt', 'a') as f:
            f.write('epoch time:{:.4f}\n'
                    .format(np.mean(epoch_time)))

        epoch_G_loss = np.mean(batch_G_loss_list)
        epoch_D_loss = np.mean(batch_D_loss_list)
        epoch_reg_loss = np.mean(batch_reg_loss_list)
        epoch_loss = np.mean(batch_loss_list)
        epoch_self_loss = np.mean(batch_self_loss_list)
        epoch_cross_loss = np.mean(batch_cross_loss_list)
        epoch_JRL_Loss = np.mean(batch_JRL_loss_list)

        G_loss_list.append(epoch_G_loss)
        D_loss_list.append(epoch_D_loss)
        loss_list.append(epoch_loss)
        self_loss_list.append(epoch_self_loss)
        cross_loss_list.append(epoch_cross_loss)
        reg_loss_list.append(epoch_reg_loss)
        JRL_loss_list.append(epoch_JRL_Loss)

        print('epoch:{}, self loss:{:.4f}, cross loss:{:.4f}, G loss:{:.4f}, D loss:{:.4f}, reg loss:{:.4f}, loss:{:.4f}'.format(epoch,epoch_self_loss,
                                                                                                        epoch_cross_loss,
                                                                                                        epoch_G_loss,
                                                                                                        epoch_D_loss,
                                                                                                        epoch_reg_loss,
                                                                                                        epoch_loss))
        with open(log + '/tmp.txt', 'a') as f:
            f.write('epoch:{}, self loss:{:.4f}, cross loss:{:.4f}, G loss:{:.4f}, D loss:{:.4f}, reg loss:{:.4f}, loss:{:.4f}\n'.format(
                epoch,
                epoch_self_loss, epoch_cross_loss, epoch_G_loss, epoch_D_loss, epoch_reg_loss,epoch_loss))

        if epoch % 1 == 0:
            model.eval()

            avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2 = test_process(model, train_loader, feed_data,
                                                                                      args.cuda, topK_list[1], mode='val')

            val_hr1_list.append(avg_hr1)
            val_ndcg1_list.append(avg_ndcg1)
            val_mrr1_list.append(avg_mrr1)
            val_hr2_list.append(avg_hr2)
            val_ndcg2_list.append(avg_ndcg2)
            val_mrr2_list.append(avg_mrr2)

            print('test: movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
                  .format(avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2))
            with open(log + '/tmp.txt', 'a') as f:
                f.write('test: movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                        .format(avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2))

            if avg_hr1 > best_hr1:
                best_hr1 = avg_hr1
                torch.save(model.state_dict(), os.path.join(log, 'best_hr1.pkl'))

            if avg_ndcg1 > best_ndcg1:
                torch.save(model.state_dict(), os.path.join(log, 'best_ndcg1.pkl'))
                best_ndcg1 = avg_ndcg1
            if avg_mrr1 > best_mrr1:
                torch.save(model.state_dict(), os.path.join(log, 'best_mrr1.pkl'))
                best_mrr1 = avg_mrr1
            if avg_hr2 > best_hr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_hr2.pkl'))
                best_hr2 = avg_hr2
            if avg_ndcg2 > best_ndcg2:
                torch.save(model.state_dict(), os.path.join(log, 'best_ndcg2.pkl'))
                best_ndcg2 = avg_ndcg2
            if avg_mrr2 > best_mrr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_mrr2.pkl'))
                best_mrr2 = avg_mrr2


    print('best val movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
          .format(best_hr1, best_ndcg1, best_mrr1, best_hr2, best_ndcg2, best_mrr2))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('best val movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
          .format(best_hr1, best_ndcg1, best_mrr1, best_hr2, best_ndcg2, best_mrr2))

    # # save necessary plot loss and validation metric
    # pickle.dump(loss_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_loss_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(JRL_loss_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_JRL_loss_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_hr1_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_val_hr1_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_ndcg1_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_val_ndcg1_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_mrr1_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_val_mrr1_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_hr2_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                             '{}_{}_val_hr2_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_ndcg2_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                               '{}_{}_val_ndcg2_list'.format(args.dataset, args.gan)), 'wb'))
    # pickle.dump(val_mrr2_list, open(os.path.join(os.getcwd(), 'saved_embeddings', method_name,
    #                                              '{}_{}_val_mrr2_list'.format(args.dataset, args.gan)), 'wb'))

    print('mean epoch time:{:.4f}'.format(np.mean(epoch_time_list)))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('mean epoch time:{:.4f}\n'
                .format(np.mean(epoch_time_list)))
    print('Val process over!')
    print('Test process......')
    for topK in topK_list:
        model.load_state_dict(torch.load(os.path.join(log, 'best_hr1.pkl')))
        test_hr1, _, _, _, _, _ = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')

        # if topK==10:
        #     user_embedding1, user_embedding2 = save_embedding_process(model, save_loader, feed_data, args.cuda)
        #     pickle.dump(user_embedding1, open(
        #         os.path.join(os.getcwd(), 'saved_embeddings', method_name, '{}_Z_movie.pkl'.format(args.dataset)), 'wb'))
        #     pickle.dump(user_embedding2, open(
        #         os.path.join(os.getcwd(), 'saved_embeddings', method_name, '{}_Z_book.pkl'.format(args.dataset)), 'wb'

        model.load_state_dict(torch.load(os.path.join(log, 'best_ndcg1.pkl')))
        _, test_ndcg1, _, _, _, _ = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_mrr1.pkl')))
        _, _, test_mrr1, _, _, _ = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_hr2.pkl')))
        _, _, _, test_hr2, _, _ = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_ndcg2.pkl')))
        _, _, _, _, test_ndcg2, _ = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_mrr2.pkl')))
        _, _, _, _, _, test_mrr2 = test_process(model, train_loader, feed_data, args.cuda, topK, mode='test')
        print('Test TopK:{} ---> movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
              .format(topK, test_hr1, test_ndcg1, test_mrr1, test_hr2, test_ndcg2, test_mrr2))
        with open(log + '/tmp.txt', 'a') as f:
            f.write('Test TopK:{} ---> movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                    .format(topK, test_hr1, test_ndcg1, test_mrr1, test_hr2, test_ndcg2, test_mrr2))



if __name__ == "__main__":
    print(args)
    main()
    print(args)