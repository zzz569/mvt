import argparse
import torch
from data_loader import load_data
from TAHIN import TAHIN

#初始化信息
parser = argparse.ArgumentParser(
    description="Parser For Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    default="movielens",
    help="Dataset to use, default: movielens",
)
parser.add_argument(
    "--path", default="./data", help="Path to save the data"
)
parser.add_argument("--model", default="TAHIN", help="Model Name")

parser.add_argument("--batch", default=128, type=int, help="Batch size")
parser.add_argument(
    "--gpu",
    type=int,
    default="0",
    help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
)
parser.add_argument(
    "--epochs", type=int, default=5, help="Maximum number of epochs"
)
parser.add_argument(
    "--wd", type=float, default=0, help="L2 Regularization for Optimizer"
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="Number of processes to construct batches",
)
parser.add_argument(
    "--early_stop", default=15, type=int, help="Patience for early stop."
)

parser.add_argument(
    "--in_size",
    default=128,
    type=int,
    help="Initial dimension size for entities.",
)
parser.add_argument(
    "--out_size",
    default=128,
    type=int,
    help="Output dimension size for entities.",
)

parser.add_argument(
    "--num_heads", default=1, type=int, help="Number of attention heads"
)
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")

args = parser.parse_args()

print(args)


#设备选择
if args.gpu >= 0 and torch.cuda.is_available():
    device = "cuda:{}".format(args.gpu)
else:
    device = "cpu"

#加载数据
(
    g,
    train_loader,
    eval_loader,
    test_loader,
    meta_paths,
    user_key,
    item_key,
) = load_data(args.dataset, args.batch, args.num_workers, args.path)  #amazon 128 2 /data
g = g.to(device)            #cuda:0
print("Data loaded.")

#创建模型
model = TAHIN(
        g, meta_paths, args.in_size, args.out_size, args.num_heads, args.dropout
    )
model = model.to(device)
print("Model created.")

model.eval()
with torch.no_grad():
    model.load_state_dict(torch.load("TAHIN" + "_" + args.dataset))   #加载模型
    user_idx =  [0, 1, 2]
    item_idx =  [3, 4, 5]
    user_emb, item_emb,user_rela_emb,item_rela_emb, user_path_emb, item_path_emb = model.forward(g, 'user', 'item', user_idx, item_idx)



def getItemEmb():
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load("TAHIN" + "_" + args.dataset))   #加载模型
        user_idx =  [0, 1, 2]
        item_idx =  [3, 4, 5]
        user_emb, item_emb,user_rela_emb,item_rela_emb, user_path_emb, item_path_emb = model.forward(g, 'user', 'item', user_idx, item_idx)
    return item_emb


