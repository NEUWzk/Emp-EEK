import logging
import argparse
import torch

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
SEP_idx = 4
USR_idx = 5
SYS_idx = 6
KG_idx = 7
CLS_idx = 8
CLSs_idx = 9
CLSl_idx = 10
CLSp_idx = 11
Y_idx = 12

if (torch.cuda.is_available()):
    USE_CUDA = True
else:
    USE_CUDA = False
# USE_CUDA = False
parser = argparse.ArgumentParser()

parser.add_argument("--hidden_dim", type=int, default=300, help='隐藏层维度')
parser.add_argument('--d_model', type=int, default=300, help='节点转换的维度')
parser.add_argument("--emb_dim", type=int, default=300, help='嵌入层维度')
parser.add_argument("--emo_dim", type=int, default=40, help='情绪嵌入维度')
parser.add_argument("--act_dim", type=int, default=30, help='对话行为嵌入维度')
parser.add_argument("--batch_size", type=int, default=16 , help='输入的batch大小')
parser.add_argument("--num_epochs", type=int, default=12, help='训练的轮数')
parser.add_argument("--lr", type=float, default=1e-4, help='学习率')
parser.add_argument("--olr", type=float, default=1.8e-4, help='学习率')
parser.add_argument("--emos_lr", type=float, default=5e-7, help='学习率')
parser.add_argument("--emol_lr", type=float, default=1e-5, help='学习率')
parser.add_argument("--act_lr", type=float, default=1e-7, help='学习率')
parser.add_argument("--lr_decay", type=float, default=0.6, help='学习率衰减')
# parser.add_argument("--max_grad_norm", type=float, default=5.0, help='最大梯度裁剪，在优化器更新之前进行梯度裁剪操作')
parser.add_argument("--beam_size", type=int, default=5, help='集束搜索大小')
parser.add_argument("--beam_search", type=bool, default=True, help='集束搜索大小')
parser.add_argument("--save_path", type=str, default='./save/test/',help='保存路径')
parser.add_argument("--print_file", type=str, default='./result/',help='保存模型的输出结果')
parser.add_argument("--save_path_pretrained", type=str, default='result/checkpoint', help='保存预训练结果模型路径')
parser.add_argument("--save_path_", type=str, default='./save/final/', help='保存结果模型路径')
parser.add_argument("--pointer_gen", action="store_true", default=True, help='copy mechanism')

parser.add_argument("--pretrain_emb", action="store_true",default=True)
parser.add_argument("--emb_file", type=str, default='/home/wangzikun/gaopan_code/data/glove.6B.300d.txt')
parser.add_argument("--weight_sharing", action="store_true", default=True, help='生成器与词嵌入共享权重')
parser.add_argument("--noam", action="store_true", default=False, help='warmup')
parser.add_argument("--load_optim", action="store_true", default=False, help='加载历史优化器')
parser.add_argument("--universal", action="store_true", default=False)
parser.add_argument("--act", action="store_true", default=False)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
##cvae
parser.add_argument("--latent_dim", type=int, default=200, help='隐变量维度')
parser.add_argument("--full_kl_step", type=int, default=15000, help='KL损失的权重达到1.0之前有多少iter')
parser.add_argument("--num_var_layers", type=int, default=0)
parser.add_argument("--kl_ceiling", type=float, default=0.1)
parser.add_argument("--aux_ceiling", type=float, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help= '梯度累积步长')

## transformer
parser.add_argument("--hop", type=int, default=1, help= 'transformer的层数')
parser.add_argument("--heads", type=int, default=2, help= '多头数量')
parser.add_argument("--depth", type=int, default=40,help= 'KQV的维度')
parser.add_argument("--filter", type=int, default=50, help= 'FFN中间层的隐藏层大小')
parser.add_argument("--dropout", type=float, default=0.0, help= 'dropout')

## knowledge
parser.add_argument("--multi_hop", type=int, default=8, help= '进行多跳推理时的最大多跳数')
parser.add_argument("--K_num", type=int, default=10, help= '每个关键概念不能有超过K_num个实体')
parser.add_argument("--k_num", type=int, default=8, help= '进行多跳推理时的最大候选实体')
parser.add_argument("--rel_num", type=int, default=44, help= '路径中关系的种类数量')
parser.add_argument("--rel_dim", type=int, default=100, help= '关系嵌入维度')
# parser.add_argument("--path_num", type=int, default=15, help= '每个对话的最多路径个数')

#model
parser.add_argument("--model", type=str, default="GEKCM", help='model name, [TCM_Wo_cog, TCM_Wo_emo, TCM_Wo_act, GEKCM, GEKCM_Wo_cog_act, GEKCM_Wo_cog, GEKCM_Wo_cog_kl, GEKCM_Wo_kl, GEKCM_Wo_pointer, GEKCM_Wo_emo, GEKCM_Wo_act]')
parser.add_argument("--test", action="store_true", default=False, help='true for inference, false for training')
parser.add_argument("--train_then_test", action="store_true", default=True, help='test model if the training finishes')
parser.add_argument("--weight", type=str, default='dwa', help='Dynamically adjust weights')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

args = parser.parse_args()
print_opts(args)
# Hyperparameters
model = args.model
hidden_dim= args.hidden_dim
emb_dim= args.emb_dim
emo_dim = args.emo_dim
act_dim = args.act_dim
batch_size= args.batch_size
num_epochs = args.num_epochs
lr=args.lr
lr_decay = args.lr_decay
beam_size=args.beam_size
save_path = args.save_path
save_path_pretrained = args.save_path_pretrained
# device = torch.device("cuda" if args.cuda else "cpu")

test = args.test
pretrain_emb = args.pretrain_emb
emb_file = args.emb_file
weight_sharing = args.weight_sharing
noam = args.noam
load_optim = args.load_optim
universal = args.universal
act = args.act
max_grad_norm = args.max_grad_norm

# max_grad_norm=arg.max_grad_norm
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

eps = 1e-12
epochs = 5

##cvae
latent_dim = args.latent_dim
full_kl_step = args.full_kl_step
num_var_layers = args.num_var_layers
kl_ceiling = args.kl_ceiling
aux_ceiling = args.aux_ceiling
gradient_accumulation_steps = args.gradient_accumulation_steps
### transformer
hop = args.hop
heads = args.heads
depth = args.depth
filter = args.filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))







