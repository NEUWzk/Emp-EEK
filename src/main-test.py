import config

from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_
import os
import torch

from Mydataloader import prepare_data_seq1
from common1 import evaluate
from model_2 import EmpEEK  #加入指针网络之后的新的网络
from model_woGCN import EEK_woGNN
from model_wo_Decoder import EEK_woDecoder
from model_wo_exem import EEK_woExem

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def make_model(vocab, dec_num):
    is_eval = config.test #false表示开启训练
    print(config.model == 'MOEL')
    if config.model == 'MOEL':
        model = EmpEEK(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    elif config.model == 'wo_GNN':
        model = EEK_woGNN(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    elif config.model == 'wo_decoder':
        model = EEK_woDecoder(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    elif config.model == 'wo_exem':
        model = EEK_woExem(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


class GData:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens
     #在这个类下定义的函数，可以用self直接访问
    def index_words(self, sentence):#通过传入的句子（由多个词组成），更新w2i，i2w，w2c，n_words
        for word in sentence:
            self.index_word(word.strip()) #22行

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


if __name__ == "__main__":  #用于测试多个已经保存的模型，看看dist和PPL
    # train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(batch_size=16)
    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq1(batch_size=16)

    model_save_path = '/home/wangzikun/EmpEEK-GNN/save_pointer/840B-0.9-0.9'
    model = make_model(vocab, 3)  # wo_decoder时这里改成1
    model = model.to(config.device)
    isTest = False  # 测试时，isTest = True
    if isTest is False:
        model = model.train()
        best_ppl = 1000
        weights_best = deepcopy(model.state_dict())
        for epoch in range(13):
            for idx, batch in tqdm(enumerate(train_set), total=len(train_set)):
                n_iter = epoch * len(train_set) + idx  # 越来越大的
                model = model.train()

                loss, ppl, bce, acc, div_loss = model.train_one_batch(batch, n_iter)
                # if idx % 500 == 0:
                #     print(" ")
                #     print(f'loss is : {loss}')
                #     print(f'ppl is : {ppl}')
                #     print(f'bce is : {bce}')
                #     print(f'acc is : {acc}')
                #     print(f'div_loss is : {div_loss}')

            # 每一代执行完进行评估
            model.eval()
            loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                model, dev_set, ty="valid", max_dec_step=50
            )

            if ppl_val <= best_ppl or ppl_val < 41:
                best_ppl = ppl_val
                torch.save({"model": model.state_dict(),
                            "result": [loss_val, ppl_val, bce_val, acc_val]},
                           os.path.join(model_save_path, 'model_{}_{:.4f}.tar'.format(epoch, best_ppl)))
                weights_best = deepcopy(model.state_dict())
                print("best_ppl: {}".format(best_ppl))
            print(f'epoch{epoch}计算完成')

    else:  #验证时需要手动修改保存的模型名称
        print("Start Testing !!!")
        model = model.to(config.device)
        model = model.eval()
        checkpoint = torch.load("/home/wangzikun/EmpEEK-GNN/save_pointer/840B-1-1/model_8_39.0477.tar",
                                map_location=lambda storage, location: storage)
        weights_best = checkpoint['model']
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        loss_test, ppl_test, bce_test, acc_test,_= evaluate( model, test_set, ty="test", max_dec_step=50)

    print("======end======")

      #验证时需要手动修改保存的模型名称
    # model1 = make_model(vocab, 3) #wo_decoder时这里改成1
    # model1 = model1.to(config.device)
    # print("Start Testing !!!")
    # model1 = model1.eval()
    # checkpoint1 = torch.load("../save/840B-1-1/model_8_38.8479.tar",
    #                             map_location=lambda storage, location: storage)
    # weights_best = checkpoint1['model']
    # model1.load_state_dict({name: weights_best[name] for name in weights_best})
    # model1.eval()
    # loss_test, ppl_test, bce_test, acc_test,_= evaluate( model1, test_set, ty="test", max_dec_step=50)
    #
    # print("======第一个模型生成结束======")
    # model2 = make_model(vocab, 3) #wo_decoder时这里改成1
    # model2 = model2.to(config.device)
    # checkpoint2 = torch.load("../save/840B-1-1/model_9_39.1554.tar",
    #                          map_location=lambda storage, location: storage)
    # weights_best2 = checkpoint2['model']
    # model2.load_state_dict({name: weights_best2[name] for name in weights_best2})
    # model2.eval()
    # loss_test1, ppl_test1, bce_test1, acc_test1, _ = evaluate(model2, test_set, ty="test", max_dec_step=50)
    # print("======第二个模型生成结束======")
    #
    #
    # model3 = make_model(vocab, 3) #wo_decoder时这里改成1
    # model3 = model3.to(config.device)
    # checkpoint3 = torch.load("../save/840B-1-1/model_10_39.5530.tar",
    #                          map_location=lambda storage, location: storage)
    # weights_best3 = checkpoint3['model']
    # model3.load_state_dict({name: weights_best3[name] for name in weights_best3})
    # model3.eval()
    # loss_test3, ppl_test3, bce_test3, acc_test3, _ = evaluate(model3, test_set, ty="test", max_dec_step=50)
    # print("======第三个模型生成结束======")
    #
    #
    # model4 = make_model(vocab, 3) #wo_decoder时这里改成1
    # model4 = model4.to(config.device)
    # checkpoint4 = torch.load("../save/840B-1-1/model_11_39.9480.tar",
    #                          map_location=lambda storage, location: storage)
    # weights_best4 = checkpoint4['model']
    # model4.load_state_dict({name: weights_best4[name] for name in weights_best4})
    # model4.eval()
    # loss_test4, ppl_test4, bce_test4, acc_test4, _ = evaluate(model4, test_set, ty="test", max_dec_step=50)
    # print("======第四个模型生成结束======")
    #
    #
    # model5 = make_model(vocab, 3) #wo_decoder时这里改成1
    # model5 = model5.to(config.device)
    # checkpoint5 = torch.load("../save/840B-1-1/model_12_40.1319.tar",
    #                          map_location=lambda storage, location: storage)
    # weights_best5 = checkpoint5['model']
    # model5.load_state_dict({name: weights_best5[name] for name in weights_best5})
    # model5.eval()
    # loss_test5, ppl_test5, bce_test5, acc_test5, _ = evaluate(model5, test_set, ty="test", max_dec_step=50)
    # print("======第五个模型生成结束======")













