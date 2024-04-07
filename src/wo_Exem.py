### TAKEN FROM https://github.com/kolloldas/torchnlp
#这个是当前用的主模型
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np
import math
from gcn_layer import DialogueGCN
from common_layer import MultiHeadAttention,DecoderMultiHeadAttention,PositionwiseFeedForward
from div_function import clean_preds
from common1 import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    LabelSmoothing,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)

import config
import random
from relationNetwork import RelationNetwork
from config import args


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        if not config.project:
            x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist



class KDecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, vocab_size=None, layer_dropout=0, attention_dropout=0.1, relu_dropout=0.1):
        super(KDecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = DecoderMultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                                      hidden_size, num_heads, None, attention_dropout)

        self.multi_head_attention_node_dec = DecoderMultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                                      hidden_size, num_heads, None, attention_dropout)

        self.multi_head_attention_knowledge = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        输入是由解码器输入和编码器输出组成的元组
        """
        x, node_outputs, attention_weight, mask = inputs  # x, encoder_output, [], (mask_src,dec_mask)
        mask_src, dec_mask = mask
        # mask = (mask_src, dec_mask)
        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        # Dropout and after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, node_outputs, node_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, node_outputs, attention_weight, mask



class MulDecoder(nn.Module):
    def __init__(
        self,
        expert_num,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )
        if config.basic_learner:
            self.basic = DecoderLayer(*params)
        self.experts = nn.ModuleList([DecoderLayer(*params) for e in range(expert_num)])  #多个解码器
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        self.kdec = nn.Sequential(*[KDecoderLayer(*params) for l in range(num_layers)])
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask, attention_epxert, node_feature, node_mask): #inputs:tensor(batch,tar_len,300) encoder_output:(batch,src_len,300) mask是个元组:0：(batch,1,src_len) 1：(batch,1,tar_len)
        mask_src, mask_trg = mask #attention_epxert:32,32,1,1
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )  #32,tar_len,tar_len
        # Add input dropout
        x = self.input_dropout(inputs)  #32,tar_len,300 代表回复的句子
        if not config.project:
            x = self.embedding_proj(x)  #32,tar_len,300
        # Add timing signal
        x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
        expert_outputs = []
        if config.basic_learner:
            basic_out, _, attn_dist, _ = self.basic(
                (x, encoder_output, [], (mask_src, dec_mask))
            )

        # compute experts
        # TODO forward all experts in parrallel
        if attention_epxert.shape[0] == 1 and config.topk > 0:
            for i, expert in enumerate(self.experts):
                if attention_epxert[0, i] > 0.0001:  # speed up inference
                    expert_out, _, attn_dist, _ = expert(
                        (x, encoder_output, [], (mask_src, dec_mask))
                    )
                    expert_outputs.append(attention_epxert[0, i] * expert_out) #存放三个解码器的输出结果
            x = torch.stack(expert_outputs, dim=1)  #expert_outputs：list3  (1,tar_len,300)  #x:(1,3,tar_len,300)
            x = x.sum(dim=1)

        else:
            for i, expert in enumerate(self.experts): #走这里
                expert_out, _, attn_dist, _ = expert(
                    (x, encoder_output, [], (mask_src, dec_mask))
                ) #expert_out:32,tar_len,300    attn_dist:32,tar_len,src_len
                expert_outputs.append(expert_out)  #[]存放结果长度为decoder个数，每一个元素都是32,44,300
            x = torch.stack(
                expert_outputs, dim=1
            )  # (batch_size, expert_number, tar_len, hidden_size)
            x = attention_epxert * x  #32,32,1,1 * 32,32,44,300
            x = x.sum(dim=1)  # (batch_size, tar_len, hidden_size)
        if config.basic_learner:  #True
            x += basic_out #1,tar_len,300
        #  x:(16,trg_len,300)  y:(16,trg_len,300)
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
        # y: 16, trg_len, 300  attn_dist:16,trg_len,src_len
        # Final layer normalization

        ky, _, node_att_dist, _ = self.kdec((x, node_feature, [], (node_mask, dec_mask)))

        y = self.layer_norm(y)
        ky = self.layer_norm(ky)
        return y, attn_dist, ky, node_att_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)
        self.k_gen_linear = nn.Linear(args.hidden_dim, 1)

    def forward(
        self,
        x,  #(16,trg_len,300)
        kx,  #(16,trg_len,300)
        attn_dist=None,  #(16, trg_len, src_len)
        node_attn_dist=None,
        node_batch=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if kx is not None:
            lamda = 0.8  #走这里
        else:
            lamda = 1

        if args.pointer_gen1:  #True
            p_gen = self.p_gen_linear(x)  #(16, trg_len 1)
            alpha = torch.sigmoid(p_gen)  #(16, trg_len 1)
            alpha = lamda * alpha  #(16, trg_len 1)
            if kx is not None:
                k_gen = self.k_gen_linear(kx)  #(16, trg_len 1)
                beta = torch.sigmoid(k_gen)   #(16, trg_len 1)
                beta =(1-lamda) * beta


        logit = self.proj(x)  #(16,trg_len,24292)

        if args.pointer_gen1:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist
            if kx is not None:
                node_attn_dist = F.softmax(node_attn_dist/temp, dim=-1)
                node_attn_dist_ = beta * node_attn_dist
                node_batch_ = torch.cat([node_batch.unsqueeze(1)]*x.size(1),1) ## extend for all seq

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            if kx is not None:
                attn_dist_ = (1 - alpha - beta) * attn_dist
            else:
                attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq

            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                if beam_search:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros.repeat(5,1,1)], 2)
                else:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)
            vocab_dist_ = vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            assert not torch.isnan(vocab_dist_).any()

            if kx is not None:
                logit = torch.log(vocab_dist_.scatter_add(2, node_batch_, node_attn_dist_) + 1e-18)
            else:
                logit = torch.log(vocab_dist_ + 1e-18)

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class EEK_woExem(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
    ):
        super(EEK_woExem, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.word_freq = np.zeros(self.vocab_size)
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.gcn = DialogueGCN(self.vocab, self.embedding)
        self.rn = RelationNetwork(self.embedding)
        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )
        self.decoder_number = decoder_number
        ## multiple decoders
        self.decoder = MulDecoder(
            decoder_number,
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.fc2 = nn.Linear(600,300) #将GCN和act_encoder_outpus进行融合的线性层
        self.decoder_key = nn.Linear(config.hidden_dim, 3, bias=False)
        self.decoder_key1 = nn.Linear(config.hidden_dim, 32, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emotion_embedding = nn.Linear(32, 300)
        self.fc1 = nn.Linear(300,300)
        self.fc_cat_hidden_states = nn.Linear(600, 300)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion1 = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        self.criterion1.weight = torch.ones(self.vocab_size)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)  #ctx_loss

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        # else:
        #     self.attention_activation = nn.Sigmoid()  # nn.Softmax()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )


    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)

    def forward(self, batch):  # 外部知识
        enc_batch = batch['context_batch']  # 对话上下文的向量表示 Tensor(batch,len)
        dec_batch = batch['target_batch']  # gold response Tensor(batch,len)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # batch,1,src_len
        input_mask = self.embedding(batch['mask_context'])  # 理论上(batch,len,300),对话状态嵌入
        act_encoder_outputs = self.encoder(self.embedding(enc_batch) + input_mask, mask_src) #batch,src_len,300

        # 与concept相关的：
        key_concepts = batch["concepts_batch"]  # 关键词
        node_batch = batch["node_id"]
        update_nodes = self.gcn(batch)  #(sum_node_num, 100)

        update_paths, node_feature, node_mask \
            = self.rn(update_nodes, act_encoder_outputs, batch['dia_triples'],  batch['node_number'])  # 关系网络
        # update_paths : (16,300)
        # node_feature :(16, max_node2id_len, 300)
        # node_mask : (16, 1, max_node2id_len)
        # act_encoder_outputs : (16,src_len,300)

        context_add_concept = torch.cat([act_encoder_outputs,
                              update_paths.unsqueeze(1).expand(-1,act_encoder_outputs.shape[1],-1)],dim=2)  #(16,src_len,600)
        context_add_concept1 = self.fc2(context_add_concept)   #(16,src_len,600)
        return mask_src, act_encoder_outputs, context_add_concept1, node_feature, node_mask, node_batch


    def exemplars_forward(self, batch):
        exem_batch = batch['exem_batch']  # list[0,1,2..batch-1]
        exem_batch_len = len(exem_batch)  #batch_size
        exem_batch_mask = []
        exemp_representations = []  # list,len=batch_size,list[i]=tensor(10,exem_len,300)
        for i in range(exem_batch_len):
            mask_exem = exem_batch[i].data.eq(config.PAD_idx).unsqueeze(1)
            exem_batch_mask.append(mask_exem)
        for j in range(exem_batch_len): #对列表中的每一个范例进行编码
            exem_encoder_outputs = self.encoder(self.embedding(exem_batch[j]), exem_batch_mask[j]) #编码
            exemp_representations.append(exem_encoder_outputs) #[tensor(10,exem_len,300);tensor(10,exem_len,300)]
        batch_exem_representations = []
        for each_exem_tensor in exemp_representations:
            exem_hidden_states = torch.sum(each_exem_tensor, dim=1)
            final_hidden_states = torch.mean(exem_hidden_states,dim=0) #300维列向量
            batch_exem_representations.append(final_hidden_states)

        return torch.stack(batch_exem_representations, 0) #范例编码完成


    def train_one_batch(self, batch, iter, train=True):  #修改后的，自己的
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        mask_src, act_encoder_outputs, context_add_concept, node_feature, node_mask, node_batch = self.forward(batch)  #context_add_concept:batch,src_len,300

        final_encode_rep = context_add_concept

        if config.noam:  #True
            self.optimizer.optimizer.zero_grad() #True
        else:
            self.optimizer.zero_grad()
        ## Encode

        q_h = ( #Attention over decoder 根据语义的编码信息计算得到的
            torch.mean(act_encoder_outputs, dim=1)
            if config.mean_query
            else act_encoder_outputs[:, 0] #运行这个
        )  #batch,300
        q_h1 = final_encode_rep[:, 0]
        # q_h = encoder_outputs[:,0]  (batch,300)
        logit_prob = self.decoder_key(q_h)  # (bsz, 3) #为3个解码器赋予不同权重
        emo_prob = self.decoder_key1(q_h1)  #情感分类矩阵 batch,32
        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk) #32，3
            a = np.empty([logit_prob.shape[0], self.decoder_number]) #16,3
            a.fill(float("-inf"))
            mask = torch.Tensor(a).to(config.device)
            logit_prob_ = mask.scatter_(
                1, k_max_index.to(config.device).long(), k_max_value
            ) #32,32
            attention_parameters = self.attention_activation(logit_prob_)
        else: #如果不想topk，走这里
            attention_parameters = self.attention_activation(logit_prob)
        # print("===============================================================================")
        # print("listener attention weight:",attention_parameters.data.cpu().numpy())
        # print("===============================================================================")
        if config.oracle: #False  batch["target_program"]:one-hot向量，为了更快收敛？
            attention_parameters = self.attention_activation(
                torch.FloatTensor(batch["empathy_label"]) * 1000  #batch["target_program"]：[[ ]]
            ).to(config.device)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)

        # Decode
        # sos_token = (
        #     torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
        #     .unsqueeze(1)
        #     .to(config.device)
        # ) #tensor(batch,1)  batch个SOS_token
        # dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1) #batch,tar_len
        #
        # mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1) #和 mask_src格式相同

        #decode with emo_signal: 这个model用来测试加入指针网络的结果
        sos_emb = self.emotion_embedding(emo_prob).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])  # (bsz, tgt_len-1, emb_dim) 将情感信号和真实的回复进行拼接
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, tgt_len, emb_dim)
        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, 1, tgt_len)


        pre_logit, attn_dist, pre_klogit, node_att_dist = self.decoder( # y, attn_dist, ky, node_att_dist
            dec_emb,  #gold response
            final_encode_rep,   #编码器输出,包含原始文本+范例+GCN
            (mask_src, mask_trg),  #mask
            attention_parameters,
            node_feature, node_mask  #加入GCN中的信息
        )  #pre_logit:32,44,300
        ## compute output dist  attn_dist:32,44,74
        # torch.set_printoptions(profile="full")
        # print(attn_dist)
        logit = self.generator(   #logit : (16, trg_len, vocab_len)
            pre_logit, pre_klogit,
            attn_dist, node_att_dist,
            node_batch,
            enc_batch,
            extra_zeros,
            attn_dist_db=None,
        )
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        #随着iter的增加,0.001 + (1 - 0.001) * math.exp(-1.0 * iter / config.schedule)越来越小(退火)
        if train and config.schedule > 10:  #True 训练初始阶段基本上都是True
            if random.uniform(0, 1) <= ( #True  0-1之间的小数
                0.001 + (1 - 0.001) * math.exp(-1.0 * iter / config.schedule)
            ):
                config.oracle = True
            else:
                config.oracle = False

        if config.softmax:  #True
            loss = self.criterion(  #True,带label_smooth的
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ) + nn.CrossEntropyLoss()(
                emo_prob, batch["emotion_label"]
            )
            loss_bce_program = nn.CrossEntropyLoss()(
                emo_prob, batch["emotion_label"]
            ).item()  #torch.LongTensor(batch["program_label"]).to(config.device)
        else:
            loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ) + nn.BCEWithLogitsLoss()(
                emo_prob, torch.FloatTensor(batch["empathy_label"]).to(config.device)
            )
            loss_bce_program = nn.BCEWithLogitsLoss()(
                emo_prob, torch.FloatTensor(batch["empathy_label"]).to(config.device)
            ).item()
        pred_program = np.argmax(emo_prob.detach().cpu().numpy(), axis=1)  #预测情感标签
        emo_pre = batch["emotion_label"].cpu().numpy().tolist()  #list
        program_acc = accuracy_score(emo_pre, pred_program)

        if config.label_smoothing: #True
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ).item()

        #最后需要加入div_loss测试
        _, preds = logit.max(dim=-1)
        preds = clean_preds(preds)
        self.update_frequency(preds)
        self.criterion1.weight = self.calc_weight()
        not_pad = dec_batch.ne(config.PAD_idx)
        target_tokens = not_pad.long().sum().item()
        div_loss = self.criterion1(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        div_loss /= target_tokens
        total_loss = 1 * loss + 1 * div_loss

        if train:
            total_loss.backward()
            self.optimizer.step()

        if config.label_smoothing:  #True
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc,div_loss.item()
        else:
            return (
                loss.item(),
                math.exp(min(loss.item(), 100)),
                loss_bce_program,
                program_acc,
            )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        mask_src, act_encoder_outputs, context_add_concept, node_feature, node_mask, node_batch = self.forward(batch)  # 外部知识

        final_encode_rep = context_add_concept


        ## Attention over decoder
        q_h = (
            torch.mean(act_encoder_outputs, dim=1)
            if config.mean_query
            else act_encoder_outputs[:, 0]
        )
        q_h1 = final_encode_rep[:, 0]
        # q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h)
        emo_prob = self.decoder_key1(q_h1)

        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float("-inf"))
            mask = torch.Tensor(a).to(config.device)
            logit_prob = mask.scatter_(
                1, k_max_index.to(config.device).long(), k_max_value
            )

        attention_parameters = self.attention_activation(logit_prob)

        if config.oracle:
            attention_parameters = self.attention_activation(
                torch.FloatTensor(batch["empathy_label"]) * 1000
            ).to(config.device)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(
            -1
        )  # (batch_size, expert_num, 1, 1)

        # ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        # mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        # decoded_words = []

        ys = torch.ones(enc_batch.shape[0], 1).fill_(config.SOS_idx).long()
        ys = ys.to(config.device)  # tensor([3])  (1,1)
        ys_emb = self.emotion_embedding(emo_prob).unsqueeze(1)  # (bsz=1, 1, emb_dim)
        ys_emb = ys_emb.to(config.device)

        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        decoded_words = []

        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(act_encoder_outputs),
                    (mask_src, mask_trg),
                    attention_parameters,
                )
            else:  #Here
                out, attn_dist, k_out, node_attn_dist = self.decoder(
                    ys_emb,
                    final_encode_rep,  #编码器最终的输出
                    (mask_src, mask_trg),
                    attention_parameters,
                    node_feature, node_mask
                )

            logit = self.generator(
                out, k_out,
                attn_dist, node_attn_dist,
                node_batch,
                enc_batch, extra_zeros, attn_dist_db=None
            )
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            _, next_word = torch.max(logit[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
            ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).to(config.device))),
                               dim=1)  # 当前生成阶段结束后，在后面拼接新的单词

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        ## Attention over decoder
        q_h = (
            torch.mean(encoder_outputs, dim=1)
            if config.mean_query
            else encoder_outputs[:, 0]
        )
        # q_h = encoder_outputs[:,0]
        logit_prob = self.decoder_key(q_h)

        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float("-inf"))
            mask = torch.Tensor(a).to(config.device)
            logit_prob = mask.scatter_(
                1, k_max_index.to(config.device).long(), k_max_value
            )

        attention_parameters = self.attention_activation(logit_prob)

        if config.oracle:
            attention_parameters = self.attention_activation(
                torch.FloatTensor(batch["target_program"]) * 1000
            ).to(config.device)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(
            -1
        )  # (batch_size, expert_num, 1, 1)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                    attention_parameters,
                )
            else:

                out, attn_dist = self.decoder(
                    self.embedding(ys),
                    encoder_outputs,
                    (mask_src, mask_trg),
                    attention_parameters,  #注意力分数
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1], top_k=5, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            next_word = torch.multinomial(
                F.softmax(filtered_logit, dim=-1), 1
            ).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.item()
            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent


### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(
        self,
        state,
        inputs,
        fn,
        time_enc,
        pos_enc,
        max_hop,
        encoder_output=None,
        decoding=False,
    ):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).to(
            config.device
        )
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).to(config.device)
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).to(config.device)
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).to(config.device)

        step = 0
        # for l in range(self.num_layers):
        while (
            ((halting_probability < self.threshold) & (n_updates < max_hop))
            .byte()
            .any()
        ):
            # Add timing signal
            state = state + time_enc[:, : inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(
                1, inputs.shape[1], 1
            ).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (
                halting_probability + p * still_running > self.threshold
            ).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (
                halting_probability + p * still_running <= self.threshold
            ).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if decoding:
                state, _, attention_weight = fn((state, encoder_output, []))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = (state * update_weights.unsqueeze(-1)) + (
                previous_state * (1 - update_weights.unsqueeze(-1))
            )
            if decoding:
                if step == 0:
                    previous_att_weight = torch.zeros_like(attention_weight).to(
                        config.device
                    )  ## [B, S, src_size]
                previous_att_weight = (
                    attention_weight * update_weights.unsqueeze(-1)
                ) + (previous_att_weight * (1 - update_weights.unsqueeze(-1)))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1

        if decoding:
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)
