import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from transformers import BertConfig, BertModel

from modules.encoders import (CPC, MMILB, ConcatFusion, CrossAttention,
                              Encoder, FinalFusionSelfAttention, FusionTrans,
                              LanguageEmbeddingLayer, RNNEncoder,
                              SelfAttention, SubNet, SumFusion)
from modules.InfoNCE import InfoNCE


class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )



        # CPC MI bound   d_prjh是什么？？？
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout + hp.model_dim_cross * 2  # 计算所有模态输出后的维度和 用于后期融合操作
        # Trimodal Settings   三模态融合
        self.fusion_prj = SubNet(
            in_size=dim_sum,  # 三个单模态输出维度和
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,  # 最终分类类别
            dropout=hp.dropout_prj
        )

        unimodal_sum=hp.d_aout+hp.d_tout+hp.d_vout
        # self.unimodal_fusion_MLP=SubNet(in_size=unimodal_sum,hidden_size=hp.d_prjh,n_class=hp.n_class,dropout=hp.dropout_prj)
        #
        crossmodal_sum=hp.model_dim_cross * 2
        # self.crossmodal_fusion_MLP=SubNet(in_size=crossmodal_sum,hidden_size=hp.d_prjh,n_class=hp.n_class,dropout=hp.dropout_prj)

        assert hp.fusion in ['sum','concat']
        if hp.fusion=="sum":
            self.fusion_module = SumFusion(input_dim=hp.model_dim_cross,output_dim=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)
        else:
            self.fusion_module=ConcatFusion(input_dim=crossmodal_sum,output_dim=hp.d_prjh,n_class=hp.n_class)

        self.classifier_ap_ta=nn.Sequential(nn.Linear(hp.model_dim_cross,hp.model_dim_cross),nn.ReLU(),nn.Linear(hp.model_dim_cross,2))
        self.classifier_ap_tv = nn.Sequential(nn.Linear(hp.model_dim_cross, hp.model_dim_cross), nn.ReLU(),nn.Linear(hp.model_dim_cross, 2))
        self.prob=self.hp.prob
        self.num_for_ta=self.hp.num
        self.num_for_tv=self.hp.num

        # 用MULT融合
        # self.fusion_trans = FusionTrans(
        #     hp,
        #     n_class=hp.n_class,  # 最终分类类别
        # )

        self.uni_audio_encoder = SelfAttention(hp,d_in=hp.d_ain, d_model=hp.model_dim_self, nhead=hp.num_heads_self,
                                     dim_feedforward=4 * hp.model_dim_self,dropout=hp.attn_dropout_self,
                                     num_layers=hp.num_layers_self)
        self.uni_vision_encoder = SelfAttention(hp,d_in=hp.d_vin, d_model=hp.model_dim_self, nhead=hp.num_heads_self,
                                     dim_feedforward=4 * hp.model_dim_self,dropout=hp.attn_dropout_self ,
                                     num_layers=hp.num_layers_self)

        self.ta_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_self,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)
        self.tv_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_self,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)

        # self.layer_wise_tv=InfoNCE(hp.d_tin,hp.model_dim_self,hp.embed_dropout_infonce_layer)
        # self.layer_wise_av = InfoNCE(hp.model_dim_self, hp.model_dim_self, hp.embed_dropout_infonce_layer)
        # self.layer_wise_ta = InfoNCE(hp.d_tin, hp.model_dim_self, hp.embed_dropout_infonce_layer)

        self.uni_infonce_tv=InfoNCE(hp.d_tin,hp.d_vin,hp.embed_dropout_infonce)
        self.uni_infonce_ta = InfoNCE(hp.d_tin, hp.d_ain, hp.embed_dropout_infonce)

        # self.layer_wise_cross=InfoNCE(hp.model_dim_cross,hp.model_dim_cross,hp.embed_dropout_infonce_layer_cross)
        self.infonce_cross=InfoNCE(hp.model_dim_cross,hp.model_dim_cross,hp.embed_dropout_infonce_cross)

    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)
            # 特征全为0的时刻加mask
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None,
                mem=None,v_mask=None,a_mask=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        sentences: torch.Size([0, 32])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        For Bert input, the length of text is "seq_len + 2"
        """
        with torch.no_grad():
            maskT = (bert_sent_mask == 0)
            maskV = self.gen_mask(visual.transpose(0,1),v_len)
            maskA = self.gen_mask(acoustic.transpose(0,1),a_len)

        # 1. 三个模态 分别 进行特征提取
        enc_word= self.uni_text_enc(sentences, bert_sent, bert_sent_type,
                                 bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)
        visual=self.uni_vision_encoder(visual) #[seq_len,bs,dim]
        acoustic= self.uni_audio_encoder(acoustic) #[seq_len,bs,dim]

        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)
        vision_trans = visual
        audio_trans = acoustic
        ###layer-wise-contrastive
        # nce_layer_wise_tv=self.layer_wise_tv(hidden_states_text[6],hidden_states_visual[3].transpose(1,0))
        # nce_layer_wise_ta = self.layer_wise_ta(hidden_states_text[6], hidden_states_acoustic[3].transpose(1,0))
        # nce_layer_wise_av = self.layer_wise_av(hidden_states_acoustic[3].transpose(1,0), hidden_states_visual[3].transpose(1,0))
        # layer_infonce=nce_layer_wise_tv+nce_layer_wise_ta+nce_layer_wise_av

        ###2.pre-contrastive 消融实验点 text-centered unimodal contrastive learning

        nce_ta,logits_ta=self.uni_infonce_ta(text_trans.transpose(1,0),audio_trans.transpose(1,0))
        nce_tv,logits_tv=self.uni_infonce_tv(text_trans.transpose(1,0),vision_trans.transpose(1,0))
        if not self.hp.no_ta: #不对齐ta
            info_nce=nce_tv
        elif not self.hp.no_tv: #不对齐tv
            info_nce=nce_ta
        elif not self.hp.use_none: #啥都不对齐
            info_nce=torch.tensor(0)
        else: #啥都对齐
            info_nce=nce_ta+nce_tv

        ###3.text-centered random fusion
        if self.training and not self.hp.ta_ap: #不用ta_ap
            text_for_cross, vision_for_cross, vision_mask_for_cross, label_tv = self.random_pair_with_label_prob(text_trans.transpose(1, 0), vision_trans.transpose(1, 0), maskV, self.prob, logits_tv, self.num_for_tv)
            cross_tv = self.tv_cross_attn(text_for_cross.transpose(1, 0), vision_for_cross.transpose(1, 0),Tmask=maskT,Vmask=vision_mask_for_cross)
            cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Vmask=maskA)
            logits_ap_tv = self.classifier_ap_tv(cross_tv.mean(dim=0))
            tv_ap_loss = F.cross_entropy(logits_ap_tv, label_tv)
            ap_loss =tv_ap_loss

        elif self.training and not self.hp.tv_ap:#不用tv_ap
            text_for_cross,audio_for_cross,audio_mask_for_cross,label_ta=self.random_pair_with_label_prob(text_trans.transpose(1,0),audio_trans.transpose(1,0),maskA,self.prob,logits_ta,self.num_for_ta)
            cross_tv = self.tv_cross_attn(text_trans, vision_trans, Tmask=maskT, Vmask=maskV)
            cross_ta = self.ta_cross_attn(text_for_cross.transpose(1, 0), audio_for_cross.transpose(1, 0),Tmask=maskT,Amask=audio_mask_for_cross)
            logits_ap_ta = self.classifier_ap_ta(cross_ta.mean(dim=0))
            ta_ap_loss = F.cross_entropy(logits_ap_ta, label_ta)
            ap_loss =ta_ap_loss

        elif self.training and not self.hp.no_ap:
            cross_tv = self.tv_cross_attn(text_trans, vision_trans,Tmask=maskT,Vmask=maskV)
            cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Amask=maskA)
            ap_loss=torch.tensor(0)

        elif self.training:
            if not self.hp.ta_hard_neg:
                self.num_for_ta=self.hp.batch_size
            elif not self.hp.tv_hard_neg:
                self.num_for_tv=self.hp.batch_size
            elif not self.hp.no_hard_neg:
                self.num_for_ta = self.hp.batch_size
                self.num_for_tv=self.hp.batch_size
            else:
                pass
            text_for_cross,audio_for_cross,audio_mask_for_cross,label_ta=self.random_pair_with_label_prob(text_trans.transpose(1,0),audio_trans.transpose(1,0),maskA,self.prob,logits_ta,self.num_for_ta)
            text_for_cross, vision_for_cross, vision_mask_for_cross, label_tv = self.random_pair_with_label_prob(text_trans.transpose(1, 0), vision_trans.transpose(1, 0), maskV, self.prob, logits_tv, self.num_for_tv)
            cross_ta = self.ta_cross_attn(text_for_cross.transpose(1, 0), audio_for_cross.transpose(1, 0),Tmask=maskT,Amask=audio_mask_for_cross)
            cross_tv = self.tv_cross_attn(text_for_cross.transpose(1, 0), vision_for_cross.transpose(1, 0),Tmask=maskT,Vmask=vision_mask_for_cross)
            logits_ap_ta = self.classifier_ap_ta(cross_ta.mean(dim=0))
            logits_ap_tv = self.classifier_ap_tv(cross_tv.mean(dim=0))
            ta_ap_loss = F.cross_entropy(logits_ap_ta, label_ta)
            tv_ap_loss = F.cross_entropy(logits_ap_tv, label_tv)
            ap_loss = ta_ap_loss+tv_ap_loss
        else:
            cross_tv = self.tv_cross_attn(text_trans, vision_trans,Tmask=maskT,Vmask=maskV)
            cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Amask=maskA)
            ap_loss=None

        ###4.fused modal bidirectional contrastive learning
        # cross_layer_wise_cross=self.layer_wise_cross(hidden_state_tv[3].transpose(1,0),hidden_state_ta[3].transpose(1,0))
        nce_cross_ta,_=self.infonce_cross(cross_ta,cross_tv)
        nce_cross_tv, _ = self.infonce_cross(cross_tv, cross_ta)
        nce_cross=nce_cross_ta+nce_cross_tv if self.hp.stage2 else torch.tensor(0)


        ###contrastive all-in-one Self-Attention
        #fusion,preds=self.final_fusion_trans(text_trans,audio_trans,vision_trans,cross_ta,cross_tv)


        text = enc_word[:, 0, :]  # 32*768 (batch_size, emb_size)
        acoustic = self.uni_acoustic_enc(acoustic, a_len)  # 32*16
        visual = self.uni_visual_enc(visual, v_len)  # 32*16

        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:  # 默认进这
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息
            # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

        # Linear proj and pred
        # text:32*769   acoustic,visual:32*16   ->  cat后：[32, 801]
        # low_level,_=self.unimodal_fusion_MLP(torch.cat([text, acoustic, visual],dim=1))
        # high_level,_=self.crossmodal_fusion_MLP(torch.cat([cross_ta.mean(dim=0),cross_tv.mean(dim=0)], dim=1))

        fusion,preds=self.fusion_module(cross_ta.mean(dim=0),cross_tv.mean(dim=0))

        #fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual,cross_ta.mean(dim=0),cross_tv.mean(dim=0)], dim=1))
        # 32*128  32*1

        #fusion, preds = self.fusion_trans(text_trans, audio_trans, vision_trans)
        # torch.Size([32, 180]) torch.Size([32, 1])

        nce_t = self.cpc_zt(text, fusion)  # 3.4660
        nce_v = self.cpc_zv(visual, fusion)  # 3.4625
        nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

        nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

        pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
        # {'tv': {'pos': None, 'neg': None}, 'ta': {'pos': None, 'neg': None}, 'va': None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        return lld, nce, preds, pn_dic, H,info_nce,nce_cross,ap_loss

    def random_pair_with_label_prob(self,modal1, modal2,mask2, prob, logits, num):
        batch_size = modal1.size(0)
        labels = []
        selected_modal2 = []
        selected_modal2_mask=[]
        for i in range(batch_size):
            # 获取当前 modal1 对应的相似度向量
            similarity_vector = logits[i]

            # 获取与 modal1 最相似的前 num 个 modal2 的索引
            most_similar_indices = torch.argsort(similarity_vector)[-num:]
            rand_key=torch.rand(1)
            # 根据概率 prob 决定是否替换 modal2
            if rand_key < prob :
                #可以被替换的向量nn
                replacement_indices = [idx for idx in range(batch_size) if idx != i and idx in most_similar_indices]
                selected_idx = np.random.choice(replacement_indices)
                selected_modal2.append(modal2[selected_idx])
                selected_modal2_mask.append(mask2[selected_idx])
            else:
                selected_idx = i
                selected_modal2.append(modal2[selected_idx])
                selected_modal2_mask.append(mask2[selected_idx])

            label = 0 if selected_idx != i else 1
            labels.append(label)

        return modal1, torch.stack(selected_modal2), torch.stack(selected_modal2_mask),torch.tensor(labels).cuda(0)



if __name__=="__main__":
    net=Encoder(4, 8, 2,32,0.1,'relu',2)
    data=torch.randn(30,32,4)
    data_mask=pad_sequence([torch.zeros(torch.FloatTensor(sample).size(0)) for sample in data])
    data_mask[:,4:].fill_(float(1.0))
    output=net(data,data_mask.transpose(1,0))
    print(data_mask,data)