import torch
import torch.nn as nn
from torch.autograd import Function

from .modules.encoder import MLPEncoder, LSTMEncoder
from .misa import MISA
from .attention_topn import Attention_TOPN_Discriminator
from .modules.transformers_encoder.multihead_attention import MultiheadAttention


class MISAwithDiscriminator(nn.Module):
    def __init__(self, args, attention_dim=1024, num_heads=2):
        super(MISAwithDiscriminator, self).__init__()

        self.args = args
        # discriminator
        self.topn_discriminator = Attention_TOPN_Discriminator(input_dims=args.hidden_dim, hidden_dim=args.hidden_dim,
                                                               dropout=args.dropout)
        # cross-attention
        self.self_att = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_TA = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_TV = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_VA = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_VT = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_AT = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )
        self.cross_att_AV = MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=num_heads,
            attn_dropout=args.dropout
        )

        # model misa
        self.misa = MISA(args)

    def forward(self, batch):
        feat_text = batch['texts']  # shape: (B, D)
        feat_audio = batch['audios']  # shape: (B, D)
        feat_video = batch['videos']  # shape: (B, D)

        # discriminate the main modality
        (t, a, v), main_index = self.topn_discriminator(feat_text, feat_audio, feat_video)

        # process new features batch with main modality
        main_feat_list, other_feat_list1, other_feat_list2 = [], [], []
        for ii in range(main_index.shape[0]):
            if main_index[ii] == 0:
                main_feat_list.append(t[ii])
                other_feat_list1.append(a[ii])
                other_feat_list2.append(v[ii])
            elif main_index[ii] == 1:
                main_feat_list.append(a[ii])
                other_feat_list1.append(t[ii])
                other_feat_list2.append(v[ii])
            elif main_index[ii] == 2:
                main_feat_list.append(v[ii])
                other_feat_list1.append(a[ii])
                other_feat_list2.append(t[ii])

        main_feat = torch.cat(main_feat_list, dim=0)
        sec_feat1 = torch.cat(other_feat_list1, dim=0)
        sec_feat2 = torch.cat(other_feat_list2, dim=0)

        # attention fusion
        main_self = self.self_att(main_feat, main_feat, main_feat)[0]
        cross1 = self.cross_att1(main_feat, sec_feat1, sec_feat1)[0]
        cross2 = self.cross_att2(main_feat, sec_feat2, sec_feat2)[0]

        # concat and to misa
        batch_for_misa = {
            'texts': main_self,
            'audios': cross1,
            'videos': cross2
        }
        return self.misa(batch_for_misa)