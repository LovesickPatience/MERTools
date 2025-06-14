import torch
import torch.nn as nn
import torch.nn.functional as F

from .lf_dnn import *
from .modules.encoder import MLPEncoder, LSTMEncoder
from .attention_topn import Attention_TOPN_Discriminator
from .modules.transformers_encoder.multihead_attention import MultiheadAttention


class Topn_LF_DNN(nn.Module):
    def __init__(self, args, num_heads=4, attn_embed_dim=512):
        super(Topn_LF_DNN, self).__init__()

        self.args = args
        self.attn_embed_dim = attn_embed_dim
        # discriminator
        self.topn_discriminator = Attention_TOPN_Discriminator(args).cuda()
        # self-attention
        self.self_att_a = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.audio_dim,
                                             kv_dim=args.audio_dim, attn_dropout=args.dropout)
        self.self_att_v = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.video_dim,
                                             kv_dim=args.video_dim, attn_dropout=args.dropout)
        self.self_att_t = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.text_dim,
                                             kv_dim=args.text_dim, attn_dropout=args.dropout)
        # cross-attention
        self.cross_att_TA = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.text_dim,
                                               kv_dim=args.audio_dim, attn_dropout=args.dropout)
        self.cross_att_TV = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.text_dim,
                                               kv_dim=args.video_dim, attn_dropout=args.dropout)
        self.cross_att_VA = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.video_dim,
                                               kv_dim=args.audio_dim, attn_dropout=args.dropout)
        self.cross_att_VT = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.video_dim,
                                               kv_dim=args.text_dim, attn_dropout=args.dropout)
        self.cross_att_AT = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.audio_dim,
                                               kv_dim=args.text_dim, attn_dropout=args.dropout)
        self.cross_att_AV = MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads, query_dim=args.audio_dim,
                                               kv_dim=args.video_dim, attn_dropout=args.dropout)

        self.lf_dnn = LF_DNN(args, self.attn_embed_dim)
        self.grad_clip = args.grad_clip

    def forward(self, batch):
        feat_text = batch['texts']  # shape: (B, D)
        feat_audio = batch['audios']  # shape: (B, D)
        feat_video = batch['videos']  # shape: (B, D)

        # discriminate the main modality
        (t, a, v), main_index = self.topn_discriminator(feat_text, feat_audio, feat_video)

        # process new features batch with main modality
        fused_main, fused_cross1, fused_cross2 = [], [], []
        for ii in range(main_index.shape[0]):
            if main_index[ii] == 0:
                m = t[ii].unsqueeze(0).unsqueeze(0)
                s1_m = self.cross_att_TA(t[ii].unsqueeze(0).unsqueeze(0), a[ii].unsqueeze(0), a[ii].unsqueeze(0))[0]
                s1_a = self.cross_att_AT(a[ii].unsqueeze(0).unsqueeze(0), t[ii].unsqueeze(0), t[ii].unsqueeze(0))[0]
                s1 = torch.cat([s1_m, s1_a], dim=-1).squeeze(1)

                s2_m = self.cross_att_TV(t[ii].unsqueeze(0).unsqueeze(0), v[ii].unsqueeze(0), v[ii].unsqueeze(0))[0]
                s2_a = self.cross_att_VT(v[ii].unsqueeze(0).unsqueeze(0), t[ii].unsqueeze(0), t[ii].unsqueeze(0))[0]
                s2 = torch.cat([s2_m, s2_a], dim=-1).squeeze(1)

                m_self = self.self_att_t(m, m, m)[0]
            elif main_index[ii] == 1:
                m = a[ii].unsqueeze(0).unsqueeze(0)
                s1_m = self.cross_att_AT(a[ii].unsqueeze(0).unsqueeze(0), t[ii].unsqueeze(0), t[ii].unsqueeze(0))[0]
                s1_a = self.cross_att_TA(t[ii].unsqueeze(0).unsqueeze(0), a[ii].unsqueeze(0), a[ii].unsqueeze(0))[0]
                s1 = torch.cat([s1_m, s1_a], dim=-1).squeeze(1)

                s2_m = self.cross_att_AV(a[ii].unsqueeze(0).unsqueeze(0), v[ii].unsqueeze(0), v[ii].unsqueeze(0))[0]
                s2_a = self.cross_att_VA(v[ii].unsqueeze(0).unsqueeze(0), a[ii].unsqueeze(0), a[ii].unsqueeze(0))[0]
                s2 = torch.cat([s2_m, s2_a], dim=-1).squeeze(1)

                m_self = self.self_att_a(m, m, m)[0]
            elif main_index[ii] == 2:
                m = v[ii].unsqueeze(0).unsqueeze(0)
                s1_m = self.cross_att_VA(v[ii].unsqueeze(0).unsqueeze(0), a[ii].unsqueeze(0), a[ii].unsqueeze(0))[0]
                s1_a = self.cross_att_AV(a[ii].unsqueeze(0).unsqueeze(0), v[ii].unsqueeze(0), v[ii].unsqueeze(0))[0]
                s1 = torch.cat([s1_m, s1_a], dim=-1).squeeze(1)

                s2_m = self.cross_att_VT(v[ii].unsqueeze(0).unsqueeze(0), t[ii].unsqueeze(0), t[ii].unsqueeze(0))[0]
                s2_a = self.cross_att_TV(t[ii].unsqueeze(0).unsqueeze(0), v[ii].unsqueeze(0), v[ii].unsqueeze(0))[0]
                s2 = torch.cat([s2_m, s2_a], dim=-1).squeeze(1)
                m_self = self.self_att_v(m, m, m)[0]

            fused_main.append(m_self.squeeze(0))
            fused_cross1.append(s1)
            fused_cross2.append(s2)

        main_feat = torch.cat(fused_main, dim=0)
        sec_feat1 = torch.cat(fused_cross1, dim=0)
        sec_feat2 = torch.cat(fused_cross2, dim=0)

        # concat and to misa
        batch_for_misa = {
            'self_attn': main_feat,
            'cross_attn_1': sec_feat1,
            'cross_attn_2': sec_feat2
        }

        return self.lf_dnn(batch_for_misa)