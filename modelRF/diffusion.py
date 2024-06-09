from .GVP import *
from .layer import *
from .transformer import *

import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import torch_geometric
from torch_geometric.data import Batch
import fm


def geo_batch(batch):
    data_list = []
    # print(len(batch['z_t']))
    batch_size, length = batch['z_t'].shape[:2]

    for i in range(batch_size):
        data_list.append(torch_geometric.data.Data(
            z_t=batch['z_t'][i],
            seq=batch['seq'][i],  # num_res x 1
            coords=batch['coords'][i],  # num_res x 3 x 3
            node_s=batch['node_s'][i],  # num_res x num_conf x 4
            node_v=batch['node_v'][i],  # num_res x num_conf x 4 x 3
            edge_s=batch['edge_s'][i],  # num_edges x num_conf x 32
            edge_v=batch['edge_v'][i],  # num_edges x num_conf x 1 x 3
            edge_index=batch['edge_index'][i],  # 2 x num_edges
            mask=batch['mask'][i]  # num_res x 1
        ))

    return Batch.from_data_list(data_list), batch_size, length

class GVPTransCond(torch.nn.Module):
    '''
    GVP + Transformer model for RNA design

    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in encoder/decoder
    :param drop_rate: rate to use in all dropout layers
    :param out_dim: output dimension (4 bases)
    '''

    def __init__(self, node_in_dim = (1301, 4), node_h_dim = (512, 32), edge_in_dim = (32, 1), edge_h_dim = (128, 1),
                num_layers = 4, ):
        super().__init__()
        self.node_in_dim = (1301, 4)  # node_in_dim
        self.node_h_dim = (512, 32)  # node_h_dim
        self.edge_in_dim = (32, 1)  # edge_in_dim
        self.edge_h_dim = (128, 16)  # edge_h_dim
        self.num_layers = 6
        self.out_dim = 4
        self.time_cond = True
        self.dihedral_angle = True
        self.num_trans_layer = 8
        self.drop_struct = -1
        self.rna_encoder, _ = fm.pretrained.rna_fm_t12()
        self.rna_encoder.eval()

        # freeze rna_encoder, prevent it from training
        for para in self.rna_encoder.parameters():
            para.requires_grad = False
                
        drop_rate = .1
        activations = (F.relu, F.relu)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                         activations=activations, vector_gate=True,
                         drop_rate=drop_rate)
            for _ in range(self.num_layers))

        # Output
        self.W_out = GVP(self.node_h_dim, (self.node_h_dim[0], 0), activations=(None, None))

        # Transformer Layers
        self.seq_res = nn.Linear(1280, self.node_h_dim[0])  #### (RNA seq, cond) expend dimension for cross attention with protein node features

        # linear module for binding site prediction
        # self.pred_out = nn.Linear(self.node_h_dim[0], 1)
        # self.pred_out_sig = nn.Sigmoid()
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.node_h_dim[0],
            -1,
        )
        self.trans_layers = nn.ModuleList(
            TransformerCrossAttentionLayer(self.node_h_dim[0], self.node_h_dim[0] * 2)
            for _ in range(self.num_trans_layer))
        self.MLP_out = nn.Sequential(
            nn.Linear(self.node_h_dim[0], self.node_h_dim[0]),
            nn.ReLU(),
            nn.Linear(self.node_h_dim[0], self.out_dim)
        )

        # Time conditioning
        if self.time_cond:
            learned_sinu_pos_emb_dim = 16
            time_cond_dim = 1024
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
            self.to_time_hiddens = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, self.node_h_dim[0]),
            )

        # Dihedral angle
        if self.dihedral_angle:
            self.embed_dihedral = DihedralFeatures(self.node_h_dim[0])

    def struct_forward(self, batch, batch_size, length, **kwargs):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        if self.dihedral_angle:
            dihedral_feats = self.embed_dihedral(batch.coords).reshape_as(h_V[0])
            h_V = (h_V[0] + dihedral_feats, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        gvp_output = self.W_out(h_V)
        return gvp_output

    def forward(self, batch, cond_drop_prob=0., **kwargs):
        # construct extra node and edge features
        batch, batch_size, length = geo_batch(batch)
        print('geo', batch, batch_size, length )
        z_t = batch.z_t  # latent vector for RNA sequence
        cond_x = kwargs.get('cond_x', None)  # previous predicted RNA sequence
        if cond_x is None:
            cond_x = torch.zeros_like(batch.z_t)
        else:
            cond_x = cond_x.reshape_as(batch.z_t)

        
        z_t_encoded = self.rna_encoder(torch.argmax(z_t, dim = -1).unsqueeze(0), repr_layers = [12])["representations"][12]
        print('z_t_encoded', z_t_encoded)
        z_t_encoded = z_t_encoded.squeeze()
        # print('z_t_encoded', z_t_encoded.shape)

        cond_x_encoded = self.rna_encoder(torch.argmax(cond_x, dim = -1).unsqueeze(0), repr_layers = [12])["representations"][12]
        print('cond_x_encoded', cond_x_encoded)
        cond_x_encoded = cond_x_encoded.squeeze()
        # print('cond_x_encoded', cond_x_encoded.shape)

        init_seq = torch.cat([z_t_encoded, cond_x_encoded], -1)  # (len_RNA) x 1280
        print('init_seq', init_seq)
        # init_seq = init_seq.squeeze(0)
        print('init_seq', init_seq.shape)
        # print()
        if self.training:
            if self.drop_struct > 0 and random.random() < self.drop_struct:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                gvp_output = self.struct_forward(batch, batch_size, length, **kwargs)

        else:
            if cond_drop_prob == 0.:
                gvp_output = self.struct_forward(batch, batch_size, length, **kwargs)
            elif cond_drop_prob == 1.:
                gvp_output = torch.zeros(batch_size, length, self.node_h_dim[0], device=batch.z_t.device)
            else:
                raise ValueError(f'Invalid cond_drop_prob: {cond_drop_prob}')
        print(gvp_output)
        # trans_x = torch.cat([gvp_output, self.seq_res(init_seq.reshape(batch_size, length, -1))], dim=-1)
        # pred_out = self.pred_out_sig(self.pred_out(gvp_output))  # output for predciting RNA binding site of protein
        RNA_base_seq = self.seq_res(init_seq.reshape(batch_size, length, -1))  # (batch_size, length, hidden_dim)
        print("RNA_base_seq:\n", RNA_base_seq)
        if self.time_cond:
            noise_level = kwargs.get('noise_level')
            time_cond = self.to_time_hiddens(noise_level)  # [B, d_s]
            time_cond = time_cond.unsqueeze(1).repeat(1, length, 1)  # [B, length, d_s]
        else:
            time_cond = None

        # add position embedding
        seq_mask = torch.ones((batch_size, length), device=batch.z_t.device)
        
        pos_emb = self.embed_positions(seq_mask)
        trans_x = RNA_base_seq + pos_emb
        trans_x = trans_x.transpose(0, 1)

        # transformer layers
        print("trans_x:\n", trans_x)

        # for layer in self.trans_layers:
        #     trans_x = layer(trans_x, gvp_output.unsqueeze(1), gvp_output.unsqueeze(1), None,
        #                     cond=time_cond.transpose(0, 1))
        for idx, layer in enumerate(self.trans_layers):
            if idx == 0:
                trans_x = layer(trans_x, gvp_output.unsqueeze(1), gvp_output.unsqueeze(1), None,
                                cond=time_cond.transpose(0, 1))
            else:
                trans_x = layer(trans_x, trans_x, trans_x, None,
                                cond=time_cond.transpose(0, 1))
            print("trans_x:\n", trans_x)
        logits = self.MLP_out(trans_x.transpose(0, 1))
        print("logits:\n", logits)
        # logits = logits.reshape(batch_size, -1, self.out_dim)
        return logits


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x = rearrange(x, 'b -> b 1')
        x = x.unsqueeze(-1)
        # freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x = rearrange(x, 'b -> b 1')
        x = x.unsqueeze(-1)
        # freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
