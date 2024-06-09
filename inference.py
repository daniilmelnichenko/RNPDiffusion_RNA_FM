from data_prepare.utils_data import *
from modelRF.diffusionV3 import *
from util import *
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from diffusion.noise_schedule import NoiseScheduleVP
import ml_collections
import pickle
import os
import matplotlib.pyplot as plt


from RNAFMDecoder.decoder import fnafmDecoder, tensor_to_sequence

# get configuration for optimizer
config = ml_collections.ConfigDict()
config.optim = optim = ml_collections.ConfigDict()
optim.weight_decay = 0
optim.optimizer = 'AdamW'
optim.lr = 2e-4
optim.beta1 = 0.9
optim.eps = 1e-8
optim.warmup = 20000
optim.grad_clip = 20.
optim.disable_grad_log = True

model = GVPTransCond().cuda()
optimizer = get_optimizer(config, model.parameters())
noise_scheduler = NoiseScheduleVP('linear', continuous_beta_0 = 0.1,
                                  continuous_beta_1=20.0)
criterion = nn.MSELoss(reduction="sum").cuda()

eps = 1e-3
time_steps = torch.linspace(1, eps, 200)

tot_tr_loss = []
# tot_eval_loss = []

train_data_dir = "./data/pkl/train"
test_data_dir = "./data/pkl/test"
train_pdb_lst = os.listdir(train_data_dir)
test_pdb_lst = os.listdir(test_data_dir)
selected_pdbs = random.sample(test_pdb_lst, 20)
save_dir = "./saved_modelRF_0609_3"

rnaFMdecoder = fnafmDecoder()
rnaFMdecoder.to("cuda")
rnaFMdecoder.load_state_dict(torch.load('./RNAFMDecoder/fnafmDecoder_weights.pth'))
rnaFMdecoder.eval()
for para in rnaFMdecoder.parameters():
    para.requires_grad = False

model.load_state_dict(torch.load('./saved_modelRF_0609_3/model_epoch_50.pt'))


sampled_seq_lst = []
true_seq = []
protein_seq = []
bs = 1

for data in os.listdir(test_data_dir):
    with torch.no_grad():
        with open(os.path.join(test_data_dir, data, "protein.pkl"), "rb") as f:
            protein = pickle.load(f)
            for key in ['coords', 'node_s', 'node_v', 'edge_s', 'edge_v', 'edge_index']:
                protein[key] = protein[key].cuda()

        with open(os.path.join(test_data_dir, data, "RNA.pkl"), "rb") as f:
            RNA = pickle.load(f)
        for rna_key in RNA.keys():
            if len(RNA[rna_key]["RNA_seq_str"]) > 281:
                continue
            elif len(RNA[rna_key]["RNA_seq_str"]) < 20:
                continue
            protein_seq.append(protein["seq_str"])
            true_seq.append(RNA[rna_key]["RNA_seq_str"])

            RNA[rna_key]["seq_one_hot"] = (
                        RNA[rna_key]["RNA_seq"].unsqueeze(-1) == torch.arange(4).unsqueeze(0)).float().unsqueeze(
                0).cuda()
            model.eval()
            sampled_seq = []
            tot_eval_loss = 0
            for _ in range(10):
                eval_time_steps = torch.linspace(1, eps, 200).cuda()

                eval_alpha_t, _ = noise_scheduler.marginal_prob(eval_time_steps[0])
                eval_sigma_t, _ = noise_scheduler.marginal_prob(eval_time_steps[1])

                eval_t_array = eval_time_steps
                eval_s_array = torch.cat([eval_time_steps[1:], torch.zeros(1).cuda()])


                _, _, RNA_token = model.alphabet_converter([(f"{rna_key}",f"{RNA[rna_key]['RNA_seq_str']}")])
                RNA_token = RNA_token.cuda()
                token_embeddings = model.rna_encoder(RNA_token, repr_layers=[12])
                token_embeddings = token_embeddings["representations"][12]
                token_embeddings = token_embeddings[:, 1:-1, :]

                eval_z_T = torch.randn_like(token_embeddings).cuda()
                eval_batch = protein
                eval_batch['z_t'] = eval_x =  eval_z_T
                eval_cond_x = torch.zeros_like(eval_z_T).cuda()

                for eval_ts in range(len(eval_time_steps)):
                    eval_t = eval_t_array[eval_ts]
                    eval_s = eval_s_array[eval_ts]

                    eval_alpha_t, eval_sigma_t = noise_scheduler.marginal_prob(eval_t)
                    eval_alpha_s, eval_sigma_s = noise_scheduler.marginal_prob(eval_s)

                    alpha_t_given_s = eval_alpha_t / eval_alpha_s
                    sigma2_t_given_s = eval_sigma_t ** 2 - alpha_t_given_s ** 2 * eval_sigma_s ** 2
                    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
                    sigma = sigma_t_given_s * eval_sigma_s / eval_sigma_t

                    eval_noise_level = torch.log(eval_alpha_t ** 2 / eval_sigma_t ** 2).cuda()

                    eval_pred_t = model(eval_batch, time=eval_t.unsqueeze(0), noise_level=eval_noise_level.unsqueeze(0),
                                        cond_x=eval_cond_x)
                    eval_cond_x = eval_pred_t.detach().clone()
                    # print(eval_pred_t)
                    if eval_pred_t.shape != eval_x.shape:
                        eval_pred_t = eval_pred_t.unsqueeze(-2)
                    # print(eval_x.shape)
                    # print(eval_x)
                    eval_x_mean = expand_dims((alpha_t_given_s * eval_sigma_s ** 2 / eval_sigma_t ** 2).repeat(bs),
                                                eval_x.dim()) * eval_x \
                                    + expand_dims((eval_alpha_s * sigma2_t_given_s / eval_sigma_t ** 2).repeat(bs),
                                                eval_pred_t.dim()) * eval_pred_t

                    eval_batch['z_t'] = eval_x = (eval_x_mean + expand_dims(sigma.repeat(bs), eval_x_mean.dim()) \
                                                    * torch.randn_like(eval_x_mean, device=eval_x.device))
                                    
                tot_eval_loss += torch.sum(torch.square(eval_x_mean.detach() - token_embeddings.detach())).item()
                sampled_seq.append(tensor_to_sequence(rnaFMdecoder(eval_x_mean)))
            sampled_seq_lst.append(sampled_seq)
            rr_list= []
            for idx in range(10):
                pred = sampled_seq[idx]
                total_correct = 0
                total_correct = sum([1 if pred[i] == true_seq[-1][i] else 0 for i in range(len(true_seq[-1]))])
                rr_list.append(round(total_correct / len(true_seq[-1]), 4))
            
            rr_list = np.array(rr_list)
            sorted_idx_lst = np.argsort(rr_list)[::-1]


            print(data)
            print(rna_key)
            print("loss         :", tot_eval_loss)
            print("protein_seq  : ", protein_seq[-1])
            print("true         : ", RNA[rna_key]["RNA_seq_str"])
            print("true recoded : ", tensor_to_sequence(rnaFMdecoder(token_embeddings)))

            for i in sorted_idx_lst[:3]:
                print(f"pred {rr_list[i]}  : ", sampled_seq[i])