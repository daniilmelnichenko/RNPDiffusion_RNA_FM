from utils_data import *
from Bio.PDB import PDBParser
from tqdm.auto import tqdm
import torch

device = 'cpu' if torch.cuda.is_available() else 'cpu'

p = PDBParser(QUIET=True)
dataset_dir = "C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\temp_dataset_txt"
NA_dataset = [
    'RNA-662_Train.txt',
    'RNA-156_Test.txt'
]


def read_label(line):
    chain_data = line.split("\t")[0]
    label_data = line.split("\t")[1].strip().split(":")
    pdb_code = line[:4]
    target_chain = chain_data.split(":")[1]
    RNA_chain = chain_data.split(":")[-1].split("_")
    label = " ".join([i[4:] for i in line.split("\t")[1].split(":")]).strip()
    label_lst = list(set(label.split(" ")))
    pos_label = [i[1:] for i in label_lst]
    pos_res = [i[0] for i in label_lst]

    if target_chain.islower():
        protein_chain = target_chain + target_chain
    else:
        protein_chain = target_chain

    pdb_name = pdb_code + "_" + protein_chain

    return pdb_name, protein_chain, target_chain, RNA_chain, pos_label, pos_res, label_data


import pickle
import esm
esm_model_650M, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model_650M.to(device).eval()
batch_converter = alphabet.get_batch_converter()

for data_txt in NA_dataset:
    ligand, split = data_txt.split("-")[0], data_txt.split("-")[1].split("_")[1][:-4]
    print(ligand, split, data_txt) # split = Train or Test, ligand is always RNA
    path = f"C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\pkl\{split.lower()}"

    # read each line of dataset_txt
    with open(os.path.join(dataset_dir, data_txt), 'r') as f:
        text = f.readlines()
        len_protein = len(text)

    for line in tqdm(text, position=0, desc=f"{ligand}-{split}"):

        pdb_name, protein_chain, target_chain, RNA_chain, pos_label, pos_res, label_data = read_label(line) # get info of protein
        structure = p.get_structure(pdb_name, path + pdb_name.split("_")[0] + ".pdb")[0] # get structure
        RNA_dict = {}

        for idx, RNA in enumerate(RNA_chain): # for each RNA chain that binds to protein chain,
            target = {}
            RNA_structure = structure[RNA]
            RNA_seq = "".join(
                [i.get_resname() for i in RNA_structure.get_residues() if i.get_resname() in ["A", "U", "G", "C"]])
            target["RNA_seq_str"] = RNA_seq
            target["RNA_seq"] = torch.tensor([RNA_LETTER_TO_NUM[base] for base in RNA_seq], dtype=torch.long)
            target["Pos_res"] = [i[1:] for i in label_data[0].split(" ")[1:]]
            target['mask'] = torch.tensor([True] * len(RNA_seq))
            RNA_dict[RNA] = target

        # get protein feature from structure using method in utils_data.py
        protein_dict = PDBtoData(structure[target_chain], 16, 16, 16)

        # get ESM-2 feature (protein language model)
        batch_labels, batch_strs, batch_tokens = batch_converter([(pdb_name, protein_dict["seq_str"])])
        with torch.no_grad():
            results = esm_model_650M(batch_tokens, repr_layers=[33], return_contacts=True)
        esm2_650M_feature = results["representations"][33][0][1:-1]

        # concat node scalar feature, which is ESM feature and one-hot-encoding of a.a.
        protein_dict["node_s"] = torch.cat([protein_dict["node_s"], esm2_650M_feature], dim=1)
        protein_dict["res_id"] = torch.tensor([i.get_id()[1] for i in structure[target_chain].get_residues()])

        for key in ['coords', 'node_s', 'node_v', 'edge_s', 'edge_v', 'edge_index']:
            protein_dict[key] = protein_dict[key].unsqueeze(0)

        # make dir for pkl
        make_dir(path)
        make_dir(os.path.join(path, pdb_name))

        # save protein and RNA dictionary in pkl format
        with open(os.path.join(path, pdb_name, "protein.pkl"), 'wb') as f:
            pickle.dump(protein_dict, f)
        with open(os.path.join(path, pdb_name, "RNA.pkl"),  'wb') as f:
            pickle.dump(RNA_dict, f)