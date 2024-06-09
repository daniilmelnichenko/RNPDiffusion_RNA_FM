import shutil

import numpy as np
import torch.nn.functional as F
import torch
import torch_geometric
import torch_cluster
import os
from Bio.PDB import PDBParser

RNA_NUM_TO_LETTER = np.array(['A', 'G', 'C', 'U'])
RNA_LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3}

PROTEIN_NUM_TO_LETTER = np.array(
    ['G', 'A', 'V', 'I', 'L', 'F', 'P', 'M', 'W', 'C', 'S', 'T', 'N', 'Q', 'Y', 'H', 'D', 'E', 'K', 'R', 'X'])
PROTEIN_LETTER_TO_NUM = {'G': 0, 'A': 1, 'V': 2, 'I': 3, 'L': 4, 'F': 5, 'P': 6, 'M': 7, 'W': 8, 'C': 9, 'S': 10,
                         'T': 11, 'N': 12, 'Q': 13, 'Y': 14, 'H': 15, 'D': 16, 'E': 17, 'K': 18, 'R': 19, 'X': 20, }

res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'TRP': 'W',
            'CYS': 'C', "MSE": "M",
            'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
            'ARG': 'R', "UNK": "X"}

p = PDBParser(QUIET=True)


def get_posenc(edge_index, num_posenc=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_posenc = num_posenc
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )

    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def get_orientations(X):
    # X : num_conf x num_res x 3
    forward = normalize(X[:, 1:] - X[:, :-1])
    backward = normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_orientations_single(X):
    # X : num_res x 3
    forward = normalize(X[1:] - X[:-1])
    backward = normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(X):
    # X : num_conf x num_res x 3 x 3
    p, origin, n = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    n, p = normalize(n - origin), normalize(p - origin)
    return torch.cat([n.unsqueeze_(-2), p.unsqueeze_(-2)], -2)


def get_sidechains_single(X):
    # X : num_res x 3 x 3
    p, origin, n = X[:, 0], X[:, 1], X[:, 2]
    n, p = normalize(n - origin), normalize(p - origin)
    return torch.cat([n.unsqueeze_(-2), p.unsqueeze_(-2)], -2)


def normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    TODO switch to DimeNet RBFs
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


@torch.no_grad()
def construct_data_single(coords, seq=None, mask=None, num_posenc=16, num_rbf=16, knn_num=10):
    coords = torch.as_tensor(coords, dtype=torch.float32)  # num_res x 3 x 3
    seq_str = seq
    # seq is np.array/string, convert to torch.tensor
    if isinstance(seq, np.ndarray):
        seq = torch.as_tensor(seq, dtype=torch.long)
    seq = torch.as_tensor([PROTEIN_LETTER_TO_NUM[residue] for residue in seq], dtype=torch.long)

    # Compute features
    # node positions: num_res x 3
    coord_C = coords[:, 1].clone()
    # Construct merged edge index
    edge_index = torch_cluster.knn_graph(coord_C, k=knn_num)
    edge_index = torch_geometric.utils.coalesce(edge_index)

    # Node attributes: num_res x 2 x 3, each
    orientations = get_orientations_single(coord_C)
    sidechains = get_sidechains_single(coords)

    # Edge displacement vectors: num_edges x  3
    edge_vectors = coord_C[edge_index[0]] - coord_C[edge_index[1]]

    # Edge RBF features: num_edges x num_rbf
    edge_rbf = rbf(edge_vectors.norm(dim=-1), D_count=num_rbf)
    # Edge positional encodings: num_edges x num_posenc
    edge_posenc = get_posenc(edge_index, num_posenc)

    node_s = (seq.unsqueeze(-1) == torch.arange(21).unsqueeze(0)).float()
    node_v = torch.cat([orientations, sidechains], dim=-2)
    edge_s = torch.cat([edge_rbf, edge_posenc], dim=-1)
    edge_v = normalize(edge_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(
        torch.nan_to_num,
        (node_s, node_v, edge_s, edge_v)
    )

    # add mask for invalid residues
    if mask is None:
        mask = coords.sum(dim=(2, 3)) == 0.
    mask = torch.tensor(mask)

    return {'seq': seq,
            'seq_str': seq_str,
            'coords': coords,
            'node_s': node_s,
            'node_v': node_v,
            'edge_s': edge_s,
            'edge_v': edge_v,
            'edge_index': edge_index,
            'mask': mask}


def parse_pdb_direct(pdb_structure, temp_save_path=None, chain=None):
    # structure = p.get_structure("temp", pdb_path)
    xyz = {}
    seq = ""
    resn_lst = []

    for residue in pdb_structure.get_residues():

        # residue name
        resn = residue.get_resname()
        # residue number
        res_id = residue.get_id()[1]

        if resn not in res_dict.keys():
            continue

        # get sequence
        seq += res_dict[resn]
        resn_lst.append(res_id)

        xyz[res_id] = {}
        xyz[res_id][resn] = {}

        # get coord for all atom
        for atom in residue.get_atoms():
            xyz[res_id][resn][atom.get_name()] = atom.get_coord()

    # convert to numpy arrays, fill in missing values
    seq_, xyz_, mask = [], [], []
    for resn in resn_lst:
        ## xyz coordinates [L, 3, 3]
        coords_tmp = np.zeros((3, 3))
        if resn in xyz:
            for k in sorted(xyz[resn]):
                if "N" in xyz[resn][k]: coords_tmp[0] = xyz[resn][k]["N"]
                if "CA" in xyz[resn][k]: coords_tmp[1] = xyz[resn][k]["CA"]
                if "C" in xyz[resn][k]: coords_tmp[2] = xyz[resn][k]["C"]

        xyz_.append(coords_tmp)
        mask.append(np.all(coords_tmp != 0.))

    assert len(seq) == len(xyz_), f"seq : {len(seq)} xyz : {len(xyz)}"
    xyz_ = np.array(xyz_, dtype=np.float32)
    mask = np.array(mask)

    return seq, xyz_, mask


def PDBtoData(pdb_structure, num_posenc, num_rbf, knn_num):
    seq, coords, mask = parse_pdb_direct(pdb_structure)
    return construct_data_single(
        coords,
        seq,
        mask,
        num_posenc=num_posenc,
        num_rbf=num_rbf,
        knn_num=knn_num,
    )

def make_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)


if __name__ == "__main__":
    path = "/home/joohyun/Desktop/RiboDiffusion/data/pdb"  # fill in your path to pdb folders
    pdb_code_list = os.listdir(path)

    complex_dir = os.path.join(path, pdb_code_list[2])
    complex = os.listdir(complex_dir)
    protein = [i for i in complex if i.startswith("protein")]

    print("protein:", protein)

    protein_pdbs = [os.path.join(complex_dir, i) for i in protein]

    # parse_pdb_direct(protein_pdbs[0])
    # PDBtoData(protein_pdbs[0], 16, 16, 16)