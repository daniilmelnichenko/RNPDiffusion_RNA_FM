import os
import urllib
from tqdm.auto import tqdm

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):

    try:
        pdbfn = pdbcode + ".pdb"
        url = downloadurl + pdbfn
        outfnm = os.path.join(datadir, pdbfn)
        urllib.request.urlretrieve(url, outfnm)
        return True
    except:
        try:
            pdbfn = pdbcode + ".cif"
            url = downloadurl + pdbfn
            outfnm = os.path.join(datadir, pdbfn)
            urllib.request.urlretrieve(url, outfnm)
            return True
        except:
            return False

with open("C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\test_cdhit_2d_50.fasta", "r") as f:
    # dataset that has asimilar structure to test pdbs
    fasta = f.readlines()
num_RNA = 0
RNA_name_lst = []
for line in fasta[::2]:
    if line.strip().split(":")[1] == "RNA":
        num_RNA += 1
        RNA_name_lst.append(line.strip().split(":")[0][1:])

# txt file for RNA binding protein datset
biolip_txt = "C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\BioLiP_RNA.txt"

with open(biolip_txt, "r") as f:
    text = f.readlines()
data_line = []
for line in text:
    if line == "\n":
        break
    pdb_chain = "_".join(line.strip().split("\t")[:2])
    pdb_id, protein_chain, res, _, _, RNA_chain, _, pdb_bs_resnum = line.strip().split("\t")[:8]
    if pdb_chain in RNA_name_lst:
        data_line.append([pdb_id, protein_chain, res, RNA_chain, pdb_bs_resnum])

# save RNP data to txt file
with open("C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\RNA_fasta.txt", "w") as f:
    for data in data_line:
        f.write("\t".join(data) + "\n")

# dir for saveing pdbs
download_dir = "C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\complex"

# download pdb (if not, cif)
with open("C:\Users\User\Desktop\temp\RNPdiffusion\Cho\data\RNA_fasta.txt", "r") as f:
    text = f.readlines()
    pdb_id = []
    for line in text:
        pdb_id.append(line.strip().split("\t")[0])
    for pdb in tqdm(list(set(pdb_id))):
        check_pdb = download_pdb(pdb, download_dir)
        if check_pdb == False:
            print(pdb, "not downloaded")
# in total, we have 817 complex pdb and cif combined, and total 2227 complex (Protein + RNA)
# which means we have average 3 RNA binding to each protein