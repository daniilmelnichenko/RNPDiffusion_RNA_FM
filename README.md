# RNPDiffusion: Inverse Problem for RNA Sequence Prediction in Protein-RNA Complexes through Diffusion Process



<img src="/figure/model.png" alt="model" style="zoom:20%;" />

Experimental validation of ribonucleoprotein (RNP) complex structures demands extensive resources. This study proposes RNPDiffusion, a novel diffusion-based method for predicting RNA sequences that bind to well-characterized proteins, leveraging latent space embeddings from the RNA-FM model. This approach offers a promising alternative to traditional methods, accelerating RNA-based diagnostics and therapeutics development. RNPDiffusion demonstrates significant alignment with known sequences and accurate protein interactions, even with a lower recovery rate, indicating its potential for precise RNA sequence design in RNP complexes.



## Results



<img src="/figure/inference2.png" alt="inference2" style="zoom:33%;" />

<img src="/figure/inference1.png" alt="inference1" style="zoom:33%;" />



## Python packages

* python=3.10.2
* tqdm=4.66.4
* torch=2.3.1 (Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks)
* torch-geometric=2.5.3 and pyg-lib=0.4.0 (For geometric neural networks
* biopython=1.78 (To parse PDB files)
* rna-fm=0.2.2 (For RNA encoder)
* ml-collections=0.1.1 (For configuration)

We also provided rnpdiffuse.yml for our working environment.



## Specific usage

### 1. Download and install packages, or use rnpdiffuse.yml to create environment.



### 2. Download .pkl files for training/inference

a. All pdb files and pkl files are uploaded to (https://drive.google.com/drive/folders/1Wlz0FgugTbF4TGDsiMlqRgfqjNMrgxxK?usp=sharing). Download the Data.zip, move it to working directory and then unzip.

b. You don't have to process any pdbs, all .pkl data is off-the-shelf.



### 3. Download weights for RNPDiffusion

a. Download pretrained model's weight is also in same google drive link. Move 'saved_modelRF_0609_3' folder into your working directory.



### 4. Train from scratch

```python
python main.py
```



### 5. Inference on test dataset

```python
python inference.py
```

* During inference each protein sequence will produce 10 de-novo RNA sequence. We only print out 3 RNA sequence with highest recovery rate.



