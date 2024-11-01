from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import SimilarityMaps
import torch
from tdc.multi_pred import DTI
import numpy as np


data = DTI(name = 'KIBA')
split = data.get_split()
split.keys()


train_drug_id=np.array(split['train']['Drug_ID']).tolist()
train_target_id=np.array(split['train']['Target_ID']).tolist()
train_drug=np.array(split['train']['Drug']).tolist()
train_target=np.array(split['train']['Target']).tolist()
train_y=np.array(split['train']['Y'])

valid_drug_id=np.array(split['valid']['Drug_ID']).tolist()
valid_target_id=np.array(split['valid']['Target_ID']).tolist()
valid_drug=np.array(split['valid']['Drug']).tolist()
valid_target=np.array(split['valid']['Target']).tolist()
valid_y=np.array(split['valid']['Y'])

test_drug_id=np.array(split['test']['Drug_ID']).tolist()
test_target_id=np.array(split['test']['Target_ID']).tolist()
test_drug=np.array(split['test']['Drug']).tolist()
test_target=np.array(split['test']['Target']).tolist()
test_y=np.array(split['test']['Y'])

drug_id_sets={}
target_id_sets={}
re_drug_id_sets={}
re_target_id_sets={}
target_id2text={}
drug_id2mol={}
for ids,i in enumerate(train_drug_id):
    if i not in re_drug_id_sets:
        idx=len(drug_id_sets)
        drug_id_sets[idx]=i
        re_drug_id_sets[i]=idx
        drug_id2mol[idx]=train_drug[ids]
for ids,i in enumerate(valid_drug_id):
    if i not in re_drug_id_sets:
        idx=len(drug_id_sets)
        drug_id_sets[idx]=i
        re_drug_id_sets[i]=idx
        drug_id2mol[idx]=valid_drug[ids]
for ids,i in enumerate(test_drug_id):
    if i not in re_drug_id_sets:
        idx=len(drug_id_sets)
        drug_id_sets[idx]=i
        re_drug_id_sets[i]=idx
        drug_id2mol[idx]=test_drug[ids]

for ids,i in enumerate(train_target_id):
    if i not in re_target_id_sets:
        idx=len(target_id_sets)
        target_id_sets[idx]=i
        re_target_id_sets[i]=idx
        target_id2text[idx]=train_target[ids]
for ids,i in enumerate(valid_target_id):
    if i not in re_target_id_sets:
        idx=len(target_id_sets)
        target_id_sets[idx]=i
        re_target_id_sets[i]=idx
        target_id2text[idx]=valid_target[ids]
for ids,i in enumerate(test_target_id):
    if i not in re_target_id_sets:
        idx=len(target_id_sets)
        target_id_sets[idx]=i
        re_target_id_sets[i]=idx
        target_id2text[idx]=test_target[ids]
print(len(drug_id_sets),len(target_id_sets))

# drug_id_sets

class RDKit_2D:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles
        
    def compute_2Drdkit(self, name):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in range(len(self.mols)):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc,columns=header)
        df.insert(loc=0, column='smiles', value=self.smiles)
        df.to_csv(name[:-4]+'_RDKit_2D.csv', index=False)

from rdkit.Chem import Descriptors    # make sure to import it if you haven't done so
descriptors_list = [x[0] for x in Descriptors._descList]
# print(descriptors_list)

# Chosen drug features

[x[0] for x in Descriptors._descList[110:120]]

def check_smiles(smiles):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList[110:120]])
    mol =  Chem.MolFromSmiles(smiles)
    x = calc.CalcDescriptors(mol)
    return x

for idx in drug_id2mol.keys():
    drug_id2mol[idx]=check_smiles(drug_id2mol[idx])

# original drug features 

drug_id2mol

# Reconstruct feature 

newdrugs = []
for i in range(10):   # number of features
    newdrugs.append([])

for drug in drug_id2mol.values():
    for i in range(0,len(drug)):
        if drug[i] not in newdrugs[i]:
            newdrugs[i].append(drug[i])
            newdrugs[i].sort()

# Drug Feature_dict build 

feature_dict = {}

for i in range(len(newdrugs)):
    for j in range(len(newdrugs[i])):
        idx=len(feature_dict) +len(drug_id_sets)+len(target_id_sets)
        feature_dict["feature" + str(i)+ "_"+ str(newdrugs[i][j])] = idx  

# Drug attributes build

new_drugs = {}
for (drug_id,drug_value) in drug_id2mol.items():

    drug_feature = []
    for i in range(0,len(drug_value)):
        drug_feature.append(feature_dict[str("feature" + str(i)+ "_"+ str(drug_value[i]))])
    new_drugs[drug_id] = drug_feature

# drug_dict construction

import pickle

drug_ids=set(drug_id2mol.keys())
drug_dict={}
for i in drug_id2mol.keys():
    drug_dict[i]={}
    drug_dict[i]['name']=i
    drug_dict[i]['attribute']=new_drugs[i]

with open(f'./data/KIBA/drug_dict.pkl','wb') as f:
    pickle.dump(drug_dict,f)
    
# Protein Feature build 

import peptides
import pandas as pd
pd.options.display.max_rows = 4000

def check_target(fasta):
    pro_descrip = []
    pro_descrip.append(int(round(peptides.Peptide(fasta).aliphatic_index()))) # 肽的脂肪族指数
    pro_descrip.append(int(round(peptides.Peptide(fasta).boman()))) #（潜在的肽相互作用）指数,所有残基的溶解度值进行平均计算得出的指数.大于 2.48的值表明蛋白质具有高结合潜力。
    pro_descrip.append(int(round(peptides.Peptide(fasta).hydrophobicity(scale="Aboderin"))))  # 计算蛋白质序列的疏水性指数。
    pro_descrip.append(int(round(peptides.Peptide(fasta).instability_index())))  # 计算蛋白质序列的不稳定性指数。
    pro_descrip.append(int(round(peptides.Peptide(fasta).isoelectric_point(pKscale="Murray"))))  # 计算蛋白质序列的等电点。
    pro_descrip.append(peptides.Peptide(fasta).structural_class("Nakashima", distance="mahalanobis")) # 计算修饰肽的质量差异。
    return pro_descrip

for idx in target_id2text.keys():
    target_id2text[idx]=check_target(target_id2text[idx])
target_id2text

# Reconstruct protein feature 

newtargets = []
for i in range(6):   # number of features
    newtargets.append([])

for target in target_id2text.values():
    for i in range(0,len(target)):
        if target[i] not in newtargets[i]:
            newtargets[i].append(target[i])
            newtargets[i].sort()

# Protein Feature_dict build 


for i in range(len(newtargets)):
    for j in range(len(newtargets[i])):
        idx=len(feature_dict) +len(drug_id_sets)+len(target_id_sets)
        feature_dict["protein_feature" + str(i)+ "_"+ str(newtargets[i][j])] = idx  

# Target attributes build

new_targets = {}
for (target_id,target_value) in target_id2text.items():

    target_feature = []
    for i in range(0,len(target_value)):
        target_feature.append(feature_dict[str("protein_feature" + str(i)+ "_"+ str(target_value[i]))])
    new_targets[target_id] = target_feature

# target_dict construction

drug_ids=set(drug_id2mol.keys())  
target_ids=set(target_id2text.keys())  

lens=len(drug_ids)
new_target_id=[]
for t in list(target_ids):
    new_target_id.append(t+lens)
    
target_dict={}
for i in target_id2text.keys():
    target_dict[i]={}
    target_dict[i]['title']=i+lens
    target_dict[i]['attribute']=new_targets[i]
    
with open(f'./data/KIBA/target_dict.pkl','wb') as f:
    pickle.dump(target_dict,f)


for u in drug_dict.keys():
    feature_dict[f'user{u}'] = u
for i in target_dict.keys():
    feature_dict[f'item{i}'] = i+len(drug_ids)
with open(f'./data/KIBA/feature_dict.pkl','wb') as f:
    pickle.dump(feature_dict,f)



