import os.path,sys
import pandas as pd
import numpy as np
import pickle
from tdc.multi_pred import DTI
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Descriptors    
import torch

# For data split
# setting for generate the split
data_op =0
split_type = 3

#data source information
data_set=['davis','KIBA']
data = DTI(name = data_set[data_op])

if data_op ==0:
    print("davis")
    data.convert_to_log(form = "standard")
    
base_path = f'./data/{data_set[data_op]}/'
for split_type in range(4):
    # split type : 0 for random, 1 for S2 split:cold Drug, 2 for S3 split:cold target, 3 for S4 split:cold drug+taregt
    if split_type == 0:
        split = data.get_split(method='random')
        path = base_path+'split_s1/'
        if not os.path.exists(path):
            os.mkdir(path)

    elif split_type ==1:
        split = data.get_split(method = 'cold_split', column_name = 'Drug')
        path = base_path + 'split_s2/'
        if not os.path.exists(path):
            os.mkdir(path)

    elif split_type ==2:
        split = data.get_split(method = 'cold_split', column_name = 'Target')
        path = base_path + 'split_s3/'
        if not os.path.exists(path):
            os.mkdir(path)

    else:
        split = data.get_split(method = 'cold_split', column_name = ['Target','Drug'])
        path = base_path + 'split_s4/'
        if not os.path.exists(path):
            os.mkdir(path)

    # get the control of splited dataset 
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


    #out put the data to csv for training
    if data_op == 0:#choose davis
        thrshold = 5
    else:
        thrshold = 12.1

    new_train=[]
    for (i,j,k) in zip(train_drug_id,train_target_id,train_y):
        if k>=thrshold:
            new_train.append([re_drug_id_sets[i],re_target_id_sets[j],1])
        else:
            new_train.append([re_drug_id_sets[i],re_target_id_sets[j],0])
    new_data=pd.DataFrame(new_train)
    new_data.to_csv(path + 'train_data.csv',index=False,header=False)


    new_valid=[]
    for (i,j,k) in zip(valid_drug_id,valid_target_id,valid_y):
        if k>=thrshold:
            new_valid.append([re_drug_id_sets[i],re_target_id_sets[j],1])
        else:
            new_valid.append([re_drug_id_sets[i],re_target_id_sets[j],0])
    new_data=pd.DataFrame(new_valid)
    new_data.to_csv(path + 'valid_data.csv',index=False,header=False)


    new_test=[]
    for (i,j,k) in zip(test_drug_id,test_target_id,test_y):
        if k>=thrshold:
                new_test.append([re_drug_id_sets[i],re_target_id_sets[j],1])
        else:
                new_test.append([re_drug_id_sets[i],re_target_id_sets[j],0])
        # new_test.append([re_drug_id_sets[i], re_target_id_sets[j], k])
    new_data=pd.DataFrame(new_test)
    new_data.to_csv(path + 'test_data.csv',index=False,header=False)


# For feature construction

