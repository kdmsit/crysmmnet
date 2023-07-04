#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time
import torch
import pickle as pkl
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
import numpy as np
import pickle as pkl
from transformers import AutoTokenizer
from transformers import AutoModel
from tokenizers.normalizers import BertNormalizer

csv.field_size_limit(sys.maxsize)


"""**VoCab Mapping and Normalizer**"""

f = open('vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

"""**Custom Dataset**"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert',model_max_length=512)

parser = argparse.ArgumentParser(description="Atomistic Line Graph Neural Network")
parser.add_argument("--root_dir",default="../data_1000/",help="Folder with id_props.csv, structure files")
parser.add_argument("--config_name",default="config.json",help="Name of the config file")
parser.add_argument("--file_format", default="cif", help="poscar/cif/xyz/pdb file format.")
parser.add_argument("--keep_data_order",default=False,help="Whether to randomly shuffle samples, True/False",)
parser.add_argument("--batch_size", default=64, help="Batch size, generally 64")
parser.add_argument("--epochs", default=10, help="Number of epochs, generally 300")
parser.add_argument("--output_dir",default="./",help="Folder to save outputs",)
parser.add_argument("--dataset",default="toy", help="Name of the Dataset") #Jarvis,MP,toy
parser.add_argument("--property",default="bulk", help="Name of the Property")
parser.add_argument("--n_train", type=int, default=None, help="Train data size")
parser.add_argument("--n_val", type=int, default=None, help="Val data size")
parser.add_argument("--n_test", type=int, default=None, help="Test data size")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Train Ratio")
parser.add_argument("--val_ratio", type=float, default=0.1, help="Val Ratio")
parser.add_argument("--test_ratio", type=float, default=0.1, help="Test Ratio")
parser.add_argument("--resume", type=int, default=0, help="Test Ratio")


def train_for_folder(root_dir="examples/sample_data",config_name="config.json",batch_size=None,epochs=None,output_dir=None,args=None):
    """Train for a folder."""
    if args.dataset == 'MP':
        if args.property == 'bulk' or args.property == 'shear':
            cif_dir = os.path.join(root_dir, 'mp_2018_small/cif/')
            id_prop_dat = os.path.join(root_dir, 'mp_2018_small/description.csv')
        elif args.property == 'formation_energy' or args.property == 'band_gap':
            cif_dir = os.path.join(root_dir, 'mp_2018_new/')
            id_prop_dat = os.path.join(root_dir, 'mp_2018_new/mat_text.csv')
    elif args.dataset == 'Jarvis':
        if args.property == 'mbj_bandgap':
            cif_dir = os.path.join(root_dir, 'jarvis/mbj_bandgap/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/mbj_bandgap/description.csv')
        elif args.property == 'bulk_modulus_kv':
            cif_dir = os.path.join(root_dir, 'jarvis/bulk_modulus_kv/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/bulk_modulus_kv/description.csv')
        elif args.property == 'shear_modulus_gv':
            cif_dir = os.path.join(root_dir, 'jarvis/shear_modulus_gv/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/shear_modulus_gv/description.csv')
        elif args.property == 'fe':
            cif_dir = os.path.join(root_dir, 'jarvis/formation_energy_peratom/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/formation_energy_peratom/description.csv')
        elif args.property == 'opt_bandgap':
            cif_dir = os.path.join(root_dir, 'jarvis/optb88vdw_bandgap/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/optb88vdw_bandgap/description.csv')
        elif args.property == 'total_energy':
            cif_dir = os.path.join(root_dir, 'jarvis/optb88vdw_total_energy/cif/')
            id_prop_dat = os.path.join(root_dir, 'jarvis/optb88vdw_total_energy/description.csv')
    elif args.dataset == 'toy':
        cif_dir = os.path.join('../../data_1000/cif/')
        id_prop_dat = os.path.join('../../data_1000/description.csv')

    config = loadjson(config_name)

    config['keep_data_order'] = False
    if output_dir is not None:
        config['output_dir'] = output_dir+args.property+'/'
    if batch_size is not None:
        config['batch_size'] = int(batch_size)
    if epochs is not None:
        config['epochs'] = int(epochs)

    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)




    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]
    print(root_dir)
    print(config.batch_size )
    dataset = []
    print('DataSize:',len(data))

    # count=0
    for j in tqdm(range(len(data))):  # 69239
    # for j in tqdm(range(300)):  # 69239
        if args.dataset == 'MP':
            if args.property == 'formation_energy':
                id, composition, target, _, crys_desc_full, _ = data[j]
            elif args.property == 'band_gap':
                id, composition, _, target, crys_desc_full, _ = data[j]
            elif args.property == 'shear':
                id, composition, target, _, crys_desc_full, _ = data[j]
            elif args.property == 'bulk':
                id, composition, _, target, crys_desc_full, _ = data[j]
        elif args.dataset == 'Jarvis':
            id, composition, target, crys_desc_full, _= data[j]
        elif args.dataset == 'toy':
            id, composition, target, crys_desc_full,_ = data[j]
        info = {}
        file_name = id
        file_path = os.path.join(cif_dir, file_name + '.cif')
        atoms = Atoms.from_cif(file_path)
        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name
        info["text"] = crys_desc_full

        info["target"] = float(target)
        if args.dataset == 'MP':
            if args.property in ['shear','bulk']:
                info["target"] = np.log10(float(target))
        dataset.append(info)
        # norm_sents = normalize(crys_desc_full)
        # tokenized_sents = tokenizer.decode(tokenizer.encode(norm_sents))
        # if len(tokenized_sents) <512:
        #     dataset.append(info)
        #     count =count+1
        #     if count==1000: break;



    (train_loader,val_loader,test_loader,prepare_batch,) = get_train_val_loaders(dataset_array=dataset,target=config.target,n_train=args.n_train,n_val=args.n_val,n_test=args.n_test,
                                                                                 train_ratio=args.train_ratio,val_ratio=args.val_ratio,test_ratio=args.test_ratio,batch_size=config.batch_size,
                                                                                 atom_features=config.atom_features,neighbor_strategy=config.neighbor_strategy,id_tag=config.id_tag,pin_memory=config.pin_memory,
                                                                                 workers=config.num_workers,save_dataloader=config.save_dataloader,use_canonize=config.use_canonize,filename=config.filename,
                                                                                 cutoff=config.cutoff,max_neighbors=config.max_neighbors,target_multiplication_factor=config.target_multiplication_factor,
                                                                                 standard_scalar_and_pca=config.standard_scalar_and_pca,keep_data_order=config.keep_data_order,output_dir=config.output_dir)
    t1 = time.time()
    train_dgl(config,train_val_test_loaders=[train_loader,val_loader,test_loader,prepare_batch],resume = args.resume)
    t2 = time.time()
    print("Time taken (s):", t2 - t1)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    train_for_folder(root_dir=args.root_dir,config_name=args.config_name,output_dir=args.output_dir,batch_size=(args.batch_size),epochs=(args.epochs),args=args)
