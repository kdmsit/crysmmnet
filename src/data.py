"""Jarvis-dgl data loaders and DGLGraph utilities."""

import random
from pathlib import Path
from typing import Optional

# from typing import Dict, List, Optional, Set, Tuple

import os
import torch
import dgl
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

# from sklearn.pipeline import Pipeline
import pickle as pk

# from sklearn.decomposition import PCA  # ,KernelPCA
from sklearn.preprocessing import StandardScaler

# use pandas progress_apply
tqdm.pandas()



# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_graphs(df: pd.DataFrame,name: str = "dft_3d",neighbor_strategy: str = "k-nearest",cutoff: float = 8,max_neighbors: int = 12,
                cachedir: Optional[Path] = None,use_canonize: bool = False):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(structure,cutoff=cutoff,atom_features="atomic_number",max_neighbors=max_neighbors,compute_line_graph=False,use_canonize=use_canonize)

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_id_train_val_test(total_size=1000,split_seed=123,train_ratio=None,val_ratio=0.1,test_ratio=0.1,n_train=None,n_test=None,n_val=None,keep_data_order=False):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_torch_dataset(dataset=[],id_tag="jid",target="",neighbor_strategy="",atom_features="",use_canonize="",
                      name="",line_graph="",cutoff=8.0,max_neighbors=12):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    vals = df[target].values
    print("data range", np.max(vals), np.min(vals))
    graphs = load_graphs(df,name=name,neighbor_strategy=neighbor_strategy,use_canonize=use_canonize,cutoff=cutoff,max_neighbors=max_neighbors)
    data = StructureDataset(df,graphs,target=target,atom_features=atom_features,line_graph=line_graph,id_tag=id_tag)
    return data


def get_train_val_loaders(dataset: str = "dft_3d",dataset_array=[],target: str = "formation_energy_peratom",atom_features: str = "cgcnn",neighbor_strategy: str = "k-nearest",
                          n_train=None,n_val=None,n_test=None,train_ratio=None,val_ratio=0.1,test_ratio=0.1,batch_size: int = 5,line_graph: bool = True,split_seed: int = 123,
                          workers: int = 0,pin_memory: bool = True,save_dataloader: bool = False,filename: str = "sample",id_tag: str = "jid",use_canonize: bool = False,
                          cutoff: float = 8.0,max_neighbors: int = 12,target_multiplication_factor: Optional[float] = None,standard_scalar_and_pca=False,keep_data_order=False,
                          output_dir=None):
    """Help function to set up JARVIS train and val dataloaders."""
    print('Batch Size:',batch_size)
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (os.path.exists(train_sample) and os.path.exists(val_sample) and os.path.exists(test_sample) and save_dataloader):
        print("Loading DataLoader from saved file...")
        print("Make sure all the DataLoader params are same.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
    else:
        if not dataset_array:
            d = jdata(dataset)
        else:
            d = dataset_array
        dat = []
        all_targets = []
        for i in d:
            if isinstance(i[target], list):  # multioutput target
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)
            elif (i[target] is not None and i[target] != "na" and not math.isnan(i[target])):
                dat.append(i)
                all_targets.append(i[target])

        id_train, id_val, id_test = get_id_train_val_test(total_size=len(dat),split_seed=split_seed,train_ratio=train_ratio,val_ratio=val_ratio,test_ratio=test_ratio,
                                                          n_train=n_train,n_test=n_test,n_val=n_val,keep_data_order=keep_data_order)
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

        if standard_scalar_and_pca:
            y_data = [i[target] for i in dataset_train]
            # pipe = Pipeline([('scale', StandardScaler())])
            if not isinstance(y_data[0], list):
                print("Running StandardScalar")
                y_data = np.array(y_data).reshape(-1, 1)
            sc = StandardScaler()

            sc.fit(y_data)
            print("Mean", sc.mean_)
            print("Variance", sc.var_)
            try:
                print("New max", max(y_data))
                print("New min", min(y_data))
            except Exception as exp:
                print(exp)
                pass
            # pc = PCA(n_components=output_features)
            # pipe = Pipeline(
            #    [
            #        ("scale", StandardScaler()),
            #        ("reduce_dims", PCA(n_components=output_features)),
            #    ]
            # )
            pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))
            # pc = PCA(n_components=10)
            # pc.fit(y_data)
            # pk.dump(pc, open("pca.pkl", "wb"))

        train_data = get_torch_dataset(dataset=dataset_train,id_tag=id_tag,atom_features=atom_features,target=target,neighbor_strategy=neighbor_strategy,
                                       use_canonize=use_canonize,name=dataset,line_graph=line_graph,cutoff=cutoff,max_neighbors=max_neighbors)

        val_data = get_torch_dataset(dataset=dataset_val,id_tag=id_tag,atom_features=atom_features,target=target,neighbor_strategy=neighbor_strategy,
                                     use_canonize=use_canonize,name=dataset,line_graph=line_graph,cutoff=cutoff,max_neighbors=max_neighbors)

        test_data = get_torch_dataset(dataset=dataset_test,id_tag=id_tag,atom_features=atom_features,target=target,neighbor_strategy=neighbor_strategy,
                                      use_canonize=use_canonize,name=dataset,line_graph=line_graph,cutoff=cutoff,max_neighbors=max_neighbors)

        collate_fn = train_data.collate
        if line_graph:
            collate_fn = train_data.collate_line_graph

        # use a regular pytorch dataloader
        train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,drop_last=True,num_workers=workers,pin_memory=pin_memory)

        val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,drop_last=True,num_workers=workers,pin_memory=pin_memory)

        test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn,drop_last=False,num_workers=workers,pin_memory=pin_memory)

        if save_dataloader:
            torch.save(train_loader, train_sample)
            torch.save(val_loader, val_sample)
            torch.save(test_loader, test_sample)

    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))

    return (train_loader,val_loader,test_loader,train_loader.dataset.prepare_batch)
