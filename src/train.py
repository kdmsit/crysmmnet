"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from data import get_train_val_loaders
from config import TrainingConfig
from models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import os

# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)

    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(config: Union[TrainingConfig, Dict[str, Any]],model: nn.Module = None,train_val_test_loaders=[],resume=0):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    # print(config)
    # if type(config) is dict:
    #     try:
    #         print(config)
    #         config = TrainingConfig(**config)
    #     except Exception as exp:
    #         print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    # print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    # pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = False
    alignn_models = {"alignn","dense_alignn","alignn_cgcnn","alignn_layernorm"}
    if config.model.name == "clgn":
        line_graph = True
    if config.model.name == "cgcnn":
        line_graph = True
    if config.model.name == "icgcnn":
        line_graph = True
    if config.model.name in alignn_models and config.model.alignn_layers > 0:
        line_graph = True

    # print ('output_dir train', config.output_dir)
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]

    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True

    # define network, optimizer, scheduler
    _model = {"alignn": ALIGNN}
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    model_number=[]
    if resume ==1:
        for f in os.listdir(config.output_dir):
            # print(config.output_dir)
            if f.startswith('checkpoint_'):
                # print(f)
                model_number.append(int(f.split('.')[0].split('_')[1]))
    # print(model_number)
    # exit()
    if resume ==1:
        checkpoint = torch.load(config.output_dir+'checkpoint_'+str(max(model_number))+'.pt')
        net.load_state_dict(checkpoint["model"])

    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if resume ==1:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config.learning_rate,epochs=config.epochs,steps_per_epoch=steps_per_epoch,pct_start=0.3)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer)

    if resume ==1:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # select configured loss function
    criteria = {"mse": nn.MSELoss(),}
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

    trainer = create_supervised_trainer(net,optimizer,criterion,prepare_batch=prepare_batch,device=device,deterministic=deterministic)

    if resume ==1:
        trainer.load_state_dict(checkpoint["trainer"])

    evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    train_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    test_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: scheduler.step())

    best_loss = float('inf')

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {"train": {m: [] for m in metrics.keys()},"validation": {m: [] for m in metrics.keys()},"test": {m: [] for m in metrics.keys()}}

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)
        test_evaluator.run(test_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        tstmetrics = test_evaluator.state.metrics

        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            tstm = tstmetrics[metric]

            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()
                tstm = tstm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)
            history["test"][metric].append(tstm)

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(filename=os.path.join(config.output_dir, "history_val.json"),data=history["validation"])
            dumpjson(filename=os.path.join(config.output_dir, "history_train.json"),data=history["train"])
        if config.progress:
            pbar = ProgressBar()
            pbar.log_message(f"Epoch: {engine.state.epoch:.1f}")
            pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
            pbar.log_message(f"Test_MAE: {tstmetrics['mae']:.4f}")

        nonlocal best_loss
        if tstmetrics['mae'] < best_loss:
            best_loss = tstmetrics['mae']
        print("Best_mae",best_loss)
        print("\n")

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    # Write Predictions
    net.eval()
    f = open(os.path.join(config.output_dir, "prediction_results_test_set.csv"),"w")
    f.write("id,target,prediction\n")
    targets = []
    predictions = []
    with torch.no_grad():
        ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
        for dat, id in zip(test_loader, ids):
            # g, lg, target = dat
            # out_data = net([g.to(device), lg.to(device)])
            g, lg, text, target = dat
            out_data = net([g.to(device), lg.to(device), text])
            out_data = out_data.cpu().numpy().tolist()
            if config.standard_scalar_and_pca:
                sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
                out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                    0
                ][0]
            target = target.cpu().numpy().flatten().tolist()
            if len(target) == 1:
                target = target[0]
            for k in range(len(target)):
                f.write("%s, %6f, %6f\n" % (id, target[k], out_data[k]))
                targets.append(target[k])
                predictions.append(out_data[k])
    f.close()
    from sklearn.metrics import mean_absolute_error
    # print(targets)
    # print(predictions)
    print("Test MAE:",mean_absolute_error(np.array(targets), np.array(predictions)))


    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
