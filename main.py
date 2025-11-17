import os
import random
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
import torch
import json
import time
import argparse

from tools.gradcam_utils import grad_cam_timeseries
from tools.plots import save_results, plot_selection_distribution, plot_module_on_series
from tools.metrics import metric

from exp.exp_main import Exp_Main


DEFAULT_CONFIG_PATH = Path("configs/linearModel.json")

#### General parameters ####
# model         --> DLinear, PatchMixer, PatchTST
# input_len     --> sequence length
# pred_len      --> prediction length
# input_sze     --> number of features (piezo=4)
# split_ratio   --> train data split ratio (train, val=0.1, test=1-train-val)
# lr            --> learning rate
# epochs        --> number of training epochs
# patience      --> early stopping patience
# batch_size    --> batch size
# debug         --> debug mode (True/False)
# save_name     --> auto name for saving results

#### visualization parameters ####
# tracking_gradcam  --> whether to track gradcam during testing
# attn_topk       --> top k% of attention to consider for visualization

#### Mixer Model parameters ####
# label_len     --> start token length
# enc_in        --> encoder input size
# seq_len       --> input sequence length
# patch_len     --> patch length
# stride        --> patch stride
# mixer_kernel_size  --> mixer layer kernel size
# d_model       --> dimension of model
# dropout       --> dropout rate
# head_dropout  --> dropout rate last
# e_layers      --> number of encoder layers

#### PatchTST Model parameters ####
# fc_dropout    --> dropout rate for the final fully connected layer
# padding_patch --> whether to pad the last patch if needed (end)
# revin         --> whether to use RevIN normalization
# affine        --> whether to use affine transformation in RevIN
# substract_last--> whether to substract the last value in the series for trend modeling
# decomposition --> whether to use series decomposition block
# kernel_size   --> convolutional kernel size for series decomposition
# d_ff          --> dimension of feedforward layer

#set Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_parser():
    parser = argparse.ArgumentParser(description="Models args")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="config file path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    known_args = parser.parse_known_args()[0]

    current_config_path = known_args.config

    current_config = {}
    with open(current_config_path, 'r') as file:
        current_config = json.load(file)

    for key, value in current_config.items():
        # if key not in parser arguments, add it
        if not any(arg.dest == key for arg in parser._actions):
            parser.add_argument(f'--{key}', type=type(value), default=value)

    # set defaults again
    parser.set_defaults(**current_config)

    args = parser.parse_args()
    return args


def main(args, file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.debug :
        print(f"[main] run on {device}")
    file_name = "{}".format(file.split("/")[-1].split(".")[0])

    folder_path = "results/"
    figs_path = "figs/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
    if not os.path.exists(folder_path+file_name+"/"):
        os.makedirs(folder_path+file_name+"/")

    args.save_name = "{}_{}_sl{}_pl{}_BS{}".format(file_name,args.model,args.seq_len,args.pred_len,args.batch_size)
    
    exp = Exp_Main(args, device, file)
            
    start_time = time.time()
    print("-------- learning for {}".format(file_name))
    
    # training phase
    exp.training()
    
    # testing phase
    preds, trues, preds_last, trues_last, global_focus_gradcam = exp.testing()

    # validation phase
    metrics_val = exp.validating()
    
    # total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    secondes = int(elapsed_time % 60)
    print(f"{minutes}min{secondes:02d}")
    
    # test metrics
    metrics_test = metric(trues, preds)
    print(f"test : {metrics_test[0]:.3f} / {metrics_test[1]:.3f} - {metrics_test[-1]:.3f}")
    print(f"val  : {metrics_val[0]:.3f} / {metrics_val[1]:.3f} - {metrics_val[-1]:.3f}")
    
    # save results
    save_results(args, file_name, metrics_test, elapsed_time, preds_last, trues_last, metrics_val)
    
    if args.tracking_gradcam:
        plot_module_on_series(trues_last, global_focus_gradcam, args, file_name, "gradcam")
        
    return metrics_test, metrics_val

if __name__ == '__main__':
    args = make_parser()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    set_seed(args.seed)

    tracker = EmissionsTracker() #allow_multiple_runs=True
    tracker.start()
    try:
        metrics_test, metrics_val = main(args, args.data)
    finally:
        tracker.stop()