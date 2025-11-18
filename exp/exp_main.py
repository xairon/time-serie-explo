import json
from data_provider.data_loader import load_data_train_val_test
from models import PatchTST, PatchMixer, DLinear
from tools.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
# import torchvision
# from thop import profile

warnings.filterwarnings('ignore')

class Exp_Main:
    def __init__(self, args, device, file):
        self.args = args
        self.device = device
        self.file = file
        self.file_name = "{}".format(file.split("/")[-1].split(".")[0])
        self.model_pred = self._build_model().to(self.device)
        self.optim_pred = torch.optim.AdamW(self.model_pred.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.scheduler_pred = torch.optim.lr_scheduler.StepLR(self.optim_pred, step_size=5, gamma=0.5)

    def _build_model(self):
        model_dict = {
            #'Autoformer': Autoformer,
            #'Transformer': Transformer,
            #'Informer': Informer,
            'DLinear': DLinear,
            #'NLinear': NLinear,
            #'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchMixer': PatchMixer,
            #'MHTA': MHTA,
            #'FSnet': FSnet,
            #'TimesNet': TimesNet,
        }
        model = model_dict[self.args.model].Model(self.args)
        return model
        
    def _early_stopping_check(self, val_loss, best_loss, patience_counter, best_model_state):
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in self.model_pred.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        return best_loss, patience_counter, best_model_state
    
    def _process_one_batch(self, model, batch_x, batch_y):
        if self.args.debug :
            print(f"[process_one_batch]")
            print(f"  batch_x, batch_y size :{batch_x.shape}, {batch_y.shape}")
        
        proj = lambda t: t.reshape(t.size(0), -1)
        pred = model(batch_x)
        return proj(pred), proj(batch_y)

    def _evaluate_loss(self, loader):
        if self.args.debug :
            print(f"[evaluate_loss]")
            
        self.model_pred.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_y, batch_idx = batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred, true = self._process_one_batch(self.model_pred, batch_x, batch_y)
                loss = self.loss_fn(pred, true).mean()
                total_loss += loss.item() * batch_x.size(0)
                count += batch_x.size(0)
        
        return total_loss / count if count > 0 else float('inf')
    
    def training(self):
        if self.args.debug :
            print(f"[training]")
        train_loader, valid_loader, _, self.scaler = load_data_train_val_test(self.args, self.file)
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        epoch_loss = float('inf')
        val_loss = float('inf')

        pbar = tqdm(range(self.args.epochs),desc="train ep")
        for ep in pbar:
            time_update = 0
            self.model_pred.train()
            num_batches = 0
            loss_accum = 0.0

            batch_i = 0
            for batch in train_loader:
                batch_x, batch_y, batch_idx = batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                pred, true = self._process_one_batch(self.model_pred, batch_x, batch_y)
                loss_pred = self.loss_fn(pred, true).mean()
                loss_accum += loss_pred.item()

                if batch_i % 10 == 0:
                    pbar.set_description(f"batch loss: {loss_pred.item():.4f} | train loss: {epoch_loss:.4f} | val loss: {val_loss:.4f}")
    
                self.optim_pred.zero_grad()
                loss_pred.backward()
                self.optim_pred.step()
                num_batches += 1
                batch_i += 1

            self.scheduler_pred.step()

            epoch_loss = loss_accum / num_batches if num_batches > 0 else float('inf')
    
            val_loss = self._evaluate_loss(valid_loader)

            best_loss, patience_counter, best_model_state = self._early_stopping_check(
                val_loss, best_loss, patience_counter, best_model_state
            )
            if patience_counter >= self.args.patience:
                print(f"Early stopping triggered at epoch {ep+1}")
                break
    
        if best_model_state is not None:
            self.model_pred.load_state_dict(best_model_state)
    
    def testing(self):
        if self.args.debug:
            print(f"[testing]")
                
        self.model_pred.eval()
        train_loader, val_loader, test_loader, self.scaler = load_data_train_val_test(self.args, self.file)
        
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        offset = n_train + n_val
    
        preds, trues = [], []
        preds_last, trues_last = [], []
    
        # timelines globales
        total_len = len(test_loader.dataset)
        data_time_lag = []
        global_focus_gradcam = np.zeros(total_len)
        counts_gradcam = np.zeros(total_len)
        
        for batch in tqdm(test_loader, desc="test batch"):
            batch_x, batch_y, batch_idx = batch
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    
            # ---- forward prédiction ----
            with torch.no_grad():
                pred, true = self._process_one_batch(self.model_pred, batch_x, batch_y)
    
            # ---- Grad-CAM ----
            if self.args.tracking_gradcam:
                try:
                    #target_fn = lambda out: out[..., -1, :].mean()
                    target_fn = lambda out: out[..., -1, 0].mean()
                    #target_fn = lambda out: out.norm(dim=-1).mean()
                    #target_fn = lambda out: out.mean()
                    cams = grad_cam_timeseries(self.args, self.model_pred, batch_x,target=target_fn,device=self.device)
    
                    for i, start in enumerate(batch_idx):
                        start = start.item() - offset
                        for t, score in enumerate(cams[i]):
                            global_idx = start + t
                            if global_idx < total_len:
                                global_focus_gradcam[global_idx] += score
                                counts_gradcam[global_idx] += 1
                except Exception as e:
                    if self.args.debug:
                        print("Grad-CAM failed:", e)
    
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.cpu().numpy())
    
            # ---- prédictions "last step" ----
            num_samples = pred.shape[0]
            preds_last.append(
                pred.detach().cpu().numpy().reshape(num_samples, self.args.pred_len, self.args.input_size)[:, -1, :]
            )
            trues_last.append(
                true.detach().cpu().numpy().reshape(num_samples, self.args.pred_len, self.args.input_size)[:, -1, :]
            )
    
        # concat
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_last = np.concatenate(preds_last, axis=0)
        trues_last = np.concatenate(trues_last, axis=0)
    
        # normalisation
        global_focus_gradcam = np.divide(global_focus_gradcam, counts_gradcam,out=np.zeros_like(global_focus_gradcam),where=counts_gradcam > 0)
    
        return (
            preds, trues, preds_last, trues_last, global_focus_gradcam
        )


    def validating(self):
        if self.args.debug:
            print(f"[testing]")
                
        self.model_pred.eval()
        _, val_loader, _, self.scaler = load_data_train_val_test(self.args, self.file)
    
        preds, trues = [], []
    
        for batch in tqdm(val_loader,desc="val batch"):
            batch_x, batch_y, batch_idx = batch
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    
            with torch.no_grad():
                pred, true = self._process_one_batch(self.model_pred, batch_x, batch_y)
    
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.cpu().numpy())
    
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
    
        return metric(trues, preds)
