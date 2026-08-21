import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator  # base class; see __sklearn_is_fitted__ below
from torch import nn

from .helpers import flatten_batch


class Pytorch_wrapper(nn.Module, BaseEstimator):
    def __init__(self, model, classes: np.ndarray | None = None, device: str|torch.device = "cpu"):
        super().__init__()
        self.model = model
        self.__is_fitted = True
        # Materialized here, not as the default value: a mutable default (np.arange(4)) would be
        # evaluated once at class-definition time and shared, by reference, across every instance
        # constructed without an explicit classes= - see .reports/2026-08-19_ruff_rung1_completion.md.
        self.classes_ = classes if classes is not None else np.arange(4)
        self.device = torch.device(device)
        self.model.to(self.device)

    def __sklearn_is_fitted__(self):
        return self.__is_fitted
    
    def forward(self, x):
        x=x.to(self.device)
        model_out = self.model(x)
        pred_proba = F.softmax(model_out, dim=1)
        seg = torch.argmax(pred_proba, dim=1)
        return seg
        
    def fit(self, X, y):  # TODO: not implemented
        self.__is_fitted = True
        return True
    
    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            model_out = self.model(X)
            pred_proba = F.softmax(model_out,dim=1)
            seg = torch.argmax(pred_proba, dim=1).unsqueeze(1)
        return flatten_batch(seg)#.cpu())
    
    def predict_proba(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            model_out = self.model(X)
        return flatten_batch(torch.softmax(model_out, dim=1))#.cpu())


