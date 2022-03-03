import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import torchmetrics

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule

class FuncApproximator(LightningModule):
    def __init__(self, input_size=1, output_size=1):
        self.input_size = input_size
        self.output_size = output_size
        super(FuncApproximator, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size),
        )
        if output_size == 1:
            self.loss_fn = F.l1_loss # F.mse_loss F.l1_loss
        else:
            self.loss_fn = F.cross_entropy
            self.accuracy = torchmetrics.Accuracy()
            self.accuracy.mode = "multi-class"

        # set after training
        self.x_scaler = None
        self.y_scaler = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        if self.output_size > 1:
            self.log('train_acc_epoch', self.accuracy)

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     out = self(x)
        
    #     # log step metric
    #     val_loss = self.loss_fn(out, y)
    #     self.log("val_loss", val_loss)
        
    #     if self.output_size > 1:
    #         self.accuracy(out, y)
    #         self.log('val_acc', self.accuracy)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        
        # log step metric
        test_loss = self.loss_fn(out, y)
        self.log("test_loss", test_loss)
        
        if self.output_size > 1:
            self.accuracy(out, y)
            self.log('test_acc', self.accuracy)


    def predict(self, x):
        if self.x_scaler:
            x = self.x_scaler.transform(x)
        y_pred = self(x)
        if self.y_scaler and self.output_size == 1:
            y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred
        

class MinMaxScaler(object):
    """MinMax Scaler
    Transforms each channel to the range [a, b].
    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not 'feature_range' in kwargs:
            self.feature_range = [0, 1]

    def fit(self, tensor):
        self.min_ = tensor.min(dim=0, keepdim=True)[0]
        self.max_ = tensor.max(dim=0, keepdim=True)[0]
        dist = self.max_ - self.min_
        dist[dist == 0.0] = 1.0
        self.scale_ = 1.0 / dist
        return self

    def transform(self, tensor):
        tensor = torch.clone(tensor)
        a, b = self.feature_range
        tensor = (tensor - self.min_) * self.scale_
        tensor = tensor * (b - a) + a
        return tensor

    def inverse_transform(self, tensor):
        tensor = torch.clone(tensor)
        tensor /= self.scale_
        tensor += self.min_
        return tensor