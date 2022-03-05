import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import torchmetrics

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
        train_loss = self.loss_fn(out, y)

        # log step metric
        self.log("train_loss", train_loss)

        if self.output_size > 1:
            self.accuracy(out, y)
            self.log('train_acc', self.accuracy)

        return train_loss

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
        test_loss = self.loss_fn(out, y)
        
        # log step metric
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


if __name__ == '__main__':

    import time
    import torch
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from subject_programs.functions_to_approximate import *
    from dataset_generator import *
    from function_approximator import *

    fns = [
        sin_fn, 
        square_fn, 
        log_fn, 
        poly_fn,
        pythagorean_fn, 
        fahrenheit_to_celcius_fn, 
        dl_textbook_fn, 
        square_disc_fn, 
        log_disc_fn, 
        neuzz_fn, 
        fahrenheit_to_celcius_disc_fn,
        log_sin_fn,
        f_of_g_fn,
        arcsin_sin_fn
    ]

    results = []
    for fg in [False]: # [True, False]
        for fn in fns:

            dg = DatasetGenerator(fn)

            train_loader, test_loader = dg(
                scaler=MinMaxScaler if dg.num_outputs == 1 else None, 
                num_examples_per_arg = 1000, 
                max_dataset_size = 1000, 
                batch_size=10, 
                fuzz_generate=False)

            model = FuncApproximator(
                input_size=dg.num_inputs,
                output_size=dg.num_outputs)
            
            tb_logger = pl_loggers.TensorBoardLogger("./logs/", name=fn.__name__)
            escb = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=2, verbose=False, mode="min")

            trainer = Trainer(
                max_epochs=3,
                gpus=torch.cuda.device_count(),
                logger=tb_logger,
                log_every_n_steps=1,
                flush_logs_every_n_steps=1,
                callbacks=[escb]
            )

            tic = time.perf_counter()
            trainer.fit(model, train_loader)
            toc = time.perf_counter()

            if 'x_scaler' in dg.__dict__:
                model.x_scaler = dg.x_scaler
            if 'y_scaler' in dg.__dict__:
                model.y_scaler = dg.y_scaler

            out = trainer.test(model, test_loader)[0]
            out['model'] = model
            out['dg'] = dg
            out['train_loader'] = train_loader
            out['test_loader'] = test_loader
            out['fn'] = fn
            out['fn_name'] = fn.__name__
            if 'test_acc' not in out:
                out['test_acc'] = 'NA'
            out['train_time_in_sec'] = toc - tic
            out['type'] = 'continous' if dg.num_outputs == 1 else 'discontinous'
            out['fuzz_generate'] = fg
           
            results.append(out)

    df = pd.DataFrame(results)
    display_cols = ['fn_name', 'type', 'test_loss', 'test_acc', 'train_time_in_sec', 'fuzz_generate']
    print(df[display_cols])


    for result in results:
        if result['test_acc'] != 'NA' or result['dg'].num_inputs > 1:
            continue

        x, y_true = result['test_loader'].dataset[:]
        x = result['model'].x_scaler.inverse_transform(x)
        y_true = result['model'].y_scaler.inverse_transform(y_true)
        y_pred = result['model'].predict(x)

        sns.lineplot(x=x.view(-1), y=y_true.view(-1), label = 'true')
        sns.lineplot(x=x.view(-1), y=y_pred.view(-1).detach(), linestyle='--', label = 'approx')
        plt.title(result['fn_name'] + ' | Fuzz Generate: ' + repr(result['fuzz_generate']))

        plt.show()