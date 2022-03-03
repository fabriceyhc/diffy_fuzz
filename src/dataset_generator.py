import inspect
import itertools

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np

import pyfuzz


class DatasetGenerator:
    def __init__(self, fn):
      
        self.fn = fn

        # extract fn argument details
        argspecs = inspect.getfullargspec(self.fn)
        self.args = argspecs.args
        self.defaults = argspecs.defaults

        self.num_inputs = len(self.args)
        self.num_outputs = inspect.getsource(self.fn).split().count("return")

    @staticmethod
    def fuzz_inputs(num_inputs = 1000, 
                    input_range = (-255, 255), 
                    seed = None):
        if not seed:
            seed = [bytearray(range(10))]
        fuzzer = pyfuzz.MutationFuzzer(seed, mutator=pyfuzz.mutate_bytes)
        input_bytes = [fuzzer.fuzz() for _ in range(num_inputs)]
        inputs = []
        for in_ in input_bytes:
            fdi = pyfuzz.FuzzedDataInterpreter(in_)
            inputs.append(fdi.claim_float_in_range(input_range[0], input_range[1]))
        return inputs

    def __call__(self, 
                 input_range = (-255, 255), 
                 num_examples_per_arg = 1000,
                 scaler = None,
                 train_test_split = 0.9,
                 batch_size = 10,
                 max_dataset_size = 10000,
                 fuzz_generate = True):
      
        inputs = {}
        for a in self.args:

            if fuzz_generate:
                inputs[a] = self.fuzz_inputs(num_inputs=num_examples_per_arg, 
                                            input_range=input_range)
            else:
                inputs[a] = np.linspace(start=input_range[0], 
                                        stop=input_range[1], 
                                        num=num_examples_per_arg)

        X = torch.Tensor(list(itertools.product(*inputs.values())))

        # enforce dataset size limit
        if len(X) > max_dataset_size:
            idx = torch.randperm(len(X))
            X = X[idx]
            X = X[:max_dataset_size]

        y = torch.Tensor([self.fn(*x) for x in X])

        # filter out inf
        X = X[~torch.isinf(y)]
        y = y[~torch.isinf(y)]

        # scale dataset if provided
        if scaler:
            self.x_scaler = scaler()
            self.y_scaler = scaler()

            self.x_scaler.fit(X)
            self.y_scaler.fit(y)

            X = self.x_scaler.transform(X)
            y = self.y_scaler.transform(y)
            
        if self.num_outputs == 1:
            y = y.float().reshape(-1, 1)
        else:
            y = torch.flatten(y.long())

        full_dataset = TensorDataset(X, y)

        # split dataset for train & test
        train_size = int(train_test_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # package as dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader