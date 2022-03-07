import torch
from torch.nn import functional as F

import numpy as np
from collections import deque

from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear

class GradientInputGenerator:
    def __init__(self, 
                 eps=1., 
                 eps_iter=0.1, 
                 nb_iter=1000, 
                 norm=2,
                 target_scaler=255,
                 num_seeds=1):
      
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.norm = norm
        self.target_scaler = target_scaler
        self.num_seeds = num_seeds

    def __call__(self, 
                 model,
                 op,
                 target,
                 seed=None):
      
        if not seed:
            # create default seed at midpoint in input space [0,1]
            seed = torch.rand((self.num_seeds, model.input_size))
        else:
            # scale provided input seed for model
            seed = model.x_scaler.transform(seed)

        # set target op for pgd
        if op == ">" or op == ">=":
            # make larger
            target = 1. if target == 0 else target
            target = target * self.target_scaler
        elif op == "<" or op == "<=":
            # make smaller
            target = 1. if target == 0 else target
            target = target * -self.target_scaler
        elif op == "==":
            # equal the target value
            target = target
        else:
            raise ValueError("Unhandled op!")

        target = torch.full((self.num_seeds, 1), target) 

        # loss function + target transform based on model output 
        if model.output_size == 1:
            loss_fn = F.l1_loss  
            if op != "==":
                target *= torch.rand_like(target)
            target  = model.y_scaler.transform(target)
        else:
            loss_fn = F.cross_entropy
            target  = target.reshape(-1).long()

        # generate input via pgd
        x_adv = projected_gradient_descent(
            model_fn=model,
            x=seed,
            y=target,
            targeted=True,
            loss_fn=loss_fn,
            eps=self.eps,
            eps_iter=self.eps_iter, 
            nb_iter=self.nb_iter,
            norm=self.norm,
            clip_min=0,
            clip_max=1,
            rand_init=True,
            rand_minmax=None,
            sanity_checks=False,
            early_stopping=True
        ).detach()

        x_adv = model.x_scaler.inverse_transform(x_adv)

        return x_adv

def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    loss_fn=F.cross_entropy
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    out = model_fn(x)
    loss = loss_fn(out, y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted: 
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward(retain_graph=True)
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation
    # print(adv_x, optimal_perturbation, out, loss)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
    loss_fn=F.cross_entropy,
    early_stopping=True
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    if early_stopping:
        prev_advs = deque([], 5)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
            loss_fn=loss_fn
        )

        if early_stopping:
            if any([(adv_x.detach() == adv_p).all() for adv_p in prev_advs]):
                eps_iter /= 2.
                prev_advs.clear()
                if eps_iter < 1e-6:
                    break
            else:
                prev_advs.append(adv_x.detach())
                # print(prev_advs)

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

if __name__ == '__main__':

    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    from subject_programs.functions_to_approximate import *
    from dataset_generator import *
    from function_approximator import *

    fn = fahrenheit_to_celcius_fn

    # train approximator
    dg = DatasetGenerator(fn)

    train_loader, test_loader = dg(
        scaler=MinMaxScaler, 
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
        # log_every_n_steps=1,
        # flush_logs_every_n_steps=1,
        callbacks=[escb]
    )
    trainer.fit(model, train_loader)
    # trainer.test(model, test_loader)

    if 'x_scaler' in dg.__dict__:
        model.x_scaler = dg.x_scaler
    if 'y_scaler' in dg.__dict__:
        model.y_scaler = dg.y_scaler

    # generate target input
    if model.output_size == 1:
        op_targets = [
            (">", 0.),
            ("<", 0.),
            ("==", 100.)
        ]
    else:  
        op_targets = [
            ("==", 0),
            ("==", 1)
        ]

    generator = GradientInputGenerator(num_seeds=10)
    for op, target in op_targets:
        x_adv = generator(model=model, op=op, target=target)

        print("op:", op, 'target:', target)
        print('x_adv:', x_adv)
        if model.input_size > 1:
            print('fn(x_adv):', [fn(*x_.numpy().tolist()[0]) for x_ in x_adv])
        else:
            print('fn(x_adv):', [fn(*x_) for x_ in x_adv])