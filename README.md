# DiffyFuzz

DiffyFuzz is a novel testing tool that approximates program logic differentiably so that inputs can be crafted to access tricky branches. 

This tool can be used directly or incorporated as an extension to other techniques, like symbolic and concolic execution. When the base tool can no longer improve coverage statistics, DiffyFuzz activates to expand coverage for numerically guarded branches. 

## How does it work?

At a high-level, there are four primary phases to our approach:

1. Run a simple fuzzing routine to initialize coverage profile
2. Indentify blocking code logic that inhibits branch exploration 
3. Approximate that functionality with a differentiable function (e.g. a neural network)
4. Generate new tests view gradient-optimization (e.g. adversarial attack)

These generated inputs are specifically designed to satisfy uncovered branch conditions and improve coverage. 

Additional details for each phase can be seen in the following flow chart.

![DiffyFuzz Overview](/imgs/overview.png?raw=true)

### Initializing the coverage profile

TODO: Apoorv, briefly describe your approach and provide runnable code snippet. Include data you provide to step 2 for Aish.

### Identifying blocking code

TODO: Aish, briefly describe your approach and provide runnable code snippet. Include data you provide to step 3-4 for Fabrice.

### Function Approximation

Once we have identified the blocking code, we must then approximate it differentiably. For this we use a shallow 2-layer neural network implemented in PyTorch (`FunctionApproximator`). To generate the data necessary to train the model, we pass the blocking code to our `DatasetGenerator`, which performs the following steps:

1. Inspects the function `argspecs`
2. Uses either `torch.linspace` or `pyfuzz` to generate inputs
4. Passes these inputs into the original code to derive the ground truth label
3. Optionally (but crucially for training) scales both the inputs and outputs separately
5. Packages the datasets into train and test `torch.utils.data.DataLoader` classes

For an example of this substep, see the [code](src/dataset_generator.py) or run this command:
```
> python dataset_generatory.py
```
This outputs a sample of both the training / testing data for the `square_fn`:
```
train_loader: [tensor([[0.3263],
        [0.5265]]), tensor([[0.1206],
        [0.0028]])]
test_loader: [tensor([[0.1231],
        [0.2813]]), tensor([[0.5681],
        [0.1914]])]
```

Now that we have a model and a dataset, it's training time! We use `pytorch-lightning` to greatly simplify this process. Even without a GPU, most of our studied functions can be well approximated in 1-3 seconds when trained on 900 datapoints for 3 epochs. 

For example, the following command yields:

```
python function_approximator.py
```

|index|fn\_name|type|test\_loss|test\_acc|train\_time_in_sec|fuzz\_generate|
|---|---|---|---|---|---|---|
|0|sin\_fn|continous|0\.34587|NA|1\.60|false|
|1|square\_fn|continous|0\.00689|NA|1\.57|false|
|2|log\_fn|continous|0\.03056|NA|1\.58|false|
|3|poly\_fn|continous|0\.01067|NA|1\.62|false|
|4|pythagorean\_fn|continous|0\.01103|NA|1\.61|false|
|5|fahrenheit\_to_celcius_fn|continous|0\.00041|NA|1\.61|false|
|6|dl\_textbook_fn|continous|0\.00299|NA|1\.61|false|
|7|square\_disc_fn|discontinous|1\.36842e-06|1|1\.62|false|
|8|log\_disc_fn|discontinous|0\.00073|1|1\.63|false|
|9|neuzz\_fn|discontinous|1\.28657e-05|1|1\.61|false|
|10|fahrenheit\_to_celcius_disc_fn|discontinous|0\.26717|0\.91|1\.60|false|
|11|log\_sin_fn|discontinous|2\.69615e-06|1|1\.60|false|
|12|f\_of_g_fn|discontinous|1\.43994e-06|1|1\.66|false|
|13|arcsin\_sin_fn|discontinous|7\.15119e-06|1|1\.62|false|

![sin_fn](/imgs/sin_fn.png?raw=true) ![square_fn](/imgs/square_fn.png?raw=true) ![log_fn](/imgs/log_fn.png?raw=true) ![poly_fn](/imgs/poly_fn.png?raw=true)          ![fahrenheit_to_celcius_fn](/imgs/fahrenheit_to_celcius_fn.png?raw=true) ![dl_textbook_fn](/imgs/dl_textbook_fn.png?raw=true)

### Generating Inputs

Now that we have a fitted approximator, we can use gradient-optimization techniques to generate an input that satisfies the constraints for the uncovered branches. For this, we repurpose a projected gradient descent (PGD) algorithm originally developed for adversarial attacks. The attack is unbounded so that the seed input can be perturbed as far as necessary to generate the target inputs. 

The following code runs through a full example where the target function is `fahrenheit_to_celcius_fn`. If the rounded output of this function is 100, then it reaches a bug (see [subject_programs/program_6](/src/subject_programs/program_6.py)). The testing goal is to find an input that causes `fahrenheit_to_celcius_fn` to output 100.

```
> python pgd.py
```
This trains a model, then uses PGD to arrive at a value that accesses the branch and will discover the bug!

```
x_adv: tensor([[212.2676]])
target: 100
fn(x_adv): tensor([100.1487])
```
Running the program with this value results in:
 ```
>python subject_programs\program_6.py --input 212.2676
You've got water!
Traceback (most recent call last):
  File "C:\Users\fabriceyhc\Documents\GitHub\diffy_fuzz\src\subject_programs\program_6.py", line 27, in <module>
    raise Exception("You found a hard-to-reach bug (and steam)!")
Exception: You found a hard-to-reach bug (and steam)!
 ```
