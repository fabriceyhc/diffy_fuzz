from target_programs.program_12 import program_12_sym, program_12
from SymbolicFuzzer import SimpleSymbolicFuzzer
import ast
import sys
import csv
import inspect
import astor
from condition_extractor import FunctionAndBranchConditionsExtractor
from target_programs.functions_to_approximate import *
from input_generator import *
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from subject_programs.functions_to_approximate import *
from dataset_generator import *
from function_approximator import *
from legend import get_subject_programs_config

filename = 'final_results.csv'

if __name__ == '__main__':

    fields = ['Program Name', 'Tries', 'Branch Coverage(Symbolic)', 'Execution Time Symbolic(in seconds)', 'Branch Coverage(Func Approximator)'] 
    rows = []

    configs = get_subject_programs_config()
    for config in configs:
        row = []
        
        # phase 1

        #get transformed symbolic program
        print(config)
        symbolic_target_program = config['symbolic_target_program']
        target_program = config['target_program']
        precision = config['precision']
        external_func_length = config['external_func_length']
        symfz_ct = SimpleSymbolicFuzzer(symbolic_target_program, precision = precision, external_func_length = external_func_length)
        symbolic_execution = config['symbolic']
        #check if symbolic execution can be performed
        if symbolic_execution:
            symfz_ct.start_execution(tries=100)
        else:
            symfz_ct.collect_branch_conditions()
        # print(symfz_ct.conditions_covered)
        # print(symfz_ct.branches)
        print(symfz_ct.calculate_branch_coverage())
        print(symfz_ct.branches_uncovered, "hey")
        row.append(target_program.__name__)
        row.append(100)
        row.append(str(symfz_ct.calculate_branch_coverage())+'%')
        row.append(str(symfz_ct.execution_time)+" seconds")
        print(row)
        if(len(symfz_ct.branches_uncovered) == 0):
            row.append('NA')
            rows.append(row)
            continue
        # phase 2
        """Source AST generation for subject program fun"""
        source_ast = ast.parse(inspect.getsource(target_program))

        funCondExtractor = FunctionAndBranchConditionsExtractor(symfz_ct)

        """Pass the function ast to extract branch conditions and target function"""
        funCondExtractor.collect_conditionComponents(source_ast)

        conditionComponents = funCondExtractor.collect_conditionComponents(source_ast)
        print(conditionComponents)
        # sys.exit(0)

        "Process the condition components to obtain the function in memory and extract operands and target from branch conditions"
        processed_conditionComponentsArray = []
        processed_conditionComponentsDict = {}
        processed_conditionComponentsDict["branch_conditions"] = []

        target = conditionComponents[0]["target_fn"]
        targetNew = ""

        for char in target:
            if char != "(":
                targetNew += char
            elif char == "(":
                break 
        # print(locals())
        processed_conditionComponentsDict["target_fn"] = eval(targetNew)

        for idx in range(len(conditionComponents)):
            processed_conditionComponentsDict["branch_conditions"].append(conditionComponents[idx]["branch_conditions"][0])

        processed_conditionComponentsArray.append(processed_conditionComponentsDict)

        print("processed_conditionComponentsArray", processed_conditionComponentsArray)

        #phase 3

        fn = processed_conditionComponentsArray[0]['target_fn']

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

        # tb_logger = pl_loggers.TensorBoardLogger("./logs/", name=fn.__name__)
        escb = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=2, verbose=False, mode="min")

        trainer = Trainer(
            max_epochs=3,
            gpus=torch.cuda.device_count(),
            # logger=tb_logger,
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

        generator = GradientInputGenerator(num_seeds=1)

        op_targets = []
        # print("hey", processed_conditionComponentsArray)
        for cond in processed_conditionComponentsArray[0]['branch_conditions']:
            arr = (cond['operator'], float(cond['target']))
            op_targets.append(arr)

        for op, target in op_targets:
            x_adv = generator(model=model, op=op, target=target)

            print("op:", op, 'target:', target)
            print('x_adv:', x_adv)
            symfz_ct.collect_additional_covergae(x_adv, model.input_size)
            # if model.input_size > 1:
            #     print('fn(x_adv):', [fn(*x_.numpy().tolist()) for x_ in x_adv])
            # else:
            #     print('fn(x_adv):', [fn(*x_) for x_ in x_adv])
        row.append(str(symfz_ct.calculate_branch_coverage())+'%')
        rows.append(row)
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)
