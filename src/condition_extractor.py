import ast
import astor
import inspect

from uncovered_branches_extractor import *
from target_programs import functions_to_approximate

class FunctionAndBranchConditionsExtractor():
    """Extract function for dataset generation and condition components"""

    def __init__(self, sub_program):
        self.var_map = {}
        self.sub_program = sub_program

    def collectVariables(self, tree):
        """Explores the AST and stores function assignment in a dictionary"""
        def traverse(node):
            if isinstance(node, ast.AnnAssign):
                self.var_map[node.target.id] = node
            for child in ast.iter_child_nodes(node):
                traverse(child)
        traverse(tree)
        return self.var_map

    def extractVariables(self, tree):
        """Explores the AST and returns the function name used for variable assignment"""
        variables = []
        
        def traverse(node):
            if isinstance(node, ast.Name):
                if node.id in self.var_map:
                    variables.append(astor.to_source(self.var_map[node.id].value).strip())
            for child in ast.iter_child_nodes(node):
                traverse(child)
        traverse(tree)
        return variables[0]

    def collect_conditionComponents(self, tree):
        """Explores the AST and extracts branch conditions and target function"""
        conditionComponents = []
        self.collectVariables(tree)
        """Extract uncovered branches"""
        branches = [b for b, _ in self.sub_program.branches_uncovered]

        def traverse(node):
            if isinstance(node, ast.If):
                if node.lineno in branches:
                    conditionComponentsDict = {}
                    conditionComponentsDict["target_fn"] = self.extractVariables(node.test)
                    processed_branchConditions = self.process_branchConditions(astor.to_source(node.test).strip())
                    conditionComponentsDict["branch_conditions"] = [{}]
                    conditionComponentsDict["branch_conditions"][0]["operator"] = processed_branchConditions[0]
                    conditionComponentsDict["branch_conditions"][0]["target"] = processed_branchConditions[1]
                    conditionComponents.append(conditionComponentsDict)
            for child in ast.iter_child_nodes(node):
                traverse(child)
        traverse(tree)
        return conditionComponents

    def process_branchConditions(self, branchCondition):
        "Extract operand and target from branch conditions"
        branchCondition = branchCondition.replace("(","").replace(")","")
        branchConditionArray = branchCondition.split()
        branchConditionArray.pop(0)
        return branchConditionArray


if __name__ == '__main__':

    """Source AST generation for subject program fun"""
    source_ast = ast.parse(inspect.getsource(program_1))

    funCondExtractor = FunctionAndBranchConditionsExtractor(symfz_ct)

    """Pass the function ast to extract branch conditions and target function"""
    funCondExtractor.collect_conditionComponents(source_ast)

    conditionComponents = funCondExtractor.collect_conditionComponents(source_ast)

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

    processed_conditionComponentsDict["target_fn"] = locals()[targetNew]

    for idx in range(len(conditionComponents)):
        processed_conditionComponentsDict["branch_conditions"].append(conditionComponents[idx]["branch_conditions"][0])

    processed_conditionComponentsArray.append(processed_conditionComponentsDict)

    print("processed_conditionComponentsArray", processed_conditionComponentsArray)






    



    
