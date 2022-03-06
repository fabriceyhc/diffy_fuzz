import ast
import astor
import inspect

from uncovered_branches_extractor import *

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
                    conditionComponentsDict["branch_conditions"] = astor.to_source(node.test).strip()
                    conditionComponents.append(conditionComponentsDict)
            for child in ast.iter_child_nodes(node):
                traverse(child)
        traverse(tree)
        return conditionComponents

if __name__ == '__main__':

    """Source AST generation for subject program fun"""
    source_ast = ast.parse(inspect.getsource(program_14))

    funCondExtractor = FunctionAndBranchConditionsExtractor(symfz_ct)

    """Pass the function ast to extract branch conditions and target function"""
    funCondExtractor.collect_conditionComponents(source_ast)

    print(funCondExtractor.collect_conditionComponents(source_ast))
