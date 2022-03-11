#!/usr/bin/env python
# coding: utf-8
import ast
# from bookutils import YouTubeVideo
import inspect
import z3
from Fuzzer import Fuzzer
from ControlFlow import PyCFG
from contextlib import contextmanager
import sys
import time
import numpy
from types import FrameType
from typing import Any, Optional, Callable

SYM_VARS = {
    int: (
        z3.Int, z3.IntVal), float: (
            z3.Real, z3.RealVal), str: (
                z3.String, z3.StringVal)}

SYM_VARS_STR = {
    k.__name__: ("z3.%s" % v1.__name__, "z3.%s" % v2.__name__)
    for k, (v1, v2) in SYM_VARS.items()
}

def translate_to_z3_name(v):
    return SYM_VARS_STR[v][0]

@contextmanager
def checkpoint(z3solver):
    z3solver.push()
    yield z3solver
    z3solver.pop()

def to_src(astnode):
    return ast.unparse(astnode).strip()

def declarations(astnode, hm=None):
    if hm is None:
        hm = {}
    if isinstance(astnode, ast.Module):
        for b in astnode.body:
            declarations(b, hm)
    elif isinstance(astnode, ast.FunctionDef):
        # hm[astnode.name + '__return__'] = \
        # translate_to_z3_name(astnode.returns.id)
        for a in astnode.args.args:
            hm[a.arg] = translate_to_z3_name(a.annotation.id)
        for b in astnode.body:
            declarations(b, hm)
    elif isinstance(astnode, ast.Call):
        # get declarations from the function summary.
        n = astnode.function
        assert isinstance(n, ast.Name)  # for now.
        name = n.id
        hm.update(dict(function_summaries[name]['vars']))
    elif isinstance(astnode, ast.AnnAssign):
        assert isinstance(astnode.target, ast.Name)
        hm[astnode.target.id] = translate_to_z3_name(astnode.annotation.id)
    elif isinstance(astnode, ast.Assign):
        # verify it is already defined
        for t in astnode.targets:
            assert isinstance(t, ast.Name)
            assert t.id in hm
    elif isinstance(astnode, ast.AugAssign):
        assert isinstance(astnode.target, ast.Name)
        assert astnode.target.id in hm
    elif isinstance(astnode, (ast.If, ast.For, ast.While)):
        for b in astnode.body:
            declarations(b, hm)
        for b in astnode.orelse:
            declarations(b, hm)
    elif isinstance(astnode, ast.Return):
        pass
    elif isinstance(astnode, ast.Raise):
        pass
    elif isinstance(astnode, ast.Expr):
        pass
    else:
        raise Exception(str(astnode))
    return hm

def used_vars(fn):
    return declarations(ast.parse(inspect.getsource(fn)))

def rename_variables(astnode, env):
    if isinstance(astnode, ast.BoolOp):
        fn = 'z3.And' if isinstance(astnode.op, ast.And) else 'z3.Or'
        return ast.Call(
            ast.Name(fn, None),
            [rename_variables(i, env) for i in astnode.values], [])
    elif isinstance(astnode, ast.BinOp):
        return ast.BinOp(
            rename_variables(astnode.left, env), astnode.op,
            rename_variables(astnode.right, env))
    elif isinstance(astnode, ast.UnaryOp):
        if isinstance(astnode.op, ast.Not):
            return ast.Call(
                ast.Name('z3.Not', None),
                [rename_variables(astnode.operand, env)], [])
        else:
            return ast.UnaryOp(astnode.op,
                               rename_variables(astnode.operand, env))
    elif isinstance(astnode, ast.Call):
        return ast.Call(astnode.func,
                        [rename_variables(i, env) for i in astnode.args],
                        astnode.keywords)
    elif isinstance(astnode, ast.Compare):
        return ast.Compare(
            rename_variables(astnode.left, env), astnode.ops,
            [rename_variables(i, env) for i in astnode.comparators])
    elif isinstance(astnode, ast.Name):
        if astnode.id not in env:
            env[astnode.id] = 0
        num = env[astnode.id]
        return ast.Name('_%s_%d' % (astnode.id, num), astnode.ctx)
    elif isinstance(astnode, ast.Return):
        return ast.Return(rename_variables(astnode.value, env))
    else:
        return astnode

def used_identifiers(src):
    def names(astnode):
        lst = []
        if isinstance(astnode, ast.BoolOp):
            for i in astnode.values:
                lst.extend(names(i))
        elif isinstance(astnode, ast.BinOp):
            lst.extend(names(astnode.left))
            lst.extend(names(astnode.right))
        elif isinstance(astnode, ast.UnaryOp):
            lst.extend(names(astnode.operand))
        elif isinstance(astnode, ast.Call):
            for i in astnode.args:
                lst.extend(names(i))
        elif isinstance(astnode, ast.Compare):
            lst.extend(names(astnode.left))
            for i in astnode.comparators:
                lst.extend(names(i))
        elif isinstance(astnode, ast.Name):
            lst.append(astnode.id)
        elif isinstance(astnode, ast.Expr):
            lst.extend(names(astnode.value))
        elif isinstance(astnode, (ast.Num, ast.Str, ast.Tuple, ast.NameConstant)):
            pass
        elif isinstance(astnode, ast.Assign):
            for t in astnode.targets:
                lst.extend(names(t))
            lst.extend(names(astnode.value))
        elif isinstance(astnode, ast.Module):
            for b in astnode.body:
                lst.extend(names(b))
        else:
            raise Exception(str(astnode))
        return list(set(lst))
    return names(ast.parse(src))

MAX_DEPTH = 10000
MAX_TRIES = 100
MAX_ITER = 1000

class SimpleSymbolicFuzzer(Fuzzer):
    """Simple symbolic fuzzer"""

    def __init__(self, fn, **kwargs):
        """Constructor.
        `fn` is the function to be fuzzed.
        Possible keyword parameters:
        * `max_depth` - the depth to which one should attempt
          to trace the execution (default 100) 
        * `max_tries` - the maximum number of attempts
          we will try to produce a value before giving up (default 100)
        * `max_iter` - the number of iterations we will attempt (default 100).
        """
        self.fn_name = fn.__name__
        py_cfg = PyCFG()
        py_cfg.gen_cfg(inspect.getsource(fn))
        self.fnenter, self.fnexit = py_cfg.functions[self.fn_name]
        self.used_variables = used_vars(fn)
        self.fn_args = list(inspect.signature(fn).parameters)
        z3.set_option(precision=30)
        self.z3 = z3.Solver()
        self.func = fn
        self.paths = None
        self.last_path = None
        self.conditions_covered = {}
        self.branches = []
        self.branches_uncovered = []
        self.no_of_inputs = 0
        self.options(kwargs)
        self.process()
        self.execution_time = 0
        self.time_coverage = [[0,0,0, True]]
        self.start_time = 0

    def options(self, kwargs):
        self.max_depth = kwargs.get('max_depth', MAX_DEPTH)
        self.max_tries = kwargs.get('max_tries', MAX_TRIES)
        self.max_iter = kwargs.get('max_iter', MAX_ITER)
        self.precision = kwargs.get('precision', 6)
        self.external_func_length = kwargs.get('external_func_length', 0)
        self._options = kwargs
    def get_all_paths(self, fenter):
        path_lst = [PNode(0, fenter)]
        completed = []
        for i in range(self.max_iter):
            new_paths = [PNode(0, fenter)]
            for path in path_lst:
                # explore each path once
                if path.cfgnode.children:
                    np = path.explore()
                    for p in np:
                        if path.idx > self.max_depth:
                            break
                        new_paths.append(p)
                else:
                    completed.append(path)
            path_lst = new_paths
        return completed + path_lst
    def process(self):
        self.paths = self.get_all_paths(self.fnenter)
        self.last_path = len(self.paths)
    def extract_constraints(self, path):
#         print(path, " $$$$")
        return [to_src(p) for p in to_single_assignment_predicates(path) if p]
    def solve_path_constraint(self, path):
        # re-initializing does not seem problematic.
        # a = z3.Int('a').get_id() remains the same.
        constraints = self.extract_constraints(path)
        identifiers = [
            c for i in constraints for c in used_identifiers(i)]  # <- changes
        with_types = identifiers_with_types(
            identifiers, self.used_variables)  # <- changes
        decl = define_symbolic_vars(with_types, '')
        exec(decl)

        solutions = {}
        with checkpoint(self.z3):
            st = 'self.z3.add(%s)' % ', '.join(constraints)
            eval(st)
            if self.z3.check() != z3.sat:
                return {}
            m = self.z3.model()
            solutions = {}
            for d in m.decls():
                sol = m[d]
                if hasattr(sol, 'numerator_as_long'):
                    sol =  sol.numerator_as_long()/sol.denominator_as_long()
                else:
                    sol = str(sol)
                    if len(sol)>1 and sol[len(sol)-1] == '?':
                        sol = sol[:len(sol)-2]
                solutions[d.name()] = round(float(sol), self.precision)
            my_args = {k: solutions.get(k, None) for k in self.fn_args}

        predicate = 'z3.And(%s)' % ','.join(
            ["%s == %s" % (k, v) for k, v in my_args.items()])
        eval('self.z3.add(z3.Not(%s))' % predicate)

        return my_args

    def get_next_path(self):
        self.last_path -= 1
        if self.last_path == -1:
            self.last_path = len(self.paths) - 1
        return self.paths[self.last_path].get_path_to_root()
    def fuzz(self):
        """Produce one solution for each path.
        Returns a mapping of variable names to (symbolic) Z3 values."""
        for i in range(self.max_tries):
            res = self.solve_path_constraint(self.get_next_path())
            if res:
                return res

        return {}
    def collect_branch_conditions(self):
        if self.start_time == 0:
            self.start_time = time.time()
        tree = ast.parse(inspect.getsource(self.func))
#         print(tree, "fdfds")
        paths = []
        def traverse_tree(node):
            if isinstance(node, ast.If):
#                 print(node)
                if hasattr(node, 'lineno'):
                    self.branches.append(node.lineno)

            for child in ast.iter_child_nodes(node):
                traverse_tree(child)
        traverse_tree(tree)
        for line in self.branches:
            obj = str(line) + "~0"
            self.conditions_covered[obj] = False
            obj_1 = str(line) + "~1"
            self.conditions_covered[obj_1] = False
            
    def start_execution(self, tries):
        self.collect_branch_conditions()
        self.start_time = time.time()
        self.lap_time = time.time()
        self.time_coverage = [[0,0,0, True]]
        self.no_of_inputs = 0
        for i in range(0, tries):
            args = self.fuzz()
            global coverage
            coverage = []
            # print(args[', "this is x", 'x' in args)
            # print(args)
            if not 'x' in args:
                continue
            sys.settrace(traceit)  # Turn on
            try:
                self.func(float(args['x']))
            except:
                pass
            sys.settrace(None)
            self.no_of_inputs+=1
            new_branch_discovered = 0
            # print(coverage)
            for j in range(len(coverage)):
                if coverage[j][0] in self.branches:
                    #next line is executed
                    if j+1 < len(coverage) and coverage[j][0] + 1 == coverage[j+1][0]:
                        if self.conditions_covered[str(coverage[j][0]) + "~1"] == False:
                            new_branch_discovered = True
                            self.conditions_covered[str(coverage[j][0]) + "~1"] = True
                    else:
                        if self.conditions_covered[str(coverage[j][0]) + "~0"] == False:
                            new_branch_discovered = True
                            self.conditions_covered[str(coverage[j][0]) + "~0"] = True
            if new_branch_discovered:
                new_branch_discovered = False
                print("coming here", self.func.__name__)
                self.time_coverage.append([self.calculate_total_branch_coverage(), round(time.time() - self.start_time, 6), self.no_of_inputs, True])
        self.execution_time = round(time.time() - self.start_time, 6)
        self.time_coverage.append([self.calculate_total_branch_coverage(), self.execution_time, self.no_of_inputs, True])
        self.collect_uncovered_branches()
    
    def collect_uncovered_branches(self):
        self.branches_uncovered = []
        for key, value in self.conditions_covered.items():
            # print(key.split('~'), not value)
            if not value:
                line_no = int(key.split('~')[0]) - self.external_func_length
                if line_no > 0:
                    branch_uncovered = [line_no, int(key.split('~')[1])]
                    self.branches_uncovered.append(branch_uncovered)

    def calculate_branch_coverage(self):
        self.collect_uncovered_branches()
        num_of_branches = 0
        for i in self.branches:
            if i - self.external_func_length > 1:
                num_of_branches+=1
        num_of_branches*=2
        return round((num_of_branches - len(self.branches_uncovered))/num_of_branches * 100,2)

    def collect_total_uncovered_branches(self):
        self.total_branches_uncovered = []
        self.branches_uncovered = []
        for key, value in self.conditions_covered.items():
            # print(key.split('~'), not value)
            if not value:
                line_no = int(key.split('~')[0]) - self.external_func_length
                if line_no > 0:
                    branch_uncovered = [line_no, int(key.split('~')[1])]
                    self.branches_uncovered.append(branch_uncovered)
        for key, value in self.conditions_covered.items():
            # print(key.split('~'), not value)
            if not value:
                line_no = int(key.split('~')[0])
                branch_uncovered = [line_no, int(key.split('~')[1])]
                self.total_branches_uncovered.append(branch_uncovered)

    def calculate_total_branch_coverage(self):
        self.collect_total_uncovered_branches()
        num_of_branches = len(self.branches)
        num_of_branches*=2
        return round((num_of_branches - len(self.total_branches_uncovered))/num_of_branches * 100,2)

    def preprocess_coverage(self,coverage):
        new_coverage = []
        first_occurence = -1
        for cov in coverage:
            if cov[1] == self.func.__name__:
                # print("######", first_occurence)
                if first_occurence == -1:
                    first_occurence = cov[0]-2
                    cov[0] = 2
                else:
                    cov[0] = cov[0]-first_occurence
                new_coverage.append(cov)

        return new_coverage

    def collect_additional_covergae(self,inputs, size):
        if self.start_time == 0:
            self.start_time = time.time()
        if size > 1:
            for x_ in inputs:
                global coverage
                coverage = []
                new_branch_discovered = False
                # print(args['x'], "this is x", 'x' in args)
                sys.settrace(traceit)  # Turn on
                try:
                    t = self.func(*x_.numpy().tolist())
                    # print(t, "Dwdw")
                except Exception as e:
                    print("error", e)
                sys.settrace(None)
                # print(coverage)
                coverage = self.preprocess_coverage(coverage)
                for j in range(len(coverage)):
                    if coverage[j][0] in self.branches:
                        #next line is executed
                        if j+1 < len(coverage) and coverage[j][0] + 1 == coverage[j+1][0]:
                            if self.conditions_covered[str(coverage[j][0]) + "~1"] == False:
                                new_branch_discovered = True
                                self.conditions_covered[str(coverage[j][0]) + "~1"] = True
                        else:
                            if self.conditions_covered[str(coverage[j][0]) + "~0"] == False:
                                new_branch_discovered = True
                                self.conditions_covered[str(coverage[j][0]) + "~0"] = True
                if new_branch_discovered:
                    new_branch_discovered = False
                    print("coming here", self.func.__name__)
                    self.time_coverage.append([self.calculate_total_branch_coverage(), round(time.time() - self.start_time, 6), self.no_of_inputs, False])
        else:
            for x_ in inputs:
                coverage = []
                # print(args['x'], "this is x", 'x' in args)
                new_branch_discovered = False
                sys.settrace(traceit)  # Turn on
                try:
                    t = self.func(float(*x_))
                    # print(t, "Dwdw")
                except Exception as e:
                    print("error", e)
                sys.settrace(None)
                self.no_of_inputs += 1
                coverage = self.preprocess_coverage(coverage)
                # print(coverage)
                # print(self.branches)
                for j in range(len(coverage)):
                    if coverage[j][0] in self.branches:
                        #next line is executed
                        if j+1 < len(coverage) and coverage[j][0] + 1 == coverage[j+1][0]:
                            if self.conditions_covered[str(coverage[j][0]) + "~1"] == False:
                                new_branch_discovered = True
                                self.conditions_covered[str(coverage[j][0]) + "~1"] = True
                        else:
                            if self.conditions_covered[str(coverage[j][0]) + "~0"] == False:
                                new_branch_discovered = True
                                self.conditions_covered[str(coverage[j][0]) + "~0"] = True
                if new_branch_discovered:
                    new_branch_discovered = False
                    print("coming here", self.func.__name__)
                    self.time_coverage.append([self.calculate_total_branch_coverage(), round(time.time() - self.start_time, 6), self.no_of_inputs, False])
                   
def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        """Trace program execution. To be passed to sys.settrace()."""
    #     if event == 'line':
        if event == 'line':
            global coverage
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            coverage.append([lineno, function_name, event])

        return traceit

def to_single_assignment_predicates(path):
    env = {}
    new_path = []
    for i, node in enumerate(path):
        ast_node = node.cfgnode.ast_node
        new_node = None
        if isinstance(ast_node, ast.AnnAssign) and ast_node.target.id in {
                'exit'}:
            new_node = None
        elif isinstance(ast_node, ast.AnnAssign) and ast_node.target.id in {'enter'}:
            args = [
                ast.parse(
                    "%s == _%s_0" %
                    (a.id, a.id)).body[0].value for a in ast_node.annotation.args]
            new_node = ast.Call(ast.Name('z3.And', None), args, [])
        elif isinstance(ast_node, ast.AnnAssign) and ast_node.target.id in {'_if', '_while'}:
            new_node = rename_variables(ast_node.annotation, env)
            if node.order != 0:
                assert node.order == 1
                new_node = ast.Call(ast.Name('z3.Not', None), [new_node], [])
        elif isinstance(ast_node, ast.AnnAssign):
            assigned = ast_node.target.id
            val = [rename_variables(ast_node.value, env)]
            env[assigned] = 0 if assigned not in env else env[assigned] + 1
            target = ast.Name('_%s_%d' %
                              (ast_node.target.id, env[assigned]), None)
            new_node = ast.Expr(ast.Compare(target, [ast.Eq()], val))
        elif isinstance(ast_node, ast.Assign):
            assigned = ast_node.targets[0].id
            val = [rename_variables(ast_node.value, env)]
            env[assigned] = 0 if assigned not in env else env[assigned] + 1
            target = ast.Name('_%s_%d' %
                              (ast_node.targets[0].id, env[assigned]), None)
            new_node = ast.Expr(ast.Compare(target, [ast.Eq()], val))
        elif isinstance(ast_node, (ast.Return, ast.Pass)):
            new_node = None
        else:
            s = "NI %s %s" % (type(ast_node), ast_node.target.id)
            raise Exception(s)
        new_path.append(new_node)
    return new_path

class PNode:
    def __init__(self, idx, cfgnode, parent=None, order=0, seen=None):
        self.seen = {} if seen is None else seen
        self.max_iter = MAX_ITER
        self.idx, self.cfgnode, self.parent, self.order = idx, cfgnode, parent, order

    def __repr__(self):
        return "PNode:%d[%s order:%d]" % (self.idx, str(self.cfgnode),
                                          self.order)
    def copy(self, order):
        p = PNode(self.idx, self.cfgnode, self.parent, order, self.seen)
        assert p.order == order
        return p
    
    def explore(self):
        ret = []
        for (i, n) in enumerate(self.cfgnode.children):
            key = "[%d]%s" % (self.idx + 1, n)
            ccount = self.seen.get(key, 0)
            if ccount > self.max_iter:
                continue  # drop this child
            self.seen[key] = ccount + 1
            pn = PNode(self.idx + 1, n, self.copy(i), seen=self.seen)
            ret.append(pn)
        return ret
    
    def get_path_to_root(self):
        path = []
        n = self
        while n:
            path.append(n)
            n = n.parent
        return list(reversed(path))
    
    def __str__(self):
        path = self.get_path_to_root()
        ssa_path = to_single_assignment_predicates(path)
        return ', '.join([to_src(p) for p in ssa_path])

def identifiers_with_types(identifiers, defined):
    with_types = dict(defined)
    for i in identifiers:
        if i[0] == '_':
            nxt = i[1:].find('_', 1)
            name = i[1:nxt + 1]
            assert name in defined
            typ = defined[name]
            with_types[i] = typ
    return with_types

def z3_names_and_types(z3_ast):
    hm = {}
    children = z3_ast.children()
    if children:
        for c in children:
            hm.update(z3_names_and_types(c))
    else:
        # HACK.. How else to distinguish literals and vars?
        if (str(z3_ast.decl()) != str(z3_ast.sort())):
            hm["%s" % str(z3_ast.decl())] = 'z3.%s' % str(z3_ast.sort())
        else:
            pass
    return hm

def define_symbolic_vars(fn_vars, prefix):
    sym_var_dec = ', '.join([prefix + n for n in fn_vars])
    sym_var_def = ', '.join(["%s('%s%s')" % (t, prefix, n)
                             for n, t in fn_vars.items()])
    return "%s = %s" % (sym_var_dec, sym_var_def)



