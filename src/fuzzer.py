#!/usr/bin/env python
# coding: utf-8

# In[409]:


import math
from bookutils import YouTubeVideo
import ast
from showast import show_ast
import inspect
import astor
import z3
from Fuzzer import Fuzzer
from ControlFlow import PyCFG, to_graph, gen_cfg
from contextlib import contextmanager
import sys
from types import FrameType, TracebackType
from typing import Any, Optional, Callable, List, Type, Set, Tuple


# In[104]:


SYM_VARS = {
    int: (
        z3.Int, z3.IntVal), float: (
            z3.Real, z3.RealVal), str: (
                z3.String, z3.StringVal)}


# In[105]:


SYM_VARS_STR = {
    k.__name__: ("z3.%s" % v1.__name__, "z3.%s" % v2.__name__)
    for k, (v1, v2) in SYM_VARS.items()
}
def translate_to_z3_name(v):
    return SYM_VARS_STR[v][0]


# In[106]:


@contextmanager
def checkpoint(z3solver):
    z3solver.push()
    yield z3solver
    z3solver.pop()


# In[107]:


def to_src(astnode):
    return ast.unparse(astnode).strip()


# In[108]:


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
    else:
        raise Exception(str(astnode))
    return hm


# In[109]:


def used_vars(fn):
    return declarations(ast.parse(inspect.getsource(fn)))


# In[453]:


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

        self.options(kwargs)
        self.process()

    def options(self, kwargs):
        self.max_depth = kwargs.get('max_depth', MAX_DEPTH)
        self.max_tries = kwargs.get('max_tries', MAX_TRIES)
        self.max_iter = kwargs.get('max_iter', MAX_ITER)
        self._options = kwargs
    def get_all_paths(self, fenter, depth=0):
        if depth > self.max_depth:
            raise Exception('Maximum depth exceeded')
        if not fenter.children:
            return [[(0, fenter)]]

        fnpaths = []
        for idx, child in enumerate(fenter.children):
            child_paths = self.get_all_paths(child, depth + 1)
            for path in child_paths:
                # In a conditional branch, idx is 0 for IF, and 1 for Else
                fnpaths.append([(idx, fenter)] + path)
        return fnpaths
    def process(self):
        self.paths = self.get_all_paths(self.fnenter)
        self.last_path = len(self.paths)
    def extract_constraints(self, path):
        predicates = []
        for (idx, elt) in path:
            if isinstance(elt.ast_node, ast.AnnAssign):
                if elt.ast_node.target.id in {'_if', '_while'}:
                    s = to_src(elt.ast_node.annotation)
                    predicates.append(("%s" if idx == 0 else "z3.Not(%s)") % s)
                elif isinstance(elt.ast_node.annotation, ast.Call):
                    assert elt.ast_node.annotation.func.id == self.fn_name
                else:
                    node = elt.ast_node
                    t = ast.Compare(node.target, [ast.Eq()], [node.value])
                    predicates.append(to_src(t))
            elif isinstance(elt.ast_node, ast.Assign):
                node = elt.ast_node
                t = ast.Compare(node.targets[0], [ast.Eq()], [node.value])
                predicates.append(to_src(t))
            else:
                pass
        return predicates
    def solve_path_constraint(self, path):
        # re-initializing does not seem problematic.
        # a = z3.Int('a').get_id() remains the same.
        constraints = self.extract_constraints(path)
#         print(constraints)
#         print(self.used_variables)
        decl = define_symbolic_vars(self.used_variables, '')
        exec(decl)
        solutions = {}
        with checkpoint(self.z3):
            st = 'self.z3.add(%s)' % ', '.join(constraints)
#             print(st, "hey this is")
            eval(st)
#             print(self.z3.check(), "Hey this is the result")
            if self.z3.check() != z3.sat:
                return {}
            m = self.z3.model()
#             print("this is m", m['x'])
            solutions = {}
            for d in m.decls():
#                 print(type(m[d]),"hey")
#                 print(m[d])
                sol = m[d]
#                 print(type(sol), "this is the type")
                if hasattr(sol, 'numerator_as_long'):
                    sol =  sol.numerator_as_long()/sol.denominator_as_long()
                else:
                    sol = str(sol)
                    if len(sol)>1 and sol[len(sol)-1] == '?':
                        sol = sol[:len(sol)-2]
                solutions[d.name()] = sol
            
#             solutions = {d.name(): m[d].numerator_as_long()/m[d].denominator_as_long() for d in m.decls()}
            my_args = {k: solutions.get(k, None) for k in self.fn_args}
        predicate = 'z3.And(%s)' % ','.join(
            ["%s == %s" % (k, v) for k, v in my_args.items()])
        eval('self.z3.add(z3.Not(%s))' % predicate)
        return my_args
    def get_next_path(self):
        self.last_path -= 1
        if self.last_path == -1:
            self.last_path = len(self.paths) - 1
        return self.paths[self.last_path]
    def fuzz(self):
        """Produce one solution for each path.
        Returns a mapping of variable names to (symbolic) Z3 values."""
        for i in range(self.max_tries):
#             print(self.get_next_path())
#             print(self.get_next_path())
            res = self.solve_path_constraint(self.get_next_path())
#             res = self.solve_path_constraint(self.paths[0])
#             print(res)
            if res:
                return res

        return {}
    def collect_branch_conditions(self):
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
            
    def collect_uncovered_branches(self):
        for key, value in self.conditions_covered.items():
            print(key.split('~'), value)
            if not value:
                branch_uncovered = [int(key.split('~')[0]), int(key.split('~')[1])]
                self.branches_uncovered.append(branch_uncovered)

                    
                    
                    
def traceit(frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        """Trace program execution. To be passed to sys.settrace()."""
    #     if event == 'line':
        if event == 'line':
            global coverage
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            coverage.append([lineno, function_name, event])

        return traceit


# In[454]:


MAX_DEPTH = 10000
MAX_TRIES = 100
MAX_ITER = 1000


# In[455]:


def sin_fn(x: float) -> float:
    return x - (x*x*x)/6
def fun(x: float) -> float:
    y: float = x - (x*x*x)/6
    cb:float = 2.5
    if y > 0:
        b:float = 3.5
    elif y < 0:
        cbc:float = 5.5
    else:
        cbx:float = 16.3
    # x ~ 0.8595900002387481
    if y == 0.9375:
        return 12
    return 0


# In[456]:


symfz_ct = SimpleSymbolicFuzzer(fun)


# In[457]:


symfz_ct = SimpleSymbolicFuzzer(fun)
all_paths = symfz_ct.get_all_paths(symfz_ct.fnenter)


# In[458]:


len(all_paths)
symfz_ct.collect_branch_conditions()
symfz_ct.branches
symfz_ct.conditions_covered


# In[459]:


symfz_ct = SimpleSymbolicFuzzer(fun)
all_paths = symfz_ct.get_all_paths(symfz_ct.fnenter)
symfz_ct.extract_constraints(all_paths[5])


# In[460]:


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


# In[461]:


def define_symbolic_vars(fn_vars, prefix):
    sym_var_dec = ', '.join([prefix + n for n in fn_vars])
    sym_var_def = ', '.join(["%s('%s%s')" % (t, prefix, n)
                             for n, t in fn_vars.items()])
    return "%s = %s" % (sym_var_dec, sym_var_def)


# In[463]:


x = None
symfz_ct = SimpleSymbolicFuzzer(fun)
symfz_ct.collect_branch_conditions()
print(symfz_ct.conditions_covered)
for i in range(1, 1000):
    args = symfz_ct.fuzz()
#     print(type(args['x']))
    global coverage
    coverage = []
    sys.settrace(traceit)  # Turn on
    fun(float(args['x']))
    sys.settrace(None) 
    for i in range(len(coverage)):
        if coverage[i][0] in symfz_ct.branches:
            #next line is executed
            if coverage[i][0] + 1 == coverage[i+1][0]:
                symfz_ct.conditions_covered[str(coverage[i][0]) + "~1"] = True
            else:
                symfz_ct.conditions_covered[str(coverage[i][0]) + "~0"] = True

print(symfz_ct.conditions_covered)
symfz_ct.collect_uncovered_branches()
print(symfz_ct.branches_uncovered)


# In[ ]:





# In[ ]:




