from target_programs.program_1 import program_1_sym
from target_programs.program_2 import program_2_sym
from target_programs.program_4 import program_4_sym
from target_programs.program_6 import program_6_sym
from target_programs.program_9 import program_9_sym
from target_programs.program_11 import program_11_sym
from target_programs.program_12 import program_12_sym
from target_programs.program_14 import program_14_sym
from SymbolicFuzzer import SimpleSymbolicFuzzer

if __name__ == '__main__':
    # symfz_ct = SimpleSymbolicFuzzer(program_1_sym)
    # symfz_ct.start_execution(tries=100)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_2_sym)
    # symfz_ct.start_execution(tries=100)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_4_sym)
    # symfz_ct.start_execution(tries=100)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_6_sym, precision = 0)
    # symfz_ct.start_execution(tries=10)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_9_sym, precision = 4, external_func_length = 5)
    # symfz_ct.start_execution(tries=100)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_11_sym, precision = 2, external_func_length = 9)
    # symfz_ct.start_execution(tries=0)
    # print(symfz_ct.branches_uncovered)
    # symfz_ct = SimpleSymbolicFuzzer(program_12_sym, precision = 2, external_func_length = 7)
    # symfz_ct.start_execution(tries=0)
    # print(symfz_ct.branches_uncovered)
    symfz_ct = SimpleSymbolicFuzzer(program_14_sym, precision = 0, external_func_length = 5)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)