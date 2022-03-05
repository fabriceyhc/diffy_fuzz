from target_programs.program_1 import program_1_sym
from target_programs.program_2 import program_2_sym
from SymbolicFuzzer import SimpleSymbolicFuzzer

if __name__ == '__main__':
    symfz_ct = SimpleSymbolicFuzzer(program_1_sym)
    symfz_ct.start_execution(tries=1000)
    print(symfz_ct.branches_uncovered)
    symfz_ct = SimpleSymbolicFuzzer(program_2_sym)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)