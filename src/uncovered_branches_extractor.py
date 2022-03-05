from target_programs.program_1 import *
from target_programs.program_2 import *

from SymbolicFuzzer import SimpleSymbolicFuzzer

if __name__ == '__main__':
    symfz_ct_program_1 = SimpleSymbolicFuzzer(program_1_sym)
    symfz_ct_program_1.start_execution(tries=1000)
    print(symfz_ct_program_1.branches_uncovered)

    symfz_ct_program_2 = SimpleSymbolicFuzzer(program_2_sym)
    symfz_ct_program_2.start_execution(tries=100)
    print(symfz_ct_program_2.branches_uncovered)