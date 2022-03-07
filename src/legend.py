from target_programs.program_1 import *
from target_programs.program_2 import *
from target_programs.program_3 import *
from target_programs.program_4 import *
from target_programs.program_5 import *
from target_programs.program_6 import *
from target_programs.program_7 import *
from target_programs.program_8 import *
from target_programs.program_9 import *
from target_programs.program_10 import *
from target_programs.program_11 import *
from target_programs.program_12 import *
from target_programs.program_13 import *
from target_programs.program_14 import *
from target_programs.program_15 import *

def get_subject_programs_config():
    return [
        # {'symbolic': False, 'symbolic_target_program': program_1, 'target_program': program_1, 'precision': 6, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_2_sym, 'target_program': program_2, 'precision': 6, 'external_func_length': 0},
        {'symbolic': False, 'symbolic_target_program': program_3, 'target_program': program_3, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_4_sym, 'target_program': program_4, 'precision': 6, 'external_func_length': 0},
        # {'symbolic': False, 'symbolic_target_program': program_5, 'target_program': program_5, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_6_sym, 'target_program': program_6, 'precision': 2, 'external_func_length': 0},
        # {'symbolic': False, 'symbolic_target_program': program_7, 'target_program': program_7, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': False, 'symbolic_target_program': program_8, 'target_program': program_8, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_9_sym, 'target_program': program_9, 'precision': 6, 'external_func_length': 5},
        # {'symbolic': False, 'symbolic_target_program': program_10, 'target_program': program_10, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_11_sym, 'target_program': program_11, 'precision': 2, 'external_func_length': 9},
        # {'symbolic': True, 'symbolic_target_program': program_12_sym, 'target_program': program_12, 'precision': 2, 'external_func_length': 7},
        # {'symbolic': False, 'symbolic_target_program': program_13, 'target_program': program_13, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': True, 'symbolic_target_program': program_14_sym, 'target_program': program_14, 'precision': 0, 'external_func_length': 0},
        # {'symbolic': False, 'symbolic_target_program': program_15, 'target_program': program_15, 'precision': 0, 'external_func_length': 0}
    ]