import csv
from target_programs.program_2 import program_2_sym
from target_programs.program_4 import program_4_sym
from target_programs.program_6 import program_6_sym
from target_programs.program_9 import program_9_sym
from target_programs.program_11 import program_11_sym
from target_programs.program_12 import program_12_sym
from target_programs.program_14 import program_14_sym
from SymbolicFuzzer import SimpleSymbolicFuzzer

fields = ['Program', 'Branch Coverage', 'Tries', 'Execution Time(in seconds)'] 
rows = []

filename = "evaluation_results.csv"

if __name__ == '__main__':
    symfz_ct = SimpleSymbolicFuzzer(program_2_sym)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 2', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_4_sym)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 4', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_6_sym, precision = 2)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 6', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_9_sym, precision = 4, external_func_length = 5)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 9', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_11_sym, precision = 2, external_func_length = 9)
    symfz_ct.start_execution(tries=0)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 11', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_12_sym, precision = 2, external_func_length = 7)
    symfz_ct.start_execution(tries=0)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 12', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])
    symfz_ct = SimpleSymbolicFuzzer(program_14_sym, precision = 0, external_func_length = 5)
    symfz_ct.start_execution(tries=100)
    print(symfz_ct.branches_uncovered)
    rows.append(['Program 14', str(symfz_ct.calculate_branch_coverage())+"%", 100, str(symfz_ct.execution_time)+" seconds"])

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)