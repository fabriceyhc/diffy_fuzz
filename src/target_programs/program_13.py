from target_programs.functions_to_approximate import log_sin_fn

def program_13(x: float):
   y:float = log_sin_fn(x)
   
   if y == 0:
      print("log_sin_fn: 0")
   if y == 1:
      print("log_sin_fn: 1")
