from target_programs.functions_to_approximate import log_disc_fn

def program_10(x: float):
   y:float = log_disc_fn(x)
   
   if y == 0:
      print("log_disc_fn: 0")
   if y == 1:
      print("log_disc_fn: 1")
