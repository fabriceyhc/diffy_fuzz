from target_programs.functions_to_approximate import arcsin_sin_fn

def program_15(x: float):
   y:float = arcsin_sin_fn(x)
   
   if y == 0:
      print("arcsin_sin_fn: 0")
   if y == 1:
      print("arcsin_sin_fn: 1")
