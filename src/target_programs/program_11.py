def program_11_sym(x: int):
   z:int = 3**x
   y:int = 0
   if z < 1:
      y = 0
   elif z < 2:
      y = 1
   elif z < 4:
      y = 2
   else:
      y = 3
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   elif y == 2:
      r = 4.432
   elif y == 3:
      r = 423.2332
   
   return r

from target_programs.functions_to_approximate import neuzz_fn

def program_11(x: int):
   y:int = neuzz_fn(x)
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   elif y == 2:
      r = 4.432
   elif y == 3:
      r = 423.2332
   
   return r
