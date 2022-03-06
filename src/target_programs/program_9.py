def program_9_sym(x: float):
   z:float = x**2
   y:int = -1.0
   if z < 1:
      y = 0
   else:
      y = 1
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   
   return r

from target_programs.functions_to_approximate import square_disc_fn

def program_9(x: float):
   y:int = square_disc_fn(x)
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   
   return r
