def program_2_sym(x: float):
   y:float = x*x
   r:float = 0.0
   if y < x:
      r = 1.75
   elif y > x:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if y == 0.757575:
      raise Exception("You found a hard-to-reach bug!")
   
   return r
from target_programs.functions_to_approximate import square_fn

def program_2(x: float):
   y:float = square_fn(x)
   r:float = 0.0
   if y < x:
      r = 1.75
   elif y > x:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if round(y, 6) == 0.757575:
      raise Exception("You found a hard-to-reach bug!")
   
   return r
