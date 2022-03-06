def program_1_sym(x: float):
   y:float = x - (x**3)/6
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if y == 0.757575:
      raise Exception("You found a hard-to-reach bug!")
   
   return r

from target_programs.functions_to_approximate import sin_fn

def program_1(x: float):
   y:float = sin_fn(x)
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if round(y, 6) == 0.757575:
      raise Exception("You found a hard-to-reach bug!")
   
   return r
