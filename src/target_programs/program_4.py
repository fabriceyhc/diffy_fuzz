def program_4_sym(x: float):
   y:float = 20*x + 3*x**2 + 0.1*x**3
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if y == 10.762532:
      r = 1.0
   
   return r

from target_programs.functions_to_approximate import poly_fn

def program_4(x: float):
   y:float = poly_fn(x)
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if round(y, 6) == 10.762532:
      r = 1.0
   
   return r