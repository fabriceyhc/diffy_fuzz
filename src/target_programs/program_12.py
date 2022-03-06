def program_12_sym(x: float):
   c:float = (5/9) * (x-32)
   y:float = 0
   if c < 0:
        y = 0
   elif c < 100:
        y = 1
   else:
        y = 2
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   elif y == 2:
      r = 4.432
   
   return r

from target_programs.functions_to_approximate import fahrenheit_to_celcius_disc_fn

def program_12(x: float):
   y:float = fahrenheit_to_celcius_disc_fn(x)
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633
   elif y == 2:
      r = 4.432
   
   return r
