def program_6_sym(x: float):
   y:float = (5/9) * (x-32)
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if y == 100:
      raise Exception("You found a hard-to-reach bug (and steam)!")
   
   return r

from target_programs.functions_to_approximate import fahrenheit_to_celcius_fn

def program_6(x: float):
   y:float = fahrenheit_to_celcius_fn(x)
   r:float = 0.0
   if y > 0:
      r = 1.75
   elif y < 0:
      r = 0.633
   else:
      r = 322.22

   # x ~ 0.8595900002387481
   if round(y, 0) == 100:
      raise Exception("You found a hard-to-reach bug (and steam)!")
   
   return r