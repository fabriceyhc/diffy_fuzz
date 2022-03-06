def program_14_sym(x: float):
   z:float = (x-2) ** 2 - x + 2
   y:float = 0
   if z < 0:
        y = 0
   else:
        y = 1
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633

   return r

from target_programs.functions_to_approximate import f_of_g_fn

def program_14(x: float):
   y:float = f_of_g_fn(x)
   r:float = 0.0
   if y == 0:
      r = 1.75
   elif y == 1:
      r = 0.633

   return r

