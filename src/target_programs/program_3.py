from target_programs.functions_to_approximate import log_fn

def program_3(x: float):      
   y:float = log_fn(x)
   
   if y > 0:
      print("log_fn returned a positive value!")
   elif y < 0:
      print("log_fn returned a negative value!")
   else:
      print("log_fn returned zero!")

   # x ~ 0.5893
   if y == -0.528820:
      raise Exception("You found a hard-to-reach bug!")