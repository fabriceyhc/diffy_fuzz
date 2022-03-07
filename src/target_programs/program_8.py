from target_programs.functions_to_approximate import sin_disc_fn

def program_8(x: float):
   y:float = sin_disc_fn(x)
   
   if y == 0:
      print("sin_disc_fn was positive!")
   if y == 1:
      print("sin_disc_fn was negative or zero!")
