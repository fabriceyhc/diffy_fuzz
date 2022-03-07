from target_programs.functions_to_approximate import pythagorean_fn

def program_5(x: float, y: float):
   z:float = pythagorean_fn(x, y)
   print(x, y)

   # x ~ [0.5, 0.75]
   if round(z, 6) == 0.901388:
      raise Exception("You found a hard-to-reach bug!")
