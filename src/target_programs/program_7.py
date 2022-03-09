from target_programs.functions_to_approximate import dl_textbook_fn

def program_7(x: float):
   y:float = dl_textbook_fn(x)
   print(y)
   
   if y > 100:
      return "dl_textbook_fn returned a positive value!"
   elif y < 100:
      return "dl_textbook_fn returned a negative value!"

   if round(y, 0) == 100:
      raise Exception("You found a hard-to-reach bug!")