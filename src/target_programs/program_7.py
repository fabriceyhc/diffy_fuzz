from target_programs.functions_to_approximate import dl_textbook_fn

def program_7(x: float):
   y:float = dl_textbook_fn(x)
   print(y)
   
   if y > 0:
      print("dl_textbook_fn returned a positive value!")
   elif y < 0:
      print("dl_textbook_fn returned a negative value!")
   else:
      print("dl_textbook_fn returned zero!")

   # x ~ 0.45
   if round(y, 6) == 0.372348:
      raise Exception("You found a hard-to-reach bug!")