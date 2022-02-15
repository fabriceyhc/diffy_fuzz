import sys
import os

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import sin_fn

if __name__ == '__main__':
   x = float(input("Enter a numeric value for sin_fn(x):"))
   y = sin_fn(x)
   
   if y > 0:
      print("sin_fn returned a positive value!")
   elif y < 0:
      print("sin_fn returned a negative value!")
   else:
      print("sin_fn returned zero!")

   # x ~ 0.8595900002387481
   if round(y, 6) == 0.757575:
      raise Exception("You found a hard-to-reach bug!")