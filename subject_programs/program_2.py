import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import square_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into square_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for square_fn(x):"))

   y = square_fn(x)
   
   if y < x:
      print("square_fn made the input smaller!")
   elif y > x:
      print("square_fn made the input larger!")
   else:
      print("square_fn didn't have any effect at all!")

   # x ~ 0.87038784458424050305248922720075
   if round(y, 6) == 0.757575:
      raise Exception("You found a hard-to-reach bug!")