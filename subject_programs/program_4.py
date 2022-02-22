import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import poly_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into poly_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for poly_fn(x):"))

   y = poly_fn(x)
   print(x, y)
   
   if y > 0:
      print("poly_fn returned a positive value!")
   elif y < 0:
      print("poly_fn returned a negative value!")
   else:
      print("poly_fn returned zero!")

   # x ~ 0.5
   if y == 10.7625:
      raise Exception("You found a hard-to-reach bug!")