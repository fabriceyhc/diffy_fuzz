import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import sin_disc_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into sin_disc_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for sin_disc_fn(x):"))

   y = sin_disc_fn(x)
   
   if y == 0:
      print("sin_disc_fn was positive!")
   if y == 1:
      print("sin_disc_fn was negative or zero!")
