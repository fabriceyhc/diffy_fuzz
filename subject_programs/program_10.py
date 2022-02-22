import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import log_disc_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into log_disc_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for log_disc_fn(x):"))

   y = log_disc_fn(x)
   
   if y == 0:
      print("log_disc_fn: 0")
   if y == 1:
      print("log_disc_fn: 1")
