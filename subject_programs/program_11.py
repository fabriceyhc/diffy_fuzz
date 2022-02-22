import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import neuzz_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", nargs='+', type=float, help="values to pass into neuzz_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = input("Enter two values (separated by a space) for neuzz_fn(x):").split()
      x = [float(x_) for x_ in x]

   y = neuzz_fn(*x)
   
   if y == 0:
      print("neuzz_fn: 0!")
   if y == 1:
      print("neuzz_fn: 1!")
   if y == 2:
      print("neuzz_fn: 2!")
   if y == 3:
      print("neuzz_fn: 3!")
