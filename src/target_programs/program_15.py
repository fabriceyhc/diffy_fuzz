import argparse

from functions_to_approximate import arcsin_sin_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into arcsin_sin_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for arcsin_sin_fn(x):"))

   y = arcsin_sin_fn(x)
   
   if y == 0:
      print("arcsin_sin_fn: 0")
   if y == 1:
      print("arcsin_sin_fn: 1")
