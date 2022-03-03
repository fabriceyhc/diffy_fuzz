import argparse

from functions_to_approximate import log_sin_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into log_sin_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for log_sin_fn(x):"))

   y = log_sin_fn(x)
   
   if y == 0:
      print("log_sin_fn: 0")
   if y == 1:
      print("log_sin_fn: 1")
