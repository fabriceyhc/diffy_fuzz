import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import pythagorean_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", nargs='+', type=float, help="values to pass into pythagorean_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = input("Enter two values (separated by a space) for pythagorean_fn(x):").split()
      x = [float(x_) for x_ in x]

   y = pythagorean_fn(*x)
   print(x, y)

   # x ~ [0.5, 0.75]
   if round(y, 6) == 0.901388:
      raise Exception("You found a hard-to-reach bug!")