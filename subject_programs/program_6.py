import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import fahrenheit_to_celcius_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into fahrenheit_to_celcius_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for fahrenheit_to_celcius_fn(x):"))

   y = fahrenheit_to_celcius_fn(x)
   
   if y > 0:
      print("You've got water!")
   elif y < 0:
      print("You've got ice!")
   else:
      print("You also have ice!")

   # x ~ 212
   if y == 100:
      raise Exception("You found a hard-to-reach bug (and steam)!")