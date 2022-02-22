import sys
import os
import argparse

sys.path.insert(1, os.path.abspath("."))

from functions_to_approximate import dl_textbook_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into dl_textbook_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for dl_textbook_fn(x):"))

   y = dl_textbook_fn(x)
   print(y)
   
   if y > 0:
      print("dl_textbook_fn returned a positive value!")
   elif y < 0:
      print("dl_textbook_fn returned a negative value!")
   else:
      print("dl_textbook_fn returned zero!")

   # x ~ 0.45
   if round(y, 6) == 0.372348:
      raise Exception("You found a hard-to-reach bug!")