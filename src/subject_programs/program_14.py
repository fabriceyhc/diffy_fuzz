import argparse

from functions_to_approximate import f_of_g_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into f_of_g_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for f_of_g_fn(x):"))

   y = f_of_g_fn(x)
   
   if y == 0:
      print("f_of_g_fn: 0")
   if y == 1:
      print("f_of_g_fn: 1")
