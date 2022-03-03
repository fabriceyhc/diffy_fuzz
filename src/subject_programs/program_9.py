import argparse

from functions_to_approximate import square_disc_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into square_disc_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for square_disc_fn(x):"))

   y = square_disc_fn(x)
   
   if y == 0:
      print("square_disc_fn was less than 1!")
   if y == 1:
      print("square_disc_fn was greater than or equal to 1!")
