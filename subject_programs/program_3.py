import argparse

from functions_to_approximate import log_fn

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("--input", type=float, help="value to pass into log_fn()")
   args = parser.parse_args()

   x = args.input

   if not args.input:
      x = float(input("Enter a numeric value for log_fn(x):"))

   if x == 0:
      x = 1e-30
      
   y = log_fn(x)
   
   if y > 0:
      print("log_fn returned a positive value!")
   elif y < 0:
      print("log_fn returned a negative value!")
   else:
      print("log_fn returned zero!")

   # x ~ 0.5893
   if round(y, 6) == -0.528820:
      raise Exception("You found a hard-to-reach bug!")