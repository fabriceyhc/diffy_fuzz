import math
import numpy as np

#################################################################
# "simple" math functions (regression) ##########################
#################################################################

def sin_fn(x):
    return np.sin(x)

def square_fn(x):
    return x**2

def log_fn(x):
    return np.log(abs(x))

def poly_fn(x):
    return 20*x + 3*x**2 + 0.1*x**3

def pythagorean_fn(a, b):
    return np.sqrt(a ** 2 + b ** 2)

def fahrenheit_to_celcius_fn(x):
    return (5/9) * (x-32)

def dl_textbook_fn(x):
    return (0.2) + \
           (0.4 * x**2) + \
           (0.3 * np.sin(15 * x)) + \
           (0.05 * np.cos(50 * x))

# def laplacian(img):
#     if len(img.shape) == 4:
#         img = img[0]
#     if len(img.shape) == 3 and img.shape[0] == 3:
#         img = transforms.Grayscale(num_output_channels=1)(img).squeeze()
#     laplace = [[0,1,0],
#                [1,-4,1],
#                [0,1,0]]
#     size = len(laplace)
#     height, width = img.shape

#     score = 0
#     for x in range(height - size + 1):
#         for y in range(width - size + 1):
#             result = 0
#             for i in range(size):
#                 for j in range(size):
#                     result += (img[x + i][y + j] & 0xFF) * laplace[i][j]
#             if (result > LAPLACE_THRESHOLD):
#                 score += 1
#     return score


# def weierstrass_fn(x, a = 0.1, b = 0):
#     if 0 >= a >= 1:
#         raise ValueError("Variable `a` must be in range (0, 1) non-inclusive.")
#     # set b
#     min_b = (1 + ((3 * math.pi) / 2)) / a
#     if b < min_b:
#         while b % 2 == 0:
#             b = random.uniform(min_b, min_b + 1) # sys.maxsize

#     return np.sum([(a**n) * (math.cos((b**n) * math.pi * x)) for n in range(100)])

#################################################################
# discretized functions #########################################
#################################################################

def sin_disc_fn(x):
    z = np.sin(x)
    if 0 < z < 1:
        return 0
    else:
        return 1

def square_disc_fn(x):
    z = x**2
    if z < 1:
        return 0
    else:
        return 1

def log_disc_fn(x):
    z = np.log(abs(x))
    if z < 1:
        return 0
    else:
        return 1

def neuzz_fn(x):
    z = math.pow(3, x)
    if z < 1:
        return 0
    elif z < 2:
        return 1
    elif z < 4:
        return 2
    else:
        return 3

def fahrenheit_to_celcius_disc_fn(x):
    c = (5/9) * (x-32)
    if c < 0:
        return 0
    elif c < 100:
        return 1
    else:
        return 2

#################################################################
# compositions of functions #####################################
#################################################################

def log_sin_fn(x):
    z = np.log(np.sin(x))
    if abs(z) == np.inf:
        return 0
    else:
        return 1

def f_of_g_fn(x):
    f = lambda x: x - 2
    g = lambda y: y ** 2 - y 
    z = f(g(x))
    if z < 0:
        return 0
    else:
        return 1

def arcsin_sin_fn(x):
    z = np.arcsin(np.sin(x))
    if z == x:
        return 0
    else:
        return 1