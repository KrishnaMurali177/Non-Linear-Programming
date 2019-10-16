import math
from sympy import *  #Sympy library is needed for the symbolic representation of functions and differentiations

def newton_method(f,start):
    '''The parameters to the newton_method function to compute the minimum are
    f - objective function
    start - starting point'''
    func = lambdify(x, f, "math")               # Convert f to a lambda function for easy math processing
    fgradient = f.diff(x)                       # Gradient of the objective function with respect to x
    gradient = lambdify(x, fgradient, "math")   # Convert gradient to a lambda function for easy math processing
    fhessian = fgradient.diff(x)                # Hessian of the objective function with respect to x
    hessian = lambdify(x, fhessian, "math")     # Convert hessian to a lambda function for easy math processing
    n = 0
    error_x = 1.0                                            #Set absolute difference of x(k) and x(k+1) as 1
    error_fx = 1.0                                           #Set absolute difference of f(x(k)) and f(x(k+1)) as 1
    x_prev = start                                           #Set the value of x_prev as the starting value
    x_next = 0
    while error_fx and error_x > math.exp(-8):
        n=n+1
        try:                                                                #Try block to gracefully exit if second derivative is 0
            x_next = x_prev - gradient(x_prev)/hessian(x_prev)              #x(k) = x(k-1) - (f'(x(k-1)) / f"(x(k-1))
            error_x = abs(x_next - x_prev)                                  # delta x* = |x(k) - x(k-1)|
            error_fx = abs(func(x_next) - func(x_prev))                     # delta f*(x) = |f(x(k)) - f(x(k-1))|
            x_prev = x_next
        except ZeroDivisionError:
            print("Second Derivative is zero. Newton's method is not possible for this objective function")
            break
        except OverflowError:
            print("Value of the function tends to infinity at the successive points. Try a different starting point")
            x_prev = None
            break
    print("Number of iterations = {}".format(n))
    return x_prev

###################
#Example#
###################

x = Symbol('x')                                 #Representing x as a symbol
f1 = log(exp(x)+exp(-x))
print("Solution for {}".format(f1))
start = [0,0.5,1]                               #Passing a list of starting points
for i in start:
    print("Starting point at {}".format(i))
    minimum = newton_method(f1,i)
    print("Minimum point is {}".format(minimum))                          #Printing the minimum point
    print("Value of the function is {}".format(f1.subs(x,minimum)))       #Printing the minimum value of the function
print("----------------------")

f1 = log(exp(x)+exp(-x))
start = 1.1                                     #Passing a starting point of 1.1 makes the Newton's method to overshoot
print("Starting point at {}".format(start))
minimum = newton_method(f1,start)
print("Minimum point is {}".format(minimum))                          #Printing the minimum point
print("Value of the function is {}".format(f1.subs(x,minimum)))       #Printing the minimum value of the function


