import math
import numpy as np
from sympy import *
from numpy import linalg as la

def goldstein_armijo_line_search(f, dims, st):
    '''The parameters to the goldstein_armijo function to compute lambda_k are
    f - objective function
    dims - list of variables in function
    start - starting point;
    Returns coeffecient of descent lambda_k'''
    fn = Matrix([f])
    f = lambdify([dims], fn, "numpy")                   # Convert f to a lambda function for easy math processing
    initial_start = Matrix(st)
    start = initial_start
    grad = fn.jacobian([dims]).transpose()
    g = lambdify([dims], grad, "numpy")                 # Convert grad to a lambda function for easy math processing
    max_iter = 50                                       # Setting the maximum iterations to 50
    alpha = math.exp(-4)                                # Setting the alpha value to e-4
    lambda_k = 0.5                                      # Setting the initial lambda value to 1/2
    eta = 0.9                                           # Setting the eta value which is used in the second condition to 0.9
    dk = -1 * grad                                      # direction of descent = -grad(xk)
    d = lambdify([dims], dk, "numpy")                     # Convert dk to a lambda function for easy math processing
    l_iter = 1
    output_dict = {}
    for i in range(1, max_iter):
        # Goldstein-Armijo Condition #1 - f(xk + lambda*dk) <= f(xk) + alpha*lambda*dk'*grad(xk)
        rhs_1 = f(st) + alpha * np.power(lambda_k, l_iter) * d(st).transpose().dot(g(st))
        next_pt = start + np.power(lambda_k, l_iter) * d(st)    #Set next point to xk + lambda_k*dk
        next_pt_array = np.array(next_pt.tolist()).astype(np.float64)
        lhs_1 = f(next_pt_array)
        if lhs_1 <= rhs_1:
            # If Condition 1 is satisfied check Condition 2:
            # dk'*grad(xk+1) >= eta*dk'*grad(xk)
            rhs_2 = eta * d(st).transpose().dot(g(st))
            lhs_2 = d(st).transpose().dot(np.squeeze(g(next_pt_array), axis=1))
            if lhs_2 >= rhs_2:
                lambda_k = np.power(lambda_k, l_iter)   # If both the conditions are satisfied finalize the lambda value
                break
        else:
            l_iter += 1                                 # If lambda value does not satisfy the 1st condition, trying with 1/2^i in the next iteration
    return lambda_k


def descent_direction_verification(f, dims, start, direction):
    '''The parameters to the descent_direction_verification function to verify the direction of descent are
    f - objective function
    dims - list of variables in function
    start - starting point
    direction - direction vector
    Returns whether the direction vector chosen is a direction of descent'''
    fn = Matrix([f])
    grad = fn.jacobian([dims]).transpose()
    g = lambdify([dims], grad, "numpy")
    dk = Matrix(direction)
    if dk.transpose().dot(g(start)) < 0:
        print('The point {} is verified as a direction of descent because'
              ' product of direction vector and gradient is less than 0'.format(direction))
    else:
        print('Product of direction vector and gradient is greater than 0. Choose a different direction of descent')

def cauchy_steepest_descent(fnc, dims, start):
    '''The parameters to the cauchy function to compute lambda_k are
    f - objective function
    dims - list of variables in function
    start - starting point;
    Returns dictionary containing the initial starting point,
    minimum point, number of iterations, lambda value and minimum value of function'''
    fn = Matrix([fnc])
    f = lambdify([dims], fn, "numpy")                           # Convert f to a lambda function for easy math processing
    initial_start = Matrix(start)
    lambda_k = goldstein_armijo_line_search(fnc, dims, start)   #Calling Goldstein armijo function to determine lambda_k
    start = initial_start
    start_array = np.array(start.tolist()).astype(np.float64)
    grad = fn.jacobian([dims]).transpose()
    g = lambdify([dims], grad, "numpy")                         # Convert grad to a lambda function for easy math processing
    max_iter = 1
    dk = -1 * grad                                              # direction of descent = -grad(xk)
    d = lambdify([dims], dk, "numpy")
    while max_iter <= 1000:
        if la.norm(g(start_array))/la.norm((1+f(start_array))) >= math.exp(-8):
            start_array = start_array + lambda_k * np.squeeze(d(start_array),axis=1)      #np.squeeze is used to re-shape the start array 
            max_iter+=1
        else: break
    output_dict = {'start':initial_start.tolist(), 'new':np.squeeze(start_array), 'iterations': max_iter, 'lambda': lambda_k, 'final_val': np.squeeze(f(start_array))}
    return output_dict

################
# Example of function with 2 variables x,y
################

x,y = Symbol('x'), Symbol('y')                                      #Symbolic representation of x and y
f =  2*x**2 + y**2 - 2*x*y + 2*x**3 +x**4                           #Setting objective functions
start = [0,-2]                                                      #Setting the starting point
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())     #Setting the sorted list of symbols dims = [x,y]
c = cauchy_steepest_descent(f, dims, start)                         #Calling the cauchy's method of steepest descent
print(f)
direction = [0,1]                                                   #Setting the direction vector for verification
descent_direction_verification(f, dims, start, direction)           #Verify direction of descent
print("Starting point at {}".format(c.get('start')))
print("Number of iterations = {}".format(c.get('iterations')))
print("Minimum point is {}".format(c.get('new')))                   #Printing the minimum point
print("Value of the function is {}".format(c.get('final_val')))     #Printing the minimum value of the function
