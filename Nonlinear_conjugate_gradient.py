import math
import numpy as np
from sympy import *
from numpy.linalg import linalg as la

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
            lhs_2 = d(st).transpose().dot(np.squeeze(g(next_pt_array)).reshape(len(dims),1))
            if lhs_2 >= rhs_2:
                lambda_k = np.power(lambda_k, l_iter)   # If both the conditions are satisfied finalize the lambda value
                break
        else:
            l_iter += 1                                 # If lambda value does not satisfy the 1st condition, trying with 1/2^i in the next iteration
    return lambda_k


def nonlinear_cg(f, dims, start):
    '''The parameters to the nonlinear_cg function to compute minimum are
    f - the objective function
    dims - list of variables in the function
    start - starting point;
    Returns dictionary containing the initial starting point,
    minimum point, number of iterations and the final value'''
    fn = Matrix([f])
    fnc = lambdify([dims], fn, 'numpy')                                 #Lambdifying the function for numpy processing
    gradient = fn.jacobian([dims]).transpose()                          #Computing the gradient
    grad = lambdify([dims],gradient,'numpy')
    x_o = start.reshape(len(dims),1)                                    #Setting the values of x_o, d_o, r_o
    d_o = np.zeros((len(dims),1))
    r_o = -1*grad(start.squeeze().tolist())
    d_new = r_o
    beta_o = 0
    alpha = goldstein_armijo_line_search(f,dims,list(start.squeeze()))  #Using Goldstein-Armijo to determine alpha
    x_new = start.reshape(len(dims),1)+ alpha*d_new
    max_iter = 10
    n = 0
    pt_list = []
    for i in range(0,max_iter):
        if la.norm(grad(x_new)) < 0: break                              #Checking if the norm of gradient at the point is less than tolerance
        else:
            beta_new = np.squeeze(grad(x_new).transpose(),axis=1).dot(np.squeeze(grad(x_new),axis=1))/(np.squeeze(grad(x_o).transpose(),axis=1).dot(np.squeeze(grad(x_o),axis=1)))  #Computing beta value
            d_new = -1*grad(x_new.squeeze().tolist()) + beta_new*(d_o)  #Computing new d value
            x_o = x_new
            d_o = d_new
            alpha = goldstein_armijo_line_search(f, dims, list(x_o.squeeze()))  #Computing new alpha using Goldstein-Armijo criteria
            x_new = x_new.reshape(len(dims),1) + alpha*d_new
            n = n+1
            pt_list.append(x_new)
    output_dict = {'start': start, 'prev': x_o, 'new': x_new, 'iterations': n, 'final_value': fnc(x_new.squeeze().tolist()), 'point_list': pt_list}
    return output_dict


#Example
x,y,z,w = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('w')
f = (w - 2*x**2)**2 + (x - 2*y**2)**2 + (y - 2*z**2)**2
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())
start = np.array([1,1,1,1]).reshape(1,4)
c = nonlinear_cg(f,dims,start)
print("Starting point at {}".format(c.get('start')))
print("Number of iterations = {}".format(c.get('iterations')))
print("Min point at {}".format(c.get('new')))
print("Min value = {}".format(c.get('final_value')))
