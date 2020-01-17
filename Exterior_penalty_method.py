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
    max_iter = 100                                       # Setting the maximum iterations to 50
    alpha = math.exp(-4)                                # Setting the alpha value to e-4
    lambda_k = 0.5                                      # Setting the initial lambda value to 1/2
    eta = 0.9                                           # Setting the eta value which is used in the second condition to 0.9
    dk = -1 * grad                                      # direction of descent = -grad(xk)
    d = lambdify([dims], dk, "numpy")                     # Convert dk to a lambda function for easy math processing
    l_iter = 1
    flag = False
    output_dict = {}
    for i in range(1, max_iter):
        # Goldstein-Armijo Condition #1 - f(xk + lambda*dk) <= f(xk) + alpha*lambda*dk'*grad(xk)
        rhs_1 = np.squeeze(f(st)) + alpha * np.power(lambda_k, l_iter) * d(st).reshape(len(dims),1).transpose().dot(g(st).reshape(len(dims),1))
        next_pt = st + np.power(lambda_k, l_iter) * d(st).reshape(len(dims),1)    #Set next point to xk + lambda_k*dk
        next_pt_array = np.array(next_pt.tolist()).astype(np.float64)
        lhs_1 = f(next_pt_array)
        if lhs_1 <= rhs_1:
            # If Condition 1 is satisfied check Condition 2:
            # dk'*grad(xk+1) >= eta*dk'*grad(xk)
            rhs_2 = eta * d(st).reshape(len(dims),1).transpose().dot(g(st).reshape(len(dims),1))
            lhs_2 = d(st).reshape(len(dims),1).transpose().dot(g(next_pt_array).reshape(len(dims),1))
            if lhs_2 >= rhs_2:
                lambda_k = np.power(lambda_k, l_iter)   # If both the conditions are satisfied finalize the lambda value
                flag = True
                break
        else:
            l_iter += 1                                 # If lambda value does not satisfy the 1st condition, trying with 1/2^i in the next iteration
    if flag:
        return lambda_k
    else: return np.power(lambda_k, l_iter)

def bfgs(f, f1, f2, dims, c):
    '''The parameters to the bfgs function to compute minimum are
        f - objective function
        f1 - Constraint function 1
        f2 - Constraint function 2
        dims - list of variables in function
        c - penalty;
        Returns dictionary containing minimum point, number of iterations and list of points traversed'''
    fnc1 = lambdify([dims], f1, "numpy")
    fnc2 = lambdify([dims], f2, "numpy")
    x_o = np.array([1,1]).reshape(2,1)
    if fnc1(x_o) < 0: f1 = 0
    if fnc2(x_o) < 0: f2 = 0
    fn = Matrix([f + c*(f1**2 + f2**2)])
    fnc = lambdify([dims], fn, "numpy")                                     # Lambdifying for easy numpy processing
    grad = fn.jacobian([dims]).transpose()                                  # Computing the gradient
    g = lambdify([dims], grad, "numpy")
    j_o = np.array([1,0,0,1]).reshape(2,2)                                  #Hessian approximation to identity matrix
    d_o = -1*j_o.dot(j_o.transpose()).dot(g(x_o).reshape(len(dims),1))      #Computing d_o from the gradient
    max_iter = 100
    n = 0
    x_dict = {}
    for i in range(0,max_iter):
        if la.norm(g(x_o))/(1+la.norm(fnc(x_o))) <= math.exp(-8): break     #Stop if norm(g(xo))/(1+norm(f(xo))) < tolerance
        else:
            lambda_k = goldstein_armijo_line_search(fn,dims,x_o)             #Compute lambda using goldstein-armijo criteria
            x_new = x_o + lambda_k*d_o                                      #Computing next point
            s_new = lambda_k*d_o                                            #Compute s_new
            y_new = g(x_new).reshape(len(dims),1) - g(x_o).reshape(len(dims),1)     #Compute y_new
            # j_new = j_o + (s_new.dot(s_new.transpose()))/(s_new.transpose().dot(y_new)) - \
            #        (j_o.dot(y_new).dot(y_new.transpose()).dot(j_o))/(y_new.transpose().dot(j_o).dot(y_new))    #Compute j_new
            j_new = j_o + ((s_new-j_o.dot(y_new)).dot(s_new.transpose()) + s_new.dot((s_new - j_o.dot(y_new)).transpose()))/(y_new.transpose().dot(s_new))  \
                    - ((s_new-j_o.dot(y_new)).dot(y_new.transpose()).dot(s_new).dot((s_new).transpose()))/((y_new.transpose().dot(s_new))**2)
            x_o = x_new
            d_new = -1*j_new.dot(j_new.transpose()).dot(g(x_new).reshape(len(dims),1))      #Compute d_new
            d_o = d_new
            n = n+1
            x_dict.update({n:{'Point': x_new, 'value':fnc(x_new).squeeze()}})
    output_dict = {'new': x_new, 'iterations': n, 'list':x_dict}
    return x_new

def ext_pen_method(f,f1,f2,dims):
    '''The parameters to the ext_pen_method function to compute minimum by incrementing penalty in exponents of 10 are
        f - objective function
        f1 - Constraint function 1
        f2 - Constraint function 2
        dims - list of variables in function;
        Returns dictionary containing minimum point and function value'''
    fnc = lambdify([dims], f, "numpy")
    i = 1
    c = 10**i
    output_dict = {}
    while(c < 10**7):
        x_n = bfgs(f,f1,f2,dims,c)
        output_dict.update({i:{'point':x_n, 'fnc_value': fnc(x_n)}})
        i = i+1
        c = 10**i
    return output_dict

#Example
x,y = Symbol('x'), Symbol('y')
f = x**2 + y**2 - 6*x - 8*y + 10
f1 = 4*x**2 + y**2 - 16
f2 = 3*x + 5*y - 4
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())
min = ext_pen_method(f,f1,f2,dims)
print(min)
print('Minimum point is {}'.format(min[max(min.keys())]['point']))
print('Minimum value of function is {}'.format(min[max(min.keys())]['fnc_value']))