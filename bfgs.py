import numpy as np
from sympy import *
import numpy.linalg as la                                   #Required imports

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
                break
        else:
            l_iter += 1                                 # If lambda value does not satisfy the 1st condition, trying with 1/2^i in the next iteration
    return lambda_k

def bfgs(f, dims, hessian_approximation, start):
    '''The parameters to the bfgs function to compute minimum are
        f - the objective function
        dims - list of variables in the function
        hessian_approximation - Identity Matrix of the size of dims
        start - Starting point;
        Returns dictionary containing the initial starting point,
        minimum point, number of iterations and the final value'''
    fn = Matrix([f])
    fnc = lambdify([dims], fn, "numpy")                     #Lambdifying for easy numpy processing
    grad = fn.jacobian([dims]).transpose()                  #Computing the gradient
    g = lambdify([dims], grad, "numpy")
    x_o = start
    j_o = hessian_approximation
    d_o = -1*j_o.dot(j_o.transpose()).dot(g(x_o).reshape(len(dims),1))      #Computing do from the gradient
    max_iter = 100
    n = 0
    x_dict = {}
    for i in range(0,max_iter):
        if la.norm(g(x_o))/(1+la.norm(fnc(x_o))) <= math.exp(-8): break     #Stop if norm(g(xo))/(1+norm(f(xo))) < tolerance
        else:
            lambda_k = goldstein_armijo_line_search(f,dims,x_o)             #Compute lambda using goldstein-armijo criteria
            x_new = x_o + lambda_k*d_o                                      #Computing next point
            s_new = lambda_k*d_o                                            #Compute s_new
            y_new = g(x_new).reshape(len(dims),1) - g(x_o).reshape(len(dims),1)     #Compute y_new
            j_new = j_o + (s_new.dot(s_new.transpose()))/(s_new.transpose().dot(y_new)) - \
                    (j_o.dot(y_new).dot(y_new.transpose()).dot(j_o))/(y_new.transpose().dot(j_o).dot(y_new))    #Compute j_new
            x_o = x_new
            d_new = -1*j_new.dot(j_new.transpose()).dot(g(x_new).reshape(len(dims),1))      #Compute d_new
            d_o = d_new
            n = n+1
            x_dict.update({n:{'Point': x_new, 'value':fnc(x_new).squeeze()}})
    output_dict = {'start': start, 'new': x_new, 'iterations': n, 'list':x_dict}
    return output_dict


#Example
x,y = Symbol('x'), Symbol('y')
print('Question (i)')
f = x**2 + 2*y**2 - 2*x*y - 2*y                                          #Setting the objective function
start = np.zeros((2,1))                                                  #Setting the starting point
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())
hess = np.array([1,0,0,1]).reshape(2,2)                                  #Setting the hessian as the identity matrix
c = bfgs(f,dims, hess,start)                                             #Calling bfgs method
print("Starting point at {}".format(c.get('start')))
print("Number of iterations = {}".format(c.get('iterations')))
print("Minimum point is {}".format(c.get('new')))
print("Points are {}".format(c.get('list')))
print('-------------------------')
print('Question (ii)')
for c in [1,10,100]:
    f1 = (x-1)**2 + (y-1)**2 + c*(x**2 + y**2 - 0.25)**2
    start1 = np.array([1,-1]).reshape(2,1)
    dims1 = sorted(list(f1.free_symbols), key=lambda x: x.sort_key())
    hess1 = np.array([1,0,0,1]).reshape(2,2)
    c1 = bfgs(f1,dims1, hess1,start1)
    print("c value = {}".format(c))
    print("Starting point at {}".format(c1.get('start')))
    print("Number of iterations = {}".format(c1.get('iterations')))
    print("Minimum point is {}".format(c1.get('new')))
    print("Points are {}".format(c1.get('list')))