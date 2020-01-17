import math
import numpy as np
from sympy import *
import numpy.linalg as la
from scipy.optimize import linprog

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
    lambda_k = 0.05                                      # Setting the initial lambda value to 1/2
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

def Zoutendijk(f,f1,f2,dims,start):
    '''The parameters to the Zoutendijk function to compute minimum are
        f - objective function
        f1 - Constraint function 1
        f2 - Constraint function 2
        dims - list of variables in function
        start - starting point;
        Returns dictionary containing minimum point and function value'''
    x_o = start                                                     #Set starting point
    fnc = lambdify([dims], f, "numpy")                              #Lambdifying for easy numpy processing
    fnc1 = lambdify([dims], f1, "numpy")
    fnc2 = lambdify([dims], f2, "numpy")
    grad = Matrix([f]).jacobian([dims]).transpose()
    grad1 = Matrix([f1]).jacobian([dims]).transpose()
    grad2 = Matrix([f2]).jacobian([dims]).transpose()
    g = lambdify([dims], grad, "numpy")
    g1 = lambdify([dims], grad1, "numpy")
    g2 = lambdify([dims], grad2, "numpy")
    max_iter = 7
    output_dict = {}
    for i in range(0,max_iter):
        if fnc1(x_o) < 0 and fnc2(x_o) < 0:
            s = -g(x_o).reshape(2,1)                                                    #Settin s
        elif fnc1(x_o) < 0 and fnc2(x_o) >=0:
            A = np.array([[g(x_o)[0], g(x_o)[1], -1],[g2(x_o)[0], g2(x_o)[1], -1]])     #Computing A for linear solving
            B = np.array([[-(g(x_o)[0]+g(x_o)[1]), -(g2(x_o)[0]+g2(x_o)[1])]])          #Computing B for linear solving
        elif fnc1(x_o) >= 0 and fnc2(x_o) >=0:
            A = np.array([[g1(x_o)[0], g1(x_o)[1], -1],[g2(x_o)[0], g2(x_o)[1], -1],[g(x_o)[0], g(x_o)[1], -1]])
            B = np.array([[-(g1(x_o)[0]+g1(x_o)[1]), -(g2(x_o)[0]+g2(x_o)[1]), -(g(x_o)[0]+g(x_o)[1])]])
        C = [0,0,1]
        x_min = linprog(C,A,B,bounds=(0,10000))                                         #Minimizing alpha as a linear program
        if x_min.get('x')[2] < 10**-4: break
        s = np.array([x_min.get('x')[0]-1,x_min.get('x')[1]-1]).reshape(2,1)
        lambda_k = goldstein_armijo_line_search(f,dims,x_o)                             #Calling goldstein armijo to compute lambda
        x_new = x_o + lambda_k*s                                                        #Computing next point
        output_dict.update({i:{'point':x_new,'fnc_value':fnc(x_new)}})
        if abs((fnc(x_o)-fnc(x_new))/fnc(x_o)) < 10**-4 and la.norm(x_new-x_o) < 10**-4: break
        x_o = x_new
    return output_dict

#Example
x, y = Symbol('x'), Symbol('y')
f = x ** 2 + y ** 2 - 6 * x - 8 * y + 10                                #Objective function
f1 = 4 * x ** 2 + y ** 2 - 16                                           #Constraints
f2 = 3 * x + 5 * y - 4
start = np.array([1.2,0.6]).reshape(2,1)
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())
min = Zoutendijk(f,f1,f2,dims,start)
print(min)
print('Minimum point is {}'.format(min[max(min.keys())]['point']))
print('Minimum value of function is {}'.format(min[max(min.keys())]['fnc_value']))
