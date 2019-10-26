import math
from sympy import *         #Sympy library is needed for the symbolic representation of functions and jacobian computations
import numpy as np
from numpy import linalg as la

def newton_method(f, dims, start):
    '''The parameters to the newton_method2d function to compute the minimum are
    f - list of objective functions
    dims - list of variables in the function
    start - list of starting point
    '''
    n = 0
    max_iter = 50
    fn = Matrix(f)                                                    #Setup function matrix
    f = lambdify([dims], fn, "numpy")
    start = Matrix(start)                                       #Setup starting point matrix
    start_array = np.array(start.tolist()).astype(np.float64)
    j = fn.jacobian([dims])                                           #Compute Jacobian of function matrix
    jacobian = lambdify([dims], j, "numpy")
    if la.det(np.squeeze(jacobian(start_array), axis = len(dims))) == 0:                #Check if Jacobian = 0
        print("Determinant of Jacobian is 0, Try different starting points")
        return None
    while n!=max_iter:
        d = la.inv((np.squeeze(jacobian(start_array), axis = len(dims)))).dot(np.squeeze(f(start_array), axis = len(dims))) #Computing inverse(Jacobian(x))*f(x) and converting the values in the matrix to float by applyfunc method
        new = start_array - d                                     #x(k+1) = x(k) - f(x(k))/f'(x(k))
        start_array = new                                         #Set the starting point as the computed point
        n = n + 1                                           #Increment iterations by 1
        if la.norm(d) < math.exp(-8): break                   #Check if the norm of the distance matrix between old and new points is less than the tolerance(1e-8)
    output_dict = {'start': start.tolist(), 'new': np.squeeze(start_array), 'iterations': n,
                    'final_val': np.squeeze(f(start_array))}
    return output_dict                 #Return dictionary


x,y = Symbol('x'), Symbol('y')                          #Symbolic representation of x and y
f1 = 3*x*y + 7*x + 2*y - 3                              #Setting objective functions
f2 = 5*x*y - 9*x - 4*y + 6
f = [f1, f2]
start = [1,1]                                                       #Setting start point
dims = sorted(list(f1.free_symbols), key=lambda x: x.sort_key())    #Setting the sorted list of symbols dims = [x,y]
nm = newton_method(f, dims, start)                                #Calling Newton_Method2D to compute minimum
print("Starting point at {}".format(nm.get('start')))
print("Number of iterations = {}".format(nm.get('iterations')))
print("Minimum point is {}".format(nm.get('new')))                          #Printing the minimum point
print("Value of the function is {}".format(nm.get('final_val')))            #Printing the minimum value of the function
