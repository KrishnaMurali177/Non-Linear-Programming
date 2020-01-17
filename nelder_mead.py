import math
import numpy as np
from sympy import *
import numpy.linalg as la                                                   #Required imports

def sort_vert(f, dims, vertices):
    '''Sorts the vertices in the ascending order of the function value'''
    fnc = lambdify([dims], f, "numpy")
    vertices = sorted(vertices, key = lambda x: fnc(x))
    return vertices

def validate_mod_vert(vertices):
    '''Vaidates the vertices and modifies them if the are over the 10 radius ball'''
    a,b = Symbol('a'), Symbol('b')
    z = a**2 + b**2
    dims = sorted(list(z.free_symbols), key=lambda x: x.sort_key())
    zn = lambdify([dims], z, "numpy")
    for i in range(0,len(vertices)):
        if zn(vertices[i]) > 100:
            x_new = sqrt((100*vertices[i][0]**2)/zn(vertices[i]))
            y_new = sqrt(100-x_new**2)
            if x_new < 0: x_new = -x_new
            if y_new < 0: y_new = -y_new
            vertices[i] = [x_new,y_new]
    return vertices

def stopping_criteria(f, dims, vertices):
    '''Evaluates the stopping criteria'''
    fnc = lambdify([dims], f, "numpy")
    value = 0
    vertices = sort_vert(f,dims,vertices)
    for i in range(0,len(vertices)):
        value = value + (fnc(vertices[i])-fnc((vertices[0]+vertices[1])/2))**2
    return sqrt(value/len(vertices))

def nelder_mead(f,dims,vertices):
    '''The parameters to the nelder_mead function to compute minimum are
    f - the objective function
    dims - list of variables in the function
    vertices - starting vertices list;
    Returns dictionary containing the
    minimum point, number of iterations and the final value table at each simplex'''
    fnc = lambdify([dims], f, "numpy")                              # Lambdifying for easy numpy processing
    output_dict = {}
    output_dict.update({'start_vert':vertices})
    vertices = sort_vert(f,dims, vertices)                          # Sorting the vertices
    alpha = 2                                                       # reflection coefficient
    gamma = 2                                                       # expansion coefficient
    beta = 0.2                                                      # contraction coefficient
    sigma = 0.5                                                     # shrink coefficient
    epsilon = 0.002                                                  # Tolerance
    max_iter = 50
    n = 0
    vert_list = {}
    for i in range(0,max_iter):
        vertices = validate_mod_vert(vertices)                      #Validating and modifying vertices
        if stopping_criteria(f,dims,vertices) <= epsilon: break     #Check if stopping criteria is satisfied
        ref = (1+alpha)*((vertices[0]+vertices[1])/2) - alpha*vertices[2]   #Computing the reflection point
        if fnc(vertices[0]) <= fnc(ref) and fnc(ref) <= fnc(vertices[2]):
            vertices[2] = ref                                               #Setting xh as xr
        if fnc(ref) <= fnc(vertices[0]):
            exp = gamma*ref + (1-gamma)*((vertices[0]+vertices[1])/2)       #Computing the expansion point
            if fnc(ref) >= fnc(exp):
                vertices[2] = exp                                           #Setting xh to xe
            else:
                vertices[2] = ref                                           #Setting xh to  xr
        if fnc(vertices[2]) <= fnc(ref):
            con = (1-beta)*((vertices[0]+vertices[1])/2) + beta*vertices[2] #Computing the contraction point
            if fnc(con) <= fnc(vertices[2]):
                vertices[2] = con                                           #Setting xh to xc
        vertices = sort_vert(f, dims, vertices)                             #Sorting the vertices again for next iteration
        vertices = np.array(vertices).squeeze()
        vert_list.update({n:[vertices, fnc(vertices[0]).squeeze()]})        #Updating the final dictionary
        n = n+1
    output_dict.update({'vert': vertices, 'vert_list':vert_list, 'iterations': n})
    return output_dict

#Example
x,y = Symbol('x'), Symbol('y')
f = x - y + 2*x**2 + 2*x*y                                                  #Setting the objective function
dims = sorted(list(f.free_symbols), key=lambda x: x.sort_key())             #Getting list of dimensions
vertices = np.array([[5,4],[4,4], [4,5]])                                   #Setting the value of the vertices
min = nelder_mead(f,dims,vertices)                                          #Calling nelder_mead method
print('Starting vertices = {}'.format(min.get('start_vert')))
print('The minimum point is {}'.format(min.get('vert')[0]))
print('The table with vertices and simplex values at each point = {}'.format(min.get('vert_list')))
print('Number of iterations = {}'.format(min.get('iterations')))
print('------------------')
print('Question 3(ii)')
f1 = x**2 + 2*y**2 - 2*y - 2*x*y
min1 = nelder_mead(f1,dims,vertices)
print('Starting vertices = {}'.format(min1.get('start_vert')))
print('The minimum point is {}'.format(min1.get('vert')[0]))
print('The table with vertices and simplex values at each point = {}'.format(min1.get('vert_list')))
print('Number of iterations = {}'.format(min1.get('iterations')))
