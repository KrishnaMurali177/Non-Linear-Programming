import math
import numpy as np
from numpy import linalg as la

def conjugate_gradient(a, b, start):
    '''The parameters to the conjugate_gradient function to solve ((xT * A * x)/2)  - bT * x are
    a - Matrix A in the form of a numpy array
    b - Matrix B in the form of a numpy array
    start - starting point;
    Returns dictionary containing the initial starting point,
    minimum point, number of iterations'''
    x_o = start
    r_o = b - a.dot(x_o)                                #Setting the value of r_o
    d_o = np.zeros(start.shape)                         #Setting the value of d_o
    d_new = r_o
    beta_new = 0
    alpha_o = r_o.transpose().dot(r_o)/(d_new.transpose().dot(a).dot(d_new))    #Computing alpha_o
    x_new = x_o + alpha_o*(d_new)                                               #Computing the next point
    r_new = r_o - alpha_o*(a).dot(d_new)                                        #Computing the new r value
    max_iter = 10                                                               #Setting the maximum iteration
    n = 0
    for i in range(0, max_iter):
        if la.norm(r_o) <= math.exp(-5):break                                   #Process until norm of r is less than tolerance
        else:
            beta_new = r_new.transpose().dot(r_new)/(r_o.transpose().dot(r_o))  #Computing the beta value
            d_new = r_new + beta_new*(d_o)
            alpha_new = r_new.transpose().dot(r_new)/(d_new.transpose().dot(a).dot(d_new))  #Computing new alpha
            x_new = x_new + alpha_new*(d_new)                                               #Computing the next point
            r_new = r_new - alpha_new*(a).dot(d_new)
            d_o = d_new
            r_o = r_new
            n = n+1
    output_dict = {'start': x_o, 'new': x_new, 'iterations': n}
    return output_dict

#Example
a = np.array([4,0,0,1,0,0,0,4,0,0,1,0,0,0,5,0,0,1,1,0,0,5,0,0,0,1,0,0,6,0,0,0,1,0,0,6])
a = a.reshape(6,6)
b = np.array([4,-8,16,1,-2,9])
b = b.reshape(6,1)
start = np.zeros((6,1))
c = conjugate_gradient(a,b,start)
print("Starting point at {}".format(c.get('start')))
print("Number of iterations = {}".format(c.get('iterations')))
print("Minimum point is {}".format(c.get('new')))
