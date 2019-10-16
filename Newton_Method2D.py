import math
from sympy import *         #Sympy library is needed for the symbolic representation of functions and jacobian computations

def newton_method2d(f1,f2,start_1, start_2):
    '''The parameters to the newton_method2d function to compute the minimum are
    f1, f2 - objective functions
    start_1 - starting point(1)
    start_2 - starting point(2)'''
    n = 0
    max_iter = 50
    fn = Matrix([f1,f2])                                                    #Setup function matrix
    start = Matrix([start_1,start_2])                                       #Setup starting point matrix
    jacobian = fn.jacobian([x,y])                                           #Compute Jacobian of function matrix
    if jacobian.subs({x: start[0], y: start[1]}).det() == 0:                #Check if Jacobian = 0
        print("Determinant of Jacobian is 0, Try different starting points")
        return None
    while n!=max_iter:
        d = (jacobian.subs({x:start[0],y:start[1]}).applyfunc(lambda x: float(x)).inv())        \
            *fn.subs({x:start[0],y:start[1]}).applyfunc(lambda x: float(x)) #Computing inverse(Jacobian(x))*f(x) and converting the values in the matrix to float by applyfunc method
        new = start - d                                     #x(k+1) = x(k) - f(x(k))/f'(x(k))
        start = new                                         #Set the starting point as the computed point
        n = n + 1                                           #Increment iterations by 1
        if d.norm() < math.exp(-8): break                   #Check if the norm of the distance matrix between old and new points is less than the tolerance(1e-8)
    print("Number of iterations = {}".format(n))            #Print number of iterations
    return float(start[0]), float(start[1])                 #Return minimum

#######################
#Example#
#######################

x,y = Symbol('x'), Symbol('y')                          #Symbolic representation of x and y
f1 = 3*x*y + 7*x + 2*y - 3                              #Setting objective functions
f2 = 5*x*y - 9*x - 4*y + 6
start_1 = 1                                             #Setting start points
start_2 = 1
x1,x2 = newton_method2d(f1,f2,start_1,start_2)          #Calling Newton_Method2D to compute minimum
print("x1 : {}".format(x1))                             #Printing the minimum point
print("x2 : {}".format(x2))
print("f1 value : {}".format(f1.subs({x:x1,y:x2})))
print("f2 value : {}".format(f2.subs({x:x1,y:x2})))


