import math

def bisection(f,a,b):
    '''The parameters to the bisection function to compute the minimum are
    f - objective function
    a - starting point
    b - ending point'''
    n = 0                               #Set counter of iterations to 0
    while abs(b-a)>=math.exp(-8):       #Absolute difference between a and b is checked with the tolerance level (1e-8)
        m = (a+b)/2                     #Mid point of a and b
        if f(m) < f(b):
            b = m                       #Set b to the value of mid point
        else:
            a = m                       #Set a to the value of mid point
        n = n+1                         #increment iteration count by 1
    print("Number of iterations = {}".format(n))
    return (a+b)/2                      #Return the minimum point

#########################
#Example#
#########################
f = lambda x: (math.exp(-x)) + (x**2)   #Setting the objective function
a = 0
b = 100
minimum = bisection(f,a,b)            #Calling the bisection method to compute minimum
print("Minimum point is {}".format(minimum))                          #Printing the minimum point
print("Minimum value of the function is {}".format(f(minimum)))       #Printing the minimum value of the function
