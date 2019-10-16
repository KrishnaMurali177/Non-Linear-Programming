import math

def golden_section(f,a,b):
    '''The parameters to the golden_section function to compute the minimum are
    f - objective function
    a - starting point
    b - ending point'''
    # Setting the Golden Numbers
    g = (3 - math.sqrt(5)) / 2
    gn = 1 - g
    n = 0                         #Setting the count of iterations to 0
    while abs(b-a)>=math.exp(-8): #Checking if the difference of a and b is greater than the tolerance level (1e-8)
        l = a + g*(b-a)           #Setting the value for lambda
        m = a + gn*(b-a)          #Setting the value of mu
        n = n+1                   #incrementing the number of iterations by 1
        if f(l) < f(m): #if the value of function at lambda is less than the value of the function at mu
            b = m                      #Set value of b to mu
            m = l                      #Set value of mu to lambda
            l = a + g*(b-a)            #Calculate the new lambda value
        else:
            a = l                      #Set value of a to lambda
            l = m                      #Set value of lambda to mu
            m = a + gn*(b-a)           #Calculate the new mu value
    print("Number of iterations = {}".format(n))   #Once outside the loop print the total number of iterations
    return (a+b)/2                                 #Return the minimum point

#############################
#Example#
#############################
f = lambda x: (math.exp(-x)) + (x**2)     #Setting the objective function
a = 0
b = 100
minimum = golden_section(f,a,b)        #Calling the golden_section function to compute minimum
print("Minimum point is {}".format(minimum))                          #Printing the minimum point
print("Minimum value of the function is {}".format(f(minimum)))       #Printing the minimum value of the function






