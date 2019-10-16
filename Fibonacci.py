import math

def fibonacci(n):
    a,b = 1,1
    for i in range(n):
        a,b = b,a+b
    return a

def fibonacci_method(f,a,b):
    '''The parameters to the fibonacci_method function to compute the minimum are
    f - objective function
    a - starting point
    b - ending point'''
    n = 39  # Approximate value of n from the formula -8*ln(10)/ln(0.618)
    k = 1
    while k < n:
        l = a + (fibonacci(n-k-1)/fibonacci(n-k+1))*(b-a)     #Setting the value for lambda
        m = a + (fibonacci(n-k)/fibonacci(n-k+1))*(b-a)       #Setting the value for mu
        if f(l) > f(m):     #if the value of function at lambda is greater than the value of the function at mu
            a = l                                                   #Set value of a to lambda
            l = m                                                   #Set value of lambda to mu
            m = a + (fibonacci(n-(k+1))/fibonacci(n-(k+1)+1))*(b-a) #Compute the new mu value
        else:
            b = m                                                   #Set value of b to mu
            m = l                                                   #Set value of mu to lambda
            l = a + (fibonacci(n-(k+1)-1)/fibonacci(n-(k+1)+1))*(b-a)#Compute the new lambda value
        k = k+1                                                      #Increment k value by 1 for next iteration
    print("Number of iterations = {}".format(k-1))                   #Once outside the loop print the number of iterations
    return (a+b)/2                                                   #Return the minimum

#########################
#Example#
#########################
f = lambda x: (math.exp(-x)) + (x**2)     #Setting the objective function
a = 0
b = 100
minimum = fibonacci_method(f,a,b)      #Calling the fibonacci function to compute minimum
print("Minimum point is {}".format(minimum))                          #Printing the minimum point
print("Minimum value of the function is {}".format(f(minimum)))       #Printing the minimum value of the function
