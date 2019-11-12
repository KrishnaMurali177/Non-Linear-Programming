# Non-Linear-Programming
Methods in finding the minimums of a non linear function

## Bisection Method
Works for unimodal and psuedo-convex functions. This method traverses the non linear function and computes the minimum at successive points. The successive point is found using the mid-point formula.

### Parameters
The parameters to the bisection function to compute the minimum are
  - f - objective function
  -  a - starting point
  -  b - ending point

## Golden Section Method
Works for unimodal functions. This method traverses the non linear function and computes the minimum at successive points. The successive point is found using the golden number ratio.

### Parameters
The parameters to the golden_section function to compute the minimum are
  - f - objective function
  -  a - starting point
  -  b - ending point

## Fibonacci Method
Works for unimodal functions. This method traverses the non linear function and computes the minimum at successive points. The successive point is found using the Fibonnaci sequence of descent.

### Parameters
The parameters to the fibonacci_method function to compute the minimum are
  - f - objective function
  -  a - starting point
  -  b - ending point
  
## Newton's method for 1-Dimensional functions
Works only for functions which have atleast 2 derivatives. The successive points is traversed through the formula Xnew = Xold - f'(Xold)/ f''(Xold)

### Dependency libraries
- sympy

### Parameters
The parameters to the newton_method function to compute the minimum are
  - f - objective function
  - a - starting point

## Newton's method for multiple simultaneous functions
The successive points is traversed through the formula x(k+1) = x(k) - f(x(k))/f'(x(k)) where x is the matrix of starting points and f is the matrix of the objective functions

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the newton_method2d function to compute the minimum are
  - f - list of objective functions
  - dims - list of variables in the function
  - start - list of starting point
  
## Cauchy's Steepest Descent algorithm
Works for an objective function with n number of variables. The successive points is traversed through the formula Xnew = Xold + lambda * dk. There are 3 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for lambda
#### descent_direction_verification
Check function if direction vector is valid
#### cauchy_steepest_descent
Computes the minimum of the function using lambda obtained from Goldstein-Armijo criteria

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the cauchy_steepest_descent function to compute the minimum are
  - f - objective function
  - dims - list of variables in function
  - start - starting point

## Conjugate Gradient Method
Solving for x in ((xT * A * x)/2)  - bT * x using conjugate_gradient function where A and B are matrices

### Dependency libraries
- numpy

### Parameters
The parameters to the conjugate_gradient function to compute x are
- a - Matrix A in the form of a numpy array
- b - Matrix B in the form of a numpy array
- start - starting point
