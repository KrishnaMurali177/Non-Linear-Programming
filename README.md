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
  
## Neweton's method for 1-Dimensional functions
Works only for functions which have atleast 2 derivatives. The successive points is traversed through the formula Xnew = Xold - f'(Xold)/ f''(Xold)

### Dependency libraries
- sympy

### Parameters
The parameters to the newton_method function to compute the minimum are
  - f - objective function
  - a - starting point

## Neweton's method for 2-Dimensional functions
The successive points is traversed through the formula x(k+1) = x(k) - f(x(k))/f'(x(k)) where x is the matrix of starting points and f is the matrix of the objective functions

### Dependency libraries
- sympy

### Parameters
The parameters to the newton_method2d function to compute the minimum are
  - f1, f2 - objective functions
  - start_1 - starting point(1)
  - start_2 - starting point(2)
