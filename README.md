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

## Nonlinear Conjugate Gradient algorithm
Works for an objective function with n number of variables. The successive points is traversed through the formula Xnew = Xold + alpha * dnew. There are 2 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for alpha
#### cauchy_steepest_descent
Computes the minimum of the function using alpha obtained from Goldstein-Armijo criteria

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the nonlinear_cg function to compute the minimum are
  - f - objective function
  - dims - list of variables in function
  - start - starting point
  
## BFGS algorithm
Works for an objective function with n number of variables. There are 2 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for alpha
#### bfgs
Computes the minimum of the function using alpha obtained from Goldstein-Armijo criteria

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the bfgs function to compute the minimum are
  - f - the objective function
  - dims - list of variables in the function
  - hessian_approximation - Identity Matrix of the size of dims
  - start - Starting point
  
## Exterior_penalty_method algorithm
Works for an objective function with 2 constraints with n number of variables. There are 3 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for alpha
#### bfgs
Computes the minimum of the function using alpha obtained from Goldstein-Armijo criteria
#### ext_pen_method
Computes the minimum of the function by running bfgs for increased penalty

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the ext_pen_method function to compute the minimum are
  - f - the objective function
  - f1 - Constraint function 1
  - f2 - Constraint function 2
  - dims - list of variables in the function

## Interior_penalty_method algorithm
Works for an objective function with 2 constraints with n number of variables. There are 4 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for alpha
#### check_constraints
Evaluates the constraints at the given point
#### bfgs
Computes the minimum of the function using alpha obtained from Goldstein-Armijo criteria
#### int_pen_method
Computes the minimum of the function by running bfgs for increased penalty

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the int_pen_method function to compute the minimum are
  - f - the objective function
  - f1 - Constraint function 1
  - f2 - Constraint function 2
  - dims - list of variables in the function
  
## Nelder_mead algorithm
Works for an objective function with n number of variables. Creates a triangular vertex and searches space for minimum. There are 4 functions inside the file.
#### sort_vert
Sorts the vertices
#### validate_mod_vert
Validates if the vertices are in the possible space
#### stopping_criteria
Evaluates stopping criteria
#### nelder_mead
Computes the minimum of the function

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the nelder_mead function to compute the minimum are
  - f - the objective function
  - dims - list of variables in the function
  - vertices - list of sorted vertices
  
## Zoutendijk algorithm
Works for constrained objective function with n number of variables. There are 2 functions inside the file.
#### goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for alpha
#### Zoutendijk
Computes the minimum of the function using alpha obtained from Goldstein-Armijo criteria

### Dependency libraries
- sympy
- numpy

### Parameters
The parameters to the Zoutendijk function to compute the minimum are
  - f - objective function
  - f1 - Constraint function 1
  - f2 - Constraint function 2
  - dims - list of variables in function
        start - starting point
