import numpy as np
import matplotlib.pyplot as plt

def sor_solver(A, b, omega, x0=None, max_iter = 100, tol=1e-6):
  n=len(b)
  if(x0.all()==None):
    x0 = np.zeros(n)
  x=x0.copy()
  errors = []
  for _ in range(max_iter):
    xnew = x.copy()
    for i in range(n):
      sigma = sum(A[i,j]*xnew[j] for j in range(n) if (j!=i))
      xnew[i] = (b[i]-sigma)*omega/A[i,i] + (1-omega)*xnew[i]
    error = np.linalg.norm(A@x - b)
    errors.append(error)
    err = np.linalg.norm(xnew-x)
    x = xnew
    if(err<tol):
      return x, _+1, errors
  return x, max_iter, errors