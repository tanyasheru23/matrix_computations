import numpy as np
import matplotlib.pyplot as plt

def jacobi_solver(A, b, tol = 1e-6, max_iter = 100):
  n = len(b)
  x = np.zeros(n)
  errors = []
  for k in range(max_iter):
    xnew = x.copy()
    for i in range(n):
      sigma = sum((A[i,j]*x[j]) for j in range(n) if (j!=i))
      xnew[i] = (b[i] - sigma)/A[i,i]
    error = np.linalg.norm(A@x - b)
    errors.append(error)
    err = np.linalg.norm(xnew-x)
    x = xnew
    if(err<tol):
      return x, k+1, errors
  return x, max_iter, errors
