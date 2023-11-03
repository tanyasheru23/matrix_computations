import numpy as np
import matplotlib.pyplot as plt
import linearSystemMethods.jacobi_solver as jacobi
import linearSystemMethods.sor_solver as sor
import linearSystemMethods.gauss_seidel_solver as gauss_seidel

#given system of linear equations
'''
7x + 2y = 10
2x + 5y = 12

'''
A = np.array([[7, 2], [2, 5]])
b = np.array([10, 12])

omega = 1.25  

n = len(b)

#storing solutions, no of iterations and error history
# Solving the equations by defined SOR Method
solution, iterations, error_history = sor.sor_solver(A, b, omega,np.zeros(n))
print(f"SOR Solution: {solution} in {iterations} iterations")


#calculating y values according to the given equations
x_values = np.linspace(0, 5, 100)
y1_values = (10 - 7 * x_values) / 2
y2_values = (12 - 2 * x_values) / 5

plt.figure(figsize=(10, 5))
plt.plot(x_values, y1_values, label="7x + 2y = 10", color='blue')
plt.plot(x_values, y2_values, label="2x + 5y = 12", color='red')
plt.scatter(solution[0], solution[1], marker='x', color='green', label="SOR Solution")
plt.title("System of Equations and SOR Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Solve using Jacobi and Gauss-Seidel methods
jacobi_solution, jacobi_iterations, jacobi_error_history = jacobi.jacobi_solver(A, b)
print(f"Jacobi Solution: {jacobi_solution} in {jacobi_iterations} iterations")

n= len(b)
gauss_seidel_solution, gs_iterations, gs_error_history = gauss_seidel.gauss_seidel_solver(A, b,np.zeros(n))
print(f"Gauss-Seidel Solution: {gauss_seidel_solution} in {gs_iterations} iterations")

# Plot the solutions
plt.figure(figsize=(12, 5))
plt.plot(x_values, y1_values, label="7x + 2y = 10", color='blue')
plt.plot(x_values, y2_values, label="2x + 5y = 12", color='red')
plt.scatter(solution[0], solution[1], marker='x', color='green', label="SOR Solution")
plt.scatter(jacobi_solution[0], jacobi_solution[1], marker='o', color='orange', label="Jacobi Solution")
plt.scatter(gauss_seidel_solution[0], gauss_seidel_solution[1], marker='s', color='purple', label="Gauss-Seidel Solution")
plt.title("System of Equations and Solution Comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# (c) Show the convergence graphs of all three methods
plt.figure(figsize=(12, 5))
plt.plot(range(len(error_history)), error_history, label="SOR")
plt.plot(range(len(jacobi_error_history)), jacobi_error_history, label="Jacobi")
plt.plot(range(len(gs_error_history)), gs_error_history, label="Gauss-Seidel")
plt.title("Convergence of SOR, Jacobi, and Gauss-Seidel Methods")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()