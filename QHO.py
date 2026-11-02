import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from sympy import symbols, exp, I
from sympy.physics.quantum.constants import hbar

E = 5 # number of energi states 

x = np.linspace(-4, 4, 1000) # returns evenly spaced numbers over the given interval

# calculates the n'th eigenfunktion for QHO. Hermitepolynomials and normalization constant.
# we use hermite polynomials because they appear in the QHO is the normalized eigenfunctions are given by the expression we use in our function of psi
# (in units where m = omega = hbar = 1) cause we use natural dimensionalless units, so our eigenfunctions lose the konstants, and remains the same physics, with less algebraic noise.
def psi (n, x):
    Hn = hermite(n)(x)
    N = (2.0**n * factorial(n))**(-0.5)
    N *= np.pi**(-1/4)

    return N * Hn * np.exp(-x**2/2)

V = 0.5*x**2 # defining the potential

# setting plot size
plt.figure(figsize=(10, 6))
#plotting the potential.
plt.plot(x, V, 'k--', linewidth=2, alpha=0.25, label="Potential")

# Calculating the Eigenfunktions for the energi states.
for n in range(E):
    En = n + 0.5
    plt.plot(x, 0.4*psi(n,x)+ En, label=f"n={n}")

# labeling and showing the plot.
plt.xlabel(r"Dimensionless position: $\xi$ = x/$\sqrt{\hbar/(m\omega)}$")       # x-akse titel
plt.ylabel(r"Energy Scale E/($\hbar$$\omega$)")                                 # y-akse titel
plt.title("QHO: Potential and the first 5 energy states")                       # plot titel
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))

n = 60                              # for den n=60 energy stadie

x = np.linspace(-12, 12, 5000)      # hvor functionen er defineret

plt.plot(x, np.abs(psi(n,x))**2, label=r"$|\psi_{60}|^2$")      # probability density |\psi_{60}(x)|^2

plt.xlim(-15, 15)       # limit x-akse fra -15 til 15
plt.ylim(0, 0.2)        # limit y-akse fra 0 til 0.2

plt.xlabel(r"Dimensionless position: $\xi$")                # x-akse titel
plt.ylabel(r"$|\psi(\xi)|^2$")                              # y-akse titel
plt.title(r"Probability density $|\psi_{60}(\xi)|^2$")      # plot titel  
plt.legend()
plt.grid()

plt.show()
