import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from sympy import symbols, exp, I
from sympy.physics.quantum.constants import hbar
import os

# ------------------------------------------------------
# Kode for at gemme grafer i den rigtige fil.
# ------------------------------------------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(project_dir, "Graphs")

os.makedirs(graph_dir, exist_ok=True)

# ------------------------------------------------------

E = 8 # number of energi states 

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

#--------------------------------------------------------------
# --- first 5 energy states ---
#--------------------------------------------------------------

#plotting the potential.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

ax1.plot(x, V, 'k--', linewidth=2, alpha=0.5, label="Potential")

# Calculating the Eigenfunktions for the energi states.
for n in range(E):
    En = n + 0.5

    ax1.plot(x, 0.4*psi(n,x) + En, label=f"n={n}")

    xt = np.sqrt(2*En)

    ax1.hlines(En, -xt, xt,
               colors='k',
               linestyles='-')

    ax1.hlines(En, x.min(), -xt,
               colors='k',
               linestyles='--')

    ax1.hlines(En, xt, x.max(),
               colors='k',
               linestyles='--')

# labeling
ax1.set_title(r"Wavefunctions $\psi_n(x)$")
ax1.set_xlabel(r'$\xi$ = x/$\sqrt{\hbar/(m\omega)}$')
ax1.set_ylabel(r'$E/(\hbar\omega)$')
ax1.grid()
ax1.legend()

# possability distributions
ax2.plot(x, V, 'k--', linewidth=2, alpha=0.5, label="Potential")

for n in range(E):
    En = n + 0.5

    ax2.plot(
        x,
        0.4*np.abs(psi(n,x))**2 + En,
        label=f"n={n}"
    )

    xt = np.sqrt(2*En)

    ax2.hlines(En, -xt, xt,
               colors='k',
               linestyles='-')

    ax2.hlines(En, x.min(), -xt,
               colors='k',
               linestyles='--')

    ax2.hlines(En, xt, x.max(),
               colors='k',
               linestyles='--')


# labeling
ax2.set_title(r"Probability densities $|\psi_n(x)|^2$")
ax2.set_xlabel(r'$\xi$ = x/$\sqrt{\hbar/(m\omega)}$')
ax2.set_ylabel(r'$E/(\hbar\omega)$')
ax2.grid()
ax2.legend()


plt.tight_layout()

plt.savefig(
    os.path.join(graph_dir,
    "Figure_1_QHO_states_and_distributions.png"),
    dpi=300
)

#--------------------------------------------------------------
# --- probability density function ---
#--------------------------------------------------------------

plt.figure(figsize=(10,6))

n = 120                              # for den n=120 energy stadie

x = np.linspace(-17, 17, 5000)      # hvor functionen er defineret

plt.plot(x, np.abs(psi(n,x))**2, label=fr"$|\psi_{{{n}}}(x)|^2$")      # probability density |\psi_{60}(x)|^2

plt.xlim(-16, 16)       # limit x-akse fra -15 til 15
plt.ylim(0, 0.2)        # limit y-akse fra 0 til 0.2

plt.xlabel(r"Dimensionless position: $\xi$ = x/$\sqrt{\hbar/(m\omega)}$")                # x-akse titel
plt.ylabel(fr"$|\psi_{{{n}}}(x)|^2$")                              # y-akse titel
plt.title(fr"Probability density $|\psi_{{{n}}}(x)|^2$")      # plot titel 
plt.grid()
plt.legend()

plt.savefig(
    os.path.join(graph_dir,
    "Figure_1_probability_density_60.png"),
     dpi=300
)  
# gemmer grafen i foldern Graphs

plt.show()
