import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sc
import sympy as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.special import gamma

save__graph = False
save_figs   = False

"""i due modi per ottenere i param della weibull

## Funzioni
"""

# roba di supporto
def CDF_at_mean(dato):
    ecdf = sm.distributions.ECDF(dato)
    mu = np.mean(dato)
    return ecdf(mu)

def Gamma_Func(n, k):
    return sc.special.gamma(1 + n / k)

def fitWeibull_type2_fast(dato):
    dato = np.array(dato)
    mu = np.mean(dato)
    C = CDF_at_mean(dato)
    m3_non_central = sc.stats.moment(dato, moment=3)

    # Funzione da minimizzare: residuo tra CDF empirica e teorica
    def residual(k):
        gamma_1 = Gamma_Func(1, k)
        dx = 1 - np.exp(-gamma_1**k)
        return abs(C - dx)

    # Minimizzazione numerica rapida
    res = minimize_scalar(residual, bounds=(0.1, 10), method='bounded')
    k_opt = res.x

    gamma_3 = Gamma_Func(3, k_opt)
    A_opt = np.cbrt(m3_non_central / gamma_3)

    return k_opt, A_opt

#

def save(nome):
  if save__graph==True:
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure/'+str(nome)+'.pdf', format='pdf')
    return None
  else :
    return None





def extrapolate_U2(U1, z_01, z_02, d, h=120):
    """
    This functions extrapolates the velocity from position 1 to position 2(Downwind),
    Models:
      - Elliot parametrization for h^I
      - Log law => U(z) = u*/k * ln(z/z_0)
      - Sempreviva ( Match G(hI(01)) = G(hI(02))  ) bonds u*_2 and u*_1
    input:
      - U1 is the velocity at 70 meter in the source (Sprogo)
      - z_01 surface roughness upstream of the source
      - z_02 surface roughness downstrem of the source
      - d distance (positive downwind) --> da mettere in m, positivo nella direzione del vento
      - h height of the extrapolated U2

    output:
      - U2 speed at the downstream position at distance d.

    when to use it? When I need the velocity in the more downstream position. Eg:

    . 1 -> wind   . 2
    .2  <- wind   . 1
    """
    k = 0.4 # Von Karman constant
## controlla i 70 --> 120 CHECK !!!
    u_star1 = U1*k/np.log(70/z_01)
    M = np.log(z_01/z_02)
    hI = z_02 * (0.75 + 0.003*M)*(d/z_02)**0.8
    u_star2 = u_star1*(1 + np.log(z_02/z_01) / np.log( hI/z_02)  )
    U2 = u_star2*np.log(h/z_02) / k
    return U2




def get_V50_PWM(X):
    """
    input: numpy vector of maxima per each year (Specific of the direction)
    output: float with V50 using PWM of that sector)
    """
    b0 = np.mean(X)

    N = len(X)
    # Compute weights (i-1), i goes from 1 to N → [0, 1, 2, ..., N-1]
    weights = np.arange(N)
    # Compute b1
    b1 = np.sum(weights * X) / (N**2 - N)
    alpha = (2*b1-b0)/np.log(2)
    beta = b0 - np.euler_gamma*alpha
    v50 = alpha*np.log(50/N) + beta
    return v50


def get_V50_Gumbell(X):
    """
    input: numpy vector of maxima per each year (Specific of the direction)
    output: float with V50 using Gumbell method (of that sector)
    """
    N = len(X)
    # Compute F(x)
    F = [i/(N+1) for i in range (1, N+1)]
    Y = -np.log(-np.log(F))
    # Linear fit: y = ax + b
    a, b = np.polyfit(X, Y, 1)  # Returns slope (a) and intercept (b)
    alpha = 1/a
    beta = -alpha*b
    v50 = alpha*np.log(50/N) + beta
    return v50


def get_V50_Weibull(this, X):
    """
    input: this: numpy vector with velocities (Specific of the direction, agnostic of the year)
            X: numpy vector of maxima per each year (Specific of the direction, needed just to get the number of years)
            Nu:integer number of measuraments to get the vector X

    output: float with V50 using Weibull method (of that sector)
    Generally the least relyable one ok if all other fail due to lack of datas
    """

    # Get A, k from fit Weibull of the datas
    A, k = fitWeibull_type2_fast(this)
    Nu = len(this)
    cie = 0.438*Nu
    beta = A*(np.log(cie))**(1/k)
    alpha =  A/k*(np.log(cie))**(1/(k-1))
    N = len(X)
    v50 = (alpha*np.log(50/N) + beta)
    if(abs(v50) > 30):
      v50 = 30
      # print(f"{v50 = :.2e}, {alpha = :.2e}, {beta = :.2e} \n")
    return v50



def apply_extrapolation(U_dict, z_01, z_02, d):
    U2_dict = {}
    for anno, sezioni in U_dict.items():
        U2_dict[anno] = {}
        for sezione, lista_U1 in sezioni.items():
            U2_dict[anno][sezione] = []
            for U1 in lista_U1:
                U2 = extrapolate_U2(U1, z_01, z_02, d, 70)
                U2_dict[anno][sezione].append(U2)
    return U2_dict



# dati utili
abl_height = 669.56909719781

z0_terra = 0.02         #m
z0_mare = 0.0002        #m
z_ref = 70              #m
z = 120                 #m
z_prime = abl_height *2 #m


U_rp = 12       #m/s
P_rated = 13    #MW
T = 24 * 365            #hours per year, 365 days

"""# Request 2

## Implementation

In this case we can approach the problem in 2 different ways:
  1. Obtain $V_{50}$ in Sprogo, extrapolate the data up to geostrophic speed, then by leveraging that the geostrophic velocity is equal in both places and recover by means of $z_0$ $u^*$ and finally by means of $U(z) = \frac{u^*}{k} \,  ln(z/z_0)$ and finally obtain $V_{50}$ in the derived place.
    - this means ```extrapolate_U2``` from Sprogo and then ```get_v50```.

  1. Extrapolate the velocities from 70m to G, again leverage the equivalence, extrapolate back to 70 meters the whole set as before and obtain $V_{50}$ directly with the extrapolated values.
    - this instead is ```get_V50``` from Sprogo and then ```extrapolateU2```.

The request  is not clear on wich approach to follow so we may try both.
"""

df = pd.read_csv("sprog.tsv", sep="\t", header=None)

# 2. Estrai l'anno dalla colonna 0 (timestamp yyyymmddHHMM)
df['anno'] = df[0].astype(str).str[:4].astype(int)


# FILTER
# replace vals with nan
df.replace({1: 99.99}, np.nan, inplace=True)
df.replace({2: 999.}, np.nan, inplace=True)
df.replace({3:999.}, np.nan, inplace=True)

df = df.dropna(subset=[1])

df['dirUni'] = df[2].fillna(df[3])
df = df.dropna(subset=['dirUni'])

df['slice'] = df['dirUni'].apply(lambda x: int(x / 10) )
df['vel'] = df[1]

df.head()

## Approach 1


# Matrix containing v50 for each of the 3 method for each direction
v_50_Sprogo = np.zeros([3, 36])

v_check = np.zeros([3,36])

# Prepare data to get the V50


X = np.zeros(23)

# Prepare data to get the V50

for dir in range(0, 36, 1):
  to_weib = np.array(df.loc[(df['slice']==dir), 'vel'])
  v_50_Sprogo[1][dir] = get_V50_Weibull(to_weib, X)

  for anno in range(1977, 2000):

    this = np.array(df.loc[(df['anno']==anno) & (df['slice']==dir), 'vel'])

    X[int(anno - 1977)] = np.max(this)
    # Riordino il vettore in ordine crescente
  X = np.sort(X)

  v_check[0][dir] = np.mean(X)
  # Store v_50 in each place for each direction
  v_50_Sprogo[0][dir] = get_V50_PWM(X)

  v_50_Sprogo[2][dir] = get_V50_Gumbell(X)

# Extrapolate to the desired place now

plt.plot(v_50_Sprogo[0], label = 'v 50 PWM')
plt.plot(v_50_Sprogo[1], label = 'v 50 Weibull')
plt.plot(v_50_Sprogo[2], label = 'v 50 Gumbell' )

plt.plot(v_check[0], label = 'Mean of the max' )
plt.grid()
plt.legend()
plt.show()




