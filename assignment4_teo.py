"""
Fai secondo metodo
Turbine class
"""
from math import ceil, floor
import pandas as pd
import numpy as np
import scipy as sc
import sympy as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
def Weibull(k, A, U):
  return k/A*(U/A)**(k-1)*np.exp(- (U/A)**k )

def fitWeibull_type1(dato):
  # from scipy import special.gamma
  mu = np.mean(dato)
  std = np.std(dato)
  toll = 1e-5
  res = 1
  dk = toll/10
  k = 0.1 - dk

  cost = (std/mu)**2 +1

  def residual(k):
      return abs(cost - sc.special.gamma(1+2/k)/(sc.special.gamma(1+1/k))**2)
  
  # Minimizzazione numerica rapida
  res = minimize_scalar(residual, bounds=(0.1, 10), method='bounded')
  k = res.x
  A = mu/sc.special.gamma(1+1/k)
  return k, A


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
    m3_non_central = sc.stats.moment(dato, moment=3, axis=0, center=0)
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


def save(nome):
  if save__graph==True:
    plt.tight_layout()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 10)
    plt.savefig('figure/'+str(nome)+'.pdf', format='pdf')
    return None
  else:
    return None



def h_IBL(d, z_01, z_02):
    M = np.log(z_01/z_02)
    hI = z_02**0.2 * (0.75 + 0.03*M) * d**0.8
    return hI


def extrapolate_U2(U1, z_01, z_02, downwind=True, d=11200, h=70):
    """
    This functions extrapolates the velocity from position 1(Upwind) to position 2(Downwind),
    Models:
      - Elliot parametrization for h^I
      - Log law => U(z) = u*/k * ln(z/z_0)
      - Sempreviva ( Match G(hI(01)) = G(hI(02))  ) bonds u*_2 and u*_1
    input:
      - U1 is the velocity at 70 meter in the source (Sprogo)
      - z_01 surface roughness upstream of the source
      - z_02 surface roughness downstrem of the source
      - d distance --> da mettere in m : 11,22 Km circa in entrambi i casi
      - h height of the extrapolated U2
    
      - downwind Bool, changes the behaviour of the extrapolation of the u_star2 acccordingly with
            with the Sempreviva law 

    output:
      - U2 speed at the position at distance d.

    downwind == True
    . 1  -> wind   . 2

    downwind == False
    . 1  <- wind   . 2
    """
    k = 0.4 # Von Karman constant
    u_star1 = U1*k/np.log(70/z_01)
    hI = h_IBL(d, z_01, z_02)
    if downwind == True:
        u_star2 = u_star1*( np.log(hI/z_01) / np.log( hI/z_02) )
    else:
        u_star2 = u_star1*( np.log(hI/z_02) / np.log( hI/z_01) )
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


def get_V50_Weibull(this, X, verbose=False):
    """
    input: this: numpy vector with velocities (Specific of the direction, agnostic of the year)
            X: numpy vector of maxima per each year (Specific of the direction, needed just to get the number of years)
            Nu:integer number of measuraments to get the vector X

    output: float with V50 using Weibull method (of that sector)
    Generally the least relyable one ok if all other fail due to lack of datas
    """
    # Get A, k from fit Weibull of the datas
    A, k = fitWeibull_type1(this)
    Nu = len(this)
    cie = 0.438*Nu
    beta = A*(np.log(cie))**(1/k)
    alpha =  (A/k)*(np.log(cie))**(1/(k - 1))
    N = len(X)
    v50 = alpha*np.log(50/N) + beta
    if verbose==True:
        print("************", dir )
        print(f"{k = :.2f}")
        print(f"{A = :.2f}")
        print(f"{cie = :.2f}")
        print(f"{Nu = :.2f}")
        print(f"{beta = :.2f}")
        print(f"{alpha = :.2f}")
        print(f"{v50 = :.2f}")
    """
    if(abs(v50) > 30):
      v50 = 30
      # print(f"{v50 = :.2e}, {alpha = :.2e}, {beta = :.2e} \n")
    """
    return v50*7


save__graph = True 
save_figs   = False

"""i due modi per ottenere i param della weibull

## Funzioni
"""



# dati utili
z0_terra = 2e-2        #m
z0_mare = 0.02e-2      #m
z_ref = 70             #m
z = 120                #m
width = 90 # Width of the slice


# # %%
# Request 2
# 
# ## Implementation
# 
# In this case we can approach the problem in 2 different ways:
#   1. Obtain $V_{50}$ in Sprogo, extrapolate the data up to geostrophic speed, 
# then by leveraging that the geostrophic velocity is equal in both places 
# and recover by means of $z_0$ $u^*$ and finally by means of $U(z) = \frac{u^*}{k} 
# \,  ln(z/z_0)$ and finally obtain $V_{50}$ in the derived place.
#     - this means ```extrapolate_U2``` from Sprogo and then ```get_v50```.
# 
#   1. Extrapolate the velocities from 70m to G, again leverage the equivalence,
#   extrapolate back to 70 meters the whole set as before and obtain $V_{50}$ directly with the extrapolated values.
#     - this instead is ```get_V50``` from Sprogo and then ```extrapolateU2```.
# 
# The request  is not clear on wich approach to follow so we may try both.


df = pd.read_csv("sprog.tsv", sep="\t", header=None)

df['anno'] = df[0].astype(str).str[:4].astype(int)

# FILTER
# replace vals with nan
df.replace({1: 99.99}, np.nan, inplace=True)
df.replace({2: 999. }, np.nan, inplace=True)
df.replace({3: 999. }, np.nan, inplace=True)
# Drop invalid velocities
df = df.dropna(subset=[1])

df['dirUni'] = df[2].fillna(df[3])
df = df.dropna(subset=['dirUni'])

# Account for the case dirUni = 360
df['dirUni'] = df['dirUni'].replace( 360, 0 )
df['slice'] = df['dirUni'].apply(lambda x: floor(x / width) )
dir_max = int(df['slice'].max() +1)
df['vel'] = df[1]

df.head()



## APPROACH 1 (V_50 IN SPROGO AND EXTRAPOLATE)

# Matrix containing v50 for each of the 3 method for each direction
v_50_Sprogo = np.zeros([3, dir_max])
v_50_Nyborg = np.zeros([3, dir_max])
v_50_Korsor = np.zeros([3, dir_max])
v_check     = np.zeros(dir_max)
# Initialize the vector of maxima
X = np.zeros(23)

# Plot Weibull fits
if False:
    for dir in range(0, int(df['slice'].max()) , 6):
        ax = df.loc[(df['slice']==dir), 'vel'].plot.hist(column='vel', bins=200, density=True)
        this = df.loc[df['slice']==dir, 'vel']
        k, A = fitWeibull_type2_fast(this)
        k1, A1 = fitWeibull_type1(this)
        print(f"{k = :.2f}, {A = :.2f}")
        print(f"{k1 = :.2f}, {A1 = :.2f}")
        U = np.linspace(0, 25, 100)
        ax.plot(U, Weibull(k, A, U) , label='fit 2' )
        ax.plot(U, Weibull(k1, A1, U) , label='fit 1' )
        plt.title(f"Section {dir*10} - {dir*10 + 10}")
        plt.legend()
        plt.show()

# Plot the IBL height in any case the IBL in far heigher than the 70m
if False:
    x = np.linspace(0, 11000)
    plt.hlines(70, 0, 11000, label='70 m reference')
    plt.plot(x, h_IBL(x, z0_mare, z0_mare), label=r"$h_{IBL}$ sea sea")
    plt.plot(x, h_IBL(x, z0_terra, z0_mare), label=r"$h_{IBL}$ land sea")
    plt.plot(x, h_IBL(x, z0_mare, z0_terra), label=r"$h_{IBL}$ sea land")
    plt.grid()
    plt.legend()
    plt.show()


# Get V50 in Sprogo
for dir in range(0, dir_max):
    to_weib = (df.loc[(df['slice']==dir), 'vel'])
    v_50_Sprogo[1][dir] = get_V50_Weibull(to_weib, X)
    for anno in range(1977, 2000):
        this = np.array(df.loc[(df['anno']==anno) & (df['slice']==dir), 'vel'])
        X[int(anno - 1977)] = np.max(this)
    # Riordino il vettore in ordine crescente
    X = np.sort(X)
    v_check[dir] = np.mean(X)
    # Store v_50 in each place for each direction
    v_50_Sprogo[0][dir] = get_V50_PWM(X)
    v_50_Sprogo[2][dir] = get_V50_Gumbell(X)
print("Computed V50 in Sprogo")

# Extrapolate V_50 from Sprogo to Nyborg
for dir in range(0, ceil(dir_max/2 - 0.5) ):
    # Nyborg sea Sprogo sea <- Wind
    v_50_Nyborg[0][dir] = extrapolate_U2(v_50_Sprogo[0][dir], z0_mare, z0_terra)
    v_50_Nyborg[1][dir] = extrapolate_U2(v_50_Sprogo[1][dir], z0_mare, z0_terra)
    v_50_Nyborg[2][dir] = extrapolate_U2(v_50_Sprogo[2][dir], z0_mare, z0_terra)
    # Sprogo sea Korsor land <- Wind
    v_50_Korsor[0][dir] = extrapolate_U2(v_50_Sprogo[0][dir], z0_mare, z0_terra, False)
    v_50_Korsor[1][dir] = extrapolate_U2(v_50_Sprogo[1][dir], z0_mare, z0_terra, False)
    v_50_Korsor[2][dir] = extrapolate_U2(v_50_Sprogo[2][dir], z0_mare, z0_terra, False)
print(f"Extrapolated V50 from easterly directions up to {dir = }")

# Extrapolate V_50 from Sprogo to Korsor
for dir in range(ceil(dir_max/2 - 0.5), dir_max):
    # Wind -> land Nyborg sea Sprogo
    v_50_Nyborg[0][dir] = extrapolate_U2(v_50_Sprogo[0][dir], z0_mare, z0_terra, False)
    v_50_Nyborg[1][dir] = extrapolate_U2(v_50_Sprogo[1][dir], z0_mare, z0_terra, False)
    v_50_Nyborg[2][dir] = extrapolate_U2(v_50_Sprogo[2][dir], z0_mare, z0_terra, False)
    # Wind -> sea Sprogo sea Kors
    v_50_Korsor[0][dir] = extrapolate_U2(v_50_Sprogo[0][dir], z0_mare, z0_terra )
    v_50_Korsor[1][dir] = extrapolate_U2(v_50_Sprogo[1][dir], z0_mare, z0_terra)
    v_50_Korsor[2][dir] = extrapolate_U2(v_50_Sprogo[2][dir], z0_mare, z0_terra)
print(f"Extrapolated V50 from weasterly directions up to {dir = }")

# PLot the v_50 in Sprogo Nyborg and Korsor
if True:
    # Sprogo v_50
    plt.plot(v_50_Sprogo[0],'o-',color = 'orange')
    plt.plot(v_50_Sprogo[1],'s-',color = 'orange')
    plt.plot(v_50_Sprogo[2],'^-',color = 'orange')
    # Nyborg v_50
    plt.plot(v_50_Nyborg[0],'o-',color = 'green')
    plt.plot(v_50_Nyborg[1],'s-',color = 'green')
    plt.plot(v_50_Nyborg[2],'^-',color = 'green')
    # Korsor v_50
    plt.plot(v_50_Korsor[0],'o-',color = 'red')
    plt.plot(v_50_Korsor[1],'s-',color = 'red')
    plt.plot(v_50_Korsor[2],'^-',color = 'red')
    """
    """
    plt.plot(v_check, color='black')
    max = np.maximum( v_50_Korsor, v_50_Nyborg) 
    min = np.minimum( v_50_Korsor, v_50_Nyborg) 
    y = np.linspace(np.min(min)-3 , np.max(max)+3 )
    plt.fill_betweenx( y, 0, (dir_max/2 - 0.5), facecolor='red', alpha=0.1)
    plt.fill_betweenx( y, (dir_max/2 - 0.5),dir_max-1, facecolor='green', alpha=0.1)
    plt.grid()
    from matplotlib.lines import Line2D
    # Custom legend items for methods (marker shape)
    method_legend = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='PWM'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', label='Weibull'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', label='Gumbell'),
        Line2D([0], [0], marker='.', color='orange', label='Sprogo'),
        Line2D([0], [0], marker='.', color='green', label='Nyborg'),
        Line2D([0], [0], marker='.', color='red', label='Korsor'),
    ]

    # Add method legend first, then location
    method_legend_handle = plt.legend(handles=method_legend, title='Method and location')

    plt.title(rf"$V_{{50}}$ with the 3 methods for each sector. Procedure 1($V_{{50}}$ in Sprogo and then extrapolate), slice of {width = } °")
    xticks = range(0, dir_max)
    plt.xticks( ticks=xticks, labels=[f"{(2*x+1) * width/2 :.0f}° " for x in xticks], rotation=45)
    save('2a_v50')
    plt.show()



if True:
    theta = np.radians([(2*x+1) * width/2 for x in range(dir_max)])  # Direction in degrees to radians

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Sprogo v_50
    ax.plot(theta, v_50_Sprogo[0], 'o-', color='orange')
    ax.plot(theta, v_50_Sprogo[1], 's-', color='orange')
    ax.plot(theta, v_50_Sprogo[2], '^-', color='orange')
    # Nyborg v_50
    ax.plot(theta, v_50_Nyborg[0], 'o-', color='green')
    ax.plot(theta, v_50_Nyborg[1], 's-', color='green')
    ax.plot(theta, v_50_Nyborg[2], '^-', color='green')
    # Korsor v_50
    ax.plot(theta, v_50_Korsor[0], 'o-', color='red')
    ax.plot(theta, v_50_Korsor[1], 's-', color='red')
    ax.plot(theta, v_50_Korsor[2], '^-', color='red')
    """
    """

    # Mean check line
    ax.plot(theta, v_check, '*', color='black', label='Mean of the max')

    # Optional: shaded sectors (convert sector boundaries to radians)
    theta_min = np.radians(0)
    theta_mid = np.radians((dir_max/2 - 0.5) * width)
    theta_max = np.radians((dir_max) * width)

    r = np.linspace(np.min(np.minimum(v_50_Korsor, v_50_Nyborg)) - 3,
                    np.max(np.maximum(v_50_Korsor, v_50_Nyborg)) + 3, 100)

    theta_fill = np.linspace(theta_min, theta_mid, 100)
    ax.fill_between(theta_fill, r[0], r[-1], color='red', alpha=0.1)

    theta_fill = np.linspace(theta_mid, theta_max, 100)
    ax.fill_between(theta_fill, r[0], r[-1], color='green', alpha=0.1)

    ax.set_theta_zero_location('N')  # 0° at the top
    ax.set_theta_direction(-1)       # Clockwise

    ax.set_title(rf"$V_{{50}}$ with the 3 methods for each sector. Procedure 1 ($V_{{50}}$ in Sprogo and extrapolate), slice of {width = }°", pad=20)


    from matplotlib.lines import Line2D
    # Custom legend items for methods (marker shape)
    method_legend = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='PWM'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', label='Weibull'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', label='Gumbell'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', label='Mean of the Max \n(Sprogo only)'),
        Line2D([0], [0], marker='.', color='orange', label='Sprogo'),
        Line2D([0], [0], marker='.', color='green', label='Nyborg'),
        Line2D([0], [0], marker='.', color='red', label='Korsor'),
    ]

    # Add method legend first, then location
    method_legend_handle = ax.legend(handles=method_legend, title='Method and location')


    # ax.legend()

    plt.tight_layout()
    save('2a_Polar_v50')
    plt.show()




### ## APPROACH 2 (EXTRAPOLATE FROM SPROGO AND GET  V50 IN PLACE)
### 
### # Extrapolate from Sprogo
### # Case : wind -> land Sprogo sea (d) place_
### df['Nyborg'] = df['vel'].apply(lambda x: extrapolate_U2(x, z0_terra, z0_mare) )
### df['Korsor'] = df['vel'].apply(lambda x: extrapolate_U2(x, z0_terra, z0_mare) )
### 
### #filter if the there is land upwind
### # Case : wind -> land place_ sea (d) Sprogo
### df['Nyborg'] = df.loc[ (df['dirUni'] > 158 ) & (df['dirUni'] < 338), 'Nyborg' ].apply(lambda x: extrapolate_U2(x, z0_terra, z0_mare, downwind=False))
### df['Korsor'] = df['Korsor'].loc[ (df['dirUni'] > 133 ) & (df['dirUni'] < 313) ].apply(lambda x: extrapolate_U2(x, z0_terra, z0_mare, downwind=False))
### 
### # GET V_50
### # Matrix containing v50 for each of the 3 method for each direction
### v_50_Nyborg = np.zeros([3, dir_max])
### v_50_Korsor = np.zeros([3, dir_max])
### v_check     = np.zeros(dir_max)
### # Initialize the vector of maxima
### X = np.zeros(23)
### 
### for dir in range(0, dir_max):
###     to_weib = np.array(df.loc[(df['slice']==dir), 'Nyborg'])
###     v_50_Nyborg[1][dir] = get_V50_Weibull(to_weib, X)
###     for anno in range(1977, 2000):
###         this = np.array(df.loc[(df['anno']==anno) & (df['slice']==dir), 'Nyborg'])
###         X[int(anno - 1977)] = np.max(this)
###     # Riordino il vettore in ordine crescente
###     X = np.sort(X)
###     v_check[dir] = np.mean(X)
###     # Store v_50 in each place for each direction
###     v_50_Nyborg[0][dir] = get_V50_PWM(X)
###     # v_50_Nyborg[2][dir] = get_V50_Gumbell(X)
###  
### plt.plot(v_50_Nyborg[0])
### plt.show()
