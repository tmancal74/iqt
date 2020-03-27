#
# Numerické řešení časové Schrödingerovy
# rovnice pomocí krátkočasové
# aproximace evolučního operátoru a
# pomocí diagonalizace hamiltoniánu
#
#
#
#
#

# numerický balík numpy
import numpy as np
# balík pro tvorbu grafů
import matplotlib.pyplot as plt

################################################################################
# Definice systému, který simulujeme
################################################################################

# počet stavů
N = 2

# Hamiltonián je reprezentován maticí
H = np.zeros((N,N), dtype=np.float)

# Jedna energie je nula a druhá např. 1.0
H[0,0] = 0.0
H[1,1] = 2.0

# interakční energie
H[0,1] = 0.8
# nezapomenout, že hamiltonián musí být 
# reprezentován hermitovskou maticí
H[1,0] = H[0,1]

################################################################################
# Definice časového kroku a intervalu na
# němž řešíme Schrödingerovu rovnici
################################################################################

# Časový krok zvolíme dt
dt = 0.1

# Počet časových kroků vypočten z času,
# do kterého simulujeme
Tmax = 100
Nt = int(Tmax/dt)

################################################################################

#
# Jak počítat? 
#
methods = ["n-power", "diag", "short-exp"]
method = "short-exp"

################################################################################
################################################################################



if method == methods[0]:

    #
    # Evoluční operátor exp(-a) 
    # aproximuleme jako (1-a/n)^n
    #

    # jednotková matice
    Uk = np.eye(N, dtype=np.float)
    # aproximace evoučního operátoru
    n = 5
    Utt0 = (Uk-1j*H*dt/np.float(n))
    for k in range(n):
        Uk = np.matmul(Utt0,Uk)
    Utt0 = Uk

elif method == methods[1]:

    #
    # Výpočet evolučního operátoru 
    # diagonalizací
    #

    # vlastní čísla a matice vlastních
    # vektorů hermitovské matice
    ee, S = np.linalg.eigh(H)

    # evoluční operátor v bázi vlastních
    # stavů
    Uk = np.diag(np.exp(-1j*ee*dt))

    # transformace do požadované báze
    Utt0 = np.matmul(S,np.matmul(Uk,
                     np.transpose(S)))

elif method == methods[2]:
    
    n = 6
    fac = 1.0
    Hn = np.eye(N, dtype=np.float)
    Utt0 = np.zeros((N,N), dtype=np.complex)
    for k in range(n):
        Utt0 += Hn*((-1j*dt)**k)/fac
        Hn = np.matmul(Hn, H)
        fac = fac*(k+1)
    
else:
     
    raise Exception("Unknown method")    

# Počáteční stav systému
psi = np.zeros(N, dtype=np.complex)
psi[1] = 1.0

# pravděpodobnost v čase
p = np.zeros((Nt,N), dtype=np.float)

# stavový vektor v čase
psit = np.zeros((Nt,N), dtype=np.complex)

# počáteční podmínka nastavena zde
psit[0,:] = psi
p[0,:] = np.abs(psi)**2

# časový vývoj opakovaným uplatněním 
# evolučního operátoru
t = np.zeros(Nt, dtype=np.float)
for k in range(Nt-1):
    t[k+1] = (k+1)*dt
    psit[k+1,:] = np.dot(Utt0,psit[k,:])
    p[k+1,:] = np.abs(psit[k+1,:])**2

# graf časového vývoje pravděpodobnosti
plt.plot(t,p[:,0],"-b")
plt.plot(t,p[:,1],"-r")

# test přesnosti; známe minimální hodnotu
# mixing angle
phi = 0.5*np.arctan(2*H[0,1]/(H[1,1]-H[0,0]))
# minimální hodnota pravděpodobnosti pro
# stav 1
min = (np.cos(phi)**2 - np.sin(phi)**2)**2
pmin = np.zeros(Nt, dtype=np.float)
pone = np.ones(Nt, dtype=np.float)
pmin[:] = min

# vykreslit minimální hodnoutu do grafu
plt.plot(t,pmin,"-k")
plt.plot(t,pone,"--k")

# ukaž graf
plt.show()















