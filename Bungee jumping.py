import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

l = 100
m = 80
g = 9.81
rho = 1
x = np.linspace(0, l, 100)

#Negligible mass
v1 = np.sqrt(2*g*x)
a1 = g

#Rope hangs down
v2 = np.sqrt((rho*g*(2*l*x - x**2) + 4*m*g*x)/(2*m + rho*(l-x)))
a2 = rho*(g*rho*(2*l*x - x**2) + 4*g*m*x)/(2*(rho*(l - x) + 2*m)**2) + (g*rho*(l - x) + 2*g*m)/(rho*(l - x) + 2*m)

#Rope falls with jumper
v3 = np.sqrt(4*g*x*(2*m + rho*x)/(4*m + rho*x))
a3 = 2*g*(rho**2*x**2 + 8*m*rho*x + 8*m**2)/((rho*x + 4*m)**2)


plt.figure()
plt.hlines(g, 0, l, 'k', label = 'Negligible mass')
plt.plot(x, a2, 'red', label = 'Rope hangs down')
plt.plot(x, a3, 'b', label = 'Rope starts at top')
plt.xlabel('Distance / m')
plt.ylabel('Acceleration / $ms{-2}$')
plt.legend()
#plt.savefig('bungee_jumping1.pdf')
plt.show()

plt.figure()
plt.plot(x, v1, 'k', label = 'Negligible mass')
plt.plot(x, v2, 'red', label = 'Rope hangs down')
plt.plot(x, v3, 'b', label = 'Rope starts at top')
plt.xlabel('Distance / m')
plt.ylabel('Velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('bungee_jumping2.pdf')
plt.show()

#changing mass
mm = np.linspace(0, 1, 5)
plt.figure()
for rho in mm:
    v = np.sqrt((rho*g*(2*l*x - x**2) + 4*m*g*x)/(2*m + rho*(l-x)))
    plt.plot(x, v, label = r'$\rho$ = ' + str(rho))
plt.xlabel('Distance / m')
plt.ylabel('velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('bungee_jumping3.pdf')
plt.show()

plt.figure()
for rho in mm:
    a = g*(rho**2*x**2 - 2*l*rho**2*x - 4*m*rho*x + 2*l**2*rho**2 + 8*m*l*rho + 8*m**2)/(2*(rho*(x - l) - 2*m)**2)
    plt.plot(x, a, label = r'$\rho$ = ' + str(rho))
plt.xlabel('Distance / m')
plt.ylabel('Acceleration / $ms^{-2}$')
plt.legend()
#plt.savefig('bungee_jumping4.pdf')
plt.show()

# plt.show()
