import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

l = 100
m = 80
g = 9.81
rho = 1
x = np.linspace(0, l, 100)

#x<=l
v1 = np.sqrt(2*g*x)
a1 = g

v2 = np.sqrt((rho*g*(2*l*x - x**2) + 4*m*g*x)/(2*m + rho*(l-x)))
#a2 = g*(rho**2*x**2 - 2*l*rho**2*x - 4*m*rho*x + 2*l**2*rho**2 + 8*m*l*rho + 8*m**2)/(2*(rho*(x - l) - 2*m)**2)
a2 = rho*(g*rho*(2*l*x - x**2) + 4*g*m*x)/(2*(rho*(l - x) + 2*m)**2) + (g*rho*(l - x) + 2*g*m)/(rho*(l - x) + 2*m)

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

#plt.figure()
#plt.plot(t, v1)
#plt.show()

#changing mass
i = 0
mm = np.linspace(0, 1, 5)
plt.figure()
for rho in mm:
    v = np.sqrt((rho*g*(2*l*x - x**2) + 4*m*g*x)/(2*m + rho*(l-x)))
    i = i + 1
    plt.plot(x, v, label = r'$\rho$ = ' + str(rho))
plt.xlabel('Distance / m')
plt.ylabel('velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('bungee_jumping3.pdf')
plt.show()

plt.figure()
for rho in mm:
    a = g*(rho**2*x**2 - 2*l*rho**2*x - 4*m*rho*x + 2*l**2*rho**2 + 8*m*l*rho + 8*m**2)/(2*(rho*(x - l) - 2*m)**2)
    i = i + 1
    plt.plot(x, a, label = r'$\rho$ = ' + str(rho))
plt.xlabel('Distance / m')
plt.ylabel('Acceleration / $ms^{-2}$')
plt.legend()
#plt.savefig('bungee_jumping4.pdf')
plt.show()


#Oscillations
k = 15
l = 100
m = 80
g = 9.81
rho = 1
t = np.linspace(0, 100, 100)

# w = np.sqrt(k/m)
# a = -g/w**2
# b = np.sqrt(2*g*l)/w
# #phi = np.cos(g/w + 1)
# #phi = np.sin(g**2/w**2)
# x = a*np.cos(w*t) + b*np.sin(w*t) + g/w**2
# v = -a*w*np.sin(w*t) + b*w*np.cos(w*t)
# a = -a*w**2*np.cos(w*t) - b*w**2*np.sin(w*t)

# w = np.sqrt(k/(m + rho*l/3))
# #c1 = -np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))/w**2
# #c2 = g*(rho**2*l**2 - 2*l*rho**2*l - 4*m*rho*l + 2*l**2*rho**2 + 8*m*l*rho + 8*m**2)/(2*w*(-2*m)**2)
# c1 = -(rho*(g*rho*(2*l*l - l**2) + 4*g*m*l)/(2*(rho*(l - l) + 2*m)**2) + (g*rho*(l - l) + 2*g*m)/(rho*(l - l) + 2*m))/w**2
# c2 = np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))/w
# c3 = -c1 - g/w**2
# x1 = c1*np.cos(w*t) + c2*np.sin(w*t) + g/w**2 #+ c3/w**2
# v1 = -c1*w*np.sin(w*t) + c2*w*np.cos(w*t)
# a1 = -c1*w**2*np.cos(w*t) - c2*w**2*np.sin(w*t)

# #c11 = -np.sqrt(4*g*l*(2*m + rho*l)/(w*(4*m + rho*l)))
# #c21 = 2*g*(rho**2*l**2 + 8*m*rho*l + 8*m**2)/(w*(rho*l + 4*m)**2)
# c11 = -(2*g*(rho**2*l**2 + 8*m*rho*l + 8*m**2)/((rho*l + 4*m)**2))/w**2
# c21 = np.sqrt(4*g*l*(2*m + rho*l)/(4*m + rho*l))/w
# c31 = -c11 - g/w**2
# x2 = c11*np.cos(w*t) + c21*np.sin(w*t) + g/w**2 #+ c31/w**2
# v2 = -c11*w*np.sin(w*t) + c21*w*np.cos(w*t)
# a2 = -c11*w**2*np.cos(w*t) - c21*w**2*np.sin(w*t)

# phi = np.arctan(np.sqrt(2*l/g)*w)
# c = -g/(w**2*np.cos(phi))
# x = c*np.cos(w*t + phi) + g/w**2
# v = -c*np.sin(w*t + phi)
# a = -c*w**2*np.cos(w*t + phi)

# plt.figure()
# plt.plot(t, x)
# plt.plot(t, v)
# plt.plot(t, a)
# plt.show()

# phi1 = np.arctan(np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l))))
# c1 = -g/(w**2*np.cos(phi))
# x1 = c*np.cos(w*t + phi) + g/w**2
# v1 = -c*np.sin(w*t + phi)
# a1 = -c*w**2*np.cos(w*t + phi)

# plt.figure()
# plt.plot(t, x1)
# plt.plot(t, v1)
# plt.plot(t, a1)
# plt.show()

# phi2 = np.arctan(np.sqrt(2*l/g)*w**2)
# c2 = -g/(w**2*np.cos(phi))
# x2 = c*np.cos(w*t + phi) + g/w**2
# v2 = -c*np.sin(w*t + phi)
# a2 = -c*w**2*np.cos(w*t + phi)

# plt.figure()
# plt.plot(t, x2)
# plt.plot(t, v2)
# plt.plot(t, a2)
# plt.show()

def pend(vec,t):
  x, v = vec
  dxdt = v
  if x > 0:
    dvdt = g - (k/m)*x
  else:
      dvdt = g
  return(dxdt,dvdt)

t = np.linspace(0,100,100)
initial = [0,np.sqrt(2*g*l)]
vec = odeint(pend, initial, t)
v = vec[:,1]
x = vec[:,0]

a = np.gradient(v, t)

plt.figure()
plt.plot(t, x)
plt.plot(t, v)
plt.plot(t, a)
plt.show()

def pend(vec,t):
  x, v = vec
  dxdt = v
  if x > 0:
     dvdt = g - (k/(m + rho*l/3))*x
  else:
      dvdt = g
  return(dxdt,dvdt)

t = np.linspace(0,100,100)
initial = [0,np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))]
vec = odeint(pend, initial, t)
v = vec[:,1]
x = vec[:,0]

a = np.gradient(v, t)

plt.figure()
plt.plot(t, x, label = 'ode')
plt.plot(t, v)
plt.plot(t, a)
plt.show()

# plt.figure()
# plt.plot(t, x, 'k')
# plt.plot(t, x1, 'red')
# plt.plot(t, x2, 'b')
# plt.xlabel('Time')
# plt.ylabel('x')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(t, v, 'k', label = 'Negligible mass')
# plt.plot(t, v1, 'red', label = 'Rope hangs down')
# plt.plot(t, v2, 'b', label = 'Rope starts at top')
# plt.ylabel('Velocity')
# plt.xlabel('Time')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(t, a, 'k', label = 'Negligible mass')
# plt.plot(t, a1, 'red', label = 'Rope hangs down')
# plt.plot(t, a2, 'b', label = 'Rope starts at top')
# plt.ylabel('Acceleration')
# plt.xlabel('Time')
# plt.legend()
# plt.show()