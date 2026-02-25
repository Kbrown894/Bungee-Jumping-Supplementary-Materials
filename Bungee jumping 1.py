import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
 
#negligible mass
k = 15 
m = 80
l = 100
g = 9.81
rho = 1
w = np.sqrt(k/m) 

#Negligible mass
def fun1(t, z):
    x, v = z
    dxdt = v
    if x > 0:
      dvdt = g - (k/m)*x
    else:
        dvdt = g 
    return [dxdt, dvdt]

t_span = (0, 100)
v0 = [0, np.sqrt(2*g*l)]
sol1 = solve_ivp(fun1, t_span, v0, t_eval = np.linspace(0, 100, 150))

a1 = np.gradient(sol1.y[1], sol1.t)

#Rope hangs down
def fun2(t, z):
    x, v = z
    dxdt = v
    if x > 0:
      dvdt = g - (k/(m + rho*l/2))*x
    else:
        dvdt = g
    return [dxdt, dvdt]

t_span = (0, 100)
v0 = [0, np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))]
sol2 = solve_ivp(fun2, t_span, v0, t_eval = np.linspace(0, 100, 150))

a2 = np.gradient(sol2.y[1], sol2.t)

#Rope starts at top
def fun3(t, z):
    x, v = z
    dxdt = v
    if x > 0:
      dvdt = g - (k/(m + rho*l/2))*x
    else:
        dvdt = g
    return [dxdt, dvdt]

t_span = (0, 100)
v0 = [0, np.sqrt(4*g*l*(2*m + rho*l)/(4*m + rho*l))]
sol3 = solve_ivp(fun3, t_span, v0, t_eval = np.linspace(0, 100, 150))

a3 = np.gradient(sol3.y[1], sol3.t)


plt.figure()
plt.plot(sol1.t, sol1.y[0], 'k', label = 'Negligible mass')
plt.plot(sol2.t, sol2.y[0], 'red', label = 'Rope hangs down')
plt.plot(sol3.t, sol3.y[0], 'b', label = 'Rope starts at top')
plt.xlabel('Time / s')
plt.ylabel('x / m')
plt.legend()
plt.show()

plt.figure()
plt.plot(sol1.t, sol1.y[1], 'k', label = 'Negligible mass')
plt.plot(sol2.t, sol2.y[1], 'red', label = 'Rope hangs down')
plt.plot(sol3.t, sol3.y[1], 'b', label = 'Rope starts at top')
plt.xlabel('Time / s')
plt.ylabel('Velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('Bungee_jumping5.pdf')
plt.show()

plt.figure()
plt.plot(sol1.t, a1, 'k', label = 'Negligible mass')
plt.plot(sol2.t, a2, 'red', label = 'Rope hangs down')
plt.plot(sol3.t, a3, 'b', label = 'Rope starts at top')
plt.legend()
plt.xlabel('Time / s')
plt.ylabel('Acceleration / $ms^{-2}$')
#plt.savefig('Bungee_jumping6.pdf')
plt.show()

#Changing Spring constant, rope hangs down
kk = [1, 5, 50]
plt.figure()

#Velocity
for k in kk:
    def fun2(t, z):
        x, v = z
        dxdt = v
        if x > 0:
          dvdt = g - (k/(m + rho*l/2))*x
        else:
            dvdt = g 
        return [dxdt, dvdt]

    t_span = (0, 100)
    v0 = [0, np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))]
    sol2 = solve_ivp(fun2, t_span, v0, t_eval = np.linspace(0, 100, 150))

    a2 = np.gradient(sol2.y[1], sol2.t)

    plt.plot(sol2.t, sol2.y[1], label = 'k = ' + str(k))
plt.xlabel('Time / s')
plt.ylabel('Velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('Bungee_jumping7.pdf')
plt.show()

plt.figure()

#Acceleration
for k in kk:
    def fun2(t, z):
        x, v = z
        dxdt = v
        if x > 0:
          dvdt = g - (k/(m + rho*l/2))*x #- (c/m)*v**2
        else:
            dvdt = g #- (c/m)*v**2
        return [dxdt, dvdt]

    t_span = (0, 100)
    v0 = [0, np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))]
    sol2 = solve_ivp(fun2, t_span, v0, t_eval = np.linspace(0, 100, 150))

    a2 = np.gradient(sol2.y[1], sol2.t)

    plt.plot(sol2.t, a2, label = 'k = ' + str(k))
plt.xlabel('Time / s')
plt.ylabel('Acceleration / $ms^{-2}$')
plt.legend()
#plt.savefig('Bungee_jumping8.pdf')

plt.show()
