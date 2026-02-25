import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
 
#negligible mass
k = 15 
m = 80
l = 100
g = 9.81
c = 1.1*1.2*0.8/2
c = 4
rho = 1
#m = m + rho*l/3
w = np.sqrt(k/m) 

#c1 = -np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))/w**2
#c3 = -c1 - g/w**2
 
#x0 = 0
#v0 = np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))

# x0 = 0
# v0 = np.sqrt(2*g*l)
 
# dt = 0.05
# maxt = (10*2*np.pi)/w 
 
# dtmax = 18 
# dtmin = 1e-5 
# epsilon = 0.01 
 
# xrk = x0 
# tlist = [] 
# xexactlist = [] 
# xrklist = [] 
# vexactlist = [] 
# vrklist = [] 
 
# xrk = x0 
# vrk = v0
# for t in np.arange(dt,maxt,dt): 
#     tlist.append(t)
     
#     ax = vrk 
#     av = g - (k/m)*xrk - (c/m)*vrk**2
#     bx = (vrk + (dt/2)*av) 
#     bv = g - (k/m)*(xrk + (dt/2)*ax) - (c/m)*(vrk + (dt/2)*av)**2
#     cx = (vrk + (dt/2)*bv) 
#     cv = g - (k/m)*(xrk + (dt/2)*bx) - (c/m)*(vrk + (dt/2)*bv)**2
#     dx = (vrk + (dt)*cv) 
#     dv = g - (k/m)*(xrk + (dt)*cx) - (c/m)*(vrk + (dt/2)*cv)**2
    
#     xrk  = xrk + (dt/6)*(ax + 2*bx + 2*cx + dx) 
#     xrklist.append(xrk) 
    
#     vrk = vrk + (dt/6)*(av + 2*bv + 2*cv + dv) 
#     vrklist.append(vrk) 
# plt.figure()
# plt.plot(tlist,xrklist, 'o', label = 'x') 
# plt.plot(tlist, vrklist, 'o', label = 'v') 
# plt.legend() 
# plt.show() 
 

# def fun(t, z):
#     x, v = z
#     dxdt = v
#     dvdt = g - (k/m)*x - (c/m)*v**2 + (k/m)*l
#     return [dxdt, dvdt]

# t_span = (0, 400)
# v0 = [0, np.sqrt(2*g*l)]
# sol = solve_ivp(fun, t_span, v0)
# plt.figure()
# plt.plot(sol.t, sol.y[0], label = 'x')
# plt.plot(sol.t, sol.y[1], label = 'v')
# plt.legend()
# plt.show()

def fun1(t, z):
    x, v = z
    dxdt = v
    if x > 0:
      dvdt = g - (k/m)*x #- (c/m)*v**2
    else:
        dvdt = g #- (c/m)*v**2
    return [dxdt, dvdt]

t_span = (0, 100)
v0 = [0, np.sqrt(2*g*l)]
sol1 = solve_ivp(fun1, t_span, v0, t_eval = np.linspace(0, 100, 150))

a1 = np.gradient(sol1.y[1], sol1.t)
# a11 = g - (k/m)*sol1.y[0]
# a11 = np.zeros(len(sol1.t))
# for n in np.arange(len(sol1.t)):
#     if sol1.y[0][n] > 0:
#         a11[n] = g - (k/m)*sol1.y[0][n]
#     else:
#         a11[n] = g

plt.figure()
plt.plot(sol1.t, sol1.y[0], label = 'x')
plt.plot(sol1.t, sol1.y[1], label = 'v')
plt.plot(sol1.t, a1, label = 'a')
#plt.plot(sol1.t, a11)
plt.xlabel('Time / s')
plt.legend()
plt.show()

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

plt.figure()
plt.plot(sol2.t, sol2.y[0], label = 'x')
plt.plot(sol2.t, sol2.y[1], label = 'v')
plt.plot(sol2.t, a2, label = 'a')
plt.xlabel('Time / s')
plt.legend()
plt.show()

def fun3(t, z):
    x, v = z
    dxdt = v
    if x > 0:
      dvdt = g - (k/(m + rho*l/2))*x #- (c/m)*v**2
    else:
        dvdt = g #- (c/m)*v**2
    return [dxdt, dvdt]

t_span = (0, 100)
v0 = [0, np.sqrt(4*g*l*(2*m + rho*l)/(4*m + rho*l))]
sol3 = solve_ivp(fun3, t_span, v0, t_eval = np.linspace(0, 100, 150))

a3 = np.gradient(sol3.y[1], sol3.t)

plt.figure()
plt.plot(sol3.t, sol3.y[0], label = 'x')
plt.plot(sol3.t, sol3.y[1], label = 'v')
plt.plot(sol3.t, a3, label = 'a')
plt.xlabel('Time / s')
plt.legend()
plt.show()

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
#plt.plot(0, g, 'o')
#plt.show()
#plt.figure()
plt.plot(sol2.t, a2, 'red', label = 'Rope hangs down')
#plt.plot(0, rho*(g*rho*(2*l*l - l**2) + 4*g*m*l)/(2*(2*m)**2) + (2*g*m)/(2*m), 'o')
#plt.show()
#plt.figure()
plt.plot(sol3.t, a3, 'b', label = 'Rope starts at top')
#plt.plot(0, 2*g*(rho**2*l**2 + 8*m*rho*l + 8*m**2)/((rho*l + 4*m)**2), 'o')
plt.legend()
plt.xlabel('Time / s')
plt.ylabel('Acceleration / $ms^{-2}$')
#plt.savefig('Bungee_jumping6.pdf')
plt.show()

# def fun1(t, z):
#     x, v = z
#     dxdt = v
#     if x > 0:
#       dvdt = g - (k/m)*x - (c/m)*v**2
#     else:
#         dvdt = g - (c/m)*v
#     return [dxdt, dvdt]

# t_span = (0, 400)
# v0 = [0, np.sqrt(2*g*l)]
# sol1 = solve_ivp(fun1, t_span, v0, t_eval = np.linspace(0, 400, 450))

# a1 = np.gradient(sol1.y[1], sol1.t)

# plt.figure()
# plt.plot(sol1.t, sol1.y[0], label = 'x')
# plt.plot(sol1.t, sol1.y[1], label = 'v')
# plt.plot(sol1.t, a1, label = 'a')
# plt.legend()
# plt.show()

# def fun2(t, z):
#     x, v = z
#     dxdt = v
#     if x > 0:
#       dvdt = g - (k/(m + rho*l/2))*x - (c/(m + rho*l/2))*v**2
#     else:
#         dvdt = g - (c/(m + rho*l/2))*v
#     return [dxdt, dvdt]

# t_span = (0, 400)
# v0 = [0, np.sqrt((rho*g*(2*l*l - l**2) + 4*m*g*l)/(2*m + rho*(l-l)))]
# sol2 = solve_ivp(fun2, t_span, v0, t_eval = np.linspace(0, 400, 450))

# a2 = np.gradient(sol2.y[1], sol2.t)

# plt.figure()
# plt.plot(sol2.t, sol2.y[0], label = 'x')
# plt.plot(sol2.t, sol2.y[1], label = 'v')
# plt.plot(sol2.t, a2, label = 'a')
# plt.legend()
# plt.show()

# def fun3(t, z):
#     x, v = z
#     dxdt = v
#     if x > 0:
#       dvdt = g - (k/(m + rho*l/2))*x - (c/(m + rho*l/2))*v**2
#     else:
#         dvdt = g - (c/(m + rho*l/2))*v
#     return [dxdt, dvdt]

# t_span = (0, 400)
# v0 = [0, np.sqrt(4*g*l*(2*m + rho*l)/(4*m + rho*l))]
# sol3 = solve_ivp(fun3, t_span, v0, t_eval = np.linspace(0, 400, 450))

# a3 = np.gradient(sol3.y[1], sol3.t)

# plt.figure()
# plt.plot(sol3.t, sol3.y[0], label = 'x')
# plt.plot(sol3.t, sol3.y[1], label = 'v')
# plt.plot(sol3.t, a3, label = 'a')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(sol1.t, sol1.y[0], 'k', label = 'Negligible mass')
# plt.plot(sol2.t, sol2.y[0], 'red')
# plt.plot(sol3.t, sol3.y[0], 'b')
# plt.show()

# plt.figure()
# plt.plot(sol1.t, sol1.y[1], 'k')
# plt.plot(sol2.t, sol2.y[1], 'red')
# plt.plot(sol3.t, sol3.y[1], 'b')
# plt.show()

# plt.figure()
# plt.plot(sol1.t, a1, 'k')
# plt.plot(sol2.t, a2, 'red')
# plt.plot(sol3.t, a3, 'b')
# plt.show()

kk = [1, 5, 50]
plt.figure()

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


    #plt.plot(sol2.t, sol2.y[0], label = 'k = ' + str(k))
    plt.plot(sol2.t, sol2.y[1], label = 'k = ' + str(k))
    #plt.plot(sol2.t, a2, label = 'a')
plt.xlabel('Time / s')
plt.ylabel('Velocity / $ms^{-1}$')
plt.legend()
#plt.savefig('Bungee_jumping7.pdf')
plt.show()

plt.figure()

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


    #plt.plot(sol2.t, sol2.y[0], label = 'k = ' + str(k))
    #plt.plot(sol2.t, sol2.y[1], label = 'v')
    plt.plot(sol2.t, a2, label = 'k = ' + str(k))
plt.xlabel('Time / s')
plt.ylabel('Acceleration / $ms^{-2}$')
plt.legend()
#plt.savefig('Bungee_jumping8.pdf')
plt.show()