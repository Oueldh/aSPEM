import numpy as np
import matplotlib.pyplot as plt
# the known variables
#t0 is the time when the eye starts moving in ms
t0 = -.20
#tapp is the time when the eye starts accelerating in ms
tapp = 0.100
k=5
v =- 5
b=k**2*(tapp-t0)
c=k**4*v*(tapp+t0)
d=((3*3**0.5*(4*b**3*c+27*c**2)**0.5-2*b**3-27*c)/2)**(1/3)
# eye acceleration
a2=1/3*(d+b**2/d-b)
# first acceleration
tf = a2/k**2+tapp
# anticpatory acceleration
a1 =(v-a2**2/k**2)/(tapp-t0)
# second acceleration
print(f'd={d}')
print("Anticipatory acceleration:",a1)
print("Acceleration after appearence ",a2)
print("Time where the eye reaches the target in (s): ",tf)

# eye velocity
def velocity(t, t0, tapp, tf, v, a1, a2):
    if t <= t0:
        return 0
    elif t0 < t <= tapp:
        return a1 * (t - t0)
    elif tapp < t <= tf:
        return a2 * (t - tapp) + a1 * (tapp - t0)
    else:
        return v


# eye position
def position(t, t0, tapp, tf, v, a1, a2):
    if t <= t0:
        return 0
    elif t0 < t <= tapp:
        return a1 * (t - t0) ** 2 / 2
    elif tapp < t <= tf:
        return (
            a2 * (t - tapp) ** 2 / 2
            + a1 * (t - tapp) * (tapp - t0)
            + 1 / 2 * a1 * (tapp - t0) ** 2
        )
    else:
        return v * t


# time vector
T = np.linspace(t0, 2 * tf, 100)
# position vector of the eye
X = np.zeros(len(T))
# velocity vector of the eye
VX = np.zeros(len(T))
# position vector of the target
Y = np.zeros(len(T))
# velocity vector of the target
VY = np.zeros(len(T))

for i in range(len(T)):
    X[i] = position(T[i], t0, tapp, tf, v, a1, a2)
    VX[i] = velocity(T[i], t0, tapp, tf, v, a1, a2)
    Y[i] = v * (T[i] * (T[i] > 0))
    VY[i] = v * (T[i] > 0)

# plot eye position
plt.figure()
plt.plot(T, X, label="eye position")
plt.plot(T, Y, label="target position")
plt.xlabel("time (s)")
plt.ylabel("position (deg)")
plt.legend()
plt.show()

# plot eye velocity
plt.figure()
plt.plot(T, VX, label="eye velocity")
plt.plot(T, VY, label="target velocity")
plt.xlabel("time (s)")
plt.ylabel("velocity (deg/s)")
plt.legend()
plt.show()
# plot eye acceleration
plt.figure()
plt.plot(T, a1*T, label="first acceleration")
plt.plot(T, a2*T, label="second acceleration")
plt.xlabel("time (s)")
plt.ylabel("acceleration (deg/s^2)")
plt.legend()
plt.show()
