import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

t0 = -0.000
tapp = 0.100
tf = 0.150
# target velocity
v = 5
# first acceleration
a1 = v * (tf + tapp) / ((tapp - t0) * (tf - t0))
# second acceleration
a2 = -v * (tapp + t0) / ((tf - tapp)*(tf - t0))


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
plt.xlabel("time (ms)")
plt.ylabel("position (deg)")
plt.legend()
plt.show()

# plot eye velocity
plt.figure()
plt.plot(T, VX, label="eye velocity")
plt.plot(T, VY, label="target velocity")
plt.xlabel("time (ms)")
plt.ylabel("velocity (deg/ms)")
plt.legend()
plt.show()
# plot eye acceleration
plt.figure()
plt.plot(T, a1*T, label="first acceleration")
plt.plot(T, a2*T, label="second acceleration")
plt.xlabel("time (ms)")
plt.ylabel("acceleration (deg/ms^2)")
plt.legend()
plt.show()
