import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Retinal-image acceleration in deg/s2
a = 4
# Retinal-image velocity in deg/s
v = 15

t0 = -1 / 2 * v / a
tf = 1 / 2 * v / a

T = np.linspace(t0, 2 * tf, 100)
X = np.zeros(len(T))
Y = np.zeros(len(T))


# eye movement
def x(t):
    if t <= t0:
        return 0
    elif t0 < t <= tf:
        return 1 / 2 * a * (t - t0) ** 2
    else:
        return v * t


# target movement
def y(t):
    if t <= 0:
        return 0
    else:
        return v * t


for i in range(len(T)):
    X[i] = x(T[i])
    Y[i] = y(T[i])

plt.plot(T, X, label="eye")
plt.plot(T, Y, label="target")
plt.xlabel("time (s)")
plt.ylabel("position (deg)")
plt.legend()
plt.show()


# velocity of eye movement


def vx(t):
    if t <= t0:
        return 0
    elif t0 < t <= tf:
        return a * (t - t0)
    else:
        return v


# velocity of target movement
def vy(t):
    if t <= 0:
        return 0
    else:
        return v


VX = np.zeros(len(T))
VY = np.zeros(len(T))
for i in range(len(T)):
    VX[i] = vx(T[i])
    VY[i] = vy(T[i])

plt.plot(T, VX, label="eye")
plt.plot(T, VY, label="target")
plt.xlabel("time (s)")
plt.ylabel("velocity (deg/s)")
plt.legend()
plt.show()

# probability of target going to the right
p = [1, 0.7, 0.5, 0.3]
ale1 = bernoulli.rvs(p[1], size=100)
ale2 = bernoulli.rvs(p[2], size=100)
ale3 = bernoulli.rvs(p[3], size=100)

err1 = 2 * v * tf * (1 - ale1)
err2 = 2 * v * tf * (1 - ale2)
err3 = 2 * v * tf * (1 - ale3)

# linear hypothesis
y1 = v * tf * (2 * ale1 - 1)
y2 = v * tf * (2 * ale2 - 1)
y3 = v * tf * (2 * ale3 - 1)

x1 = 2 * v * tf * (bernoulli.rvs(p[1], size=100) - 0.5)
x2 = 2 * v * tf * (bernoulli.rvs(p[2], size=100) - 0.5)
x3 = 2 * v * tf * (bernoulli.rvs(p[3], size=100) - 0.5)
# Linear errors
err4 = np.abs(x1 - y1)
err5 = np.abs(x2 - y2)
err6 = np.abs(x3 - y3)

x_theo1 = 2 * v * tf * (p[1] - 1 / 2)
x_theo2 = 2 * v * tf * (p[2] - 1 / 2)
x_theo3 = 2 * v * tf * (p[3] - 1 / 2)

temps = np.linspace(0, 100, 100)

plt.figure()
plt.plot(temps, err1, label="arb/p=0.7")
plt.plot(temps, err4, label="lin/p=0.7")
plt.xlabel("trial")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

plt.figure()
plt.plot(temps, err2, label="arb/p=0.5", ls="--")
plt.plot(temps, err5, label="lin/p=0.5", ls="--")
plt.xlabel("trial")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

plt.figure()
plt.plot(temps, err3, label="arb/p=0.3", ls=":")
plt.plot(temps, err6, label="lin/p=0.3", ls=":")
plt.xlabel("trial")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

print("p=0.7, theorical arb: ", (1 - p[1]) * 2 * v * tf)
print("p=0.7, arb: ", np.mean(err1))
print(
    "p=0.7, theorical lin: ",
    p[1] * (v * tf - x_theo1) + (1 - p[1]) * (v * tf + x_theo1),
)
print("p=0.7, lin: ", np.mean(err4))

print("\n")
print("p=0.5, theorical arb: ", (1 - p[2]) * 2 * v * tf)
print("p=0.5, arb: ", np.mean(err2))
print(
    "p=0.5, theorical lin: ",
    p[2] * (v * tf - x_theo2) + (1 - p[2]) * (v * tf + x_theo2),
)
print("p=0.5, lin: ", np.mean(err5))
print("\n")
print("p=0.3, theorical arb: ", (1 - p[3]) * 2 * v * tf)
print("p=0.3, arb: ", np.mean(err3))
print(
    "p=0.3, theorical lin: ",
    p[3] * (v * tf - x_theo3) + (1 - p[3]) * (v * tf + x_theo3),
)
print("p=0.3, lin: ", np.mean(err6))

plt.figure()
plt.plot(temps, np.mean(err1) * np.array([1] * len(temps)), label="arb/p=0.7")
plt.plot(temps, np.mean(err4) * np.array([1] * len(temps)), label="lin/p=0.7")
plt.plot(
    temps,
    (1 - p[1]) * 2 * v * tf * np.array([1] * len(temps)),
    label="theo_arb/p=0.7",
    ls="--",
)
plt.plot(
    temps,
    (p[1] * (v * tf - x_theo1) + (1 - p[1]) * (v * tf + x_theo1))
    * np.array([1] * len(temps)),
    label="theo_lin/p=0.7",
    ls="--",
)
plt.xlabel("time (s)")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

plt.figure()
plt.plot(temps, np.mean(err2) * np.array([1] * len(temps)), label="arb/p=0.5", ls="--")
plt.plot(temps, np.mean(err5) * np.array([1] * len(temps)), label="lin/p=0.5", ls="--")
plt.plot(
    temps, (1 - p[2]) * 2 * v * tf * np.array([1] * len(temps)), label="theo_arb/p=0.5"
)
plt.plot(
    temps,
    (p[2] * (v * tf - x_theo2) + (1 - p[2]) * (v * tf + x_theo2))
    * np.array([1] * len(temps)),
    label="theo_lin/p=0.5",
)

plt.xlabel("time (s)")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

plt.figure()
plt.plot(temps, np.mean(err3) * np.array([1] * len(temps)), label="arb/p=0.3", ls=":")
plt.plot(temps, np.mean(err6) * np.array([1] * len(temps)), label="lin/p=0.3", ls=":")
plt.plot(
    temps,
    (1 - p[3]) * 2 * v * tf * np.array([1] * len(temps)),
    label="theo_arb/p=0.3",
    ls="--",
)
plt.plot(
    temps,
    (p[3] * (v * tf - x_theo3) + (1 - p[3]) * (v * tf + x_theo3))
    * np.array([1] * len(temps)),
    label="theo_lin/p=0.3",
    ls="--",
)
plt.xlabel("time (s)")
plt.ylabel("error (deg)")
plt.legend()
plt.show()

