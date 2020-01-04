import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import math


def get_af(t_, TMAX):
    # af_ = 2.0 * t_ / TMAX
    # p = 0.5 * TMAX
    # if t_ > p:
    #     af_ = 2.0 * (TMAX - t_) / TMAX
    af_ = -(t_-0.5)*(t_-0.5)+0.25
    return af_


XL = 0.0  # левый конец отрезка
XR = 1.0  # правый конец отрезка
TMAX = 1.0  # максимальное время

N = 101  # число узлов сетки по пространству
M = 21  # число узлов сетки по времени
DELTA = 0.001  # уровень погрешности во входных данных


H = (XR - XL) / (N - 1)
x = np.array([XL+i*H for i in range(N)])
print('H = ', H)
TAU = TMAX / (M - 1)
print('TAU = ', TAU)


ua = np.zeros(M)
u = np.array([get_af(i*TAU, TMAX) for i in range(M)])
y, z, a, b, c, f, q = [np.zeros(N) for i in range(7)]
fiy = np.zeros(M)


def prog3(a_, c_, b_, f_, y_):
    b_[0] = b_[0] / c_[0]
    for i in range(1, N):
        c_[i] = c_[i] - b_[i - 1] * a_[i]
        b_[i] = b_[i] / c_[i]
    f_[0] = f_[0] / c_[0]
    for i in range(1, N):
        f_[i] = (f_[i] + f_[i - 1] * a_[i]) / c_[i]
    y_[-1] = f_[-1]
    for i in range(N - 2, -1, -1):
        y_[i] = b_[i] * y_[i + 1] + f_[i]
    return a_, c_, b_, f_, y_


def schema_koeff(a_, b_, c_, f_, y_, u_k, i_=0, f_i = None):
    a_[1:N - 1-i_] = 1.0 / (H * H)
    b_[1:N - 1-i_] = 1.0 / (H * H)
    c_[1:N - 1-i_] = a_[1:N - 1-i_] + b_[1:N - 1-i_] + 1.0 / TAU
    if f_i is None:
        f_[1:N - 1-i_] = y_[1:N - 1-i_] / TAU
    else:
        f_[1:N - 1-i_] = f_i

    b_[0] = 2.0 / (H * H)
    c_[0] = b_[0] + 1.0 / TAU
    f_[0] = y_[0] / TAU
    a_[-1] = 0.0
    c_[-1] = 1.0
    f_[-1] = u_k
    return a_, c_, b_, f_, y_

def non_clear_schema(a_, b_, c_, f_, y_):
    fi_ = np.zeros(M)
    fid_ = np.zeros(M)
    for k in range(1,M):
        schema_koeff(a_, b_, c_, f_, y_, u[k])
        a_, c_, b_, f_, y_ = prog3(a_, c_, b_, f_, y_)
        fi_[k] = y[0]
        fid_[k] = fi_[k] + 2 * DELTA * (random.uniform(-0.25, 0.25) - 0.5)
    return fi_, fid_

fi, fid = non_clear_schema(a, b, c,  f, y)

a, c, b, f, y = schema_koeff(a, b, c, f, y, 1, 0, 0)
a, c, b, f, q = prog3(a, c, b, f, q)

ITMAX = 100
ALPHA = 0.001
QQ = 0.75
l = 0

for it in range(ITMAX):
    y = np.zeros(N)
    ua[0] = y[-1]
    for k in range(1, M):
        a, c, b, f, y = schema_koeff(a, b, c, f, y, 0, 1)
        a, c, b, f, z = prog3(a, c, b, f, z)
        ua[k] = (fid[k] - z[0]) / (ALPHA + q[0])
        y[:] = z[:] + q[:]*ua[k]
    y = np.zeros(N)
    fiy[0] = y[0]
    k = 0
    for k in range(1, M):
        a, c, b, f, y = schema_koeff(a, c, b, f, y, ua[k], 0)
        a, c, b, f, y = prog3(a, c, b, f, y)
        fiy[k] = y[0]
    sl2 = math.sqrt(np.sum((fiy[:] - fid[:]) ** 2 * TAU))
    if (it == 0):
        if (sl2 < DELTA):
            QQ = 1.0 / QQ
    ALPHA = ALPHA * QQ

k_range = np.linspace(1, 21, M)
t_range = np.linspace(0, 1, M)

plt.plot(t_range, u, color='black')
plt.plot(t_range, ua, 'y--')
plt.plot(t_range, fiy, color='blue')
plt.plot(t_range, fid, 'r--')
plt.show()
