import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from cyclic_time_func import get_data


# Working example

# # Number of samplepoints
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
# import matplotlib.pyplot as plt
# plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
# plt.grid()
# plt.show()


# Pull disconnect sample
# NEED TO PULL SAMPLE OF PERIODIC DATA 
# (WEEKS WORTH OF DISCONNECTS, OR ALL CHURN WITH CHURN_TYPE, CONVERT DATETIME TO TIME FROM START.)
x = get_data()

# Fourier Transform
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)

#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()



