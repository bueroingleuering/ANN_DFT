import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks

"""Setup"""

f   = 50                                    # Frequency [Hz]
w   = 2*f*np.pi                             # Omega [1/s]
phase = 0*np.pi/180;                        # Phase delay
n   = 4096                                  # Time Steps
N   = 16                                    # Samples to measure
K   = int(N/2)                              # Wave number
t   = np.arange(0,1/f,1/(f*n))              # Time vector
y   = 0                                     # DC offset
y1  = y + 325*np.sin(w*t+phase)             # Sinus with base frequency
y   = y1 + 7*np.sin(3*(w*t+phase))          # Sinus with base frequency + disturbance
y   = y + 5*np.sin(5*(w*t+phase))
y   = y + 20*np.sin(7*(w*t+phase))

"""Discrete"""

td  = np.arange(0,1/f,1/(f*N))              # Discrete time vector
yd  = np.zeros(N)                           # Discrete signal vector prepare
for i in range(N):
    yd[i] = y[t.tolist().index(td[i])]      # Discrete signal vector

dftmtx = np.fft.fft(np.eye(N))              # Build DFT matrix from FFT

"""Define ANN"""
               
model = ks.Sequential()                         # Sequential ANN model
layer_N = ks.layers.Dense(N,use_bias = False)   # Input layer NxN
model.add(layer_N)                              # Add layer to model
x = tf.ones((N,2*N))                            # Input dummy Nx2N
model(x)                                        # Extend layer to Nx2N

"""Set weights"""

model.set_weights([np.vstack([dftmtx.real,dftmtx.imag])]) # Enforce weights

"""Prediction"""

P = model.predict(np.hstack([yd,yd])[np.newaxis])[0,:] /N # Prediction result

"""Reshaping"""

Y = np.append([],P[0])                      # Initialise wave number vector
for k in range(1,K):
    Y = np.append(Y,P[N-k]-P[k])            # Substract symmetric components
     
"""Plot"""    

f   = f*np.arange(0,K)
plt.figure()
plt.stem(f,Y,'k',use_line_collection=True)
plt.title('Signal freq. domain with ANN DFT',fontsize=20)
plt.grid(True)
plt.xlabel('f [Hz]',fontsize=16)
plt.show()