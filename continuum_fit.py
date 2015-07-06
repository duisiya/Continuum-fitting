import numpy as np
import asciidata
import matplotlib.pyplot as plt

    
#read synthetic spectrum
spec_synth = asciidata.open('low_res_synth_spectrum_Jband.dat')
wave_synth = spec_synth[0].tonumpy()
flux_synth = spec_synth[1].tonumpy()

#The continuum is presented as a combination of sin and cos functions.
#The matrix equation F = AX, where F is flux, A is coefficient matrix (values for sin and cos at each wavelenght) and X is solutution matrix
K = 5 #the number of components you want to fit; play around with this parameter, probably, you don't want more than 5. but it also depends on a spectral resoltuion
L = (max(wave_synth) - min(wave_synth)) *2
A = np.vstack([np.vstack([np.cos(2*np.pi*k*wave_synth/L) for k in range(K)]),
               np.vstack([np.sin(2*np.pi*k*wave_synth/L) for k in range(1,K)])]) #coefficient matrix A

ATA = np.dot(A, A.T)
ATy = np.dot(A, flux_synth)
X = np.linalg.solve(ATA, ATy)#solution
F = np.dot(X, A) #continuum

plt.plot(wave_synth, flux_synth, label='spectrum')
plt.plot(wave_synth, F, linewidth = 3, label='continuum')
plt.legend()
plt.xlabel('Wavelength (microns)')
plt.ylabel('Flux')
plt.show()

