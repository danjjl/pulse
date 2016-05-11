import imageio #mp4 reading lib  (http://imageio.github.io/)
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse, signal
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    """Implements an Asymmetric Least Squares Smoothing baseline correction
    algorithm (P. Eilers, H. Boelens 2005)

    algorithm parameters:
    lam: smoothness parameter (10^2 <= lam <= 10^9)
    p  : asymmetry (0.001 <= p <= 0.1)
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L),2))
    w = np.ones(L)

    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam*np.dot(D,D.T)
        z = spsolve(Z,w*y)
        w = p*(y>z) + (1-p)*(y<z)
    return z


""" Calculate heart rate from mp4 video of thumb using two different methods:
FFT and peak detection
"""

vid = imageio.get_reader('thumb.mp4',  'ffmpeg')
fps = vid._meta['fps']

# Extract brightness at each frame
brightness = np.zeros(len(vid))
for i, im in enumerate(vid):
    brightness[i] = im.mean()


# baseline correction
baseline = baseline_als(brightness, 1500, 0.1)

corrected = brightness - baseline
corrected -= corrected.mean()

# FFT method (maximum of FFT)
FFT = abs(scipy.fft(corrected))
freqs = scipy.fftpack.fftfreq(len(vid), 1.0/fps)
bpmFFT = abs(60*freqs[np.argmax(FFT)])

# Peak detection method (number of spikes)
pozPeaks = signal.find_peaks_cwt(corrected, np.arange(2,15))
negPeaks = signal.find_peaks_cwt(corrected*-1, np.arange(2,15))
pozBpm = fps*60.0*len(pozPeaks)/len(vid)
negBpm = fps*60.0*len(negPeaks)/len(vid)
bpmPeaks = 0.5*(pozBpm + negBpm)

# Visualize results
time = np.arange(0.0,len(vid))/fps

# Raw brightness data
fig = plt.figure()
plt.plot(time, brightness)
plt.xlabel('time [s]')
plt.ylabel('brightness')
plt.title('Raw brightness data')

# Baseline corrected brightness
fig = plt.figure()
plt.plot(time, corrected)
plt.xlabel('time [s]')
plt.ylabel('brightness')
plt.title('Baseline corrected brightness')

# FFT of baseline corrected brightness
fig = plt.figure()
plt.plot(freqs*60,20*scipy.log10(FFT), '.')
plt.plot(bpmFFT, 20*scipy.log10(max(FFT)), 'ro')
plt.plot(-1*bpmFFT, 20*scipy.log10(max(FFT)), 'ro')
plt.xlabel('BPM')
plt.ylabel('[dB]')
plt.title('FFT of baseline corrected brightness')

# Detected peaks on baseline corrected brightness
fig = plt.figure()
plt.plot(time, corrected)
plt.plot(time[pozPeaks], corrected[pozPeaks], 'ro')
plt.plot(time[negPeaks], corrected[negPeaks], 'go')
plt.xlabel('time [s]')
plt.ylabel('brightness')
plt.title('Detected peaks on baseline corrected brightness')

# Print results
print("FFT method:  %1.1f" %(bpmFFT))
print("Peak method: %1.1f" %(bpmPeaks))

plt.show()
