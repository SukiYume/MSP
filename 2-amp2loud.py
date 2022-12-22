import numpy as np
from scipy.io.wavfile import write
from scipy.interpolate import interp1d

def sin_fu(x, a, b):
    return a * np.sin(1000 * x) + b

if __name__ == '__main__':

    sr          = 48000 # 采样率
    wave_length = 2     # 音频长度，s
    int_number  = sr * wave_length

    b    = np.load('Data/Profile.npy')
    b    = np.log10((b - np.min(b)) / (b.max() - b.min()) + 1)
    a    = np.linspace(0, 2*np.pi, len(b))

    c    = interp1d(a, b)
    d    = np.linspace(a.min(), a.max(), int_number)
    e    = c(d)
    g    = sin_fu(d, e, e)
    wave = (g * (30000 / g.max())).astype(np.int16)

    write('Audio.wav', sr, wave)