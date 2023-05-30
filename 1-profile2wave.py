import wave
import numpy as np
from scipy.io.wavfile import write
from scipy import stats, signal, integrate, interpolate

def read_wave(file):
    with wave.open(file, 'rb') as f:
        nchannels, sampwidth, framerate, nframes = f.getparams()[:4]
        str_data                                 = f.readframes(nframes)
    wave_data       = np.frombuffer(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    return wave_data.T[0]

if __name__ == '__main__':

    wave_length = 10 # second
    sample_rate = 48000
    repeat_num  = 10

    data        = np.load('Data/Burst.npy')
    data        = np.mean(data, axis=1)
    data        = np.tile(data, repeat_num)

    time        = np.arange(len(data)) / (len(data) - 1) * wave_length
    f           = interpolate.interp1d(time, data, kind='linear')
    wave_time   = np.arange(0, wave_length, 1 / sample_rate)
    wave_raw    = f(wave_time)
    # wave_raw    = ((wave_raw - np.min(wave_raw)) / (wave_raw.max() - wave_raw.min()) * 6e4 - 3e4).astype(np.int16)
    wave_raw    = ((wave_raw - np.min(wave_raw)) / (wave_raw.max() - wave_raw.min()) * 6e4).astype(np.int16)

    if True:
        # 读取音频文件
        sound      = read_wave('Instruments/vio.wav')

        # 归一化乐器声音
        sounds     = stats.binned_statistic(
            x      = np.arange(0, len(sound)), 
            values = sound, 
            bins   = len(sound) * 44100 / 48000
        )[0]

        L          = integrate.simps(sounds)
        sound_norm = sounds / L

        # 卷积
        wave_raw   = signal.convolve(wave_raw, sound_norm, mode='same')
        wave_raw   = ((wave_raw - np.min(wave_raw)) / (np.max(wave_raw) - np.min(wave_raw)) * 3e4).astype(np.int16)
    
    write('Audio.wav', sample_rate, wave_raw)