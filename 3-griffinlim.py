import numpy as np
import librosa, copy
import soundfile as sf
from scipy import signal

def del_burst(data, exposure_cut=25):
    h, w            = data.shape
    data           /= np.mean(data, axis=0)
    flatten_data    = np.sort(data.flatten())
    vmin, vmax      = flatten_data[int(h*w/exposure_cut)], flatten_data[int(h*w/exposure_cut*(exposure_cut-1))]
    data[data<vmin] = vmin
    data[data>vmax] = vmax
    data            = (data - data.min()) / (data.max() - data.min())
    return data

def melspectrogram2wav(mel):
    # transpose
    mel    = mel.T
    # de-noramlize
    mel    = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    # to amplitude
    mel    = np.power(10.0, mel * 0.05)
    m      = _mel_to_linear_matrix(sr, n_fft, n_mels)
    mag    = np.dot(m, mel)
    # wav reconstruction
    wav    = griffin_lim(mag)
    # de-preemphasis
    wav    = signal.lfilter([1], [1, -preemphasis], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def spectrogram2wav(mag):
    # transpose
    mag    = mag.T
    # de-noramlize
    mag    = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db
    # to amplitude
    mag    = np.power(10.0, mag * 0.05)
    # wav reconstruction
    wav    = griffin_lim(mag)
    # de-preemphasis
    wav    = signal.lfilter([1], [1, -preemphasis], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m      = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    m_t    = np.transpose(m)
    p      = np.matmul(m, m_t)
    d      = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def griffin_lim(spectrogram):
    X_best     = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t    = invert_spectrogram(X_best)
        est    = librosa.stft(X_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase  = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t        = invert_spectrogram(X_best)
    y          = np.real(X_t)
    return y

def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')

if __name__ == '__main__':

    ### Params
    sr           = 48000  # Sample rate.
    n_iter       = 200   # Number of inversion iterations
    preemphasis  = 0.97 # or None
    max_db       = 100
    ref_db       = 20
    top_db       = 15
    n_mels       = 512  # Number of Mel banks to generate
    n_fft        = 4096

    ###### 消色散后 ######
    # data            = np.load('Data/Burst.npy')
    # data            = data.astype(np.float64)
    # data            = del_burst(data)
    # data            = data[1024: 1024+2048]
    # data            = np.mean(data.reshape(64, 32, 512, 8), axis=(1, 3))

    ###### 消色散前 ######
    data            = np.load('Data/RawBurst.npy')
    data            = np.mean(data.reshape(128, 22, 512, 8), axis=(1, 3))

    if True:
        ### Mel to Wav
        frame_length = 0.04   # seconds
        win_length   = int(sr * frame_length) # samples.
        hop_length   = win_length // 4        # samples.

        a            = melspectrogram2wav(data)
        sf.write('Audio.wav', a, sr, subtype='PCM_24')

    else:
        ### Mag to Wav
        hop_length   = 19
        win_length   = hop_length * 4

        data         = np.load('Data/RawBurst.npy')
        asd          = spectrogram2wav(data)
        sf.write('Audio.wav', asd, sr, subtype='PCM_24')