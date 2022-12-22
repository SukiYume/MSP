import numpy as np
from astropy.table import Table
from astronify.series import SoniSeries

if __name__ == '__main__':

    down_rate  = 10
    data       = np.load('Data/Burst.npy')
    data       = np.mean(data, axis=1)
    data       = (data - data.min()) / (data.max() - data.min())
    data       = np.mean(data[:len(data) // down_rate * down_rate].reshape(-1, down_rate), axis=1)

    data_table = Table({'time': np.arange(len(data)), 'flux': data})
    data_soni  = SoniSeries(data_table)
    data_soni.note_spacing = 0.01
    data_soni.sonify()
    data_soni.write('Audio.wav')