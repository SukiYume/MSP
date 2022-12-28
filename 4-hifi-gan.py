import numpy as np
import os, json, torch
from scipy.io.wavfile import write
from skimage.transform import resize

from HiFiGAN.env import AttrDict
from HiFiGAN.models import Generator

def del_burst(data, exposure_cut=25):
    h, w            = data.shape
    data           /= np.mean(data, axis=0)
    flatten_data    = np.sort(data.flatten())
    vmin, vmax      = flatten_data[int(h*w/exposure_cut)], flatten_data[int(h*w/exposure_cut*(exposure_cut-1))]
    data[data<vmin] = vmin
    data[data>vmax] = vmax
    data            = (data - data.min()) / (data.max() - data.min())
    return data

def rescale_data(data):
    data = resize(data, (data.shape[0], 80))
    data = (data - data.min()) / (data.max() - data.min())
    h, w = data.shape
    a    = np.histogram(data.flatten(), bins=int(h*w/100))
    b, c = (a[1][1:] + a[1][:-1]) / 2, a[0]
    d    = 0.6 - b[np.argmax(c)]
    data = (data + d) * 12 - 10.5
    data = np.clip(data, -11, 1.6)
    return data.T[np.newaxis, :, :]

def generate_wav(x):

    with open(config_path) as f:
        config = json.load(f)

    h          = AttrDict(config)
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator  = Generator(h).to(device)
    state_dict_g = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval().remove_weight_norm()

    with torch.no_grad():
        x      = torch.FloatTensor(x).to(device)
        audio  = generator(x).squeeze() * 65536 / 2
        audio  = audio.cpu().numpy().astype('int16')

    return audio, h.sampling_rate

if __name__ == '__main__':

    data_path   = 'Data/RawBurst.npy'
    save_path   = './RawBurst_Generated.wav'
    config_path = 'HiFiGAN/model/config.json'
    checkpoint_path = 'HiFiGAN/model/g_03000000'

    ###### 消色散前的爆发，0-axis是时间，1-axis是频率 ######
    data = np.load(data_path)
    data = np.mean(data.reshape(128, 22, 512, 8), axis=(1, 3))
    data = rescale_data(data)

    ###### ParkesBurst.npy 0-axis是频率，1-axis是时间 ######
    data = np.load('./Data/ParkesBurst.npy').T[:, ::-1]
    data = np.mean(data.reshape(data.shape[0], data.shape[1]//4, 4), axis=2)
    data = rescale_data(data)

    ###### HiFiGAN Mel to Wav ######
    wave, sr = generate_wav(data)
    write(save_path, sr, wave)