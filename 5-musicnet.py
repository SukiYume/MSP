import torch, sys
from scipy.io import wavfile
import librosa, tqdm, pathlib, platform

if platform.system() == 'Windows':
    posix_backup = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
sys.path.append('./MusicNet')

from MusicNet import utils, wavenet_models
from MusicNet.wavenet import WaveNet
from MusicNet.wavenet_generator import WavenetGenerator

def convert_audio(decoder_id, checkpoint_type, file_path, rate):

    checkpoint_path = model_path + '{}_{}.pth'.format(checkpoint_type, decoder_id)
    model_args = torch.load(model_path + 'args.pth')[0]

    encoder = wavenet_models.Encoder(model_args)
    encoder.load_state_dict(torch.load(checkpoint_path)['encoder_state'])
    encoder.eval()
    encoder = encoder.cuda()

    decoder = WaveNet(model_args)
    decoder.load_state_dict(torch.load(checkpoint_path)['decoder_state'])
    decoder.eval()
    decoder = decoder.cuda()
    decoder = WavenetGenerator(decoder, batch_size=batch_size, wav_freq=rate)

    data, _ = librosa.load(file_path, sr=rate)
    data = utils.mu_law(data)

    xs = torch.stack([torch.tensor(data).unsqueeze(0).float().cuda()]).contiguous()
    with torch.no_grad():
        zz = torch.cat([encoder(xs_batch) for xs_batch in torch.split(xs, batch_size)], dim=0)
        with utils.timeit("Generation timer"):
            audio_res = []
            for zz_batch in torch.split(zz, batch_size):
                print(zz_batch.shape)
                splits = torch.split(zz_batch, split_size, -1)
                audio_data = []
                decoder.reset()
                for cond in tqdm.tqdm(splits):
                    audio_data += [decoder.generate(cond).cpu()]
                audio_data = torch.cat(audio_data, -1)
                audio_res += [audio_data]
            audio_res = torch.cat(audio_res, dim=0)
            del decoder
    wavfile.write(save_path, rate, utils.inv_mu_law(audio_res.cpu().numpy()).squeeze())

    return None

if __name__ == '__main__':

    batch_size = 1
    split_size = 20
    model_path = './MusicNet/checkpoints/pretrained_musicnet/'
    save_path  = './MusicNet_Converted.wav'

    convert_audio(
        decoder_id      = 2,
        checkpoint_type = 'bestmodel',
        file_path       = './Data/Burst-wirfi.wav',
        rate            = 16000 # 48000 # 16000
    )