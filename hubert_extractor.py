import sys
import os
import torch
import torch.nn as nn
import soundfile as sf
from multiprocessing.pool import Pool
from utils import chunks
sys.path.append('../fairseq')
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_audio(path):
    return sf.read(path)[0]

def save_pt(args):
    if not os.path.exists(args[1]):
        torch.save(args[0], args[1])

class HubertExtractor(nn.Module):
    def __init__(self, hubert_path="./hubert_fisher.pt",
                 audio_path='./datasets/audios/'):
        super().__init__()

        encoder = HubertFeatureReader(
            checkpoint_path=hubert_path,
            layer=1,
            use_cuda=False,
        ).model
        del encoder.encoder.layers
        encoder.to('cuda').eval().half()
        self.model = encoder
        self.AUDIO_PATH = os.path.abspath(audio_path)
        self.pool = Pool(32)

    def get_conv_features(self, features):
        #features B x 1 x 160000
        pad_size = self.model.feature_extractor.conv_layers[0][0].kernel_size[0] - 1
        features = nn.functional.pad(features, (pad_size, 0, 0, 0), mode='constant', value=0)
        features = self.model.feature_extractor.conv_layers[0][0](features)
        features = self.model.feature_extractor.conv_layers[0][1](features)
        features = self.model.feature_extractor.conv_layers[0][2](features)
        
        pad_zero = torch.zeros((features.size(0), features.size(1), 1)).to(features.device).to(features.dtype)
        features = torch.cat((pad_zero, features), dim = 2)
        features = self.model.feature_extractor.conv_layers[0][3](features)
                
        for conv in self.model.feature_extractor.conv_layers[1:]:
            # Padding here
            pad_size = conv[0].kernel_size[0] - 1
            features = nn.functional.pad(features, (pad_size, 0, 0, 0), mode='constant', value=0)
            features = conv(features)
        
        features = features[:,:,1:]
        features = features.transpose(1, 2)
        return features
    
    def extract_features_from_wavform(self, wavform):
        BATCH_SIZE = 4
        INTERVAL = int(0.16 * 16000)
        CHUNK_SIZE  = int(round(12 * 16000))
        CHUNK_SIZE_STEPS = int(CHUNK_SIZE / (0.02 * 16000))
        LEFT_CONTEXT = int(round(4 * 16000))
        LEFT_CONTEXT_STEPS = int(LEFT_CONTEXT / (0.02 * 16000))
        
        # wavform normal from soundilfe read
        with torch.no_grad():
            x = torch.from_numpy(wavform).float()
            
            # Make sure it is devisible by 160
            AUDIO_LEN = len(x)
            if AUDIO_LEN % INTERVAL != 0:
                x = torch.cat((torch.zeros(INTERVAL - (AUDIO_LEN % INTERVAL)), x))
            
            AUDIO_LEN = len(x)
            x = x.to(self.model.final_proj.weight.device).to(self.model.final_proj.weight.dtype)
            
            all_chunks, full_audio_batch = [], []
            
            for f_start in range(0, AUDIO_LEN, CHUNK_SIZE):
                c_start = max(0, f_start - LEFT_CONTEXT)
                c_end   = min(f_start + CHUNK_SIZE, AUDIO_LEN)
                x_chunk = x[c_start : c_end]
                all_chunks.append(x_chunk.view(1, -1))
            
            if len(all_chunks) < 5:
                all_chunks = list(chunks(all_chunks, 1))
            else:
                all_chunks = [all_chunks[:1]] + list(chunks(all_chunks[1:-1], BATCH_SIZE)) + [all_chunks[-1:]]
            
            for idx, batch_audio in enumerate(all_chunks):
                batch_audio = torch.cat(batch_audio, dim = 0)
                features = batch_audio
                # BxT -> BxCxT (where C = 1)
                features = features.unsqueeze(1)
                features = self.get_conv_features(features)
                features = self.model.layer_norm(features)
                if self.model.post_extract_proj is not None:
                    features = self.model.post_extract_proj(features)
                features = self.model.dropout_input(features)
                
                PAD_LEN = 63
                
                padded_features = nn.functional.pad(features, (0, 0, PAD_LEN, 0))
                x_conv = self.model.encoder.pos_conv(padded_features.transpose(1, 2))
                x_conv = x_conv.transpose(1, 2)
                x_conv = x_conv[:,:-PAD_LEN,:]
                features = features + x_conv    # ---> torch.Size([8, 500, 768])
                
                if features.size(1) > LEFT_CONTEXT_STEPS and idx > 0:
                    features = features[:, LEFT_CONTEXT_STEPS:, :]
                features = features[:, -CHUNK_SIZE_STEPS:, :]
                full_audio_batch.append(features.reshape(-1, features.shape[-1]))
            
            full_audio_batch = torch.cat(full_audio_batch, dim = 0).type(torch.bfloat16)
            return full_audio_batch
    
    def extract_features(self, audios_ids):
        audios_paths   = [os.path.join(self.AUDIO_PATH, idx) for idx in audios_ids]
        audios_paths   = [path + '.wav' if '.wav' not in path else path for path in audios_paths]
        
        features_pt = []
        audio_wavs = self.pool.map(read_audio, audios_paths)
        for i in range(0,len(audios_ids)):
            x = audio_wavs[i]
            features_pt.append(self.extract_features_from_wavform(x))
        
        return features_pt
    