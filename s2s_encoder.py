import sys
sys.path.append('./fairseq')
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import HubertFeatureReader
import torch
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.transformer.transformer_config import DecoderConfig
import torch.nn as nn
from fairseq.data import Dictionary
import torch.nn.functional as F
import nltk
from fairseq.modules import GradMultiply, LayerNorm
import time
import os
import random
from hubert_extractor import HubertExtractor
from transformers.modeling_outputs import BaseModelOutput
from transformers import BartConfig
from bart_decoder import BartForConditionalGeneration, BartLearnedPositionalEmbedding

def init_xavier(m):
    nn.init.xavier_uniform_(m.weight)
    
def init_uniform(m):
    try:
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
    except:
        nn.init.uniform_(m, a=-0.1, b=0.1)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb

def streaming_mask(size, left_context = 5, right_context = 0, type_ = 'float'):
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask_value = True
    #making the unwated positions
    for i in range(0, size):
        mask[i][0 : max(0, i - left_context)] = mask_value
        mask[i][i + 1 + right_context:] = mask_value
    return mask

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

def truncate_batch(batch, max_tokens = 700):
    while True:
        max_len = max([len(seq) for seq in batch])
        num_tokens = len(batch) * max_len
        if num_tokens > max_tokens:
            print('truncated', max_len)
            max_len = int(max_len  * 0.75)
            batch = [seq[:max_len] for seq in batch]
        else:
            break
    return batch
    
def pad_to_max_length(source, pad_id = -100, batch_max_tokens = 700):
    source = truncate_batch(source, batch_max_tokens)
    max_length = max(len(inner_list) for inner_list in source)
    for inner_list in source:
        while len(inner_list) < max_length:
            inner_list.append(pad_id)
    return source

import uuid
def create_embedding_layer(num_vocab, HIDDEN_DIM):
    lines = [' '.join([str(i), '1']) for i in range(0,num_vocab)]
    file_name = uuid.uuid4().hex
    with open(file_name,'w') as f:
        f.write('\n'.join(lines))
    dictionary = Dictionary()
    dictionary = dictionary.load(file_name, add_special_symbols=False)
    dictionary.pad_index = 0
    embed = build_embedding(dictionary, HIDDEN_DIM)
    os.remove(file_name)
    return dictionary, embed

def pad_encoder_outs(encoder_outs):
    max_len = max(out.size(0) for out in encoder_outs)
    padded_encoder_outs = []
    encoder_padding_mask = []
    
    for out in encoder_outs:
        seq_len = out.size(0)
        padded_out = torch.cat([out, torch.zeros(max_len - seq_len, out.size(1)).to(out.dtype).to(out.device)], dim=0)
        padded_encoder_outs.append(padded_out)
        mask = torch.cat([torch.zeros(seq_len, dtype=torch.bool), torch.ones(max_len - seq_len, dtype=torch.bool)])
        encoder_padding_mask.append(mask)
    
    padded_encoder_outs = torch.stack(padded_encoder_outs)
    encoder_padding_mask = torch.stack(encoder_padding_mask).to(padded_encoder_outs.device)
    return padded_encoder_outs, encoder_padding_mask

def pad_right_encoder_outs(encoder_outs):
    max_len = max(out.size(0) for out in encoder_outs)
    padded_encoder_outs = []
    encoder_padding_mask = []
    is_padded = False
    for out in encoder_outs:
        seq_len = out.size(0)
        pad_len = max_len - seq_len
        if pad_len > 0:
            is_padded = True
        padded_out = torch.cat([torch.zeros(pad_len, out.size(1)).to(out.dtype).to(out.device), out], dim=0)
        padded_encoder_outs.append(padded_out)
        mask = torch.cat([torch.zeros(pad_len), torch.ones(seq_len)])
        encoder_padding_mask.append(mask)
    
    padded_encoder_outs = torch.stack(padded_encoder_outs)
    encoder_padding_mask = torch.stack(encoder_padding_mask).to(torch.int64).to(padded_encoder_outs.device)
    if is_padded:
        return padded_encoder_outs, encoder_padding_mask
    else:
        return padded_encoder_outs, None

import numpy as np

def get_sinusoidal_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]  # Shape (S_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # Shape (1, 768)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle = pos * angle_rates  # Shape (S_len, 768)    
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(angle[:, 0::2])  # Sin for even indices
    pos_enc[:, 1::2] = np.cos(angle[:, 1::2])  # Cos for odd indices
    return torch.tensor(pos_enc, dtype=torch.float32).unsqueeze(0)

class S2SEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        num_vocab = 32000
        HIDDEN_DIM = 768
        self.hidden_dim = HIDDEN_DIM
        
        encoder = HubertFeatureReader(
            checkpoint_path = './hubert_fisher.pt',
            layer = 12,
            use_cuda = True if self.device == 'cuda' else False
        )
        encoder.model.encoder.layers = encoder.model.encoder.layers[:8]
        self.encoder = encoder.model.encoder
        
        self.pad_id = 0
        self.start_id = 1
        self.eos = 2
        self.vocab_size = num_vocab
        self.text_vocab_size = 32000
        
        # unit decoder
        config = BartConfig()
        del config.id2label, config.label2id
        config.bos_token_id = 1
        config.d_model = HIDDEN_DIM*2
        config.decoder_ffn_dim = 2048
        config.decoder_layers = 10
        config.decoder_attention_heads = 12
        config.decoder_start_token_id = 1
        config.eos_token_id = 2
        config.forced_eos_token_id = 2
        config.num_hidden_layers = 6
        config.pad_token_id = 0
        config.vocab_size = num_vocab
        config.max_position_embeddings = 256 + 512
        self.decoder = BartForConditionalGeneration(config)
        
        # ----------- text decoder
        text_dictionary, text_embed = create_embedding_layer(self.text_vocab_size, HIDDEN_DIM)
        args = DecoderConfig(embed_dim = HIDDEN_DIM, ffn_embed_dim = 2048, layers = 2,
                             attention_heads=8, learned_pos=True, input_dim = HIDDEN_DIM,
                             normalize_before = False)
        args.max_target_positions = 1024
        args.dropout = 0.05
        self.text_decoder = TransformerDecoder(args, text_dictionary, text_embed)
        self.left_contexts = []
    
    def encode(
            self,
            source: torch.Tensor,
            prev_hidden_states,
            text_labels,
            reduce_layers = [2, 4, 6],
            cur_position = 0,
            alignment_layer = None,
    ):
        device = self.device
        batch_size = len(source['spk0'])
        x = source
        
        # merge x from two speakers
        x = torch.cat((x['spk0'], x['spk1']), dim = 0)  # x = [B x spk1, B x spk2] * S_len * 1024
        source_len = x.size(1)
        
        x = x.to(self.encoder.layer_norm.weight.dtype).to(self.encoder.layer_norm.weight.device)
        x = F.dropout(x, p=self.encoder.dropout, training=self.encoder.training)
        x = x.transpose(0, 1) # B x T x C -> T x B x C
        
        new_hidden_states = {'spk0': [], 'spk1': []}
        
        for i, layer in enumerate(self.encoder.layers):
            # Save cur and prepare previous states
            # new_hidden_states['spk0'] = [ B x S_len x 768]
            
            x = x[-source_len:, :, :]
            new_hidden_states['spk0'].append(x.transpose(0, 1)[:batch_size])
            new_hidden_states['spk1'].append(x.transpose(0, 1)[batch_size:])
            
            prev_states_i = torch.cat((prev_hidden_states['spk0'][i],
                                       prev_hidden_states['spk1'][i]), dim = 0)
            
            x = torch.cat((prev_states_i, x.transpose(0, 1)), dim = 1).transpose(0, 1)
            
            x, (z, lr) = layer(
                x, 
                self_attn_mask = streaming_mask(x.size(0), self.left_contexts[i]).to(device),
                need_weights=False
            )
            
            if (i+1) in reduce_layers:
                x_reshaped = x.view(int(x.size(0)/2), 2, x.size(1), x.size(2))
                x = torch.mean(x_reshaped, dim=1)
                source_len = int(source_len/2)
        
        x = x.transpose(0, 1) # B x S_len x 768
        x = self.encoder.layer_norm(x)        
        x = x[:, -source_len:, :]
        new_hidden_states['spk0'].append(x[:batch_size])
        new_hidden_states['spk1'].append(x[batch_size:])
        
        loss_fct = nn.CrossEntropyLoss(label_smoothing = 0.1)
        encoder_outs, labels, text_preds, text_truths = [], [], [], []
        loss_text = torch.tensor(0.0).to(self.device)
        
        for spk in ['spk0','spk1']:
            for i in range(0, batch_size):
                encoder_out = new_hidden_states[spk][-1][i]
                for seg in text_labels[spk][i]:
                    encoder_outs.append(encoder_out[seg['sid'] : seg['eid']])
                    labels.append(seg['tokens'][1:])
        
        if len(labels) > 0:
            encoder_outs, encoder_masks = pad_encoder_outs(encoder_outs)
            pos_emb = get_sinusoidal_positional_encoding(encoder_outs.shape[1], encoder_outs.shape[2]).to(encoder_outs.device)
            encoder_outs = encoder_outs + 0.5*pos_emb
            encoder_outs = F.layer_norm(encoder_outs, encoder_outs.size()[1:], eps=1e-6)
            labels = pad_to_max_length(labels, -100, 10000000)
            labels = torch.Tensor(labels).type(torch.int64).to(self.device)
            decoder_inputs_ids = shift_tokens_right(labels, self.pad_id, self.start_id).to(self.device)
            encoder_out = {'encoder_out':[encoder_outs.permute(1,0,2)], 'encoder_padding_mask':[encoder_masks]}
            logits, extra = self.text_decoder(decoder_inputs_ids, encoder_out)
            logits = logits.reshape(-1, self.text_vocab_size)
            labels = labels.reshape(-1)
            loss_text = loss_fct(logits, labels)
            text_preds = torch.argmax(F.softmax(logits, dim = 1), dim = 1).tolist()
            text_truths = labels.tolist()
        
        return new_hidden_states, loss_text, text_preds, text_truths
    
    def decode(
            self,
            encoder_outs,
            all_labels,
    ):
        encoder_outs, encoder_masks = pad_right_encoder_outs(encoder_outs)
        
        #multi_encoder_outs = B x 256 x [768, 768]
        x1 = encoder_outs[:,:,:self.hidden_dim]
        x2 = encoder_outs[:,:,self.hidden_dim:]
        
        pos_emd    = self.decoder.dlm_positions(x1) # 1 x S_len x 768
        pos_emd    = torch.flip(pos_emd, dims=[1])
        weight_pos = torch.linspace(0.5, 2, pos_emd.shape[1])
        weight_pos = weight_pos.unsqueeze(0).unsqueeze(2).to(pos_emd.dtype).to(pos_emd.device)
        pos_emd = pos_emd * weight_pos
        
        spk1_emd = self.decoder.speaker_embedding(torch.Tensor([0]*x1.shape[1]).long().to(x1.device)).unsqueeze(0) #1x1x768
        spk2_emd = self.decoder.speaker_embedding(torch.Tensor([1]*x2.shape[1]).long().to(x1.device)).unsqueeze(0) #1x1x768
        
        x1 = x1 + pos_emd + spk1_emd
        x2 = x2 + pos_emd + spk2_emd
        x1 = self.decoder.layernorm_combine_both(x1)
        x2 = self.decoder.layernorm_combine_both(x2)
        encoder_outs = torch.cat((x1,x2), dim = 2)
        
        labels = pad_to_max_length(all_labels, -100, self.max_token_per_batch)
        labels = torch.Tensor(labels).type(torch.int64).to(self.device)
        decoder_inputs_ids  = shift_tokens_right(labels, self.pad_id, self.start_id).to(self.device)
        decoder_inputs_mask = (decoder_inputs_ids != self.pad_id).long().to(self.device)
        
        e_outs = BaseModelOutput(last_hidden_state = encoder_outs, hidden_states = None, attentions = encoder_masks)
        outs = self.decoder(encoder_outputs = e_outs, decoder_input_ids = decoder_inputs_ids,
                            decoder_attention_mask = decoder_inputs_mask, labels = labels)
        
        return outs, labels
    
    def forward(self, features, prev_hidden_states, encoder_mask, cur_position, labels, text_labels):
        # features = {'spk0': [B X S_len, 768], 'spk1': [B X S_len, 768]}
        # prev_hidden_states = {'spk0': 12-List[[S_len,768],..,[S_len,768] ] , 'spk1': 12-List[[],[]]}
        # encoder_mask = S_len * S_len
        # labels: {'spk0' : B x [[4, 5], [6, 10], [123, 123, 123]] , 'spk1': []}
        # cur_pos
        batch_size = len(features['spk0'])
        hidden_states, loss_text, text_preds, text_truths = self.encode(features, prev_hidden_states,
                                                                        text_labels, cur_position = cur_position)
        
        # hidden_states = {'spk0': [B X S_len, 768], 'spk1': [B X S_len, 768]}
        
        tag_preds, tag_truths, seq_preds, seg_truths = [], [], [], []
        
        # 0 : SPK , 1 SIL, 2 CNT, 3 : FORC_SIL
        loss_tag_fct = nn.CrossEntropyLoss(reduction='none')
        loss_seq_fct = nn.CrossEntropyLoss(label_smoothing = 0.1)
        loss_tag = torch.tensor(0.01).to(self.device)
        loss_seq = torch.tensor(0.01).to(self.device)
        
        all_encoder_outs = []
        all_labels = []
        single_encoder_outs = []
        all_single_labels = []
        
        for k in range(0, batch_size):
            for spk in ['spk0','spk1']:
                single_labels = [value[:1] for index, value in enumerate(labels[spk][k])]
                multi_ids     = [index for index, value in enumerate(labels[spk][k]) if len(value) > 2]
                multi_labels  = [value for index, value in enumerate(labels[spk][k]) if len(value) > 2]
                
                encoder_outs = torch.cat((prev_hidden_states[spk][-1][k], hidden_states[spk][-1][k]), dim = 0) # S_lenx768
                other_spk = 'spk0' if spk == 'spk1' else 'spk1'
                encoder_outs_other = torch.cat((prev_hidden_states[other_spk][-1][k], hidden_states[other_spk][-1][k]), dim=0)
                encoder_outs = torch.cat((encoder_outs, encoder_outs_other), dim = -1)    # S_len x 1536
                
                if len(multi_labels) > 0:
                    for idx in multi_ids:
                        encoder_out = encoder_outs[~encoder_mask[idx]]
                        all_encoder_outs.append(encoder_out)
                    all_labels += multi_labels
                
                # for tags predictions
                valid_ids = [
                    i for i, label in enumerate(single_labels)
                    if not (self.training and label[0] in [4, 5] and random.random() < 0.97)
                ]
                single_labels = [single_labels[i] for i in valid_ids]
                
                for idx in valid_ids:
                    single_encoder_outs.append(encoder_outs[~encoder_mask[idx]])
                all_single_labels += single_labels
        
        if len(all_labels) > 0:
            outs, pad_labels = self.decode(all_encoder_outs, all_labels)
            logits = outs.logits
            logits_seq = logits[:,1:,:].reshape(-1, self.vocab_size)
            labels_seq = pad_labels[:,1:].reshape(-1)
            loss_seq = loss_seq_fct(logits_seq, labels_seq)
            seq_preds  = torch.argmax(F.softmax(logits_seq, dim = 1), dim = 1).tolist()
            seg_truths = labels_seq.tolist()
        else:
            seq_preds, seg_truths = [1], [1]
        
        tag_preds, tag_truths = [1], [1]
        
        ids = {'tag_preds' : tag_preds, 'tag_truths': tag_truths,
               'seq_preds' : seq_preds, 'seg_truths': seg_truths,
               'text_preds' : text_preds, 'text_truths': text_truths}
        
        return hidden_states, (loss_tag, loss_seq, loss_text), ids