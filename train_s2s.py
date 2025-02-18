#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.
import os
os.environ['TRANSFORMERS_CACHE'] = '../trans_cache/'
os.environ['HF_DATASETS_CACHE'] = '../trans_cache/'
os.environ['HF_HOME'] = '../trans_cache/'

from multiprocessing.pool import Pool
import soundfile as sf
import torch
import time
import pickle
import sys

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

from hubert_extractor import HubertExtractor
from s2s_encoder_direct import S2SEncoder

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from accelerate import DistributedDataParallelKwargs
import threading
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, AutoProcessor

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
import contextlib

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CyclicLR
        
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_file", type=str, default=None
    )
    parser.add_argument(
        "--validation_file", type=str, default=None
    )
    parser.add_argument("--tag_weight", type=float, default=0.1, required=True)
    parser.add_argument("--text_weight", type=float, default=0.25, required=True)
    parser.add_argument("--max_lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=20000)
    parser.add_argument(
        "--audio_path", type=str, default='./datasets/audios/', required=True
    )
    parser.add_argument(
        "--log_file", type=str, default='log.txt'
    )
    parser.add_argument(
        "--pretrain_file", type=str, default=None
    )
    parser.add_argument("--output_dir", type=str, default='./ckpt')
    parser.add_argument("--loss_steps", type=int, default=200)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--skip_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--frame_per_chunk", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_token_per_batch", type=int, default=None)
    parser.add_argument("--dialog_context", type=int, default=256)
    parser.add_argument("--asr_context", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--is_pretraining", action="store_true"
    )
    parser.add_argument(
        "--clip_norm", action="store_true"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max_train_steps", type=int, default=None
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
def main():
    args = parse_args()
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(filename = args.log_file,
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    feature_extractor = HubertExtractor(audio_path = args.audio_path)
    
    accelerator = Accelerator(mixed_precision='bf16',
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
                              )
    logger.info(str(args))
    logger.info(accelerator.state)
    logger.info(accelerator.device)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    args.min_frame_per_chunk = 512
    args.reduct_times = 8
    args.left_contexts   = [args.asr_context]*2 + [args.asr_context//2]*2 + [args.asr_context//4]*2 + [args.dialog_context]*3
    args.num_seq_states  = len(args.left_contexts) 
    args.frame_per_steps = int(args.frame_per_chunk / args.reduct_times)
    
    time.sleep(random.randint(0,3))
    model = S2SEncoder()
    model.left_contexts = args.left_contexts
    model.batch_size = args.batch_size
    args.hidden_dim = model.hidden_dim
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={'train': args.train_file, 'validation': args.validation_file})
    
    if args.output_dir == "None":
        args.output_dir = None
    
    def preprocess_function(examples):
        return {'ids': examples['id'], 'spk0' : examples['0'], 'spk1' : examples['1'],
                'text0': examples['text_0'], 'text1': examples['text_1']}
    
    def tokenize_batch(batch):
        return batch

    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        remove_columns = ['id','0','1']
    )
    
    valid_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        remove_columns = ['id','0','1']
    )
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=tokenize_batch, batch_size=args.per_device_train_batch_size)
    
    eval_dataloader = DataLoader(
        valid_dataset, shuffle=False, collate_fn=tokenize_batch, batch_size=args.per_device_eval_batch_size)
    
    if args.pretrain_file is not None:
        state_dict = torch.load(args.pretrain_file)
        #state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
        model.load_state_dict(state_dict, strict = False)
        print("Pretrain model Loaded !")
    
    EOS_ID = model.eos
    model.dtype = torch.float16
    if args.max_token_per_batch is None:
        args.max_token_per_batch = int(args.frame_per_chunk / 3) * args.batch_size
    model.max_token_per_batch = args.max_token_per_batch
    
    label_pad_token_id = -100
    
    def pad_to_max_length(source, pad_id = -100):
        max_length = max(len(inner_list) for inner_list in source)
        for inner_list in source:
            while len(inner_list) < max_length:
                inner_list.append(pad_id)
        return source
    
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'encoder.' in n],
            "weight_decay": args.weight_decay,
            "lr": args.max_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'encoder.' not in n],
            "weight_decay": args.weight_decay,
            "lr": args.max_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = args.max_lr, betas=(0.9, 0.98), eps=1e-08,)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch * args.per_device_train_batch_size
        overrode_max_train_steps = True
    
    #feature_extractor, model, optimizer, train_dataloader = accelerator.prepare(feature_extractor, model, optimizer, train_dataloader)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    feature_extractor.eval()
    feature_extractor = feature_extractor.to('cuda:1')
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(10000000), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    model_dtype  = accelerator.unwrap_model(model).dtype
    model_device = accelerator.unwrap_model(model).device
    
    def get_features(file_id, streaming = False):
        audio_ids = [file_id + '_0.wav', file_id + '_1.wav']
        features = accelerator.unwrap_model(feature_extractor).extract_features(audio_ids)
        full_features = {'spk0':features[0].to(model_dtype).to(model_device), 
                         'spk1':features[1].to(model_dtype).to(model_device)}
        
        min_length = min(full_features['spk0'].size(0), full_features['spk1'].size(0))
        full_features['spk0'] = full_features['spk0'][:min_length]
        full_features['spk1'] = full_features['spk1'][:min_length]
        assert min_length % args.reduct_times == 0, file_id
        
        return full_features

    def get_features_batch(file_ids):
        audio_ids = [[file_id + '_0.wav', file_id + '_1.wav'] for file_id in file_ids]
        audio_ids = [j for i in audio_ids for j in i]
        features = accelerator.unwrap_model(feature_extractor).extract_features(audio_ids)
        batch_features = {}
        for i in range(0, len(features), 2):
            full_features = {'spk0' : features[i].to(model_dtype).to(model_device), 
                             'spk1' : features[i+1].to(model_dtype).to(model_device)}
            
            min_length = min(full_features['spk0'].size(0), full_features['spk1'].size(0))
            full_features['spk0'] = full_features['spk0'][:min_length]
            full_features['spk1'] = full_features['spk1'][:min_length]
            assert min_length % args.reduct_times == 0

            batch_features[file_ids[i//2]] = full_features
            
        return batch_features
    
    def get_dummy_hidden_states(batch_size):
        hidden_states = {'spk0':[torch.empty((batch_size, 0, args.hidden_dim)).to(model_dtype).to(model_device)] * args.num_seq_states,
                         'spk1':[torch.empty((batch_size, 0, args.hidden_dim)).to(model_dtype).to(model_device)] * args.num_seq_states
                        }
        return hidden_states
    
    def prepare_data(start, end, batch_full_features, batch_labels):
        list_features = {'spk0': [], 'spk1': []}
        list_labels =  {'spk0': [], 'spk1': []}
        list_text_labels = {'spk0': [], 'spk1': []}
        
        for k in range(0,len(batch_full_features)):
            full_features = batch_full_features[k]
            batch = batch_labels[k]
            list_features['spk0'].append(full_features['spk0'][start:end,:])
            list_features['spk1'].append(full_features['spk1'][start:end,:])
            
            chunk_start = int(start//args.reduct_times)
            chunk_end = int(end//args.reduct_times)
            
            label0 = batch['spk0'][chunk_start: chunk_end]
            label1 = batch['spk1'][chunk_start: chunk_end]
            labels = {'spk0': label0, 'spk1': label1}
            
            #truncate and add eos
            for spk in labels:
                for j in range(0,len(labels[spk])):                
                    labels[spk][j] = labels[spk][j] + [EOS_ID]
                    labels[spk][j] = labels[spk][j][:args.max_length]
                list_labels[spk].append(labels[spk])
            
            text_labels = {'spk0': [], 'spk1': []}
            for spk in ['spk0', 'spk1']:
                segs = batch[spk.replace('spk','text')]
                for seg in segs:
                    frame_start = seg['start'] // 160
                    frame_end = seg['stop'] // 160
                    if frame_start >= chunk_start + 2 and frame_end <= chunk_end - 2 and len(seg['tokens']) > 1:
                        relative_start = frame_start - chunk_start - 1
                        relative_end = frame_end - chunk_start + 1
                        text_labels[spk].append({'sid': relative_start, 'eid': relative_end, 'tokens': seg['tokens']})
                list_text_labels[spk].append(text_labels[spk])
        
        list_features['spk0'] = torch.stack(list_features['spk0'], dim = 0)
        list_features['spk1'] = torch.stack(list_features['spk1'], dim = 0)
        return list_features, list_labels, list_text_labels
    
    def get_encoder_mask(start, end, left_context):
        num_seg = (end - start) // args.reduct_times
        encoder_mask = torch.ones((num_seg, left_context + num_seg), dtype=torch.bool).to(model_device)
        for i in range(0, num_seg):
            start_attn = max(0, left_context + i - args.left_contexts[-1])
            end_attn = (left_context + i + 1)
            encoder_mask[i][start_attn : end_attn] = False
        return encoder_mask
    
    import concurrent.futures    
    executor = concurrent.futures.ThreadPoolExecutor()
    def get_features_in_background(sample_ids):
        future = executor.submit(get_features_batch, sample_ids)
        return future
    
    completed_mini_steps = 1
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        tag_losses, seq_losses, text_losses = [], [], []
        
        iterator = iter(train_dataloader)
        batch = next(iterator, None)
        batch = sorted(batch, key = lambda x : len(x['spk0']))
        batch_features = get_features_batch([sample['ids'] for sample in batch])
        
        for step in range(len(train_dataloader) - 1):
            next_batch = next(iterator, None)
            if next_batch is None:
                break
            next_batch = sorted(next_batch, key = lambda x : len(x['spk0']))
            
            if completed_steps < args.skip_steps:
                progress_bar.update(len(next_batch))
                completed_steps += len(next_batch)
                continue
            
            next_batch_features = get_features_in_background([sample['ids'] for sample in next_batch])
            all_full_features, all_full_labels, all_lens = [], [], []
            for sample in batch:
                full_features = batch_features[sample['ids']]
                lable_len = min(len(sample['spk0']), len(sample['spk1']))
                full_labels = {'spk0': sample['spk0'][:lable_len],
                               'spk1': sample['spk1'][:lable_len],
                               'text0': sample['text0'],
                               'text1': sample['text1']}
                
                len_all_device = accelerator.gather(torch.Tensor([full_features['spk0'].size(0)]).to(model_device))
                len_all_device = int(torch.min(accelerator.gather(len_all_device)).item())
                
                all_full_features.append(full_features)
                all_full_labels.append(full_labels)
                all_lens.append(len_all_device)
            
            for z in range(0, len(all_lens), args.batch_size):
                list_full_features = all_full_features[z:z+args.batch_size]
                list_full_labels = all_full_labels[z:z+args.batch_size]
                list_lens = all_lens[z:z+args.batch_size]
                
                if len(list_lens) != args.batch_size:
                    break
                
                # We have the batch now 
                batch_feature_len = min(list_lens)
                step_size = min(batch_feature_len, args.frame_per_chunk)
                prev_hidden_states = get_dummy_hidden_states(args.batch_size)
                
                #prev_hidden_states = {'spk0': List of hidden_staetes of B x S_len x 764}
                
                for start in range(0, batch_feature_len - args.min_frame_per_chunk, step_size):
                    end = min(start + step_size, batch_feature_len)
                    left_context = prev_hidden_states['spk0'][-1][0].size(0)
                    encoder_mask = get_encoder_mask(start, end, left_context)
                    list_features,list_labels,list_text_labels = prepare_data(start,end,list_full_features,list_full_labels)
                    hidden_states, (loss_tag, loss_seq, loss_text), logits = model(list_features,
                                                                                   prev_hidden_states,
                                                                                   encoder_mask,
                                                                                   end // args.reduct_times,
                                                                                   list_labels,
                                                                                   list_text_labels)
                    
                    # this is the complete of single segment 0 -> 1024
                    for spk in ['spk0','spk1']:
                        for i in range(0, args.num_seq_states):
                            prev_hidden_states[spk][i] = torch.cat((prev_hidden_states[spk][i],
                                                                    hidden_states[spk][i]),
                                                                    dim=1).detach()
                            prev_hidden_states[spk][i] = prev_hidden_states[spk][i][:,-args.left_contexts[i]:,:].detach()
                    
                    loss = loss_seq + args.tag_weight*loss_tag + args.text_weight*loss_text
                    loss = loss/args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    completed_mini_steps += 1
                    
                    tag_losses.append(loss_tag.detach().item())
                    seq_losses.append(loss_seq.detach().item())
                    text_losses.append(loss_text.detach().item())
                    
                    if completed_mini_steps % args.gradient_accumulation_steps == 0:
                        if args.clip_norm:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                del prev_hidden_states, list_full_features, list_full_labels, list_lens
                
                # this is after each mini batch of 4 -- but still inside the main batch loop
                if completed_steps % args.loss_steps == 0:
                    seq_str = str(sum(seq_losses)/len(seq_losses))
                    text_str = str(sum(text_losses)/len(text_losses))
                    tag_str = str(sum(tag_losses)/len(tag_losses))
                    logger.info("Seq Loss: " + seq_str)
                    logger.info("Text Loss: " + text_str)
                    logger.info("Tag Loss: " + tag_str)
                    tag_losses, seq_losses, text_losses = [], [], []
                
                if accelerator.sync_gradients:
                    progress_bar.update(args.batch_size)
                    completed_steps += args.batch_size
                
                if completed_steps > 10 and completed_steps % args.eval_steps == 0 and completed_steps >= args.skip_steps:
                    
                    tag_preds, tag_truths = [], []
                    seq_preds, seg_truths = [], []
                    text_preds, text_truths = [], []
        
                    accelerator.wait_for_everyone()
                    model.eval()
                    print("Model evaluation !!!!!!!!!!")
                    
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        eval_batch = eval_batch[0]
                        full_features = get_features(eval_batch['ids'])
                        lable_len = min(len(eval_batch['spk0']), len(eval_batch['spk1']))
                        full_labels = {'spk0': eval_batch['spk0'][:lable_len], 'spk1': eval_batch['spk1'][:lable_len],
                                       'text0': eval_batch['text0'], 'text1': eval_batch['text1']
                                      }
                        
                        list_full_features = [full_features]
                        list_full_labels = [full_labels]
                        
                        # We have the batch now
                        batch_feature_len = full_features['spk0'].size(0)
                        step_size = min(batch_feature_len, args.frame_per_chunk)
                        prev_hidden_states = get_dummy_hidden_states(1)
                        
                        for start in range(0, batch_feature_len - args.min_frame_per_chunk, step_size):
                            end = min(start + step_size, batch_feature_len)
                            left_context = prev_hidden_states['spk0'][-1][0].size(0)
                            encoder_mask = get_encoder_mask(start, end, left_context)
                            list_features,list_labels,list_text_labels = prepare_data(start, end,
                                                                                      list_full_features,
                                                                                      list_full_labels)
                            
                            with torch.no_grad():
                                hidden_states, (loss_tag, loss_seq, loss_text), logits = model(list_features,
                                                                                               prev_hidden_states,
                                                                                               encoder_mask,
                                                                                               end // args.reduct_times,
                                                                                               list_labels,
                                                                                               list_text_labels)
                            for spk in ['spk0','spk1']:
                                for i in range(0, args.num_seq_states):
                                    prev_hidden_states[spk][i] = torch.cat((prev_hidden_states[spk][i],
                                                                            hidden_states[spk][i]),
                                                                            dim=1).detach()
                                    prev_hidden_states[spk][i] = prev_hidden_states[spk][i][:,-args.left_contexts[i]:,:].detach()
                            
                            tag_preds  += logits['tag_preds']
                            tag_truths += logits['tag_truths']
                            seq_preds  += logits['seq_preds']
                            seg_truths += logits['seg_truths']
                            text_preds  += logits['text_preds']
                            text_truths += logits['text_truths']
                    
                    seq_preds, seg_truths = map(list, zip(*[e for e in zip(seq_preds, seg_truths) if e[1] != -100]))
                    text_preds, text_truths = map(list, zip(*[e for e in zip(text_preds, text_truths) if e[1] != -100]))
                    tag_preds, tag_truths = map(list, zip(*[e for e in zip(tag_preds, tag_truths) if e[1] != -100]))
                    logger.info("Seq accuracy score: " + str(accuracy_score(seq_preds, seg_truths)))
                    logger.info("Text accuracy score: " + str(accuracy_score(text_preds, text_truths)))
                    logger.info(str(completed_steps))
                    logger.info("Tag accuracy score: " + str(accuracy_score(tag_preds, tag_truths)))
                    if args.is_pretraining is False:
                        logger.info("Tag accuracy report : \n" + str(classification_report(tag_preds, tag_truths,
                                                                                           output_dict = True)))
                        logger.info("Tag confusion matrix: \n" + str(confusion_matrix(tag_preds, tag_truths)))
                    tag_preds, tag_truths = [], []
                    seq_preds, seg_truths = [], []
                    text_preds, text_truths = [], []
                    
                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            torch.save(unwrapped_model.state_dict(),
                                       os.path.join(args.output_dir, 'ckpt_' + str(completed_steps)))
                    
                    accelerator.wait_for_everyone()
                    model.train()
            
            batch_features = next_batch_features.result()
            batch = next_batch
            accelerator.wait_for_everyone()
            
if __name__ == "__main__":
   main()