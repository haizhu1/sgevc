import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerEmotionLoader, TextAudioSpeakerEmotionCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil
from tqdm import tqdm
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main():
    hps = utils.get_hparams_from_file("./logs/ESD_chinese_semi_3_gamma_1.0_alpha_0.1/config.json")
    net_g = SynthesizerTrn(
                len(symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("./logs/ESD_chinese_semi_3_gamma_1.0_alpha_0.1/G_200000.pth", net_g, None)
    path = Path("./filelists/ch_audio_text_val_ESD_all_fix.txt")
    with path.open('r',encoding='utf-8') as rf:
        t = [line.split('|')[-1].strip('\n') for line in rf]
    with path.open('r',encoding='utf-8') as rf:
        u = [line.split('|')[0].strip('\n').strip(".wav") for line in rf]
    with path.open('r',encoding='utf-8') as rf:
        s = [line.split('|')[1].strip('\n') for line in rf]
    with path.open('r',encoding='utf-8') as rf:
        e = [line.split('|')[2].strip('\n') for line in rf]

    d_u_t = {}
    for i in range(len(t)):
        d_u_t[u[i]] = t[i]

    d_u_s = {}
    for i in range(len(t)):
        d_u_s[u[i]] = s[i]

    d_u_e = {}
    for i in range(len(t)):
        d_u_e[u[i]] = e[i]
    
    source_wav_path = "Source/Neutral/"
    target_wav_path = "./Target/"
    speakers = ['0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
    emotions={"生气":(351,370),"快乐":(701,720),"伤心":(1051,1070),"惊喜":(1401,1420)} 
 
    if not os.path.exists("./listening_test"):
        os.mkdir("./listening_test")
    if not os.path.exists("./listening_test/supervison_level_1"):
        os.mkdir("./listening_test/supervison_level_1")
    out = "./listening_test/supervison_level_1/"
    # do emotion conversion
    for speaker in speakers:
        for j in range(20):
            if j+1>=10:
                wav_name_src = speaker + "_0000" + str(1+j)
                choose_src = source_wav_path + wav_name_src
            else:
                wav_name_src = speaker + "_00000" + str(1+j)
                choose_src = source_wav_path + wav_name_src
            str_sid_src = d_u_s[wav_name_src]
            spec_src = torch.load(choose_src + ".spec.pt")
            spec_src = spec_src.unsqueeze(0)
            spec_src_lengths = torch.LongTensor([spec_src.shape[2]])
            spec_src, spec_src_lengths = spec_src.cuda(), spec_src_lengths.cuda()
            
            for k,v in emotions.items():
                if k =="生气":
                    out_emotion_VC = out + "N2A/"
                    out_emotion_VC_target = out + "Target_N2A/"
                    wav_name = speaker + "_000" + str(v[0]+j)
                    choose_trg = target_wav_path + "Angry/" + wav_name
                if k=="快乐":
                    out_emotion_VC = out + "N2H/"
                    out_emotion_VC_target = out + f"Target_N2H/"
                    wav_name = speaker + "_000" + str(v[0]+j)
                    choose_trg = target_wav_path + "Happy/" + wav_name
                if k=="伤心":
                    out_emotion_VC = out + "N2S1/"
                    out_emotion_VC_target = out + f"Target_N2S1/"
                    wav_name = speaker + "_00" + str(v[0]+j)
                    choose_trg = target_wav_path + "Sad/" + wav_name
                if k=="惊喜":
                    out_emotion_VC = out + "N2S2/"
                    out_emotion_VC_target = out + f"Target_N2S2/"
                    wav_name = speaker + "_00" + str(v[0]+j)
                    choose_trg = target_wav_path + "Surprise/" + wav_name
                if not os.path.exists(out_emotion_VC):
                    os.mkdir(out_emotion_VC)
                if not os.path.exists(out_emotion_VC_target):
                    os.mkdir(out_emotion_VC_target)
                if not os.path.exists(out_emotion_VC_target):
                    os.mkdir(out_target)
                str_sid_trg = d_u_s[wav_name]
                str_eid_trg = d_u_e[wav_name]
                spec_trg =  torch.load(choose_trg + ".spec.pt")
                spec_trg = spec_trg.unsqueeze(0)
                spec_trg_lengths = torch.LongTensor([spec_trg.shape[2]])
                spec_trg, spec_trg_lengths = spec_trg.cuda(), spec_trg_lengths.cuda()

                with torch.no_grad():
                    sid_src = torch.LongTensor([int(str_sid_src)]).cuda()
                    sid_trg = torch.LongTensor([int(str_sid_trg)]).cuda()
                    audio_vc, _, _ = net_g.voice_conversion(y=spec_src, y_lengths=spec_src_lengths, y1=spec_trg, y1_lengths=spec_trg_lengths, sid_src=sid_src, sid_trg=sid_trg)
                    audio_vc = audio_vc.data.cpu().float().numpy()
                    audio_vc *= 32768
                    print(f"convert the emotion of {wav_name_src} to {k}.")
                    #shutil.copy(choose_trg + ".wav", out_emotion_VC_target + f"{speaker}-{j+1}.wav")
                    cmd = "sox "+choose_trg+".wav"+" -b 16 -e signed-integer -r 16000 "+out_emotion_VC_target+f"{speaker}-{j+1}.wav"
                    subprocess.call(cmd,shell=True)        
                    out_path = out_emotion_VC + f"{speaker}-{j+1}.wav"
                    out_path_16k = out_emotion_VC + f"{speaker}-{j+1}-16k.wav"
                    write(out_path, 22050, audio_vc.astype(np.int16))
                    cmd = "sox "+ out_path +" -b 16 -e signed-integer -r 16000 " + out_path_16k 
                    subprocess.call(cmd,shell=True)
                    os.remove(out_path)


if __name__ == '__main__':
    main()    
