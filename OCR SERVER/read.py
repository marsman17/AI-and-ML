# read.py

import os
import pytz
import math
import argparse
from PIL import Image
from datetime import datetime

import torch
import torch.utils.data

from model import Model  # Ensure these imports work correctly
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter, Logger

def run_ocr(opt, device):
    opt.device = device
    os.makedirs("read_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'read_outputs/{datetime_now}.txt')
    
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = model.to(device)

    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    logger.log('Loaded pretrained model from %s' % opt.saved_model)
    model.eval()
    
    if opt.rgb:
        img = Image.open(opt.image_path).convert('RGB')
    else:
        img = Image.open(opt.image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * img.size(0))
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    logger.log(preds_str)
    return preds_str
