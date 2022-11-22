import os
import random

directory = "./dataset/HumanML3D/texts/"
NUM_SAMPLES = 1

contents = []

# print(f"files: {os.listdir(directory)}")

# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
import random
import clip


MOTION_LENGTH_INDEX = 3
NUMBER_OF_TAKEN_FRAMES = 40
NUMBER_OF_CUT_FRAMES = 96
SEQUENCE_LENGTH = 7


def encode_text(clip_model, raw_text):

    # raw_text - list (batch_size length) of strings with input text prompts
    # device = clip_model.device
    device = dist_util.dev()
    max_text_len = 20
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2  # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
        # print('texts', texts.shape)
        zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)
        # print('texts after pad', texts.shape, texts)
    else:
        texts = clip.tokenize(raw_text, truncate=True).to(device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
    return clip_model.encode_text(texts).float()


def load_and_freeze_clip(clip_version):
    print(f"loading clip")
    # clip_model, clip_preprocess = clip.load(clip_version, device="cpu", jit=False)  # Must set jit=False for training
    clip_model, clip_preprocess = clip.load(clip_version, device=dist_util.dev(), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def main():
    args = edit_args()
    dist_util.setup_dist(args.device)

    clip_version = "ViT-B/32"
    clip_model = load_and_freeze_clip(clip_version)

    text1 = "a person is fighting"
    text2 = "a person kicks"

    def clip_dist(text1, text2):
        enc1 = encode_text(clip_model, text1)
        enc2 = encode_text(clip_model, text2)

        VALUE_AXIS = 1
        dist = (enc1 - enc2).pow(2).sum(VALUE_AXIS).sqrt()
        return dist[0]

    print(f"dist: {clip_dist(text1, text2)}")


if __name__ == "__main__":
    main()


# print(len(contents))
# sample = random.sample(contents, NUM_SAMPLES)

# print(sample)
