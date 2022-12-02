import string
import torch
import math
import numpy as np
import pickle


# Variance schedule
def get_beta_set():
    start = 1e-5
    end = 0.4
    num_step = 60
    beta_set = 0.02 + torch.exp(torch.linspace(math.log(start), math.log(end), num_step))
    return beta_set


# Data processing
def pad_stroke_seq(x, maxlength):
    if len(x) > maxlength or np.amax(np.abs(x)) > 15:
        return None
    zeros = np.zeros((maxlength - len(x), 2))
    ones = np.ones((maxlength - len(x), 1))
    padding = np.concatenate((zeros, ones), axis=-1)
    x = np.concatenate((x, padding)).astype("float32")
    return x


def pad_img(img, width, height):
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img


def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)

    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len - len(text),))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            if x is not None and sample.shape[1] < img_width:
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples


def create_dataset(strokes, texts, samples, style_extractor, batch_size, buffer_size):
    pass


# nn utils
def get_alphas(batch_size, alpha_set):
    pass


def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    pass


def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    pass


def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new',
                        show_every=None, show_samples=True, path=None):
    pass


class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = "_" + string.ascii_letters + string.digits + ".?!,\'\"- "
        self.numbers = np.arange(2, len(self.text) + 2)
        self.create_dict()
        self.vocab_size = len(self.text) + 2

    def create_dict(self):
        for char, token, in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = " ", "<end>"  # only for decoding

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text:
                tokenized.append(self.tokens[char])
            else:
                tokenized.append(2)  # unknown character is '_', which has index 2

        tokenized.append(1)  # 1 is the end of sentence character
        return tokenized

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return "".join(text)
