import string
import torch
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt


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
    #TODO: This is currently wrong but there's no way to verify the correctness until we can run train
    alpha_indices = torch.randint(low=0, high=len(alpha_set) - 1, size=(batch_size, 1))
    lower_alphas = torch.gather(alpha_set, alpha_indices)
    upper_alphas = torch.gather(alpha_set, alpha_indices + 1)
    alphas = torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas)
    alphas += lower_alphas
    alphas = alphas.view(batch_size, 1, 1)
    return alphas


def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    xt_minus1 = (1 / torch.sqrt(1 - beta)) * (xt - (beta * eps / torch.sqrt(1 - alpha)))
    if add_sigma:
        xt_minus1 += torch.sqrt(beta) * torch.randn(xt.shape)
    return xt_minus1


def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    xt_minus1 = (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    xt_minus1 += torch.randn(xt.shape) * torch.sqrt(1 - alpha_next)
    return xt_minus1


def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new',
                        show_every=None, show_samples=True, path=None):
    if isinstance(text, str):
        text = torch.tensor([tokenizer.encode(text) + [1]])
    elif isinstance(text, list) and isinstance(text[0], str):
        tmp = []
        for i in text:
            tmp.append(tokenizer.encode(i) + [1])
        text = torch.tensor(tmp)

    bs = text.shape[0]
    L = len(beta_set)
    alpha_set = torch.cumprod(1 - beta_set)
    x = torch.randn([bs, time_steps, 2])

    for i in range(L - 1, -1, -1):
        alpha = alpha_set[i] * torch.ones([bs, 1, 1])
        beta = beta_set[i] * torch.ones([bs, 1, 1])
        a_next = alpha_set[i - 1] if i > 1 else 1.
        model_out, pen_lifts, att = model(x, text, torch.sqrt(alpha), style)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i))
        else:
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)

        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0])
                plt.show()

    x = torch.cat([x, pen_lifts], dim=-1)
    for i in range(bs):
        plt.show(x[i], scale=1, show_output=show_samples, name=path)

    return x.numpy()


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
