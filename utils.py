import string
import torch
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from nn import StyleExtractor


def generate_stroke_image(strokes, save_path=None, show_output=True, scale=1):
    strokes = strokes.squeeze()
    positions = np.cumsum(strokes, axis=0).T[:2]
    prev_ind = 0
    W, H = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    plt.figure(figsize=(scale * W / H, scale))

    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5:
            plt.plot(positions[0][prev_ind:ind], positions[1][prev_ind:ind], color='black')
            prev_ind = ind

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Image saved to {save_path}")
    if show_output:
        plt.show()
    plt.close()


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
    pad_len = width - img.shape[2]
    padding = np.full((1, height, pad_len), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=2)
    return img


def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)

    strokes, texts, samples = [], [], []
    for x, text, sample in tqdm(ds):
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len - len(text),))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            # Ignore all samples with width > img_width
            if x is not None and sample.shape[2] < img_width:
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples


def cache_style_vectors(pickle_path, max_text_len=50, max_seq_len=488, img_width=1400, img_height=96, batch_size=128,
                        save_path="data/cached_style_vec.npy"):
    strokes, texts, samples = preprocess_data(pickle_path, max_text_len, max_seq_len, img_width, img_height)
    dataset = TensorDataset(torch.tensor(samples))
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    style_extractor = StyleExtractor()
    style_extractor.to(device)
    style_vectors = []
    tqdm.write("Creating style vectors")
    for image in tqdm(data_loader):
        style_output = style_extractor(image[0].to(device))
        style_vectors.append(style_output)

    style_vectors = torch.cat(style_vectors).cpu().numpy()
    np.save(save_path, style_vectors)


# nn utils
def get_alphas(batch_size, alpha_set):
    alpha_indices = torch.randint(low=0, high=len(alpha_set) - 1, size=(batch_size, 1))
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
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


# if __name__ == "__main__":
#     cache_style_vectors("data/train_strokes.p")
