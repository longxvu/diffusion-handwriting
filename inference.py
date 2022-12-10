#import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import nn
import argparse
import os
import preprocessing
from utils import standard_diffusion_step, new_diffusion_step, generate_stroke_image, pad_img


def extract_style_from_file(path, style_extractor=None):
    if not style_extractor:
        style_extractor = nn.StyleExtractor()

    writer_img = preprocessing.read_img(path, 96)
    writer_img = pad_img(writer_img, 1400, 96)
    writer_img = torch.tensor(writer_img).unsqueeze(dim=0)
    with torch.no_grad():
        style_vector = style_extractor(writer_img)
    style_vector = style_vector.permute(0, 2, 1)
    return style_vector


def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--textstring', help='the text you want to generate', default="deep generative models")
    parser.add_argument('--writersource', help="path of the image of the desired writer, (e.g. './assets/image.png'   \
                                                will use random from ./assets if unspecified", default=None)
    parser.add_argument('--name', help="path for generated image (e.g. './assets/sample.png'), \
                                             will not be saved if unspecified", default=None)
    parser.add_argument('--diffmode', help="what kind of y_t-1 prediction to use, use 'standard' for  \
                                            Eq 9 in paper, will default to prediction in Eq 12", default='new', type=str)
    parser.add_argument('--show', help="whether to show the sample (popup from matplotlib)", default=False, type=bool)
    parser.add_argument('--weights', help='the path of the loaded weights', default='./weights/model_weights.h5', type=str)
    parser.add_argument('--seqlen', help='number of timesteps in generated sequence, default 16 * length of text', default=None, type=int)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution, \
                                                 only change this if loaded model was trained with that hyperparameter', default=2, type=int)
    parser.add_argument('--channels', help='number of channels at lowest resolution, only change \
                                                 this if loaded model was trained with that hyperparameter', default=128, type=int)
    
    args = parser.parse_args()
    time_steps = len(args.textstring) * 16 if args.seqlen is None else args.seqlen
    time_steps = time_steps - (time_steps % 8) + 8
    #must be divisible by 8 due to downsampling layers
    text = args.textstring
    diffusion_mode = args.diffmode
    save_intermediate_step_path = "temp"

    if args.writersource is None:
        assetdir = os.listdir('data/assets')
        # sourcename = 'data/assets/' + assetdir[np.random.randint(0, len(assetdir))]
        sourcename = "data/assets/r06-412z-04.tif"
    else: 
        sourcename = args.writersource

    weights_path = "weights/epoch_0500.pth"

    device = "cpu"
    L = 60
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    #alpha_set = tf.math.cumprod(1-beta_set)
    alpha_set = torch.cumprod(1-beta_set, 0)

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    show_every = None

    model = nn.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3, device=device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Define model input
    text = torch.tensor([tokenizer.encode(text) + [1]])
    style_vector = extract_style_from_file(sourcename)

    L = len(beta_set)
    alpha_set = torch.cumprod(1 - beta_set, dim=0)
    x = torch.randn([1, time_steps, 2])

    for i in range(L - 1, -1, -1):
        alpha = alpha_set[i] * torch.ones([1, 1, 1])
        beta = beta_set[i] * torch.ones([1, 1, 1])
        a_next = alpha_set[i - 1] if i > 1 else torch.tensor(1.)
        with torch.no_grad():
            model_out, pen_lifts, att = model(x, text, torch.sqrt(alpha), style_vector)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i))
        else:
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)

        if save_intermediate_step_path:
            generate_stroke_image(torch.cat([x, pen_lifts], dim=-1).numpy(), scale=1, show_output=False,
                                  save_path=os.path.join(save_intermediate_step_path, f"{i:02}.png"))

        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0])
                plt.show()

    x = torch.cat([x, pen_lifts], dim=-1)
    generate_stroke_image(x.numpy(), scale=1, show_output=False, save_path="generated_output.png")


if __name__ == '__main__':
    main()
