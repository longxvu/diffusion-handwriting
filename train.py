import torch
import argparse
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', help='number of trainsteps, default 60k', default=60000, type=int)
    parser.add_argument('--batch_size', help='default 96', default=96, type=int)
    parser.add_argument('--seq_len', help='sequence length during training, default 480', default=480, type=int)
    parser.add_argument('--text_len', help='text length during training, default 50', default=50, type=int)
    parser.add_argument('--width', help='offline image width, default 1400', default=1400, type=int)
    parser.add_argument('--warmup', help='number of warmup steps, default 10k', default=10000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_att_layers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--print_every', help='show train loss every n iters', default=1000, type=int)
    parser.add_argument('--save_every', help='save ckpt every n iters', default=10000, type=int)

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = args.seq_len
    MAX_TEXT_LEN = args.text_len
    WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATT_LAYERS = args.num_att_layers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    SAVE_EVERY = args.save_every
    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN % 8) + 8

    BUFFER_SIZE = 3000
    L = 60
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
