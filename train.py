import os

import torch
import argparse
import utils
import nn
from torch.optim import AdamW
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader
from dataset import IAMDataset
from tqdm import tqdm
import time
from nn import invSqrtSchedule


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


def train(args):
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = args.seq_len
    MAX_TEXT_LEN = args.text_len
    IMG_WIDTH = args.width
    IMG_HEIGHT = 96
    DROP_RATE = args.dropout
    NUM_ATT_LAYERS = args.num_att_layers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    C1 = args.channels
    C2 = C1 * 3 // 2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN % 8) + 8
    cached_style_vec_path = "data/cached_style_vec.npy"
    num_epoch = 200
    batch_size = 64
    SAVE_EVERY = 50
    L = 60

    save_path = os.path.join("weights/", time.strftime("%m%d_%H%M%S"))
    print(f"Saving weights and logs to {save_path}")
    os.makedirs(save_path, exist_ok=True)


    # tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    alpha_set = torch.cumprod(1 - beta_set, dim=0)
    # style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=NUM_ATT_LAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE)

    # TODO: There's a clip grad norm in the original Tensorflow Adam implementation
    # lr = nn.InvSqrtScheduler(C3, warmup_steps=WARMUP_STEPS)
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98))

    path = './data/train_strokes.p'
    print("Preprocessing data")
    strokes, texts, _ = utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, IMG_WIDTH, IMG_HEIGHT)
    # dataset = utils.create_dataset(strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE)
    dataset = IAMDataset(strokes, texts, cached_style_vec_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    score_criterion = MSELoss()
    pl_criterion = BCELoss()

    # init lazy module parameter
    # stroke = torch.randn(1, 488, 2)
    # text = torch.randint(0, 30, (1, 50))
    # alphas = torch.randn(1, 1)
    # style_vec = torch.randn(1, 14, 1280)
    # with torch.no_grad():
    #     _ = model(stroke, text, alphas, style_vec)
    # for p in model.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -100, 100))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_avg_loss = 1e5
    global_step = 1
    print("Starting training")
    for epoch in range(num_epoch):
        score_losses = []
        pl_losses = []
        for stroke, pen_lift, text, style_vec in (pbar := tqdm(train_loader)):
            stroke, pen_lift, text, style_vec = stroke.to(device), pen_lift.to(device),\
                                                text.to(device), style_vec.to(device)
            # Uncomment for lr scheduler
            # invSqrtSchedule(optimizer, WARMUP_STEPS, C3, global_step)
            # alphas has shape: [batch, 1, 1]
            alphas = utils.get_alphas(stroke.size(0), alpha_set)
            alphas = alphas.to(device)
            eps = torch.randn(stroke.size(), device=device)
            stroke_perturbed = torch.sqrt(alphas) * stroke
            stroke_perturbed += torch.sqrt(1 - alphas) * eps

            optimizer.zero_grad()
            score, pl_pred, att = model(stroke_perturbed, text, torch.sqrt(alphas), style_vec)
            score_loss = score_criterion(score, eps)
            pl_loss = torch.mean(pl_criterion(pl_pred, pen_lift) * torch.squeeze(alphas, -1))
            # score_loss, pl_loss = nn.loss_fn(eps, score, pen_lift, pl_pred, alphas, bce_loss)
            loss = score_loss + pl_loss
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch [{epoch + 1}/{num_epoch}]")
            pbar.set_postfix({
                "score_loss": score_loss.item(),
                "pl_loss": pl_loss.item()
            })
            global_step += 1

            score_losses.append(score_loss.cpu().item())
            pl_losses.append(pl_loss.cpu().item())

        avg_score_loss = sum(score_losses) / len(score_losses)
        avg_pl_loss = sum(pl_losses) / len(pl_losses)
        tqdm.write(f"avg_score_loss: {avg_score_loss:.5f}, avg_pl_loss: {avg_pl_loss:.5f}")
        torch.save(model.state_dict(), os.path.join(save_path, f"last.pth"))
        avg_loss = avg_score_loss + avg_pl_loss
        if avg_loss < best_avg_loss:
            torch.save(model.state_dict(), os.path.join(save_path, f"best.pth"))
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{epoch:04}.pth"))

        # Save log for graph later
        with open(os.path.join(save_path, "loss_log.txt"), "a") as f:
            for score_l, pl_l in zip(score_losses, pl_losses):
                f.write(f"{score_l} {pl_l}\n")
        with open(os.path.join(save_path, "train_log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: {avg_score_loss} {avg_pl_loss}\n")


if __name__ == "__main__":
    args = get_args()
    train(args)
