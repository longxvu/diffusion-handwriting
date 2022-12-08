import torch
from torch import nn
import numpy as np
from torch.nn import Dropout, Linear, LayerNorm, SiLU, Embedding, Conv1d, AvgPool1d, Upsample, AvgPool2d, LazyLinear, \
    MultiheadAttention


def create_padding_mask(seq, repeats=1):
    seq = torch.eq(seq, 0).float()
    seq = torch.repeat_interleave(seq, repeats=repeats, dim=-1)
    # mask = seq[:, torch.newaxis, torch.newaxis, :]
    mask = torch.unsqueeze(torch.unsqueeze(seq, 1), 1)
    return mask


def get_angles(pos, i, C, pos_factor=1):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(C))
    return pos * angle_rates * pos_factor


def positional_encoding(position, C, pos_factor=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(C)[np.newaxis, :], C, pos_factor=pos_factor)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32, device=device)


def ff_network(out_size, dff=768, act_before=True):
    ff_layers = [
        torch.nn.LazyLinear(dff),
        torch.nn.SiLU(),
        torch.nn.LazyLinear(out_size)
    ]
    if act_before:
        ff_layers.insert(0, torch.nn.SiLU())
    return torch.nn.Sequential(*ff_layers)


def loss_fn(eps, score_pred, pl, pl_pred, abar, bce):
    score_loss = torch.mean(torch.sum(torch.square(eps - score_pred), dim=-1))
    pl_loss = torch.mean(bce(pl, pl_pred) * torch.squeeze(abar, -1))
    return score_loss, pl_loss


def scaled_dp_attn(q, k, v, mask):
    kt = k.mT
    qk = torch.matmul(q, kt)  # batch_size, d_model, seq_len_q, seq_len_k
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_qk = qk / torch.sqrt(dk)
    if mask is not None: scaled_qk += (mask * -1e12)

    attention_weights = torch.nn.functional.softmax(scaled_qk, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


def reshape_up(x, factor=2):
    x_shape = x.size()
    x = torch.reshape(x, [x_shape[0], x_shape[1] * factor, x_shape[2] // factor])
    return x


def reshape_down(x, factor=2):
    x_shape = x.size()
    x = torch.reshape(x, [x_shape[0], x_shape[1] // factor, x_shape[2] * factor])
    return x


"""
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, C, num_heads):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.wq = Dense(C)
        self.wk = Dense(C)
        self.wv = Dense(C)
        self.dense = Dense(C)  
        
    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.C // self.num_heads))
        return torch.transpose(x, perm=[0,2,1,3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v) # (bs, sl, C)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size) #(bs, nh, sl, C // nh) for q,k,v

        attention, attention_weights = scaled_dp_attn(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (bs, sl, nh, C // nh)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.C)) # (bs, sl, c)
        output = self.dense(concat_attention)
        return output, attention_weights
"""


def invSqrtSchedule(optim, warmup_steps, d_model, step):
    arg1 = 1 / np.sqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
    lr = 1 / np.sqrt(d_model) * np.min([arg1, arg2])
    for g in optim.param_groups:
        g['lr'] = lr


class AffineTransformLayer(nn.Module):
    """
    sigma: [batch_size, in_size]
    x: [batch_size, channels, out_size)
    returns: [batch_size, channels, out_size]???
    """

    def __init__(self, filters, channel_first_input=False):
        super().__init__()
        # TODO: bias_initializer='ones' for gamma_dense
        self.gamma_dense = LazyLinear(filters)
        self.beta_dense = LazyLinear(filters)
        self.channel_first = channel_first_input

    def forward(self, x, sigma):
        gammas = self.gamma_dense(sigma)
        betas = self.beta_dense(sigma)
        if self.channel_first:
            return (x.permute(0, 2, 1) * gammas + betas).permute(0, 2, 1)
        else:
            return x * gammas + betas


class ConvSubLayer(torch.nn.Module):
    """
    x: [batch, in_channels, in_size]
    alpha: [_, in_alpha]
    filters: number of channels in output
    dils: dilation for conv1d
    activation: activation function
    drop_rate: for dropout layer
    """

    # Here only first dimension of dilation is used, why do they have 2 values?
    # In all our usage, we removed the 2nd values of dilation tuple, using the default dilation=1
    def __init__(self, in_channels, filters, dilation=1, drop_rate=0.0):
        super().__init__()
        self.swish = nn.SiLU()
        self.affine1 = AffineTransformLayer(filters // 2, channel_first_input=True)
        self.affine2 = AffineTransformLayer(filters, channel_first_input=True)
        self.affine3 = AffineTransformLayer(filters)
        self.conv_skip = Conv1d(in_channels, filters, (3,), padding='same')
        self.conv1 = Conv1d(in_channels, filters // 2, (3,), dilation=(dilation,), padding='same')
        self.conv2 = Conv1d(filters // 2, filters, (3,), dilation=(dilation,), padding='same')
        self.fc = Linear(filters, filters)
        # Why do we need Dropout when drop_rate is 0 everytime???????
        self.drop = Dropout(drop_rate)

    def forward(self, x, alpha):
        x_skip = self.conv_skip(x)
        x = self.conv1(self.swish(x))
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.swish(x))
        x = self.drop(self.affine2(x, alpha))
        # Permuting because conv layer uses channel first while fc uses channel last
        x = x.permute(0, 2, 1)
        x = self.fc(self.swish(x))
        x = self.drop(self.affine3(x, alpha))
        x = x.permute(0, 2, 1)
        x += x_skip
        return x


class StyleExtractor(torch.nn.Module):
    # takes a grayscale image (with the last channel) with pixels [0, 255]
    # rescales to [-1, 1] and repeats along the channel axis for 3 channels
    # uses a MobileNetV2 with pretrained weights from imagenet as initial weights
    def __init__(self):
        super().__init__()
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.mobilenet = self.mobilenet.features  # Only include feature extraction part
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        self.local_pool = AvgPool2d((3, 3))

    def forward(self, im):
        x = im.float()
        x = (x / 127.5) - 1
        x = torch.repeat_interleave(x, 3, dim=1)  # repeat at color channel

        x = self.mobilenet(x)
        x = self.local_pool(x)
        output = torch.squeeze(x, dim=2)  # Squeezing height dimension, output shape should be [1280, 14]
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate=0.1, pos_factor=1):
        super().__init__()
        self.text_pe = positional_encoding(2000, d_model, pos_factor=1)
        self.stroke_pe = positional_encoding(2000, d_model, pos_factor=pos_factor)
        # self.stroke_pe = self.stroke_pe.permute(0, 2, 1)
        self.drop = Dropout(drop_rate)
        self.lnorm = LayerNorm(d_model, eps=1e-6)  # TODO: set to not trainable?
        self.text_dense = LazyLinear(d_model)

        self.mha = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mha2 = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = ff_network(d_model, dff=d_model * 2)
        self.affine0 = AffineTransformLayer(d_model)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.SiLU = SiLU()

    def forward(self, x, text, sigma, text_mask, swap_channel_layer=True):
        # Basically this function combines stroke & text using 2 MHA layers
        # First MHA combines strokes and text, second MHA is self attention (with pe) (?)
        # input text is [B, 50, 384]
        text = self.text_dense(self.SiLU(text))  # This will have [B, 50, d_model in (C1, C2, C3)]
        text = self.affine0(self.lnorm(text), sigma)  # affine text transform, channel last is good
        text_pe = text + self.text_pe[:, :text.size(1)]

        # Need to permute x to have sequence dim in middle for MHA. Shape x should be: [B, L, E]
        # this is assumed x to have channel first memory structure, so we swap it to last.
        # For series of DecoderLayer (attention), this should be False
        if swap_channel_layer:
            x = x.permute(0, 2, 1)
        x_pe = x + self.stroke_pe[:, :x.size(1)]
        # text_mask should be shape [B, K]
        text_mask = text_mask.view(text_mask.size(0), -1)
        x2, att = self.mha(x_pe, text_pe, text, key_padding_mask=text_mask)
        x2 = self.lnorm(self.drop(x2))
        x2 = self.affine1(x2, sigma) + x

        x2_pe = x2 + self.stroke_pe[:, :x.size(1)]
        x3, _ = self.mha2(x2_pe, x2_pe, x2)
        x3 = self.lnorm(x2 + self.drop(x3))
        x3 = self.affine2(x3, sigma)

        x4 = self.ffn(x3)
        x4 = self.drop(x4) + x3
        out = self.affine3(self.lnorm(x4), sigma)
        return out, att


class TextStyleEncoder(nn.Module):
    def __init__(self, d_model, d_ff=512):
        super().__init__()
        # Is 73 vocabulary size?
        self.emb = Embedding(73, d_model)
        # self.text_conv = Conv1d(out_channels=d_model, kernel_size=3, padding='same')
        # 256 is from [B, 70, 256] of style_vec
        self.style_ffn = ff_network(d_model, dff=d_ff)  # [256 -> d_ff -> d_model]
        self.mha = MultiheadAttention(d_model, 8, batch_first=True)
        self.layer_norm = LayerNorm(d_model, eps=1e-6)
        self.dropout = Dropout(0.3)
        # Affine has input equal to number of channel in sigma
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_ffn = ff_network(d_model, dff=d_model * 2)

    def forward(self, text, style, sigma):
        # style and embedded text goes through MHA layer
        # output is the transformed attention combined with embedded text in text_out
        # sigma has shape [B, 1, c1//4=32]
        style = reshape_up(self.dropout(style), 5)
        style = self.affine1(self.layer_norm(self.style_ffn(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layer_norm(text), sigma)
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layer_norm(text + mha_out), sigma)
        text_out = self.affine4(self.layer_norm(self.text_ffn(text)), sigma)
        return text_out


# tse = Text_Style_Encoder(192 * 2, 192 * 4)
# text = torch.randint(0, 40, [2, 50])
# style = torch.randn([2, 14, 1280])
# sigma = torch.randn([2, 1, 32])
#
# out = tse(text, style, sigma)
# print(out.shape)

class DiffusionWriter(nn.Module):
    def __init__(self, num_layers=4, c1=128, c2=192, c3=256, drop_rate=0.1, num_heads=8, device="cuda:0"):
        super().__init__()
        # forward process:
        # enc1 -> pool -> enc2 -> enc3 -> pool -> enc4 -> enc5 -> attention ->
        # upsample + dec3 -> upsample + dec2 -> upsample + dec1 -> out_dense, pl_dense
        self.input_dense = Linear(2, c1)
        self.sigma_ffn = ff_network(c1 // 4, dff=256, act_before=False)
        self.enc1 = ConvSubLayer(c1, c1)
        self.enc2 = ConvSubLayer(c1, c2, dilation=1)
        self.enc3 = DecoderLayer(c2, 3, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c2, c3, dilation=1)
        self.enc5 = DecoderLayer(c3, 4, drop_rate, pos_factor=2)
        self.pool = AvgPool1d(2)
        # In the original code, UpSampling1D repeats the nearest neighbor with size=scale_factor
        self.upsample = Upsample(scale_factor=2, mode="nearest")

        self.skip_conv1 = Conv1d(c1, c2, 3, padding='same')
        self.skip_conv2 = Conv1d(c2, c3, 3, padding='same')
        self.skip_conv3 = Conv1d(c3, c2 * 2, 3, padding='same')
        self.text_style_encoder = TextStyleEncoder(c2 * 2, c2 * 4)
        self.att_dense = Linear(c3, c2 * 2)
        self.att_layers = [DecoderLayer(c2 * 2, 6, drop_rate) for _ in range(num_layers)]
        for layer in self.att_layers:
            layer.to(device)

        self.dec3 = ConvSubLayer(2 * c2, c3, 1)
        self.dec2 = ConvSubLayer(c3, c2, 1)
        self.dec1 = ConvSubLayer(c2, c1, 1)
        self.output_dense = Linear(c1, 2)
        self.pen_lifts_dense = Linear(c1, 1)  # sigmoid activation included in forward function

    def forward(self, strokes, text, sigma, style_vector):
        # strokes shape: [B, 488, 2]
        sigma = self.sigma_ffn(sigma)  # [batch, 1, c1//4=32]
        text_mask = create_padding_mask(text)  # [batch, 1, 1, 50]
        # text and style vector gets combined using attention here, output shape: [B, 50, 384=d_model=c2*2]
        text = self.text_style_encoder(text, style_vector, sigma)

        # We need to permute strokes since conv1d requires channel first
        x = self.input_dense(strokes)  # After input_dense, shape: [B, 488, 128=c1]
        x = x.permute(0, 2, 1)
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)  # h2 shape: [B, 244, 128=c1] -> [B, 244, 192=c2]
        h2, _ = self.enc3(h2, text, sigma, text_mask)  # Same shape
        # enc3 encode both text and stroke information. output of enc3 has shape: [B, L, C], we need channel first
        # for all pooling and conv stuff
        h2 = h2.permute(0, 2, 1)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)  # h3 shape: [B, 122, 192=c2] -> [B, 122, 256=c3]
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        h3 = h3.permute(0, 2, 1)  # same reason as h2
        x = self.pool(h3)

        # Make channel last for linear attention layers
        x = x.permute(0, 2, 1)
        x = self.att_dense(x)  # x shape: [B, 61, 256]
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask, swap_channel_layer=False)

        # After attention and FC, permute x to have channel first
        x = x.permute(0, 2, 1)
        x = self.upsample(x) + self.skip_conv3(h3)  # upsample: [B, 2*c2, 61->122], skip: [B, c3->2*c2, 122]
        x = self.dec3(x, sigma)  # [B, 2*c2->c3, 122]

        x = self.upsample(x) + self.skip_conv2(h2)  # upsample: [B, c3, 122->244], skip: [B, 2*c2->c3, 244]
        x = self.dec2(x, sigma)  # [B, c3->c2, 244]

        x = self.upsample(x) + self.skip_conv1(h1)  # upsample: [B, c2, 244->488], skip: [B, c1->c2, 488]
        x = self.dec1(x, sigma)  # [B, c2->c1, 488]

        x = x.permute(0, 2, 1)
        output = self.output_dense(x)
        pl = self.pen_lifts_dense(x)
        pl = torch.sigmoid(pl)
        return output, pl, att
