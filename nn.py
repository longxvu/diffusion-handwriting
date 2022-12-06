import torch
from torch import nn
import numpy as np
from torch.nn import Dropout, Linear, LayerNorm, SiLU, Embedding, Conv1d, AvgPool1d, Upsample

def create_padding_mask(seq, repeats=1):
    seq = torch.eq(seq, 0).float()
    seq = torch.repeat_interleave(seq, repeats=repeats, dim=-1)
    #mask = seq[:, torch.newaxis, torch.newaxis, :]
    mask = torch.unsqueeze(torch.unsqueeze(seq, 1), 1)
    return mask

def get_angles(pos, i, C, pos_factor = 1):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(C))
    return pos * angle_rates * pos_factor

def positional_encoding(position, C, pos_factor=1):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(C)[np.newaxis, :], C, pos_factor=pos_factor)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])    
    pos_encoding = angle_rads[np.newaxis, ...]    
    return torch.tensor(pos_encoding, dtype=torch.float32)
    
def ff_network(in_size, out_size, dff=768, act_before=True):
    ff_layers = [
        torch.nn.Linear(in_size, dff),
        torch.nn.SiLU(),
        torch.nn.Linear(dff, out_size)    
    ]
    if act_before: ff_layers.insert(0, torch.nn.SiLU())
    return torch.nn.Sequential(*ff_layers)   
    
def loss_fn(eps, score_pred, pl, pl_pred, abar, bce):
    score_loss = torch.mean(torch.sum(torch.pow(eps - score_pred, 2)), dim=-1)) 
    pl_loss = torch.mean(bce(pl, pl_pred) * torch.squeeze(abar, -1))
    return score_loss + pl_loss
    
def scaled_dp_attn(q, k, v, mask):
    kt = k.mT
    qk = torch.matmul(q, kt) #batch_size, d_model, seq_len_q, seq_len_k
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  
    scaled_qk = qk / torch.sqrt(dk)
    if mask is not None: scaled_qk += (mask*-1e12)
    
    attention_weights = torch.nn.functional.softmax(scaled_qk, dim=-1) # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

def reshape_up(x, factor=2):
    x_shape = torch.shape(x)
    x = torch.reshape(x, [x_shape[0], x_shape[1]*factor, x_shape[2]//factor])
    return x

def reshape_down(x, factor=2):
    x_shape = torch.shape(x)
    x = torch.reshape(x, [x_shape[0], x_shape[1]//factor, x_shape[2]*factor])
    return x
    
class AffineTransformLayer(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.gamma_emb = Dense(filters, bias_initializer='ones')
        self.beta_emb = Dense(filters)
    
    def forward(self, x, sigma):
        gammas = self.gamma_emb(sigma)
        betas = self.beta_emb(sigma)
        return x * gammas + betas

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
MultiHeadAttention = torch.nn.MultiHeadAttention # I think this is essentially the same thing???

class InvSqrtSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class AffineTransformLayer(Layer):
    """
    sigma: [batch_size, in_size]
    x: [batch_size, channels, out_size)
    returns: [batch_size, channels, out_size]???
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.gamma_dense = torch.nn.Linear(in_size, out_size, bias_initializer='ones')
        self.beta_dense = torch.nn.Linear(in_size, out_size)
    
    def forward(self, x, sigma):
        gammas = self.gamma_dense(sigma)
        betas = self.beta_dense(sigma)
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
    def __init__(self, in_size, in_channels, in_alpha, filters, dils=[1,1], activation=torch.nn.functional.silu, drop_rate=0.0):
        super().__init__()
        self.act = activation
        self.affine1 = AffineTransformLayer(in_alpha, in_size)
        self.affine2 = AffineTransformLayer(in_alpha, in_size)
        self.affine3 = AffineTransformLayer(in_alpha, in_size)
        self.conv_skip = torch.nn.Conv1d(filters, 3, padding='same')
        self.conv1 = torch.nn.Conv1d(in_channels, filters//2, 3, dilation=dils[0], padding='same')
        self.conv2 = torch.nn.Conv1d(filers//2, filters, 3, dilation=dils[0], padding='same')
        self.fc = torch.nn.Linear(in_size, in_size)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x, alpha):
        x_skip = self.conv_skip(x)
        x = self.conv1(self.act(x))
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.act(x))
        x = self.drop(self.affine2(x, alpha))
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))
        x += x_skip
        return x

class StyleExtractor(torch.nn.Module):
    #takes a grayscale image (with the last channel) with pixels [0, 255]
    #rescales to [-1, 1] and repeats along the channel axis for 3 channels
    #uses a MobileNetV2 with pretrained weights from imagenet as initial weights

    def __init__(self):
        super().__init__()
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        self.local_pool = AveragePooling2D((3,3))

    def forward(self, im, im2=None, training=False):
        x = im.float()
        x = (x / 127.5) - 1     
        x = torch.repeat_interleave(x, 3, dim=-1)

        x = self.mobilenet(x)
        x = self.local_pool(x)
        output = torch.squeeze(x, axis=1)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate=0.1, pos_factor=1):
        super().__init__()
        self.text_pe = positional_encoding(2000, d_model, pos_factor=1)
        self.stroke_pe = positional_encoding(2000, d_model, pos_factor=pos_factor)
        self.drop = Dropout(drop_rate)
        self.lnorm = LayerNorm(eps=1e-6)  # TODO: input shape and set this to not trainable
        self.text_dense = Linear(out_features=d_model)  # TODO: missing in_features

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = ff_network(d_model, d_model*2)
        self.affine0 = AffineTransformLayer(d_model)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
    
    def forward(self, x, text, sigma, text_mask):
        text = self.text_dense(SiLU(text))
        text = self.affine0(self.lnorm(text), sigma)
        text_pe = text + self.text_pe[:, :text.size(1)]

        x_pe = x + self.stroke_pe[:, :x.size(1)]
        x2, att = self.mha(x_pe, text_pe, text, text_mask)
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

class Text_Style_Encoder(nn.Module):
    def __init__(self, d_model, d_ff=512):
        super().__init__()
        self.emb = Embedding(73, d_model)
        self.text_conv = Conv1d(out_channels=d_model, kernel_size=3, padding='same')  # TODO: MI
        self.style_ffn = ff_network(d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, 8)
        self.layernorm = LayerNorm(eps=1e-6)
        self.dropout = Dropout(0.3)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_ffn = ff_network(d_model, d_model*2)

    def forward(self, text, style, sigma):
        style = reshape_up(self.dropout(style), 5)
        style = self.affine1(self.layernorm(self.style_ffn(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layernorm(text), sigma)
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layernorm(text + mha_out), sigma)
        text_out = self.affine4(self.layernorm(self.text_ffn(text)), sigma)
        return text_out

class DiffusionWriter(nn.Module):
    def __init__(self, num_layers=4, c1=128, c2=192, c3=256, drop_rate=0.1, num_heads=8):
        super().__init__()
        self.input_dense = Linear(c1)  # TODO: MI
        self.sigma_ffn = ff_network(c1//4, 2048)
        self.enc1 = ConvSubLayer(c1, [1, 2])
        self.enc2 = ConvSubLayer(c2, [1, 2])
        self.enc3 = DecoderLayer(c2, 3, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c3, [1, 2])
        self.enc5 = DecoderLayer(c3, 4, drop_rate, pos_factor=2)
        self.pool = AvgPool1d(2)
        # In the original code, UpSampling1D repeats the nearest neighbor with size=scale_factor
        self.upsample = Upsample(scale_factor=2, mode="nearest")

        # TODO: Missing a bunch of input shape here
        self.skip_conv1 = Conv1d(c2, 3, padding='same')
        self.skip_conv2 = Conv1d(c3, 3, padding='same')
        self.skip_conv3 = Conv1d(c2*2, 3, padding='same')
        self.text_style_encoder = Text_Style_Encoder(c2*2, c2*4)
        self.att_dense = Linear(c2*2)
        self.att_layers = [DecoderLayer(c2*2, 6, drop_rate) for _ in range(num_layers)]
                     
        self.dec3 = ConvSubLayer(c3, [1, 2])
        self.dec2 = ConvSubLayer(c2, [1, 1])
        self.dec1 = ConvSubLayer(c1, [1, 1])
        self.output_dense = Linear(2)
        self.pen_lifts_dense = Linear(1) # Including sigmoid activation
 
    def forward(self, strokes, text, sigma, style_vector):
        sigma = self.sigma_ffn(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        x = self.input_dense(strokes)
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)
        h2, _ = self.enc3(h2, text, sigma, text_mask)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        x = self.pool(h3)
        
        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)

        x = self.upsample(x) + self.skip_conv3(h3)
        x = self.dec3(x, sigma)

        x = self.upsample(x) + self.skip_conv2(h2)
        x = self.dec2(x, sigma)

        x = self.upsample(x) + self.skip_conv1(h1)
        x = self.dec1(x, sigma)
        
        output = self.output_dense(x)
        pl = self.pen_lifts_dense(x)
        pl = nn.functional.sigmoid(pl)
        return output, pl, att
