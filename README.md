Pytorch implementation of : https://arxiv.org/abs/2011.06704

Reference from Tensorflow implementation here: https://github.com/tcl9876/Diffusion-Handwriting-Generation

Current changes:

| Tensorflow         | Pytorch         | Notes |
|--------------------|-----------------|-------|
| Dropout            | Dropout         | SP    |
| LayerNormalization | LayerNorm       | MI    |
| Dense              | Linear          | MI    |
| swish              | SiLu            | SP    |
| Embedding          | Embedding       | SP    |
| Conv1D             | Conv1d          | MI    |
| AveragePooling1D   | AvgPool1d       | SP    |
| UpSampling1D       | Upsample        | DP    |
| gather_nd          | simple indexing ||
|                    |                 ||
|                    |                 ||


Notation:

SP: Same parameters

MI: Missing input shape, but with overall same parameters

DP: Different parameters, require different things