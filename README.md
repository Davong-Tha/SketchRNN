# SketchRNN

# Summary
This is a pytorch implementation of SketchRNN, a variational autoencoder based architecture used for modeling sequential stroke based drawing. The goal is to predict the next storke using the previous stroke conditioned on the encoder latent output. This experiment was trained on one class from google quickdraw dataset and achieve good output for both sketch reconstruction and generation. 
# Table of Contents
- [Repo Content](#repo-content)
- [Key Features](#key-features)
- [Result](#result)
- [Lesson Learnt and Troubleshooting](#lesson-learnt-and-troubleshooting)
- [Limitation](#limitation)
# Repo Content

# Features

## Architecture
The model is based on a VAE with a encoder-decoder module. The encoder is a bidirectional RNN while the decoder is an autoregressive unidirectional RNN train on teacher forcing. The encoder output a latent space that represent the overall structure of the input sketch while the decoder output parameters to a multivariant GMM. Sampling from the GMM give us the pen movement (dx and dy) and one hot pen state(pen up, pen down and drawing end).

Random cut off of the sketch is used to to train the encoder, which output a latent space z which is then used to condition every timestep of the decoder trained on teacher forcing.

During reconstruction, the mean value of dx and dy is calculated from the GMM while during generation stochastic sampling from the GMM is used to generate new stroke. 

## loss functions
There are three type of loss functions being used to train the model:
- Reconstruction loss
- pen state: this is a one hot label so a simple cross entrophy loss was used.
- pen movement (dx and dy): the horizontal and vertical movement of the stroke is model by multivariant GMM. To optimize this function its negative log likelihood was derived
- The penstate and pen movement represent reconstruction loss which tell us how similar the generated sketch is to our training data.
- KL loss
- This loss function is exclusive for the encoder. It measure how far the output latent drift from standard gaussian N(0,I). Ideally, we want it to drift from N(0, I) but not too far. The idea is that we want to encode information into the latent space so it need to drift from N(0,I) but too far of a drift mean the encoder is just memorizing the input sketch. The general behaviour of the KL loss is that it start increasing in early epoch as the model learned to rely on the latent space then peaked as the decoder get stronger and relied on teacher forcing input, the kl loss would slowly decrease as less information is being encoded in the latent space.
***insert equation for KL divergence***

## KL annealing
In order to ensure the model doesn't cheat by collapsing the latent space the following equation is used to decrease the reward for collapsing the latent during early epoch. This weight graually get bigger as training progress allowing the model to optimize the KL to ensure the encoder doesn't just memorize the input sketch.
***insert kl annealing equation***
## Encoder reparamerization trick
This is a technique used in a VAE. The encoder output the mean and variance for the latent space, however in order to ensure stochastic sampling we need to introduce randomness into this latent space but doing so would make training unstable. The trick is to separate the deterministic part of training from the stochastic part. By raparameterize the mean and variance with sample from a standard normal as below we ensure that training remain deterministic while stochatic sampling from the posterior is still possible.  


## 

# Result
## Dataset 
The cat dataset on google quickdraw was used to train the model. There are a total of ~120,000 sample was used. the quickdraw format was converted to sketchrnn format, and the pen movement (dx and dy) was normalized their joint standard deviation. Any sketch that are too long or too short are drop by considering them as outlier using IQR percentile. 

## train and validation loss
The model was trained on 8:2 train/validation split 

## reconstruction and generation

# Lesson Learnt and Troubleshooting
## Posterior Collapse
During training, the decoder used both the latent space provided by the encoder and the previous stroke provided by teacher forcing. However, this introduce a fundamental failure mode into the design since the decoder can cheat by collapsing the latent space to zero thereby minimizing the loss in earlier epoch. This can be diagnosed by the KL loss flooring at early epoch and the generated drawing are all the same regardless of the input to the encoder. The decoder essentially ignore the latent space which contain global structure information and rely solely on teacherforcing to optimized the loss. This can be prevented by KL annealing which decrease the reward for minimizing the KL loss in early epoch. Later on when reconstrcution quality improve, collpasing the latent space become costly since the model already learned to rely on it.

***Insert graph and generated sketch of posterior collapse***
## Overiftting and Generalization
Larger model tend to overfit on the training, however, smaller model seem to underfit with low quality sketch output. Therefore, a small variance gaussian noise is applied to the teacher forcing input of the decoder during training so that the model can generalized better to the validation set. A small input dropout was also used on the decoder instead of recurrent drop out which slow down training. 
## Task mismatch training between reconstruction and generatiion 
Initially, the encoder was trained on the entire sketch and the decoder receive teacher forcing input from the entire sketch as well. This allow us to achieve good reconstruction, however generation condition on an incomplete sketch abruptly end after the timestep the decoder was condition on. i.e if n timestep (incomplete sketch) was pass to the encoder, the decoder stop generation after n timestep. In order to resolve this random cutoff of the sketch was used to train the encoder instead while the decoder still receive teacher forcing on the whole sketch. This teach the model that the sketch doesn't end even after the timestep it was conditioned on. 
## custom lstm 
During earlier trails of the experiment, a custom lstm cell was also being since it can give full control over the hidden satte thereby allowing us to applied recurrent dropout. The idea is that we corrupt the hidden state of the decoder forcing the model to rely more the latent space, however, this greatly reduce the training speed and therefore keras lstm was used instead. 

# Limitation
