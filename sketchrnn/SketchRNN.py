import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from sketchrnn.encoder import Encoder
from sketchrnn.decoder import Decoder
class SketchRNN(keras.Model):
    def __init__(self, M=20, z_dim=128, enc_hidden=512, dec_hidden=1024, enc_cell_type='customlstm', dec_cell_type='customlstm', R= 0.999999, wKL=0.5, KLmin=0.2, **kwargs):
        super().__init__(**kwargs) 
        self.M = M
        self.out_dim = 6*M+3
        self.z_dim = z_dim
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.enc_cell_type = enc_cell_type
        self.dec_cell_type = dec_cell_type
        self.encoder = Encoder(self.enc_hidden, self.z_dim, dec_hidden, enc_cell_type=enc_cell_type, recurrent_dropout=False)
        #need to change the output to match stroke 
        self.decoder = Decoder(dec_hidden, self.out_dim, dec_cell_type=dec_cell_type, recurrent_dropout=False)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.max_seq_len = 50
        self.R = R
        self.wKL = wKL
        self.KLmin = KLmin
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "M": self.M,
            "z_dim": self.z_dim,
            "enc_hidden": self.enc_hidden,
            "dec_hidden": self.dec_hidden,
            "enc_cell_type": self.enc_cell_type,
            "dec_cell_type": self.dec_cell_type,
            "R": self.R,
            "wKL": self.wKL,
            "KLmin": self.KLmin,
            "max_seq_len": self.max_seq_len,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def loss_func(self, mixture_weight, meanx, meany, logsigx, logsigy, rho_hat, delX, delY, pen_gd, pen_pred, z_mean, z_logvar, mask=None):
        #gmm param -> (B, T, M)
        # delX, delY -> (B, T)
        # pen_gd, pen_pred -> (B, T, 3)
        B = tf.cast(tf.shape(delX)[0], tf.float32)
        T = tf.cast(tf.shape(delX)[1], tf.float32)
        eps = 1e-6
        pi = tf.nn.softmax(mixture_weight, axis=-1) 
        ent = -tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=-1)
        first_term = tf.math.log(2*math.pi)
        # logsigx = tf.clip_by_value(logsigx, -4.0, 2.0)
        # logsigy = tf.clip_by_value(logsigy, -4.0, 2.0)
        sigx = tf.exp(logsigx) # (B, T, M)
        sigy = tf.exp(logsigy) # (B, T, M)
        rho_hat = tf.tanh(rho_hat) # (B, T, M)
        # mixture_weight = tf.nn.softmax(mixture_weight, axis=-1)
        # pen_pred = tf.nn.softmax(pen_pred, axis=-1) # (B, T, 3)
        # need to understand this part
        norm_x = (tf.expand_dims(delX, axis=-1) - meanx) / (sigx + eps)
        norm_y = (tf.expand_dims(delY, axis=-1) - meany) / (sigy + eps)
        one_minus_rho2 = tf.maximum(1.0 - tf.square(rho_hat), eps)
        bivariant_loss = (-first_term 
                          - tf.math.log(sigx + eps) 
                          - tf.math.log(sigy + eps) 
                          -0.5 * tf.math.log(one_minus_rho2) 
                          - (1 / (2 * one_minus_rho2)) * (tf.square(norm_x) + 
                                                                    tf.square(norm_y) - 
                                                                    2 * rho_hat * norm_x * norm_y)) # (B, T, M)
        # (B,T,M)
        logsig_min = -4.0   # start here
        sigma_reg =tf.nn.relu(logsig_min - logsigx)**2 + tf.nn.relu(logsig_min - logsigy)**2  # (B, T, M)
        sigma_reg = tf.reduce_sum(sigma_reg * pi, axis=-1) # (B, T)
        LS = tf.reduce_logsumexp(bivariant_loss + tf.nn.log_softmax(mixture_weight, axis=-1), axis=-1) # (B, T)
        LR = keras.losses.categorical_crossentropy(pen_gd, pen_pred, from_logits=True)# (B, T)
        KL = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1)  * 1/self.z_dim) # (B, )
        
        mask_f = tf.cast(mask, tf.float32)  # (B, T)
        mask_d = mask_f[:, :, None]  # (B, T, 1)
        # pi = tf.nn.softmax(mixture_weight, axis=-1)   # (B,T,M)
        # dx_hat = tf.reduce_sum(pi * meanx, axis=-1)   # (B,T)
        # dy_hat = tf.reduce_sum(pi * meany, axis=-1)   # (B,T)

        time_step = tf.reduce_sum(mask_f, axis=-1)  # (B, )
        time_step = tf.maximum(time_step, 1.0)  # Avoid division by zero
        LS = tf.reduce_mean(tf.reduce_sum(LS * mask_f, axis=-1)/time_step)
        LR = tf.reduce_mean(tf.reduce_sum(LR * mask_f, axis=-1) /time_step)
        ent = tf.reduce_mean(tf.reduce_sum(ent * mask_f, axis=-1)/ time_step)
        sigma_reg = tf.reduce_mean(tf.reduce_sum(sigma_reg * mask_f, axis=-1) /time_step)
        
        
        # mse = tf.reduce_sum(tf.square(delX - dx_hat) + tf.square(delY - dy_hat), axis=-1)
        # mse = tf.reduce_mean(tf.reduce_sum(mse * mask_f, axis=-1) * 1/time_step)

        step = tf.cast(self.optimizer.iterations, tf.float32)
       
        n_min = 0.01
        
        beta = 1- (1-n_min)* self.R**step
        

        recon_loss = -LS + LR # 1e-3 * sigma_reg - 1e-3*ent
        loss = recon_loss + beta*self.wKL * tf.maximum(KL, self.KLmin)
        # metric = self.calculate_metric(mixture_weight, meanx, meany, logsigx, logsigy, rho_hat, z_mean, z_logvar, mask_f, mask_d, delX, delY)
        return loss, recon_loss, KL, None

    def calculate_metric(self, mixture_weight, meanx, meany, logsigx, logsigy, rho_hat, z_mean, z_logvar, mask_d, mask_d2, delX, delY):
       
        eps = 1e-8
        pi = tf.nn.softmax(mixture_weight, axis=-1)
        mix_entropy = tf.reduce_sum(-tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=-1) * mask_d)/ (tf.reduce_sum(mask_d) + 1e-8)
        meanx_bt = tf.reduce_sum(pi * meanx, axis=-1)
        meanx_mean = tf.reduce_sum( meanx_bt* mask_d) / (tf.reduce_sum(mask_d) + 1e-8 )
        meanx_std = tf.reduce_sum(tf.reduce_sum(pi * tf.square((meanx - meanx_bt[..., None])), axis=-1) * mask_d) / (tf.reduce_sum(mask_d) + 1e-8 )
        # residual_X_mean = tf.reduce_sum((delX - meanx_bt) * mask_d)/(tf.reduce_sum(mask_d) + 1e-8)
        # residual_X_std = tf.reduce_sum(tf.square((delX - meanx_bt) - residual_X_mean) * mask_d)/(tf.reduce_sum(mask_d) + 1e-8)
        
        
        meany_bt = tf.reduce_sum(pi * meany, axis=-1)
        meany_mean = tf.reduce_sum( meany_bt* mask_d) / (tf.reduce_sum(mask_d) + 1e-8 )
        meany_std = tf.reduce_sum(tf.reduce_sum( pi * tf.square((meany - meany_bt[..., None])), axis=-1) * mask_d) / (tf.reduce_sum(mask_d) + 1e-8 )
        # residual_Y_mean = tf.reduce_sum((delY - meany_bt) * mask_d)/(tf.reduce_sum(mask_d) + 1e-8)
        # residual_Y_std = tf.reduce_sum(tf.square((delY - meany_bt) - residual_Y_mean) * mask_d)/(tf.reduce_sum(mask_d) + 1e-8)
        sigx_bt = tf.reduce_sum(pi * logsigx, axis=-1)
        sigy_bt = tf.reduce_sum(pi * logsigy, axis=-1)
        rho_bt = tf.reduce_sum(pi * rho_hat, axis=-1)
        sigx_mean =  tf.reduce_sum(sigx_bt * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        sigx_std = tf.reduce_sum(tf.reduce_sum( pi * tf.square(logsigx - sigx_bt[..., None]), axis=-1) * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        # Ex2 = tf.reduce_sum(
        #     pi * (tf.exp(2.0 * logsigx) + tf.square(meanx)),
        #     axis=-1
        # )  
        # mu_x = tf.reduce_sum(pi * meanx, axis=-1)       # (B, T)

        # predictive variance Var_pred
        # var_pred_x = Ex2 - tf.square(mu_x)  
        # den = tf.reduce_sum(mask_d) + eps

        # pred_var_x = tf.reduce_sum(var_pred_x * mask_d) / den
        # pred_std_x = tf.sqrt(pred_var_x + eps) 
        
        # mu_y = tf.reduce_sum(pi * meany, axis=-1)

        # Ey2 = tf.reduce_sum(
        #     pi * (tf.exp(2.0 * logsigy) + tf.square(meany)),
        #     axis=-1
        # )

        # var_pred_y = Ey2 - tf.square(mu_y)

        # pred_var_y = tf.reduce_sum(var_pred_y * mask_d) / den
        # pred_std_y = tf.sqrt(pred_var_y + eps)
        
        sigy_mean = tf.reduce_sum(sigy_bt * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        sigy_std = tf.reduce_sum(tf.reduce_sum( pi * tf.square(logsigy - sigy_bt[..., None]), axis=-1) * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        
        rho_mean = tf.reduce_sum(rho_bt * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        rho_std = tf.reduce_sum(tf.reduce_sum( pi * tf.square(rho_hat - rho_bt[..., None]), axis=-1) * mask_d) / (tf.reduce_sum(mask_d) + 1e-8)
        
        z_per_dim = tf.reduce_mean(z_mean, axis=0)
        logs = {}
        for i in range(self.z_dim):
            logs[f"z_mean_dim_{i}"] = z_per_dim[i]
        return {
            "mix entrophy": mix_entropy,
            
            "sigx mean": sigx_mean,
            "sigx std": sigx_std,
            # "residual_X_mean": residual_X_mean,
            # "residual_X_std": residual_X_std,
            # "ratio_x": residual_X_std / (pred_var_x + eps),
            "sigy mean": sigy_mean,
            "sigy std": sigy_std,
            # "residual_Y_mean": residual_Y_mean,
            # "residual_Y_std": residual_Y_std,
            # "ratio_y": residual_Y_std / (pred_var_y + eps),

            "rho mean": rho_mean,
            "rho std": rho_std,  
                 
            "meanx mean": meanx_mean,
            "meanx std": meanx_std,
            "meany mean": meany_mean,
            "meany std": meany_std,
            
            "z_mean mean": tf.reduce_mean(z_mean),
            "z_mean std": tf.math.reduce_std(z_mean),
            "z_logvar mean": tf.reduce_mean(z_logvar),
            "z_logvar std": tf.math.reduce_std(z_logvar),
            **logs
        }
    def train_step(self, data, mask=None):
        X, y = data
        encoder_data = X['encoder_data']
        encoder_mask = X['encoder_mask']
        decoder_data = X['decoder_data']
        decoder_mask = X['decoder_mask']
        # mask = X['mask']
        mask_d = decoder_mask[:, :, None]
        # mask = tf.cast(mask, tf.float32)
        mask_d = tf.cast(mask_d, tf.float32)
        
        noise = tf.random.uniform(shape=(tf.shape(encoder_data)[0], 1, 2), minval=0.9, maxval=1.1)
        augmented = encoder_data[:, :, :2] * noise
        encoder_data = tf.concat([augmented, encoder_data[:, :, 2:]], axis=-1)
        augmented = decoder_data[:, :, :2] * noise
        decoder_data = tf.concat([augmented, decoder_data[:, :, 2:]], axis=-1)
        # data_aug = X
        with tf.GradientTape() as tape:
            h0, z, z_mean, z_logvar, c0 = self.encoder(encoder_data, training=True, mask=encoder_mask)
            z = tf.tile(z[:, None, :], [1, tf.shape(decoder_data)[1]-1, 1]) # z: (B, z_dim) -> (B, T, z_dim) 
            #X is mask but decoder input after concatenating z isn't mask 
            noise = tf.random.normal(shape=tf.shape(decoder_data[:, :-1, :2]) , stddev=0.05) * mask_d[:, :-1, :]
            decoder_input = tf.concat([decoder_data[:, :-1, :2] + noise,
                                       decoder_data[:, :-1, 2:], z], axis=-1)
            # print('decoder input shape: ', tf.shape(decoder_input))
            # print('decoder mask shape: ', tf.shape(mask))
            # decoder_input = decoder_input * mask_d[:, :-1]
            
            out, _, _ = self.decoder(decoder_input, initial_state=[h0, c0], training=True, mask=decoder_mask[:, :-1])

            #calculate loss here 
            # out dim is (B, T, output_dim)
            #calculate gradients and apply optimizer here
            
            gmm_raw = out[:, :, :6*self.M]
            penstate_pred = out[:, :, 6*self.M:]
            mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
            
            delX = decoder_data[:, 1:, 0]
            delY = decoder_data[:, 1:, 1]
            pen_gd = decoder_data[:, 1:, 2:]

            # Calculate the loss
            total_loss, recon_loss, kl_loss, metrics = self.loss_func(mixture_weight, meanx, meany, logsigx, logsigy, rho_hat,
                            delX=delX, delY=delY,
                            pen_gd=pen_gd, pen_pred=penstate_pred,
                            z_mean=z_mean, z_logvar=z_logvar, mask=decoder_mask[:,1:])

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Calculate gradients and apply optimizer
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # **metrics
           
             
        }
    def test_step(self, data):
        X, y = data
        encoder_data = X['encoder_data']
        encoder_mask = X['encoder_mask']
        decoder_data = X['decoder_data']
        decoder_mask = X['decoder_mask']
        # mask = X['mask']
        mask_d = decoder_mask[:, :, None]
        # mask = tf.cast(mask, tf.float32)
        mask_d = tf.cast(mask_d, tf.float32)
       
        h0, z, z_mean, z_logvar, c0 = self.encoder(encoder_data, training=False, mask=encoder_mask)
        z = tf.tile(z[:, None, :], [1, tf.shape(decoder_data)[1]-1, 1]) # z: (B, z_dim) -> (B, T, z_dim) 
        decoder_input = tf.concat([decoder_data[:, :-1, :2], decoder_data[:, :-1, 2:], z], axis=-1)
        # decoder_input = decoder_input * mask_d[:, :-1]
        out, _, _ = self.decoder(decoder_input, initial_state=[h0, c0], mask=decoder_mask[:, :-1], training=False)
        # out dim is (B, T, output_dim)
        gmm_raw = out[:, :, :6*self.M]
        penstate_pred = out[:, :, 6*self.M:]
        mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
        
        delX = decoder_data[:, 1:, 0]
        delY = decoder_data[:, 1:, 1]
        pen_gd = decoder_data[:, 1:, 2:]

        # Calculate the loss
        total_loss, recon_loss, kl_loss, metrics = self.loss_func(mixture_weight, meanx, meany, logsigx, logsigy, rho_hat,
                         delX=delX, delY=delY,
                         pen_gd=pen_gd, pen_pred=penstate_pred,
                         z_mean=z_mean, z_logvar=z_logvar, mask=decoder_mask[:,1:])

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # **metrics
           
           
        }
    def sample_output(self, mixture_weight, meanx, meany, logsigx, logsigy, rho_hat, penstate_pred):
        # shape B, M/3
        temp = 0.1
        k = tf.random.categorical(mixture_weight/temp, 1) # (B, 1)
        # k = tf.argmax(mixture_weight, axis=-1, output_type=tf.int32)[:, None]  # (B, 1)
        
        # k = tf.squeeze(k, axis=-1) # (B,)
        mu_x_k = tf.gather(meanx, k, batch_dims=1) # (B, 1)
        mu_y_k = tf.gather(meany, k, batch_dims=1 ) # (B, 1)
        logsigx_k = tf.gather(logsigx, k, batch_dims=1) # (B, 1)
        logsigy_k = tf.gather(logsigy, k, batch_dims=1) # (B, 1)
        rho_hat_k = tf.gather(rho_hat, k, batch_dims=1) # (B, 1)
        logsigx_k = tf.clip_by_value(logsigx_k, -6.0, 2.0)
        logsigy_k = tf.clip_by_value(logsigy_k, -6.0, 2.0)
        sigx_k = tf.exp(logsigx_k)# (B, 1)
        sigy_k = tf.exp(logsigy_k)  # (B, 1)
        rho_hat_k = tf.tanh(rho_hat_k) # (B, 1)
        e1 = tf.random.normal(tf.shape(mu_x_k)) # (B, 1)
        e2 = tf.random.normal(tf.shape(mu_y_k)) # (B, 1)
        one_minus_rho2 = tf.maximum(1.0 - tf.square(rho_hat_k), 1e-6) # (B, 1)

        dx = mu_x_k + sigx_k * e1  # (B, 1)
        dy = mu_y_k + sigy_k * (rho_hat_k * e1 + tf.sqrt(one_minus_rho2) * e2) # (B, 1)
        idx = tf.random.categorical(penstate_pred, 1)    
        temp_pen = 0.1
        logits = tf.reshape(penstate_pred, [tf.shape(penstate_pred)[0], 3]) / temp_pen
        idx = tf.random.categorical(logits, 1)
        idx = tf.argmax(logits, axis=-1, output_type=tf.int32)[:, None]  # (B, 1)
        idx = tf.squeeze(idx, axis=-1)# (B,) 
        pen_state = tf.one_hot(idx, depth=penstate_pred.shape[-1]) # (B, 3)
        mixture_weight = tf.nn.softmax(mixture_weight, axis=-1)
        dx = tf.reduce_sum(mixture_weight * meanx, axis=-1, keepdims=True)  # (B, 1)
        dy = tf.reduce_sum(mixture_weight * meany, axis=-1, keepdims=True)  # (B, 1)
        return dx, dy, pen_state
    
    def sample_generative(self, mixture_logits, meanx, meany, logsigx, logsigy, rho_hat, pen_logits,
                        gmm_temp=0.7, pen_temp=1.8, disabled_pen_end=False):
        k = tf.random.categorical(mixture_logits / gmm_temp, 1)  # (B,1)
        k = tf.squeeze(k, axis=-1)                               # (B,)

        mu_x_k = tf.gather(meanx, k, batch_dims=1)               # (B,)
        mu_y_k = tf.gather(meany, k, batch_dims=1)               # (B,)
        logsigx_k = tf.gather(logsigx, k, batch_dims=1)          # (B,)
        logsigy_k = tf.gather(logsigy, k, batch_dims=1)          # (B,)
        rho_hat_k = tf.gather(rho_hat, k, batch_dims=1)          # (B,)

        # logsigx_k = tf.clip_by_value(logsigx_k, -6.0, 2.0)
        # logsigy_k = tf.clip_by_value(logsigy_k, -6.0, 2.0)

        sigx_k = tf.exp(logsigx_k)
        sigy_k = tf.exp(logsigy_k)
        rho_k = tf.tanh(rho_hat_k)

        e1 = tf.random.normal(tf.shape(mu_x_k))
        e2 = tf.random.normal(tf.shape(mu_y_k))
        one_minus_rho2 = tf.maximum(1.0 - tf.square(rho_k), 1e-6)

        dx = (mu_x_k + sigx_k * e1)[:, None]  # (B,1)
        dy = (mu_y_k + sigy_k * (rho_k * e1 + tf.sqrt(one_minus_rho2) * e2))[:, None]  # (B,1)

        logits = tf.reshape(pen_logits, [tf.shape(pen_logits)[0], 3]) / pen_temp
        logits = tf.Variable(logits)

        if disabled_pen_end:
            logits[:, 2].assign(-1e9)# Force pen end to be unlikely if disabled
        idx = tf.random.categorical(logits, 1)
        idx = tf.squeeze(idx, axis=-1)
        pen_state = tf.one_hot(idx, depth=3, dtype=dx.dtype)  # (B,3)

        return dx, dy, pen_state

    def call(self, inputs, training=False):
        
       
        
        X = inputs['data']
        mask = inputs['mask']
        # mask = tf.cast(mask, tf.float32)
        mask_d = mask[:, :, None]
        mask_d = tf.cast(mask_d, tf.float32)
        B = tf.shape(X)[0]
        T = tf.shape(X)[1]
        # print('X shape: ', tf.shape(X))
        # print('mask shape: ', tf.shape(mask))

        h, z, z_mean, z_logvar, c = self.encoder(X, training=training, mask=mask) # inputs: (B, T, 5)
       
        z = tf.tile(z[:, None, :], [1, T-1, 1]) # z: (B, z_dim) -> (B, T-1, z_dim) 
        # z = tf.zeros(z.shape)  #
        # print('suffle')
        decoder_input = tf.concat([X[:, :-1, :], z], axis=-1)  # (B, 1, 5 + z_dim)
        # decoder_input = decoder_input * mask_d[:, :-1]
        
        
        

        out, h, c = self.decoder(decoder_input, initial_state=[h, c], training=training, mask=mask[:, :-1])  # out: (B, T-1, output_dim)
        gmm_raw = out[:, :, :6*self.M]
        penstate_pred = out[:, :, 6*self.M:]
        mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
        output = []
        for i in range(mixture_weight.shape[1]):
            dx, dy, pen_state = self.sample_output(mixture_weight[:, i, :], 
                                                   meanx[:, i, :], meany[:, i, :], 
                                                   logsigx[:, i, :], logsigy[:, i, :], 
                                                   rho_hat[:, i, :], penstate_pred[:, i, :])
            stroke = tf.concat([dx[:, None, :], dy[:, None, :], pen_state[:, None, :]], axis=-1) # (B, 1, 5)
            output.append(stroke)
        return tf.squeeze(tf.stack(output, axis=1)  , axis=2)  # (B, T, 5)
    def recon(self, inputs, training=False):
        X, y = inputs
        encoder_data = X['encoder_data']
        encoder_mask = X['encoder_mask']
        decoder_data = X['decoder_data']
        decoder_mask = X['decoder_mask']
        # mask = tf.cast(mask, tf.float32)
        mask_d = decoder_mask[:, :, None]
        mask_d = tf.cast(mask_d, tf.float32)
        # B = tf.shape(X)[0]
        # T = tf.shape(X)[1]
        # print('X shape: ', tf.shape(X))
        # print('mask shape: ', tf.shape(mask))

        h, z, z_mean, z_logvar, c = self.encoder(encoder_data, training=training, mask=encoder_mask) # inputs: (B, T, 5)
       
        z = tf.tile(z[:, None, :], [1, tf.shape(decoder_data)[1]-1, 1]) # z: (B, z_dim) -> (B, T-1, z_dim) 
        # z = tf.zeros(z.shape)  #
        # print('suffle')
        decoder_input = tf.concat([decoder_data[:, :-1, :], z], axis=-1)  # (B, 1, 5 + z_dim)
        # decoder_input = decoder_input * mask_d[:, :-1]
        
        
        

        out, h, c = self.decoder(decoder_input, initial_state=[h, c], training=training, mask=decoder_mask[:, :-1])  # out: (B, T-1, output_dim)
        gmm_raw = out[:, :, :6*self.M]
        penstate_pred = out[:, :, 6*self.M:]
        mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
        output = []
        for i in range(mixture_weight.shape[1]):
            dx, dy, pen_state = self.sample_output(mixture_weight[:, i, :], 
                                                   meanx[:, i, :], meany[:, i, :], 
                                                   logsigx[:, i, :], logsigy[:, i, :], 
                                                   rho_hat[:, i, :], penstate_pred[:, i, :])
            stroke = tf.concat([dx[:, None, :], dy[:, None, :], pen_state[:, None, :]], axis=-1) # (B, 1, 5)
            output.append(stroke)
        return tf.squeeze(tf.stack(output, axis=1)  , axis=2)
    def generate(self, inputs, max_len=250, training=False, pen_temp=1.8, gmm_temp=0.7, disabled_pen_end=False):
        X, y = inputs
        encoder_data = X['encoder_data']
        encoder_mask = X['encoder_mask']
        decoder_data = X['decoder_data']
        decoder_mask = X['decoder_mask']
        # mask = tf.cast(mask, tf.float32)
        mask_d = decoder_mask[:, :, None]
        mask_d = tf.cast(mask_d, tf.float32)
        B = tf.shape(encoder_data)[0]
        T = tf.shape(decoder_data)[1]
        dtype = encoder_data.dtype
        
        result = tf.zeros((B, 0, 5), dtype=dtype)  # Initialize an empty tensor to store generated points

        # Encode once
        h0, z, _, _, c0 = self.encoder(encoder_data, mask=encoder_mask, training=training)
        z = tf.tile(z[:, None, :], [1, 1, 1])  # (B, 1, z_dim)
        decoder_input = tf.concat([encoder_data[:, -1:, :], z], axis=-1)
        
        for i in range(T):
             # Start with the first point
            out, h0, c0 = self.decoder(decoder_input, initial_state=[h0, c0], training=training)  # (B, 1, output_dim)
            gmm_raw = out[:, :, :6*self.M]
            penstate_pred = out[:, :, 6*self.M:]
            mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
            dx, dy, pen_state = self.sample_generative(mixture_weight[:, 0, :], 
                                                   meanx[:, 0, :], meany[:, 0, :], 
                                                   logsigx[:, 0, :], logsigy[:, 0, :], 
                                                   rho_hat[:, 0, :], penstate_pred[:, 0, :], 
                                                   pen_temp=pen_temp, gmm_temp=gmm_temp)
            new_point = tf.concat([dx, dy, pen_state], axis=-1)  # (B, 5)
            result = tf.concat([result, new_point[:, None, :]], axis=1)  # Append new point to result
            decoder_input = tf.concat([new_point[:, None, :], z], axis=-1)  # Use the newly generated point as input for the next step
        # print('Generated sequence shape: ', tf.shape(result)) 
        output = result[:, -1:, :]  # Start with the last generated point
        for i in range(max_len):
            decoder_input = tf.concat([output[:, -1:, :], z], axis=-1)  # Use last generated point and z
            out, h0, c0 = self.decoder(decoder_input, initial_state=[h0, c0], training=training)  # (B, 1, output_dim)
            gmm_raw = out[:, :, :6*self.M]
            penstate_pred = out[:, :, 6*self.M:]
            mixture_weight, meanx, meany, logsigx, logsigy, rho_hat = tf.split(gmm_raw, num_or_size_splits=6, axis=-1)
            dx, dy, pen_state = self.sample_generative(mixture_weight[:, 0, :], 
                                                   meanx[:, 0, :], meany[:, 0, :], 
                                                   logsigx[:, 0, :], logsigy[:, 0, :], 
                                                   rho_hat[:, 0, :], penstate_pred[:, 0, :], 
                                                   pen_temp=pen_temp, gmm_temp=gmm_temp, disabled_pen_end=False)
            new_point = tf.concat([dx, dy, pen_state], axis=-1)  # (B, 5)
            output = tf.concat([output, new_point[:, None, :]], axis=1)  # Append new point to result
            if tf.reduce_all(pen_state[:, 2] == 1):  # If pen end is predicted
                break       
        return tf.concat([result, output[:, 1:, :]], axis=1)  # Combine initial generation with continued generation