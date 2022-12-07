import tensorflow as tf
import tensorflow_probability as tfp

from re_implementation.helpers import bias_helper

appy_correction = False


class Encoder(tf.keras.Model):
    def __init__(self, num_channel=1, num_filter=32, latent_dimensions=20, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.latent_dimensions = latent_dimensions

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(32, 32, num_channel))

        self.conv_layer1 = tf.keras.layers.Conv2D(filters=num_filter, kernel_size=4, strides=2, padding="same")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv_layer2 = tf.keras.layers.Conv2D(filters=2 * num_filter, kernel_size=4, strides=2, padding="same")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv_layer3 = tf.keras.layers.Conv2D(filters=4 * num_filter, kernel_size=4, strides=2, padding="same")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.conv_layer4_mean = tf.keras.layers.Conv2D(
            filters=latent_dimensions, kernel_size=4, strides=1, padding="valid"
        )
        self.conv_layer4_var = tf.keras.layers.Conv2D(
            filters=latent_dimensions, kernel_size=4,
            strides=1, activation=tf.keras.activations.softplus,
            padding="valid"
        )

    def call(self, inputs):
        x = self.input_layer(inputs)

        x = self.conv_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv_layer2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.conv_layer3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        z_mean = self.conv_layer4_mean(x)
        z_log_var = self.conv_layer4_var(x)

        return z_mean, z_log_var

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'num_channel': self.num_channel,
            'num_filter': self.num_filter,
            'latent_dimensions': self.latent_dimensions
        })
        return config

    def print_network(self):
        x = tf.keras.layers.Input(shape=(32, 32, self.num_channel))
        model = tf.keras.Model(inputs=x, outputs=self.call(x), name='encoder')
        model.summary()
        return model


class Decoder(tf.keras.Model):
    def __init__(self, num_channel=1, num_filter=32, latent_dimensions=20, decoder_dist="cBern", name="decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.num_channel = num_channel
        self.num_filter = num_filter
        self.latent_dimensions = latent_dimensions
        self.decoder_dist = decoder_dist

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(1, 1, latent_dimensions))

        self.deconv_layer1 = tf.keras.layers.Conv2DTranspose(filters=4 * num_filter, kernel_size=4, strides=1,
                                                             padding="valid")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.deconv_layer2 = tf.keras.layers.Conv2DTranspose(filters=2 * num_filter, kernel_size=4, strides=2,
                                                             padding="same")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.deconv_layer3 = tf.keras.layers.Conv2DTranspose(filters=num_filter, kernel_size=4, strides=2,
                                                             padding="same")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        if decoder_dist == "cBern":
            self.deconv_layer4 = tf.keras.layers.Conv2DTranspose(filters=num_channel, kernel_size=4, strides=2,
                                                                 padding="same")
            self.reshape_layer = tf.keras.layers.Reshape(target_shape=(32, 32, num_channel))

        elif decoder_dist == "cat":
            self.deconv_layer4 = tf.keras.layers.Conv2DTranspose(filters=num_channel * 256, kernel_size=4, strides=2,
                                                                 padding="same")
            self.reshape_layer = tf.keras.layers.Reshape(target_shape=(32, 32, num_channel, 256))

        else:
            raise ValueError("Undefined Decoder Output Distribution.")

    def call(self, inputs):
        x = self.input_layer(inputs)

        x = self.deconv_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.deconv_layer2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.deconv_layer3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.deconv_layer4(x)
        reconstruction = self.reshape_layer(x)

        return reconstruction

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'num_channel': self.num_channel,
            'num_filter': self.num_filter,
            'latent_dimensions': self.latent_dimensions,
            'decoder_dist': self.decoder_dist
        })
        return config

    def print_network(self):
        x = tf.keras.layers.Input(shape=(1, 1, self.latent_dimensions))
        model = tf.keras.Model(inputs=x, outputs=self.call(x), name='decoder')
        model.summary()
        return model


class Sampling(tf.keras.layers.Layer):
    def __init__(self, name="sampling", **kwargs):
        super(Sampling, self).__init__(name=name, **kwargs)

    def call(self, inputs, num_samples):
        z_mean, z_log_var = inputs
        z = tfp.distributions.Normal(loc=z_mean, scale=z_log_var).sample(num_samples)
        z = tf.reshape(z, (-1, z.shape[2], z.shape[3], z.shape[4]))
        return z

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config


class CVAE(tf.keras.Model):
    def __init__(self, num_channel=1, num_filter=32, latent_dimensions=20, num_samples=100, decoder_dist="cBern",
                 name="cvae", **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)

        self.num_channel = num_channel
        self.num_filter = num_filter
        self.latent_dimensions = latent_dimensions
        self.num_samples = num_samples
        self.decoder_dist = decoder_dist

        self.sampling = Sampling()
        self.encoder = Encoder(num_channel, num_filter, latent_dimensions)
        self.decoder = Decoder(num_channel, num_filter, latent_dimensions, decoder_dist)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var), self.num_samples)
        reconstruction = self.decoder(z)

        z = tf.reshape(z, (self.num_samples, -1, 1, 1, self.latent_dimensions))

        z_mean_broadcasted = tf.expand_dims(z_mean, axis=0)
        z_mean_broadcasted = tf.repeat(z_mean_broadcasted, repeats=[self.num_samples], axis=0)

        z_log_var_broadcasted = tf.expand_dims(z_log_var, axis=0)
        z_log_var_broadcasted = tf.repeat(z_log_var_broadcasted, repeats=[self.num_samples], axis=0)

        return {'reconstruction': reconstruction,
                'kl_divergence': tf.stack([z_mean_broadcasted, z_log_var_broadcasted, z])
                }

    def get_config(self):
        config = super(CVAE, self).get_config()
        config.update({
            'num_channel': self.num_channel,
            'num_filter': self.num_filter,
            'latent_dimensions': self.latent_dimensions,
            'num_samples': self.num_samples,
            'decoder_dist': self.decoder_dist
        })
        return config

    def print_network(self):
        x = tf.keras.layers.Input(shape=(32, 32, self.num_channel))
        model = tf.keras.Model(inputs=x, outputs=self.call(x), name='cvae')
        model.summary()
        return model

    def continuous_bernoulli_loss(self, x_true, reconstruction):
        reconstruction = tf.reshape(reconstruction, (self.num_samples, -1, 32, 32, self.num_channel))

        lp_x_z = tfp.distributions.ContinuousBernoulli(logits=reconstruction).log_prob(x_true)

        loss = tf.reduce_mean(
            tf.reduce_logsumexp(tf.reduce_sum(lp_x_z, axis=[2, 3, 4]), axis=0) - tf.math.log(float(self.num_samples))
        )

        if appy_correction:
            debiasing_term = bias_helper.get_bias_correction_term("cBern")
            corrected_loss = loss - debiasing_term
        else:
            corrected_loss = loss

        return -corrected_loss

    def categorical_loss(self, x_true, reconstruction):
        reconstruction = tf.reshape(reconstruction, (self.num_samples, -1, 32, 32, self.num_channel, 256))

        lp_x_z = tfp.distributions.Categorical(logits=reconstruction).log_prob(x_true)

        loss = tf.reduce_mean(
            tf.reduce_logsumexp(tf.reduce_sum(lp_x_z, axis=[2, 3, 4]), axis=0) - tf.math.log(float(self.num_samples))
        )

        if appy_correction:
            debiasing_term = bias_helper.get_bias_correction_term("cat")
            corrected_loss = loss - debiasing_term
        else:
            corrected_loss = loss

        return -corrected_loss

    def get_reconstruction_loss_func(self):
        if self.decoder_dist == 'cBern':
            loss_func = self.continuous_bernoulli_loss

        elif self.decoder_dist == 'cat':
            loss_func = self.categorical_loss

        else:
            raise ValueError("Undefined Decoder Output Distribution.")

        return loss_func

    def kl_divergence_loss(self, x_true, z_stack):
        z_mean, z_log_var, z = tf.unstack(z_stack)

        lq_z_x = tfp.distributions.Normal(loc=z_mean, scale=z_log_var).log_prob(z)
        lp_z = tfp.distributions.Normal(loc=0, scale=1).log_prob(z)

        loss = tf.reduce_logsumexp(tf.reduce_sum(lp_z - lq_z_x, axis=[2, 3, 4]), axis=0) - tf.math.log(float(self.num_samples))

        return -tf.reduce_mean(loss)
