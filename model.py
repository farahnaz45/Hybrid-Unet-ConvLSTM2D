import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class UNetModel:
    def __init__(self, input_shape=(7, 64, 64, 1), l2_lambda=0.01):
        self.input_shape = input_shape
        self.l2_lambda = l2_lambda

    def build(self):
        inputs = layers.Input(self.input_shape)

        # Encoder
        conv1 = self.conv_block(inputs, 64)
        pool1 = layers.Dropout(0.3)(layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv1))

        conv2 = self.conv_block(pool1, 128)
        pool2 = layers.Dropout(0.3)(layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv2))

        conv3 = self.conv_block(pool2, 256)
        pool3 = layers.Dropout(0.3)(layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv3))

        conv4 = self.conv_block(pool3, 512)
        pool4 = layers.Dropout(0.3)(layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv4))

        # Bottleneck
        conv5 = layers.ConvLSTM2D(512, (3, 3), activation='relu', padding='same', return_sequences=True)(pool4)
        conv5 = layers.BatchNormalization()(conv5)
        conv5 = layers.ConvLSTM2D(512, (3, 3), activation='relu', padding='same', return_sequences=True)(conv5)
        conv5 = layers.BatchNormalization()(conv5)

        # Decoder
        up6 = self.upsample_block(conv5, conv4, 512)
        up7 = self.upsample_block(up6, conv3, 256)
        up8 = self.upsample_block(up7, conv2, 128)
        up9 = self.upsample_block(up8, conv1, 64)

        outputs = layers.TimeDistributed(layers.Conv2D(1, (1, 1), activation='sigmoid'))(up9)

        model = models.Model(inputs, outputs, name='U-Net')
        return model

    def conv_block(self, inputs, filters):
        conv = layers.TimeDistributed(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))(inputs)
        conv = layers.TimeDistributed(layers.BatchNormalization())(conv)
        conv = layers.TimeDistributed(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))(conv)
        return layers.TimeDistributed(layers.BatchNormalization())(conv)

    def upsample_block(self, inputs, skip_connection, filters):
        up = layers.TimeDistributed(layers.UpSampling2D((2, 2)))(inputs)
        concat = layers.Concatenate()([up, skip_connection])
        conv = layers.TimeDistributed(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))(concat)
        conv = layers.TimeDistributed(layers.BatchNormalization())(conv)
        conv = layers.TimeDistributed(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))(conv)
        return layers.TimeDistributed(layers.BatchNormalization())(conv)

    def add_l2_regularization(self, model):
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                layer.kernel_regularizer = regularizers.l2(self.l2_lambda)
