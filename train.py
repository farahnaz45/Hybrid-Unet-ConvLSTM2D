import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

class UNetTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.callbacks = []

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mse'])

    def set_callbacks(self, checkpoint_path):
        self.callbacks = [
            ReduceLROnPlateau(factor=0.5, patience=2),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        ]

    def train(self, train_dataset, val_dataset, epochs=10):
        return self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=self.callbacks)
