from preprocessing import DataPreprocessor
from model import UNetModel
from train import UNetTrainer
from evaluate import Evaluator
import tensorflow as tf
import os


checkpoint_path = '/Unet_Precipiationnowcasting/ckim_final_new.weights.h5'
preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.load_dataset('/Unet_Precipiationnowcasting/ConvLSTM2D-master/train', num_folders=7000)
X_val, y_val = preprocessor.load_dataset('/Unet_Precipiationnowcasting/ConvLSTM2D-master/validation', num_folders=2000)
X_test, y_test = preprocessor.load_dataset('/Unet_Precipiationnowcasting/ConvLSTM2D-master/test', num_folders=3000)

def normalize(data, is_input_data=True):
    if is_input_data:
        data = data / 255.0
    else:
        pass
    return data

X_train = normalize(X_train, is_input_data=True)
y_train = normalize(y_train, is_input_data=True)
X_val = normalize(X_val, is_input_data=True)
y_val = normalize(y_val, is_input_data=True)
X_test = normalize(X_test, is_input_data=True)
y_test = normalize(y_test, is_input_data=True)

# Prepare TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(4)

# Build the U-Net model
model = UNetModel().build()

# Initialize the UNetTrainer
trainer = UNetTrainer(model)

# Check if weights exist
if os.path.exists(checkpoint_path):
    print(f"Weights found at {checkpoint_path}, loading and skipping training")
    model.load_weights(checkpoint_path)  # Load pre-existing weights
    trainer.compile_model()  # Compile the model even if weights are loaded
else:
    print("No weights found. Starting training from scratch.")

    # Set the callbacks for saving checkpoints, early stopping, etc.
    trainer.set_callbacks(checkpoint_path)

    # Compile the model
    trainer.compile_model()

    # Train the model
    trainer.train(train_dataset, val_dataset, epochs=200)

# Evaluate the model after training or loading weights
Predictions = model.predict(X_test)

# Calculate evaluation metrics
mae_scores, mse_scores, mean_mse = Evaluator.calculate_mae_mse(y_test, Predictions)

# Print the evaluation results
print("Evaluation Results:")
print(f"Mean MSE: {mean_mse}, MAE Scores: {mae_scores}, MSE Scores: {mse_scores}")

