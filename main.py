from preprocessing import DataPreprocessor
from model import UNetModel
from train import UNetTrainer
from evaluate import Evaluator
from hss_calculate import HSSCalculator
from csi_calculate import CSI_Calculator
from plotting import FrameVisualizer
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.load_dataset('/content/Unet_Precipiationnowcasting/ConvLSTM2D-master/train', num_folders=7000)
X_val, y_val = preprocessor.load_dataset('/content/Unet_Precipiationnowcasting/ConvLSTM2D-master/validation', num_folders=2000)
X_test, y_test = preprocessor.load_dataset('/content/Unet_Precipiationnowcasting/ConvLSTM2D-master/test', num_folders=3000)

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

#prepare TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(4)

#build the U-Net model
model = UNetModel().build()

#initialize the UNetTrainer
trainer = UNetTrainer(model)

checkpoint_path = '/content/Unet_Precipiationnowcasting/ckim_final_new.weights.h5'

#check if weights exist
if os.path.exists(checkpoint_path):
    print(f"Weights found at {checkpoint_path}, loading and skipping training")
    model.load_weights(checkpoint_path)
    trainer.compile_model()
else:
    print("No weights found. Starting training from scratch.")

    trainer.set_callbacks(checkpoint_path)

    #compile the model
    trainer.compile_model()

    #train the model
    trainer.train(train_dataset, val_dataset, epochs=20)

#evaluate the model after training or loading weights
Predictions = model.predict(X_test)

#calculate mse
mae_scores, mse_scores, mean_mse = Evaluator.calculate_mae_mse(y_test, Predictions)

#print the evaluation results
print("Evaluation Results:")
print(f"Mean MSE: {mean_mse}, MAE Scores: {mae_scores}, MSE Scores: {mse_scores}")
#Evaluator.plot_and_save_mse(mse_scores)

#Csi Caclulation
csi_calculator = CSI_Calculator()
csi_calculator.compute_csi_scores(y_test, Predictions)
csi_calculator.plot_csi_scores()

#HSS Calculation
timesteps = list(range(7))
dbz_thresholds = [5, 20, 40]
hss_calculator = HSSCalculator(y_test, Predictions, timesteps, dbz_thresholds)
all_hss_results = hss_calculator.compute_hss_scores()
hss_calculator.plot_hss_scores(all_hss_results)

#plotting
visualizer = FrameVisualizer(num_samples=5)
visualizer.visualize_frames(X_test, y_test, Predictions)
