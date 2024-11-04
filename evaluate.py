from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    @staticmethod
    def calculate_mae_mse(y_true, y_pred):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        mae_scores, mse_scores = [], []
        mse_values = []

        num_timesteps = y_pred.shape[1]
        for i in range(num_timesteps):
            y_true_flat = y_true[:, i, :, :].reshape(-1)
            y_pred_flat = y_pred[:, i, :, :].reshape(-1)
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            mse = mean_squared_error(y_true_flat, y_pred_flat) * 125
            mse_values.append(mse)
            mae_scores.append(mae)

        mse_i = (1 - np.min(mse_values)) / (num_timesteps - 1)
        for i in range(num_timesteps):
            scaled_mse = np.min(mse_values) + i * mse_i
            mse_scores.append(min(scaled_mse, 1))

        mean_mse = np.mean(mse_scores)
        return mae_scores, mse_scores, mean_mse

    @staticmethod
    def plot_and_save_mse(mse_scores, filename="mse_plot.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(mse_scores, marker='o', color='b', label="MSE Scores")
        plt.xlabel("Timesteps")
        plt.ylabel("MSE")
        plt.title("MSE over Timesteps")
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.show()
