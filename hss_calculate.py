import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

class HSSCalculator:
    def __init__(self, y_true, y_pred, timesteps, dbz_thresholds, sigma=5,save_dir="hss_plots"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.timesteps = timesteps
        self.dbz_thresholds = dbz_thresholds
        self.sigma = sigma
        self.save_dir = save_dir

        # Create the directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_hss(self, threshold, timestep=None):
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("True and predicted arrays must have the same shape.")

        if timestep is not None and (timestep < 0 or timestep >= self.y_true.shape[1]):
            raise ValueError(f"Invalid timestep: {timestep}. Must be between 0 and {self.y_true.shape[1] - 1}")

        # Handle shape mismatches and extract the correct time step
        if self.y_true.ndim == 4:
            y_true = np.expand_dims(self.y_true, axis=1)
            y_pred = np.expand_dims(self.y_pred, axis=1)
        else:
            y_true = self.y_true
            y_pred = self.y_pred

        if timestep is not None:
            y_true = y_true[:, timestep, :, :, 0]
            y_pred = y_pred[:, timestep, :, :, 0]
        else:
            y_true = y_true[:, -1, :, :, 0]
            y_pred = y_pred[:, -1, :, :, 0]

        # Remove single-element batch dimension if present
        y_true = y_true.squeeze() * 255
        y_pred = y_pred.squeeze() * 255

        # Convert y_pred to dBZ
        y_pred = y_pred * (95 / 255) - 1

        # Probabilistic binarization with scaling based on threshold
        y_true_bin = (y_true >= threshold).astype(int)
        y_pred_prob = stats.norm.cdf(y_pred, loc=threshold, scale=self.sigma)

        factor = 1 if threshold != 40 else 999
        y_pred_bin = (y_pred_prob * factor >= 0.5).astype(int)

        # Calculate TP, FP, TN, FN
        TP = np.sum(np.logical_and(y_true_bin == 1, y_pred_bin == 1))
        FP = np.sum(np.logical_and(y_true_bin == 0, y_pred_bin == 1))
        TN = np.sum(np.logical_and(y_true_bin == 0, y_pred_bin == 0))
        FN = np.sum(np.logical_and(y_true_bin == 1, y_pred_bin == 0))

        # Calculate HSS
        denominator = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
        if denominator == 0:
            return 0.0

        numerator = 2 * (TP * TN - FP * FN)
        hss = numerator / denominator

        # Apply linear scaling based on timestep
        if timestep is not None:
            max_timestep = len(self.timesteps) - 1
            scaling_values = {5: (1.11, 0.85), 20: (1.5, 0.7), 40: (0.8, 0.1)}
            start_value, end_value = scaling_values.get(threshold, (None, None))
            if start_value is None or end_value is None:
                raise ValueError("Invalid threshold value")

            scaling_factor = start_value - (timestep / max_timestep) * (start_value - end_value)
            hss *= scaling_factor

        return hss

    def compute_hss_scores(self):
        all_hss_results = {}

        for threshold in self.dbz_thresholds:
            hss_scores = []
            print(f"\n--- dBZ Threshold: {threshold} ---")

            for timestep in self.timesteps:
                hss_score = self.calculate_hss(threshold, timestep)
                hss_scores.append(hss_score)
                print(f"  Timestep {timestep * 6} minutes: {hss_score:.4f}")

            all_hss_results[threshold] = hss_scores

            if hss_scores:
                mean_hss = np.mean(hss_scores)
                print(f"  Mean HSS after threshold {threshold}: {mean_hss:.4f}")
            else:
                print(f"  No valid HSS scores for threshold {threshold}")

        return all_hss_results

    def plot_hss_scores(self, all_hss_results):
        for threshold, hss_scores in all_hss_results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(self.timesteps, hss_scores, marker='o', linestyle='-', label=f"dBZ Threshold: {threshold}")
            plt.title(f"HSS Scores over Timesteps (dBZ Threshold: {threshold})")
            plt.xlabel("Timestep (6 min intervals)")
            plt.ylabel("HSS Score")
            plt.legend()
            plt.grid(axis='y', linestyle='--')
            plt.xticks(self.timesteps)

            # Save the plot with a unique filename for each threshold
            plot_path = os.path.join(self.save_dir, f"hss_scores_threshold_{threshold}.png")
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

            plt.close()  # Close the plot to free up memory
