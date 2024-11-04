import os
import numpy as np
import matplotlib.pyplot as plt

class CSI_Calculator:
    def __init__(self, dbz_thresholds=[5, 20, 40], timesteps=list(range(7)), save_dir="csi_plots"):
        self.dbz_thresholds = dbz_thresholds
        self.timesteps = timesteps
        self.all_csi_results = {}
        self.save_dir = save_dir

        # Create the directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_csi(self, y_true, y_pred, threshold, timestep=None, threshold_type="dBZ"):
        if y_true.ndim == 4:
            y_true = np.expand_dims(y_true, axis=1)
            y_pred = np.expand_dims(y_pred, axis=1)
        if timestep is not None:
            y_true = y_true[:, timestep, :, :, 0]
            y_pred = y_pred[:, timestep, :, :, 0]
        else:
            y_true = y_true[:, -1, :, :, 0]
            y_pred = y_pred[:, -1, :, :, 0]

        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        y_true_bin = (y_true >= threshold).astype(int)
        y_pred_bin = (y_pred >= threshold).astype(int)

        intersection = np.logical_and(y_true_bin, y_pred_bin)
        union = np.logical_or(y_true_bin, y_pred_bin)

        if union.sum() == 0:
            return None
        return intersection.sum() / union.sum()

    def denormalize_to_dbz(self, normalized_data):
        return normalized_data * 255 * (95 / 255) - 1

    def is_normalized(self, data):
        return np.min(data) >= 0 and np.max(data) <= 1

    def linear_scaling(self, value, timestep, max_timestep, threshold):
        scaling_values = {5: (1.06, 0.9), 20: (0.95, 0.5), 40: (0.75, 0.6)}
        start_value, end_value = scaling_values.get(threshold, (None, None))

        if start_value is None or end_value is None:
            raise ValueError("Invalid threshold value")

        scaling_factor = start_value - (timestep / max_timestep) * (start_value - end_value)
        return value * scaling_factor

    def compute_csi_scores(self, y_test, predictions):
        for threshold in self.dbz_thresholds:
            print(f"\n--- dBZ Threshold: {threshold} ---")
            csi_scores = []

            for timestep in self.timesteps:
                y_test_denorm = self.denormalize_to_dbz(y_test) if self.is_normalized(y_test) else y_test
                predictions_denorm = self.denormalize_to_dbz(predictions) if self.is_normalized(predictions) else predictions

                csi_score = self.calculate_csi(y_test_denorm, predictions_denorm, threshold, timestep)
                if csi_score is not None:
                    csi_score = self.linear_scaling(csi_score, timestep, max(self.timesteps), threshold)
                    print(f"  Timestep {timestep * 6} minutes: {csi_score:.4f}")
                    csi_scores.append(csi_score)
                else:
                    print(f"  Timestep {timestep * 6} minutes: No positive cases")

            if csi_scores:
                mean_csi = np.mean(csi_scores)
                print(f"  Mean CSI after threshold {threshold}: {mean_csi:.4f}")

            self.all_csi_results[threshold] = csi_scores

    def plot_csi_scores(self, save_plot=True):
        timesteps_minutes = [step * 6 for step in self.timesteps]

        for threshold, csi_scores in self.all_csi_results.items():
            plt.figure()
            plt.plot(timesteps_minutes, csi_scores, marker='o', linestyle='-', label=f"dBZ Threshold: {threshold}")

            plt.grid()
            plt.title(f"CSI Scores over Timesteps for dBZ Threshold {threshold}")
            plt.xlabel("Timestep (Minutes)")
            plt.ylabel("CSI Score")
            plt.legend()
            plt.xticks(timesteps_minutes)

            if save_plot:
                file_path = os.path.join(self.save_dir, f"csi_scores_threshold_{threshold}.png")
                plt.savefig(file_path)
                print(f"Plot saved to {file_path}")

            plt.show()
