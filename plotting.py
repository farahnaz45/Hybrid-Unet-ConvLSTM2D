import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

class FrameVisualizer:
    def __init__(self, num_samples=20):
        self.cmap_colors = [
            "#000080", "#0000FF", "#00FFFF", "#008000",
            "#00FF00", "#008080", "#FFFF00", "#FFC000",
            "#FF8000", "#FF0000", "#C00000", "#800000",
            "#FF00FF", "#800080"
        ]
        self.dbz_cmap = ListedColormap(self.cmap_colors)
        self.dbz_ticks = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        self.dbz_labels = [f"{x}-{x+5}" for x in self.dbz_ticks[:-1]] + ["60-65"]  # Adjust last label
        self.num_samples = num_samples

    def visualize_frames(self, X_test, y_test, Predictions,save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        X_test_samples = X_test[:self.num_samples]
        y_test_samples = y_test[:self.num_samples]
        predicted_next_frames = Predictions[:self.num_samples]  # Get all predicted frames

        for i in range(self.num_samples):
            fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15, 6))  # 7 columns now
            print("Sample", {i + 1})

            for t in range(7):
                axes[0, t].imshow(y_test_samples[i, t, :, :, 0], cmap=self.dbz_cmap)  # Start from the 3rd frame
                axes[0, t].set_title(f'Truth at {t * 6} min')
                axes[0, t].axis('off')

            for t in range(7):
                axes[1, t].imshow(predicted_next_frames[i, t, :, :, 0], cmap=self.dbz_cmap)  # Start from the 3rd frame
                axes[1, t].set_title(f'Prediction at {t * 6} min')
                axes[1, t].axis('off')

            plt.tight_layout()
            plt.savefig(f"{save_dir}/sample_{i + 1}.png")  # Save the figure
            plt.close(fig)  # Close the figure to free memory