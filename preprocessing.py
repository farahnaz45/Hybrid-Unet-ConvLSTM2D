import os
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, seq_length=14, input_length=7, predict_length=7):
        self.seq_length = seq_length
        self.input_length = input_length
        self.predict_length = predict_length

    def load_image_sequence(self, folder_path):
        images = []
        for img_name in sorted(os.listdir(folder_path))[:self.seq_length]:
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(64, 64))
            img_array = img_to_array(img)
            images.append(img_array)
        return np.array(images)

    def load_dataset(self, base_dir, num_folders, visualize=True):
        X, y = [], []
        all_folders = os.listdir(base_dir)
        sampled_folders = np.random.choice(all_folders, num_folders, replace=False)

        for i, sample_folder in enumerate(sampled_folders):
            sample_path = os.path.join(base_dir, sample_folder)
            images = self.load_image_sequence(sample_path)

            if visualize and i < 2:
                self._visualize_sample(images, sample_folder)

            X.append(images[:self.input_length])
            y.append(images[self.input_length:self.input_length + self.predict_length])

        return np.array(X), np.array(y)

    def preprocess_data(self, data, is_input_data=True):
        if is_input_data:
            return data
        return data

    @staticmethod
    def _visualize_sample(images, sample_folder):
        num_images_to_display = min(5, len(images))
        fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 3))
        for j in range(num_images_to_display):
            axes[j].imshow(images[j])
            axes[j].set_title(f"Image {j + 1} from {sample_folder}")
        plt.show()

