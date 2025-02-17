# **Project Title Hybrid-Unet**
**A two-stage trained hybrid Unet-ConvLSTM2D for enhanced precipitation nowcasting**

## **Introduction**
This repository implements **Your Model Name**, a deep learning-based precipitation nowcasting model using a two-stage training approach.
- **Stage 1**: Pre-training on the MIINST dataset.
- **Stage 2**: Fine-tuning on the CIKM2017 dataset.

This model builds on deterministic and diffusion-based components for improved prediction accuracy.


# **Dataset**
## **📂 Datasets**
This project uses two datasets:

### **1️ MNIST Dataset (Pre-training)**
**📥 Download Link:**  
- [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

**📝 Description:**  
- **70,000 grayscale images** of handwritten digits (0-9).
- Each image is **28×28 pixels**.
- **Training Set:** 60,000 images
- **Testing Set:** 10,000 images

**🖼️ Example Images:**
![MNIST Example](https://www.tensorflow.org/images/MNIST.png)

---

### **2️⃣ CIKM2017 Dataset (Fine-Tuning)**
**📥 Download Link:**  
- [CIKM2017 Dataset](https://github.com/yaoyichen/CIKM-Cup-2017)

**📝 Description:**  
- Radar reflectivity maps covering a **101×101 km** region over **Shenzhen, China**.
- Each radar image consists of **101×101 pixels**, representing a **1×1 km** square.
- The dataset is divided into:
  - **Training Set:** 7,000 sequences
  - **Validation Set:** 2,000 sequences
  - **Testing Set:** 3,000 sequences
- **Each sequence spans 90 minutes**, with frames every **6 minutes**.

**📂 Dataset Structure:**

    ├── Train/       # Training data
    ├── Validation/  # Validation data
    ├── Test/        # Testing data



























This repository uses CIKM2017 dataset for precipiation nowcasting.



To run the code you can simply run the main.py file. 



There are four paths in the main.py file. Adjust them according to your own dataset paths and pretrained checkpoint file.



The dataset paths inclue paths to these folders: Train, Test and Validation from CIKM2017 dataset.



Evaluation, Score metrics and Plots will be generated from within main.py file.


