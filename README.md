# **Project Title Hybrid-Unet**
**A two-stage trained hybrid Unet-ConvLSTM2D for enhanced precipitation nowcasting**

## **Introduction**
This repository implements **Your Model Name**, a deep learning-based precipitation nowcasting model using a two-stage training approach.
- **Stage 1**: Pre-training on the MIINST dataset.
- **Stage 2**: Fine-tuning on the CIKM2017 dataset.

This model builds on deterministic and diffusion-based components for improved prediction accuracy.


---

### **5️ Add Dataset Information**
Use bullet points for clarity.

```markdown
## **Datasets**
This project uses two datasets:
- **MIINST Dataset**: Used for pre-training.
- **CIKM2017 Dataset**: Used for fine-tuning.

  **Download Datasets**:
- [MIINST Dataset](dataset-link)
- [CIKM2017 Dataset](https://competitions.codalab.org/competitions/17168#learn_the_details)


















This repository uses CIKM2017 dataset for precipiation nowcasting.



To run the code you can simply run the main.py file. 



There are four paths in the main.py file. Adjust them according to your own dataset paths and pretrained checkpoint file.



The dataset paths inclue paths to these folders: Train, Test and Validation from CIKM2017 dataset.



Evaluation, Score metrics and Plots will be generated from within main.py file.


