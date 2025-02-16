This repository uses CIKM2017 dataset for precipiation nowcasting.



To run the code you can simply run the main.py file. 



There are four paths in the main.py file. Adjust them according to your own dataset paths and pretrained checkpoint file.



The dataset paths inclue paths to these folders: Train, Test and Validation from CIKM2017 dataset.



Evaluation, Score metrics and Plots will be generated from within main.py file.

Repository Structure
/dataset/          # Dataset folder (MIINST & CIKM2017 datasets)
/models/           # Model architecture files (ConvLSTM, UNet, etc.)
/utils/            # Utility functions (data preprocessing, evaluation metrics)
/scripts/          # Training and evaluation scripts
main.py            # Main execution script
README.md          # Project documentation
requirements.txt   # Dependencies
.gitignore         # Files to be ignored by Git
**Datasets**

