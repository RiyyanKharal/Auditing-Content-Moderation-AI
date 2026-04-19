# Jigsaw Toxicity Detection with DistilBERT
## Bias & Fairness Analysis, Adversarial Attacks, and Mitigation Techniques

### Project Overview
This project implements a toxicity detection system using DistilBERT fine-tuned on the Jigsaw Toxic Comment Classification dataset. The analysis covers:

1. **Baseline Model Training** - DistilBERT fine-tuned for toxic comment classification
2. **Bias & Fairness Analysis** - Disparity analysis across identity groups (high-black vs. reference cohort)
3. **Adversarial Attacks** - Evasion attacks (character-level perturbations) and poisoning attacks (label flipping)
4. **Bias Mitigation Techniques** - Reweighing, Threshold Optimization, and Oversampling
5. **Production Pipeline** - Multi-layer moderation system with regex blocklist and calibrated model

### System Requirements

**Hardware:**
- T4 GPU colab

pip install -r requirements.txt

3. Download Dataset

Download train.csv from Jigsaw Toxic Comment Classification Challenge and place in project root.
4. Run Notebook

Open and execute Part1(5).ipynb in order.

For Google Colab:

    Select Runtime → Change runtime type → T4 GPU

    Mount Google Drive to save models

    Run all cells sequentially

Data Preparation

    Original dataset: 1.8M comments

    Training subset: 100,000 comments (stratified)

    Evaluation subset: 20,000 comments (stratified)

    Labels binarized at target ≥ 0.5

    Identity columns: black, white, muslim, jewish, homosexual_gay_or_lesbian

Key Results
Metric	Baseline	Reweighing	Threshold Opt	Oversampling
F1 Score	0.7866	0.8000	0.6493	0.7986
SPD	-0.0064	-0.0104	-0.0271	0.0338
EOD	0.0911	-0.1546	0.0844	0.1186
Model Outputs

Trained models are saved to /content/drive/MyDrive/fairness_models/:

    baseline_model/

    reweighed_model/

    oversampled_model/

    optimal_thresholds.json

File Structure
text

├── Part1(5).ipynb          # Main notebook
├── requirements.txt         # Dependencies
├── README.md               # This file
├── pipeline.py             # Moderation pipeline code
├── train_subset_100k.csv   # Training data (generated)
├── eval_subset_20k.csv     # Evaluation data (generated)
└── drive/MyDrive/
    └── fairness_models/    # Saved models

Citation

Jigsaw Toxic Comment Classification Challenge. (2018). Kaggle.

**Software:**
- Python: 3.12
- OS: Ubuntu 22.04 / Google Colab

### Setup Instructions

#### 1. Clone Repository
```bash
git clone <repository-url>
cd jigsaw-toxicity-detection  


