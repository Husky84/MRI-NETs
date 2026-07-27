MRI-NETs-Survival: Deep Learning for Survival Prediction
Status:​ Under Review / Preprint

📋 Project Overview

This repository contains the source code for training and evaluating a dual-branch deep learning model that predicts NETs (Neuroendocrine Transdifferentiation)-associated survival risk​ using pre-treatment CE-T1 and T2-FLAIR MRI scans. The model utilizes bulk RNA-seq derived NETs risk scores as the supervisory label (without accessing transcriptomic data during inference).

Key Features:

🧠 Dual-Branch 3D CNN: Processes CE-T1 and T2-FLAIR modalities separately.

📊 Survival Analysis: Outputs a continuous risk score correlated with patient survival.

🔍 Interpretability: Includes Grad-CAM visualization to highlight imaging regions contributing to the prediction.

📄 Reproducible: Clean pipeline from data loading to figure generation (Figure 7).

📂 Repository Structure

.
├── .gitignore                 # Specifies files to ignore (e.g., weights, env)
├── LICENSE                    # Apache 2.0 License
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── example_out2.csv           # 📝 Example survival data (format reference)
├── filter.py                  # 🚀 Main inference & Grad-CAM visualization script

📝 Note on Data
example_out2.csv​ is provided only as a format reference. It contains dummy data to show the expected columns (sample, risk_01, etc.).
Real patient data​ is sourced from the Chinese Glioma Genome Atlas (CGGA). Due to data use agreements and privacy regulations, raw data and model weights are not included​ in this repository. See the Data Availability section below.

🛠️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Husky84/MRI-NETs-Survival.git
cd MRI-NETs-Survival

2. Create Environment
We recommend using a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

🚀 Usage
Prepare Data (External)
Apply for CGGA data access​ at http://www.cgga.org.cn.
Download the necessary files:
Imaging section：Image-genomic data
Transcriptome data：mRNAseq_693&mRNAseq_325
Place them in the root directory or a designated data/ folder (modify filter.py paths accordingly).

Expected Output:
Grad-CAM visualizations (2D & 3D stacked plots)
Scatter plots (Observed vs. Predicted)
Bland-Altman plots
Kaplan-Meier survival curves (high-risk vs. low-risk)
These will be saved as PDF files in the results/ folder (which you can create).

📜 Data & Code Availability
Data Availability
The datasets analyzed during the current study are available in the Chinese Glioma Genome Atlas (CGGA)​ repository (http://www.cgga.org.cn). Access is subject to CGGA data use policies and requires formal application.
Derived data​ (e.g., out2.csv): Available from the corresponding author upon reasonable request, in compliance with CGGA regulations.
Example Format: example_out2.csv in this repository demonstrates the required input format.

🙏 Acknowledgements
Thanks to the CGGA consortium for providing the invaluable glioma dataset.
Thanks to the reviewers for their constructive feedback.
