MetaGAVE: Multivariate Time Series Anomaly Detection Framework

MetaGAVE is an end-to-end framework for multivariate time series anomaly detection that integrates:

*Data Imputation: GAT-based missing value imputation

*Anomaly Detection: Dual-path mechanism (VAE reconstruction + BiAT-LSTM prediction)

*Anomaly Explanation: Probabilistic graph-based root cause analysis

🚀 Features

✅ Graph Attention Networks (GAT) for learning inter-variable dependencies

✅ VAE for reconstruction-based anomaly detection

✅ BiAT-LSTM for prediction-based anomaly detection

✅ Dynamic threshold computation using POT (Peaks Over Threshold)

✅ Missing value imputation

✅ Anomaly explanation and root cause localization

✅ Support for benchmark datasets (SMAP, MSL, SMD, SWAT)

✅ Comprehensive evaluation metrics

✅ Training visualization and logging

📦 Installation

*Requirements*

-> requirements.txt

torch>=2.0.0

numpy>=1.21.0

pandas>=1.3.0

scikit-learn>=1.0.0

matplotlib>=3.4.0

seaborn>=0.11.0

tqdm>=4.62.0

scipy>=1.7.0

*Install*
pip install -r requirements.txt
