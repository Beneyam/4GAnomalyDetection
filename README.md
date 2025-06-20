# 4G Anomaly Detection

This project implements cell-level anomaly detection on 4G KPI (Key Performance Indicator) data using various machine learning models. The goal is to identify cells exhibiting abnormal behavior over time, which can help in proactive network maintenance and optimization.

## Features

- Loads and processes cleaned KPI data from CSV files
- Detects anomalies using:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
- Compares model performance using precision, recall, and F1-score
- Visualizes results with 2D and 3D plots for better interpretability

## Usage

1. **Prepare Data**  
   Place your cleaned KPI CSV file at:  
   `RealData/AugestRawData/cleaned_kpi_data.csv`

2. **Run the Notebook**  
   Open and execute `anomaly_detection_pipeline.ipynb` in VS Code or Jupyter Notebook.

3. **Review Results**  
   - Check the output metrics for model comparison.
   - Explore the generated plots for anomaly visualization.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

### Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Project Structure

```
4GAnomalyDetection/
│
├── anomaly_detection_pipeline.ipynb   # Main notebook for anomaly detection
├── README.md                          # Project documentation
└── RealData/
    └── AugestRawData/
        └── cleaned_kpi_data.csv       # Input data file (not included)
```

## License

This project is licensed under the MIT License.

---

*Created by Beneyam