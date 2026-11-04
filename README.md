# IMS Bearing Data Analysis

Vibration-based bearing condition monitoring on the IMS run-to-failure dataset, in two parts:

1. **Analysis**: signal processing, feature extraction and a PCA health indicator, to understand how bearing condition changes as failure approaches.
2. **Prediction Model**: machine-learning models for health-state detection, fault classification and Remaining Useful Life (RUL) estimation, built on the extracted features.

This was one of my first projects where I worked with **real vibration data** instead of a small, clean dataset. The main focus was on understanding the data, extracting useful features, and seeing which features were able to track bearing degradation.

## Repository Structure

```
IMS-Bearing-Data-Analysis/
│
├── README.md
│
├── Analysis/
│   ├── IMS_Bearing_Data_Analysis.ipynb
│   ├── Case_1_features.csv
│   ├── Case_2_features.csv
│   ├── Case_3_features.csv
│   └── set2_bearing1_features.csv
│
└── Prediction_Model/
    ├── 02_RUL_and_Fault_Classification_v3.ipynb
    ├── extract_features.py
    ├── ims_cli.py
    ├── README_CLI.md
    ├── case1_features_v3.csv
    ├── case2_features_v3.csv
    ├── case3_features_v3.csv
    └── IMS Bearing Run-to-Failure Health, Fault and RUL.pdf
```

| Part | What it contains |
|------|------------------|
| `Analysis/` | The exploratory notebook and the feature CSVs it uses |
| `Prediction_Model/` | The RUL and fault-classification notebook, the feature extraction script, a command-line tool, richer feature CSVs and a written report (PDF) |

## Part 1: Analysis

The analysis notebook (`Analysis/IMS_Bearing_Data_Analysis.ipynb`) covers:

- Loading and exploring raw vibration measurements
- Understanding the sampling rate and time-domain signal
- Basic time-domain features such as RMS, Peak, Crest Factor and Kurtosis
- Learning and implementing FFT for frequency-domain analysis
- Understanding spectral leakage and windowing
- Using a Hann window for the vibration signals
- Calculating theoretical bearing fault frequencies
- Comparing raw FFT with envelope analysis
- Extracting features around BPFO, BPFI, BSF and FTF
- Tracking feature values over the life of a bearing
- Comparing different features based on their degradation trends
- Using PCA to combine multiple useful features into a single health indicator

### Some important observations

One of the interesting parts of the analysis was that the expected bearing fault frequency was not always obvious in the raw FFT.

For the Set 2 bearing, the theoretical **BPFO** was around 236 Hz. Looking directly at the raw spectrum did not give a particularly clear peak at this frequency.

After applying envelope analysis, however, a much clearer component appeared around the expected BPFO.

This helped me understand why simply looking for a large peak in the FFT is not always enough for vibration-based bearing fault detection.

I also compared different time-domain and frequency-domain features over the complete bearing life instead of looking at only individual measurements, and then used PCA to combine some of the better-performing features into a single health indicator.

## Part 2: Prediction Model

Building on the Analysis, the `Prediction_Model/` folder turns the extracted features into predictive models. The main notebook is `02_RUL_and_Fault_Classification_v3.ipynb`, and the same pipeline is available as a command-line tool (`ims_cli.py`, see `README_CLI.md`).

It covers three tasks:

- **Health-state detection**: an Isolation Forest for anomaly detection, and a Random Forest classifier for healthy vs. degraded states
- **Fault diagnosis**: identifying the fault type (BPFO vs. BPFI) from envelope-spectrum features, using a simple rule and a logistic regression model
- **RUL estimation**: predicting remaining useful life as a percentage of bearing life, using Ridge regression, Random Forest and XGBoost

All models are evaluated with **Leave-One-Set-Out** validation across the three run-to-failure cases (each case is held out in turn), with no random train/test splits, so a model is never tested on a bearing it has already seen.

`extract_features.py` builds the per-file feature tables from the raw data (time-domain statistics, band energies and envelope-spectrum fault-frequency features). A written summary of the results is in `Prediction_Model/IMS Bearing Run-to-Failure Health, Fault and RUL.pdf`.

## Dataset

The project uses the **IMS Bearing Dataset**, which contains vibration measurements collected from bearings during run-to-failure experiments. Each measurement file contains vibration samples recorded at a high sampling frequency.

> Dataset: IMS Bearing Dataset from the University of Cincinnati's Center for Intelligent Maintenance Systems (IMS).

The main analysis uses **Set 2** of the dataset, and the prediction models use all three cases.

### Dataset Setup

The raw IMS Bearing Dataset is not included in this repository because of its size.

To run the code locally:

1. Download the IMS Bearing Dataset from the original dataset source.
2. Extract the dataset.
3. Place the `IMS` folder in the **root** of this repository, next to `Analysis/` and `Prediction_Model/`.

The expected structure is:

```
IMS-Bearing-Data-Analysis/
│
├── Analysis/
├── Prediction_Model/
│
└── IMS/
    ├── 1st_test/
    ├── 2nd_test/
    └── 3rd_test/
```

The extracted feature CSVs are already included, so the raw dataset is only needed to re-run the feature extraction.

## Workflow

```
Raw vibration data
        ↓
Data inspection
        ↓
Time-domain and FFT analysis
        ↓
Bearing fault frequencies + envelope analysis
        ↓
Feature extraction
        ↓
Feature trends over bearing life
        ↓
PCA-based health indicator                  (Analysis)
        ↓
Health detection, fault diagnosis, RUL      (Prediction Model)
```

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- XGBoost
- Jupyter Notebook

## What I learned

This project helped me get more comfortable with working with real-world data where the important patterns are not immediately obvious.

In particular, I got practical experience with:

- Working with large collections of measurement files
- Signal preprocessing
- FFT and frequency resolution
- Windowing and spectral leakage
- Basic vibration analysis
- Bearing fault frequency calculations
- Feature extraction and feature evaluation
- PCA and dimensionality reduction
- Building and validating models on small run-to-failure datasets
- Interpreting plots instead of just generating them

## Limitations

- Only three complete run-to-failure cases are available, so results should be read as a proof of concept rather than a general-purpose bearing failure prediction system.
- The theoretical fault frequencies were calculated assuming a constant shaft speed.
- The feature ranking in the Analysis part uses relatively simple statistical measures.
- The PCA health indicator is an exploratory indicator rather than a trained model.
- Health-state labels in the Prediction Model are derived from RMS thresholds, so models that use RMS-like features can be partly circular. This is why results are also reported without those features.

## Final Observations

Bearing condition monitoring is not just about finding a large peak in an FFT. Different features capture different changes in the vibration signal, and some faults become much easier to see after using techniques such as envelope analysis. Looking at how features change throughout the life of a bearing was also more useful than looking at a single measurement in isolation. 
