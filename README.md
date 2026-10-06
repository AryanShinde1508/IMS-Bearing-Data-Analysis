# IMS Bearing Data Analysis

An analysis of vibration data from the IMS Bearing Dataset to understand how bearing condition changes as failure approaches.

This was one of my first projects where I worked with **real vibration data** instead of a small, clean dataset. The main focus was on understanding the data, extracting useful features, and seeing which features were able to track bearing degradation.

## What I worked on

The analysis covers:

* Loading and exploring raw vibration measurements
* Understanding the sampling rate and time-domain signal
* Basic time-domain features such as:

  * RMS
  * Peak
  * Crest Factor
  * Kurtosis
* Learning and implementing FFT for frequency-domain analysis
* Understanding spectral leakage and windowing
* Using a Hann window for the vibration signals
* Calculating theoretical bearing fault frequencies
* Comparing raw FFT with envelope analysis
* Extracting features around BPFO, BPFI, BSF and FTF
* Tracking feature values over the life of a bearing
* Comparing different features based on their degradation trends
* Using PCA to combine multiple useful features into a single health indicator

## Dataset

The project uses the **IMS Bearing Dataset**, which contains vibration measurements collected from bearings during run-to-failure experiments.

For the main analysis, I worked with **Set 2** of the dataset.

The dataset contains repeated measurements from the bearings as they operate and gradually approach failure. Each measurement file contains vibration samples recorded at a high sampling frequency.

> Dataset: IMS Bearing Dataset from the University of Cincinnati's Center for Intelligent Maintenance Systems (IMS).

## Dataset Setup

The raw IMS Bearing Dataset is not included in this repository because of its size.

To run the notebook locally:

1. Download the IMS Bearing Dataset from the original dataset source.
2. Extract the dataset.
3. Place the `IMS` folder in the root directory of this repository.

The expected structure is:

```text
IMS-Bearing-Data-Analysis/
│
├── IMS_Bearing_Data_Analysis.ipynb
├── README.md
├── Case_1_features.csv
├── Case_2_features.csv
├── Case_3_features.csv
│
└── IMS/
    ├── 1st_test/
    ├── 2nd_test/
    └── 3rd_test/
```

The notebook uses relative paths, so the dataset does not need to be stored at a specific location on the user's computer.


## Project Workflow

The overall workflow was:

```text
Raw vibration data
        ↓
Data inspection
        ↓
Time-domain analysis
        ↓
FFT / frequency-domain analysis
        ↓
Bearing fault frequencies
        ↓
Envelope analysis
        ↓
Feature extraction
        ↓
Feature trends over bearing life
        ↓
Feature quality comparison
        ↓
PCA-based health indicator
```

## Some important observations

One of the interesting parts of the analysis was that the expected bearing fault frequency was not always obvious in the raw FFT.

For the Set 2 bearing, the theoretical **BPFO** was around 236 Hz. Looking directly at the raw spectrum did not give a particularly clear peak at this frequency.

After applying envelope analysis, however, a much clearer component appeared around the expected BPFO.

This helped me understand why simply looking for a large peak in the FFT is not always enough for vibration-based bearing fault detection.

I also compared different time-domain and frequency-domain features over the complete bearing life instead of looking at only individual measurements.

Finally, I used PCA to combine some of the better-performing features into a single health indicator and compared it with the individual features.

## Repository Structure

```text
IMS-Bearing-Data-Analysis/
│
├── README.md
├── IMS_Bearing_Data_Analysis.ipynb
│
├── Case_1_features.csv
├── Case_2_features.csv
├── Case_3_features.csv
└── set2_bearing1_features.csv
```

The main analysis is contained in:

`IMS_Bearing_Data_Analysis.ipynb`

The CSV files contain the extracted features used for the later feature comparison and health-index analysis.

## Tools Used

* Python
* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-learn
* Jupyter Notebook

## What I learned

This project helped me get more comfortable with working with real-world data where the important patterns are not immediately obvious.

In particular, I got practical experience with:

* Working with large collections of measurement files
* Signal preprocessing
* FFT and frequency resolution
* Windowing and spectral leakage
* Basic vibration analysis
* Bearing fault frequency calculations
* Feature extraction
* Feature evaluation
* PCA and dimensionality reduction
* Interpreting plots instead of just generating them

It also gave me a better idea of how signal processing and data analysis can be used together for condition monitoring.

## Limitations

There are a few limitations to keep in mind when interpreting the results.

* Only a small number of complete bearing failure cases were used for the cross-case comparison.
* The theoretical fault frequencies were calculated assuming a constant shaft speed.
* The feature ranking uses relatively simple statistical measures.
* The PCA health indicator is an exploratory health indicator rather than a trained predictive-maintenance model.

So, the results are useful for understanding this dataset, but they should not be treated as a general-purpose bearing failure prediction system.

## Final Observations

The main thing I took away from this project is that bearing condition monitoring is not just about finding a large peak in an FFT.

Different features capture different changes in the vibration signal, and some faults become much easier to see after using techniques such as envelope analysis.

Looking at how features change throughout the life of a bearing was also more useful than looking at a single measurement in isolation.

Overall, this project was my attempt to combine basic signal processing with data analysis to understand how vibration measurements can be used to track bearing degradation.
