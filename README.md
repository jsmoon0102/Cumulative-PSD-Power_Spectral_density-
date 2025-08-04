# **Cumulative-PSD-Power_Spectral_density-**
Super easy, Super fast technique for extracting intention from overt behaviors. 

**Overview**
This repository contains all core code and analysis scripts used in my MSc thesis:
“Cumulative Power Spectral Density (CPSD) as a Feature for Motor Intention Decoding in EEG”

The project proposes a simple but robust pipeline for extracting and analyzing cumulative PSD features from EEG data,
with a focus on distinguishing voluntary (IC) and involuntary (OC) movement intentions.

**Key Features**
**1. CPSD Feature Extraction**:
Extracts cumulative beta/gamma-band power from EEG epochs.

**2. Adaptive Thresholding**:
Computes intention detection thresholds for each trial/subject automatically.

**3. IC/OC State Classification**:
Pipeline for classifying voluntary/involuntary states in EEG using minimal features.



"""
**Sliding Window CPSD Analysis for Motor Imagery EEG**
--------------------------------------------------
- Computes cumulative PSD (CPSD) in beta/gamma bands per trial
- Optimizes threshold via ROC curve for each transition
- Outputs per-subject, per-run result CSVs
  
**Notes**
--This repository will be public for a limited period for review.
--It may be reverted to private after one week for data security.
--If you have questions or need access to non-public data, please contact me.
--All code is written in Python (3.9+), and uses standard packages:
  numpy, pandas, mne, scipy, sklearn, matplotlib.


  **Contact**
June Sung Moon
Email: jsmoon0102@gmail.com

MSc, Concordia University
