# Forecasting busy hours on Capital Bikeshare

A supervised classification project for INST414 Module 5. A decision tree trained on 2011 calendar data predicts whether each recorded hour in 2012 will have at least 400 rentals across the system. The threshold is a project definition.

## Reproduce the article

Use Python 3.11. From this folder, run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py --archive bike-sharing.zip
```

Omit `--archive bike-sharing.zip` to fetch the original archive from UCI. The other inputs are fixed in the script. The run prints the split sizes, confusion matrix counts and evaluation metrics. It writes every test prediction, the five errors discussed in the article and a publication-ready figure to `results/`.

In the CSVs, `busy` is the label from the observed count and `alert` is the calendar-based prediction. The columns `train_busy` and `train_hours` give the number of busy hours and total hours in the same terminal node of the fitted tree during 2011. The figure and five selected errors are also included in this package for review.

## Source and attribution

Hadi Fanaee-T (2013), [Bike Sharing](https://doi.org/10.24432/C5W894), UCI Machine Learning Repository, based on Capital Bikeshare rental records. The original UCI ZIP is included under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This analysis uses its hourly table and adds binary labels, model predictions and summary statistics.
