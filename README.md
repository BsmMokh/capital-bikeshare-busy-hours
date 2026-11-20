# Can a calendar predict a Capital Bikeshare rush?

A busy hour means at least **400 rentals** across the D.C. system. The model learned from 2011 trips using three clues: the hour, the month, and whether it was a workday.

In 2012, **497 of its 540 alerts were correct**. But it caught only **28% of all busy hours**, so a dispatcher could miss a rush even when the model predicts a quiet hour.

## Run it

From the project folder, run these two commands with Python 3.11. The data is included.

```sh
python3.11 -m pip install -r requirements.txt
python3.11 analysis.py
```

The script prints the scores and saves three files in `results/`:

- `busy-hours.png` compares actual busy hours with the model’s alerts by time of day.
- `five_errors.csv` contains the five wrong predictions discussed in the article.
- `test_predictions.csv` contains every 2012 prediction. Compare `busy` (what happened) with `alert` (what the model predicted).

Data from Hadi Fanaee-T’s [UCI Bike Sharing dataset](https://doi.org/10.24432/C5W894), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
