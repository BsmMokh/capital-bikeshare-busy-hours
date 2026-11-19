"""Forecast busy Capital Bikeshare hours with calendar features available in advance."""
import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FEATURES = ["hr", "workingday", "mnth"]
BUSY_THRESHOLD = 400


def load_hours(archive):
    if archive is None:
        with urlopen(DATA_URL, timeout=60) as response:
            archive = BytesIO(response.read())
    with ZipFile(archive) as package, package.open("hour.csv") as source:
        hours = pd.read_csv(source, parse_dates=["dteday"])
    assert not hours.isna().any().any()
    assert not hours.duplicated(["dteday", "hr"]).any()
    assert hours.hr.between(0, 23).all()
    assert hours.workingday.isin([0, 1]).all()
    assert hours.mnth.eq(hours.dteday.dt.month).all()
    assert hours[["casual", "registered", "cnt"]].ge(0).all().all()
    assert hours.cnt.eq(hours.casual + hours.registered).all()
    hours["busy"] = hours.cnt.ge(BUSY_THRESHOLD)
    return hours.sort_values(["dteday", "hr"])


def plot_by_hour(test, destination):
    grouped = test.groupby("hr")[["busy", "alert"]].mean() * 100
    fig, ax = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    ax.plot(grouped.index, grouped.busy, "o-", color="#006958", label="Actually busy", lw=2.5, ms=4)
    ax.plot(grouped.index, grouped.alert, "s--", color="#aa4820", label="Flagged by the tree", lw=2.2, ms=4)
    ax.set(xlabel="Hour of day", ylabel="Share of recorded hours (%)", ylim=(-3, 103),
           xticks=[0, 4, 8, 12, 16, 20, 23],
           xticklabels=["Midnight", "4 am", "8 am", "Noon", "4 pm", "8 pm", "11 pm"])
    ax.set_title("Which hours did the model overlook in 2012?", loc="left", fontsize=16, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional local copy of the UCI ZIP.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    hours = load_hours(args.archive)
    train = hours.loc[hours.dteday.dt.year == 2011].copy()
    test = hours.loc[hours.dteday.dt.year == 2012].copy()
    assert len(train) + len(test) == len(hours) and train.dteday.max() < test.dteday.min()
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=24, random_state=414)
    tree.fit(train[FEATURES], train.busy)
    test["alert"] = tree.predict(test[FEATURES])
    train["leaf"] = tree.apply(train[FEATURES])
    test["leaf"] = tree.apply(test[FEATURES])
    leaf_summary = train.groupby("leaf").busy.agg(train_busy="sum", train_hours="size")
    test = test.join(leaf_summary, on="leaf")
    assert test.alert.eq(2 * test.train_busy > test.train_hours).all()
    print(f"Recorded hours: {len(hours)}. Training: {len(train)}, busy: {train.busy.sum()}.")
    print(f"Test: {len(test)}, busy: {test.busy.sum()}.")
    tn, fp, fn, tp = confusion_matrix(test.busy, test.alert, labels=[False, True]).ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy={(tn+tp)/len(test):.4f}, precision={tp/(tp+fp):.4f}, recall={tp/(tp+fn):.4f}")
    print(f"Always-ordinary accuracy={(tn+fp)/len(test):.4f}")
    assert np.array_equal([tn, fp, fn, tp], [
        (~test.busy & ~test.alert).sum(), (~test.busy & test.alert).sum(),
        (test.busy & ~test.alert).sum(), (test.busy & test.alert).sum()])
    test.to_csv(args.out / "test_predictions.csv", index=False)
    article_hours = [
        ("2012-03-17", 12), ("2012-04-16", 8), ("2012-04-17", 9),
        ("2012-05-14", 8), ("2012-09-21", 16),
    ]
    indexed = test.set_index(["dteday", "hr"])
    examples = indexed.loc[[(pd.Timestamp(day), hour) for day, hour in article_hours]]
    assert len(examples) == 5 and examples.busy.ne(examples.alert).all()
    examples.to_csv(args.out / "five_errors.csv")
    plot_by_hour(test, args.out / "busy-hours.png")


if __name__ == "__main__":
    main()
