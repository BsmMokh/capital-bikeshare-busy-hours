"""Can the calendar tell us when Capital Bikeshare will be busy?"""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FEATURES = ["hr", "workingday", "mnth"]  # Hour, workday flag, month
ARTICLE_HOURS = [
    ("2012-03-17", 12),
    ("2012-04-16", 8),
    ("2012-04-17", 9),
    ("2012-05-14", 8),
    ("2012-09-21", 16),
]


def read_hours(archive):
    if archive is None:
        with urlopen(DATA_URL, timeout=60) as response:
            archive = BytesIO(response.read())
    with ZipFile(archive) as files:
        with files.open("hour.csv") as hourly_file:
            hours = pd.read_csv(hourly_file, parse_dates=["dteday"])

    assert not hours.isna().any().any(), "A value is missing."
    assert not hours.duplicated(["dteday", "hr"]).any(), "An hour appears twice."
    assert (hours["casual"] + hours["registered"] == hours["cnt"]).all()
    assert hours["hr"].between(0, 23).all() and hours["workingday"].isin([0, 1]).all()
    assert (hours["mnth"] == hours["dteday"].dt.month).all()
    assert (hours[["casual", "registered", "cnt"]] >= 0).all().all()
    hours["busy"] = hours["cnt"] >= 400
    return hours.sort_values(["dteday", "hr"])


def report_results(test):
    actual = test["busy"]
    predicted = test["alert"]
    correct_alerts = (actual & predicted).sum()
    missed_hours = actual.sum() - correct_alerts
    accuracy = accuracy_score(actual, predicted)
    assert accuracy == (actual == predicted).mean()

    print(f"Correct predictions: {accuracy:.1%} of all hours")
    print(f"Always guessing ordinary: {(~actual).mean():.1%} correct")
    print(f"Correct alerts: {correct_alerts} of {predicted.sum()} ({correct_alerts / predicted.sum():.1%})")
    print(f"Busy hours caught: {correct_alerts} of {actual.sum()} ({correct_alerts / actual.sum():.1%})")
    print(f"Busy hours missed: {missed_hours}")


def save_chart(test, path):
    plt.switch_backend("Agg")  # Save a PNG without opening a plot window.
    # The mean of a yes/no column is the fraction of rows marked yes.
    by_hour = test.groupby("hr")[["busy", "alert"]].mean() * 100
    fig, ax = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    ax.plot(by_hour.index, by_hour["busy"], "o-", color="#006958", label="Actually busy", lw=2.5, ms=4)
    ax.plot(by_hour.index, by_hour["alert"], "s--", color="#aa4820", label="Flagged by the tree", lw=2.2, ms=4)
    ax.set(xlabel="Hour of day", ylabel="Share of recorded hours (%)", ylim=(-3, 103))
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23], ["Midnight", "4 am", "8 am", "Noon", "4 pm", "8 pm", "11 pm"])
    ax.set_title("Which hours did the model overlook in 2012?", loc="left", fontsize=16, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(archive, output):
    # Learn from 2011. Keep 2012 separate to test on a later year.
    hours = read_hours(archive)
    year = hours["dteday"].dt.year
    train = hours[year == 2011].copy()
    test = hours[year == 2012].copy()
    assert len(train) + len(test) == len(hours)
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=24, random_state=414)
    tree.fit(train[FEATURES], train["busy"])
    test["alert"] = tree.predict(test[FEATURES])
    report_results(test)

    # A leaf is the group of hours that followed the same rules in the tree.
    train["leaf"] = tree.apply(train[FEATURES])
    test["leaf"] = tree.apply(test[FEATURES])
    groups = train.groupby("leaf")["busy"]
    test["train_busy"] = test["leaf"].map(groups.sum())
    test["train_hours"] = test["leaf"].map(groups.size())

    output.mkdir(parents=True, exist_ok=True)
    test.to_csv(output / "test_predictions.csv", index=False)
    errors = test[test["alert"] != test["busy"]].set_index(["dteday", "hr"])
    dates_and_hours = [(pd.Timestamp(date), hour) for date, hour in ARTICLE_HOURS]
    errors.loc[dates_and_hours].to_csv(output / "five_errors.csv")
    save_chart(test, output / "busy-hours.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Path to the UCI data ZIP.")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Folder for results.")
    args = parser.parse_args()
    archive = args.archive
    if archive is None and Path(__file__).with_name("bike-sharing.zip").is_file():
        archive = Path(__file__).with_name("bike-sharing.zip")
    main(archive, args.out)
