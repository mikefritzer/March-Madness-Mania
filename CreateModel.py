"""
March Madness Mania - updated model after the datathon.

What changed from the datathon version:
  1. Targets are now 0/1. Before, we multiplied the "Yes"/"No" strings by 3 and 2
     to weight them, but that just renamed the classes and didn't do anything.
  2. There are only 16 champions out of 1,077 teams, so each forest uses
     class_weight="balanced_subsample" to deal with the imbalance.
  3. The weighting for deeper runs now happens in the team strength score:
     3*P(champion) + 2*P(finalist) + 1*P(Final Four).
  4. Testing leaves out one full season at a time, so the model never trains
     and tests on the same year.
  5. Since these outcomes are rare, accuracy isn't very useful. We look at
     ROC-AUC, average precision, Brier score, and how many of the model's top 4
     teams actually made the Final Four.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGETS = ["Tournament Winner?", "Tournament Championship?", "Final Four?"]
TARGET_WEIGHTS = {"Tournament Winner?": 3, "Tournament Championship?": 2, "Final Four?": 1}
NON_FEATURES = ["Mapped ESPN Team Name", "Season"]


def load(path: str) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)
    for t in TARGETS:
        df[t] = (df[t] == "Yes").astype(int)
    features = [
        c for c in df.select_dtypes(include="number").columns
        if c not in NON_FEATURES + TARGETS
    ]
    return df, features


def make_model() -> Pipeline:
    # no scaler needed for random forests
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
    ])


def fit_all(train: pd.DataFrame, features: list) -> dict:
    return {t: make_model().fit(train[features], train[t]) for t in TARGETS}


def strength(models: dict, teams: pd.DataFrame, features: list) -> pd.Series:
    """Combine the three probabilities into one score, weighting deeper runs more."""
    score = sum(
        TARGET_WEIGHTS[t] * models[t].predict_proba(teams[features])[:, 1]
        for t in TARGETS
    )
    return pd.Series(score, index=teams.index)


def evaluate_by_season(df: pd.DataFrame, features: list) -> None:
    """For each season, train on all the other seasons and test on that one."""
    probs = {t: pd.Series(np.nan, index=df.index) for t in TARGETS}
    ff_hits, champ_top4 = [], []

    for season in sorted(df["Season"].unique()):
        test_mask = df["Season"] == season
        train, test = df[~test_mask], df[test_mask]
        models = fit_all(train, features)
        for t in TARGETS:
            probs[t][test.index] = models[t].predict_proba(test[features])[:, 1]

        top4 = strength(models, test, features).nlargest(4).index
        hits = int(test.loc[top4, "Final Four?"].sum())
        champ_in = bool(test.loc[top4, "Tournament Winner?"].any())
        ff_hits.append(hits)
        champ_top4.append(champ_in)
        logging.info(f"{season}: {hits}/4 Final Four hits, champion in top 4: {champ_in}")

    print("\nHeld-out metrics (each season predicted by a model that never saw it):")
    for t in TARGETS:
        y, p = df[t], probs[t]
        print(f"  {t:<26} ROC-AUC {roc_auc_score(y, p):.3f} | "
              f"Avg precision {average_precision_score(y, p):.3f} "
              f"(base rate {y.mean():.3f}) | Brier {brier_score_loss(y, p):.3f}")
    print(f"  Final Four hits: {sum(ff_hits)}/{4 * len(ff_hits)} "
          f"({np.mean(ff_hits):.2f} of 4 per season)")
    print(f"  Champion in model's top 4: {sum(champ_top4)}/{len(champ_top4)} seasons")


def predict_matchup(team_a: str, team_b: str, models: dict,
                    current: pd.DataFrame, features: list) -> str:
    rows = current[current["Mapped ESPN Team Name"].isin([team_a, team_b])]
    missing = {team_a, team_b} - set(rows["Mapped ESPN Team Name"])
    if missing:
        raise ValueError(f"Not in the current tournament field: {sorted(missing)}")
    s = strength(models, rows, features)
    s.index = rows["Mapped ESPN Team Name"]
    return s.idxmax()


if __name__ == "__main__":
    df, features = load("MMData/Data.csv")
    evaluate_by_season(df, features)

    # train on all past seasons, then predict this year's games
    models = fit_all(df, features)
    current, _ = load("MMData/CurrentTeams.csv")
    print(f"\nDuke vs Florida -> {predict_matchup('Duke', 'Florida', models, current, features)}")