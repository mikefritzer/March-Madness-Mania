"""
Cleans up the Kaggle NCAA data and splits it into two files:
  Data.csv         - past tournament teams, used for training
  CurrentTeams.csv - this year's 68 teams, used for predictions
"""
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CURRENT_SEASON = 2025
MIN_SEASON = 2008          # no height/experience data before 2007
EXCLUDED_SEASONS = {2020}  # covid year, no tournament
TEAM_COL = "Mapped ESPN Team Name"

# Columns that don't say anything about how good a team is.
# Note: the coach columns only reflect the current coach, copied back to every
# old season (Duke in 2002 shows Scheyer), so they can't be used as features.
COLUMNS_TO_DROP = [
    "Region", "Top 12 in AP Top 25 During Week 6?",
    "Short Conference Name", "Mapped Conference Name", "Full Team Name",
    "Current Coach", "Since", "Active Coaching Length", "Active Coaching Length Index",
    "Post-Season Tournament", "Post-Season Tournament Sorting Index",
]


def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Load the raw csv and keep only March Madness teams from the seasons we want."""
    logging.info(f"Loading raw dataset from {input_path}")
    df = pd.read_csv(input_path)
    initial_rows = len(df)

    df = df[
        (df["Post-Season Tournament"] == "March Madness")
        & (df["Season"] >= MIN_SEASON)
        & (~df["Season"].isin(EXCLUDED_SEASONS))
    ].copy()
    logging.info(f"Filtered rows from {initial_rows} to {len(df)} by tournament and season")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Get rid of columns we don't need or that aren't filled in for every season."""
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Some stats weren't tracked every year (NSTRate only since 2021, DFP is
    # missing 2017-2019, possession length starts in 2010), so drop them.
    incomplete = df.columns[df.isna().any()].tolist()
    logging.info(f"Dropping columns with missing values: {incomplete}")
    df = df.drop(columns=incomplete)

    # Seed comes in as text since non-tournament teams have labels instead of numbers
    df["Seed"] = df["Seed"].astype(int)

    df.insert(0, TEAM_COL, df.pop(TEAM_COL))
    return df.sort_values(["Season", TEAM_COL], ascending=[False, True]).reset_index(drop=True)


def execute_etl_pipeline(input_path: str, output_dir: str) -> None:
    df = clean_features(load_and_filter_data(input_path))

    historical = df[df["Season"] < CURRENT_SEASON]
    current = df[df["Season"] == CURRENT_SEASON]
    if len(current) != 68 or current[TEAM_COL].duplicated().any():
        logging.warning(f"Expected 68 unique {CURRENT_SEASON} teams, found {current[TEAM_COL].nunique()}")

    os.makedirs(output_dir, exist_ok=True)
    hist_path = os.path.join(output_dir, "Data.csv")
    current_path = os.path.join(output_dir, "CurrentTeams.csv")
    historical.to_csv(hist_path, index=False)
    current.to_csv(current_path, index=False)
    logging.info(f"Saved {len(historical)} historical teams to {hist_path} "
                 f"and {len(current)} {CURRENT_SEASON} teams to {current_path}")


if __name__ == "__main__":
    execute_etl_pipeline("MMData/MarchMadnessOriginal.csv", "MMData")
