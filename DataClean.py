import pandas as pd
import os
import logging

# Configure logging for professional output tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Loads the raw NCAA dataset and applies base filters."""
    logging.info(f"Loading raw dataset from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        raise

    # Filter for March Madness teams only
    initial_shape = df.shape
    df = df[df['Post-Season Tournament'] == 'March Madness'].copy()
    
    # Filter valid seasons (post-2007 for height stats, excluding 2020 Covid anomaly)
    df = df[(df['Season'] > 2007) & (df['Season'] != 2020)]
    logging.info(f"Filtered rows from {initial_shape[0]} to {df.shape[0]} based on tournament and season validity")
    
    return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Removes noisy columns and handles obvious missing data structures."""
    logging.info("Cleaning features and dropping low-variance/noisy columns")
    
    columns_to_drop = [
        'Region', 'Top 12 in AP Top 25 During Week 6?', 'Active Coaching Length',
        'DFP', 'NSTRate', 'RankNSTRate', 'OppNSTRate', 'RankOppNSTRate', 
        'Short Conference Name', 'Mapped Conference Name', 'Current Coach', 
        'Full Team Name', 'Since'
    ]
    
    # Drop columns if they exist in the dataframe to prevent KeyError
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Ensure Team Name is the primary identifier column (Index 0)
    if 'Mapped ESPN Team Name' in df.columns:
        team_name_col = df.pop("Mapped ESPN Team Name")
        df.insert(0, "Mapped ESPN Team Name", team_name_col)
        
    # Sort chronologically
    df = df.sort_values(by=['Season'], ascending=False).reset_index(drop=True)
    return df

def execute_etl_pipeline(input_path: str, output_dir: str):
    """Executes the complete Extract, Transform, Load pipeline."""
    df = load_and_filter_data(input_path)
    df = clean_features(df)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into historical training data and future inference data
    logging.info("Splitting dataset into Historical (Training) and Current (Inference)")
    historical_data = df[df['Season'] != 2025]
    future_data = df[df['Season'] == 2025]
    
    # Export
    hist_path = os.path.join(output_dir, 'Data.csv')
    future_path = os.path.join(output_dir, 'CurrentTeams.csv')
    
    historical_data.to_csv(hist_path, index=False)
    future_data.to_csv(future_path, index=False)
    logging.info(f"ETL Complete. Saved training data to {hist_path} and inference data to {future_path}")

if __name__ == "__main__":
    # Set to data directory
    RAW_DATA_PATH = 'MMData/MarchMadnessOriginal.csv' 
    OUTPUT_DIRECTORY = 'MMData'
    
    execute_etl_pipeline(RAW_DATA_PATH, OUTPUT_DIRECTORY)