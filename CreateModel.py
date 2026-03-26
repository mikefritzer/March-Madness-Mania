import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_and_train_model(data_path: str) -> tuple[Pipeline, list]:
    """
    Builds a scikit-learn pipeline (Imputation > Scaling -> RF) and trains it on historical NCAA data.
    """
    logging.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    # Define target variables and feature space
    targets = ['Tournament Winner?', 'Tournament Championship?', 'Final Four?']
    X = df.drop(columns=['Mapped ESPN Team Name'] + targets, errors='ignore')
    X = X.select_dtypes(include=['number'])
    y = df[targets].copy()

    # Weight the target variables to prioritize deeper tournament runs
    y['Tournament Winner?'] *= 3
    y['Tournament Championship?'] *= 2

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Construct the ML Pipeline
    # Using an imputer catches any NaNs that slipped through cleaning
    logging.info("Constructing ML Pipeline (Imputer > Scaler > MultiOutput RF)")
    model_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42, n_jobs=-1)
        ))
    ])

    # Train the model
    logging.info("Training model pipeline")
    model_pipeline.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model_pipeline.predict(X_test)
    logging.info("Model Evaluation Metrics")
    winner_acc = accuracy_score(y_test['Tournament Winner?'], y_pred[:, 0])
    runner_acc = accuracy_score(y_test['Tournament Championship?'], y_pred[:, 1])
    
    logging.info(f"Winner Prediction Accuracy:   {winner_acc:.4f}")
    logging.info(f"Runner-Up Prediction Accuracy:{runner_acc:.4f}")
    
    # Print classification report for the Championship target
    print("\nClassification Report (Tournament Winner):")
    print(classification_report(y_test['Tournament Winner?'], y_pred[:, 0], zero_division=0))
    
    return model_pipeline, list(X.columns)

def predict_matchup(team_a_data: pd.DataFrame, team_b_data: pd.DataFrame, pipeline: Pipeline, features: list) -> str:
    """
    Evaluates two teams against the trained pipeline and returns the team 
    with the higher probability of winning the tournament.
    """
    # Ensure correct feature order and extraction
    t1_features = team_a_data[features]
    t2_features = team_b_data[features]
    
    # Extract the probability of class '1' (winning) for the first target variable ('Tournament Winner?')
    # pipeline.predict_proba returns a list of arrays for MultiOutputClassifiers
    prob_t1 = pipeline.predict_proba(t1_features)[0][0][1]
    prob_t2 = pipeline.predict_proba(t2_features)[0][0][1]
    
    return 'Team A' if prob_t1 > prob_t2 else 'Team B'

if __name__ == "__main__":
    # Trail & Test
    try:
        trained_pipeline, feature_cols = build_and_train_model('MMData/Data.csv')
        logging.info("Loading Current Tournament data for inference")
    except FileNotFoundError:
        logging.error("Missing dataset, please run DataClean.py first")
        exit()
    
    df_cur = pd.read_csv('MMData/CurrentTeams.csv')
    
    # Change names to teams to be simulated
    team_a_name = 'Duke'
    team_b_name = 'Florida'
    
    team_a = df_cur[df_cur['Mapped ESPN Team Name'] == team_a_name]
    team_b = df_cur[df_cur['Mapped ESPN Team Name'] == team_b_name]
    
    if not team_a.empty and not team_b.empty:
        logging.info(f"Simulating Matchup: {team_a_name} vs {team_b_name}")
        winner = predict_matchup(team_a, team_b, trained_pipeline, feature_cols)
        
        winning_team = team_a_name if winner == 'Team A' else team_b_name
        print(f"{winning_team} Won!")
    else:
        logging.warning("One or both requested teams could not be found in the dataset")