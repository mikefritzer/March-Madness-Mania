# March Madness Mania: NCAA Tournament Predictor

## Project Overview
The primary objective of this project is to predict the outcomes of NCAA March Madness matchups using Machine Learning. By analyzing historical trends dating back to 2010 alongside current team profiles, this model identifies the underlying characteristics of teams that historically advance to the Final Four and Championship rounds.

## Data Engineering & Preprocessing
This project utilizes a comprehensive dataset of past and present NCAA tournament teams. To optimize model performance and reduce overfitting, the dataset underwent rigorous preprocessing:
* **Noise Reduction:** Eliminated non-March Madness teams (e.g., NIT tournament participants) and stripped irrelevant or highly correlated features (e.g., team conferences, week 6 AP ranks).
* **Temporal Filtering:** Excluded data prior to 2007 due to missing player height metrics, and removed the 2020 season to account for COVID-19 anomalies.
* **Dimensionality Reduction:** Dropped columns with significant missing values (e.g., Non-Steal Turnovers) to maintain dataset integrity.

## Methodology & Architecture
The predictive engine is built on a **Weighted Multi-output Random Forest Classifier (RFC)** implemented in Python (`scikit-learn`). 
* **Target Variables:** The model predicts multiple outcomes simultaneously: `Tournament Winner`, `Tournament Championship`, and `Final Four`.
* **Class Weighting:** To account for the difficulty and importance of later rounds, higher weights are assigned to the Winner and Runner-Up dependent variables during training.
* **Evaluation:** The model is evaluated utilizing Accuracy, Classification Reports, and probability thresholds to ensure realistic outcome modeling (e.g., mitigating the over-prediction of extreme upsets).

## Challenges & Limitations
The NCAA tournament is famously difficult to predict due to inherent randomness. While this model successfully identifies underlying statistical trends that out-perform a standard "highest-seed-wins" strategy, it cannot account for "cosmic noise" such as unexpected injuries, locker room dynamics, or historic anomalies (e.g., a 15-seed defeating a 2-seed). 

## Future Improvements
Given the constraints of a 48-hour Datathon, future iterations of this project will explore:
1. **Monte Carlo Simulations:** Running the bracket probabilities thousands of times to establish more robust confidence intervals for the entire tournament.
2. **Deep Learning Integration:** Implementing Neural Networks alongside the RFC to better capture non-linear relationships in team characteristics.
3. **Granular Player-Level Data:** Expanding the feature space to include individual player stats to better assess the impact of star players or localized injuries.
4. **End-to-End Automation:** Developing a comprehensive script to fully automate the data ingestion, matchup simulation, and bracket generation process.
