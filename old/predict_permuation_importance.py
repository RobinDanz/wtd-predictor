import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
import shap
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset
    """
    # Load the CSV file with semicolon delimiter and handle European number format
    df = pd.read_csv(filepath, sep=';', decimal=',')

    df = df[(df['WTD'] >= -10) & (df['WTD'] <= 75)]
    df = df[(df['pH'] >= 2.5) & (df['pH'] <= 5.5)]
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"First 2 rows:\n{df.head(2)}")
    
    # Print column names to check for extra spaces
    print("\nActual column names in the CSV file:")
    for col in df.columns:
        print(f"'{col}'")
    
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    
    # Define columns to exclude from features - handle possible whitespace in column names
    metadata_cols = []
    for col in df.columns:
        if col.strip() in ['PERSON', 'COUNTRY', 'SITE', 'SAMPLE', 'REFERENCE', 'SAMPLE CODE', 'SITE NAME', 'pH', 'LAT', 'LONG', 'WTD'] or 'Unnamed' in col:
            metadata_cols.append(col)
    
    print(f"\nMetadata columns identified: {metadata_cols}")
    
    # Extract target variable
    target_col = [col for col in df.columns if 'WTD' in col][0]
    y = df[target_col].copy()
    
    # Convert target to float type (handle European number format with comma as decimal)
    if y.dtype == 'object':
        y = y.str.replace(',', '.').astype(float)
    
    # Extract features (species data)
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X = df[feature_cols].copy()
    
    # Convert feature columns to numeric type
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                # Replace comma with dot for decimal if necessary
                X[col] = X[col].str.replace(',', '.').astype(float)
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to numeric. Error: {str(e)}")
                X = X.drop(columns=[col])
                feature_cols.remove(col)
    
    # Check for NaN values and fill them
    if X.isna().any().any():
        print(f"Warning: Found NaN values in features. Filling with zeros.")
        X = X.fillna(0)
    
    # Check for any remaining non-numeric columns
    non_numeric_cols = [col for col in X.columns if X[col].dtype == 'object']
    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns after conversion: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    print(f"\nNumber of features (species): {len(feature_cols)}")
    print(f"\nFeatures: {feature_cols}")
    print(f"Target variable (WTD) range: {y.min()} to {y.max()}")
    
    return X, y, feature_cols

def train_xgboost_model(X, y):
    """
    Train an XGBoost model and evaluate its performance
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Ensure there are no NaN values in the target
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())
    
    # Initialize and train the XGBoost model with safer default settings
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist'  # Use histogram-based algorithm for better stability
    )
    
    try:
        # Try to perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\nCross-validation RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    except Exception as e:
        print(f"\nWarning: Cross-validation failed with error: {str(e)}")
        print("Proceeding with direct model fitting instead...")
    
    # Train the model on the full training set
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model fitting: {str(e)}")
        print("Trying alternative approach with simpler model...")
        # Fall back to a simpler model configuration
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            tree_method='hist',
            random_state=42
        )
        model.fit(X_train, y_train)

    # 4. Calculer l'importance par permutation
    result = permutation_importance(model, X_test, y_test, n_repeats=100, random_state=42, n_jobs=-1)

    # 5. Créer un DataFrame pour les résultats
    perm_importances = pd.Series(result.importances_mean, index=X.columns)
    perm_importances = perm_importances.sort_values(ascending=False)

    # 6. Afficher les top features
    print(perm_importances.head(10))

    # 7. Visualiser
    plt.figure(figsize=(10, 6))
    perm_importances.head(20).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Features par Permutation Importance")
    plt.xlabel("Importance moyenne")
    plt.show()
    
    # # Make predictions on the test set
    # y_pred = model.predict(X_test)
    
    # # Calculate metrics
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)
    
    # print(f"Test set RMSE: {rmse:.4f}")
    # print(f"Test set R² score: {r2:.4f}")
    
    # return model

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    # Get feature importance scores
    importance = model.feature_importances_
    
    # Create dataframe of features and their importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    print("\nTop 15 most important species for WTD prediction:")
    print(importance_df.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Most Important Species for WTD Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved as 'feature_importance.png'")
    
    return importance_df

def main(filepath):
    """
    Main function to run the analysis
    """
    print("Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data(filepath)
    
    print("\nTraining XGBoost model...")
    model = train_xgboost_model(X, y)

    # explainer = shap.TreeExplainer(model)
    # explanation = explainer(X)

    # shap_values = explanation.values

    # shap.plots.beeswarm(explanation, max_display=20)
    
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_names)
    
    # Save feature importance to CSV
    importance_df.to_csv('species_importance.csv', index=False)
    print("Feature importance saved to 'species_importance.csv'")

if __name__ == "__main__":
    # Use the file path provided in the error message
    filepath = "All data-Tableau 1.csv"  # Update this to your actual file path
    
    try:
        main(filepath)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check the CSV format and make sure all columns that should be numeric are in the correct format")
        print("2. Check for and handle missing values in the dataset")
        print("3. If you're still encountering issues, try running with `error_score='raise'` in cross_val_score to get more detailed error information")