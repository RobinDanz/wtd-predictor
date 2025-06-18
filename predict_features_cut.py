import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
from process import load_and_preprocess_data
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test set RMSE: {rmse:.4f}")
    print(f"Test set R² score: {r2:.4f}")
    
    return model, rmse, r2, cv_rmse

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
    # print("\nTop 3 most important species for WTD prediction:")
    # print(importance_df.head(3))

    # print("\nLeast important feature:")
    # print(importance_df.tail(1))

    return importance_df

def features_cut(X, importance_df):
    to_drop = importance_df[importance_df.Importance == 0].Feature.values
    X = X.drop(to_drop, axis=1)

    least_important_feature = importance_df[importance_df.Importance > 0].tail(1).Feature.values[0]

    X = X.drop(least_important_feature, axis=1)
    return X

def main(filepath):
    """
    Main function to run the analysis
    """
    print("Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data(filepath)

    cv_rmses = []
    rmse_scores = []
    r2_scores = []
    feature_lists = []

    by_nb_species = {}

    min_features = 5
    iteration = 1
    range = []
    while X.shape[1] > min_features:
        range.append(X.shape[1])
        print("==============================")
        print(f"\nNumber of features: {X.shape[1]}")
        print("\nTraining XGBoost model...")
        model, rmse, r2, cv_rmse = train_xgboost_model(X, y)

        cv_rmses.append(cv_rmse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        feature_lists.append(list(X.columns))
        

        print("\nAnalyzing feature importance...")
        importance_df = analyze_feature_importance(model, list(X))

        by_nb_species[len(X.columns)] = (rmse, r2, importance_df)

        X = features_cut(X, importance_df)

        iteration += 1


    fix, (ax1, ax2) = plt.subplots(2)
    
    min_rmse_x = range[np.argmin(rmse_scores)]
    min_rmse_y = np.min(rmse_scores)

    rmse_10, r2_10, best_10 = by_nb_species[10]
    rmse_15, r2_15, best_15 = by_nb_species[15]
    rmse_20, r2_20, best_20 = by_nb_species[20]
    _, _, best = by_nb_species[min_rmse_x]

    print(best_10)
    print('-----')
    print(best_15)
    print('-----')
    print(best_20)
    print('-----')
    print(best)

    ax1.plot(range, rmse_scores, marker='.', c='k')
    ax1.set_ylabel('RMSE scores')
    ax1.scatter(min_rmse_x, min_rmse_y, c='r', label=f'Best, {min_rmse_x}: {min_rmse_y:.4f}')

    ax1.scatter(10, rmse_10, c='g', label=f'For 10 species: {rmse_10:.4f}')
    ax1.scatter(15, rmse_15, c='c', label=f'For 15 species: {rmse_15:.4f}')
    ax1.scatter(20, rmse_20, c='y', label=f'For 20 species: {rmse_20:.4f}')

    # ax1.plot([min_rmse_x, min_rmse_x], [min_rmse_y, min_rmse_y], 'r--')
    # ax1.plot([0, len(rng)], [min_rmse_y, min_rmse_y], 'r--')
    # ax1.plot([0, len(rng)], [min_rmse_y, min_rmse_y], 'g--')

    min_r2_x = range[np.argmax(r2_scores)]
    min_r2_y = np.max(r2_scores)

    ax2.plot(range, r2_scores, marker='.', c='k')
    ax2.set_xlabel('Number of features')
    ax2.set_ylabel('R2 scores')
    ax2.scatter(min_r2_x, min_r2_y, c='r', label=f'Best, {min_r2_x}: {min_r2_y:.4f}')

    ax2.scatter(10, r2_10, c='g', label=f'For 10 species: {r2_10:.4f}')
    ax2.scatter(15, r2_15, c='c', label=f'For 15 species: {r2_15:.4f}')
    ax2.scatter(20, r2_20, c='y', label=f'For 20 species: {r2_20:.4f}')

    print("\nFeatures for best score: ")
    print(feature_lists[np.argmin(rmse_scores)])
    print(f"\nNumber of features for best score: {len(feature_lists[np.argmin(rmse_scores)])}")
    print(f"Best CrossVal RMSE: {cv_rmses[np.argmin(rmse_scores)].mean():.4f} ± {cv_rmses[np.argmin(rmse_scores)].std():.4f}")
    print(f"Best RMSE: {np.min(rmse_scores)}")
    print(f"Best R2: {np.max(r2_scores)}")

    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')

    print(best_10)
    print('-----')
    print(best_15)
    print('-----')
    print(best_20)
    print('-----')
    print(best)

    plt.show()

if __name__ == "__main__":
    # Use the file path provided in the error message
    filepath = "data_euro.csv"  # Update this to your actual file path
    
    try:
        main(filepath)
    except Exception as e:
        # print(e)
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check the CSV format and make sure all columns that should be numeric are in the correct format")
        print("2. Check for and handle missing values in the dataset")
        print("3. If you're still encountering issues, try running with `error_score='raise'` in cross_val_score to get more detailed error information")