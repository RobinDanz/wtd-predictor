import pandas as pd


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset
    """
    # Load the CSV file with semicolon delimiter and handle European number format
    df = pd.read_csv(filepath, sep=';', decimal=',')

    df = df[(df['WTD'] >= -10) & (df['WTD'] <= 70) & (df['pH'] <= 5.5)]
    
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
        if col.strip() in ['REFERENCE', 'SAMPLE CODE', 'SITE NAME', 'pH', 'LAT', 'LONG', 'WTD', 'PERSON', 'COUNTRY', 'SITE', 'SAMPLE'] or 'Unnamed' in col:
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
                X[col].replace()
                # print(X.col)
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
    
    # X[X > 0] = 1
    print(f"\nNumber of features (species): {len(feature_cols)}")
    print(f"Target variable (WTD) range: {y.min()} to {y.max()}")
    return X, y, feature_cols
