import pandas as pd
def extract_untreated_effect(n, X_train, model):
    
    # Set features to mean value
    X_mean_mapping = {'X1': [X_train[:, 0].mean()] * n,
                      'X2': [X_train[:, 1].mean()] * n,
                      'X3': [X_train[:, 2].mean()] * n,
                      'X4': [X_train[:, 3].mean()] * n,
                      'T': [0] * n}

    # Create DataFrame
    df_scoring = pd.DataFrame(X_mean_mapping)

    # Add full range of treatment values
    df_scoring

    # Calculate outcome prediction for treated
    untreated = model.predict(df_scoring)
    
    return untreated