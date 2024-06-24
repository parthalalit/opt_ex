import pandas as pd

def extract_treated_effect(n, X_train, model):
    
    # Set features to mean value
    X_mean_mapping = {'X1': [X_train[:, 0].mean()] * n,
                      'X2': [X_train[:, 1].mean()] * n,
                      'X3': [X_train[:, 2].mean()] * n,
                      'X4': [X_train[:, 3].mean()] * n}

    # Create DataFrame
    df_scoring = pd.DataFrame(X_mean_mapping)

    # Add full range of treatment values
    df_scoring['T'] = X_train[:, 4].reshape(-1, 1)

    # Calculate outcome prediction for treated
    treated = model.predict(df_scoring)
    
    return treated, df_scoring