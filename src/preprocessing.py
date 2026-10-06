import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Target
    y = df['TARGET'].values
    
    # Features
    # Behavioral
    behavioral_cols = ['Delivery_person_Ratings', 'Time_taken_numeric', 'Task_Frequency']
    # Transactional
    transactional_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED']
    # Alternative
    alternative_cols = ['Mobility_Variance', 'Device_Consistency', 'Social_Network_Strength', 'Income_Volatility']
    
    categorical_cols = ['CODE_GENDER', 'Platform_ID']
    
    numeric_cols = behavioral_cols + transactional_cols + alternative_cols
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X = df[numeric_cols + categorical_cols]
    
    # 80-10-10 split as per paper
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42) # ~10% of total
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Also return test DataFrame for fairness evaluation
    df_test = X_test.copy()
    df_test['TARGET'] = y_test
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor, df_test
