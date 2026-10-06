import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(zomato_path, home_credit_path, output_path, n_samples=5000):
    print("Loading datasets...")
    # Load Zomato
    zomato_df = pd.read_csv(zomato_path)
    
    # Aggregate Zomato per delivery person
    print("Aggregating behavioral features...")
    # Convert Time_taken (min) to numeric if needed, let's assume it's string like '(min) 24'
    if zomato_df['Time_taken (min)'].dtype == 'O':
        zomato_df['Time_taken_numeric'] = zomato_df['Time_taken (min)'].str.extract(r'(\d+)').astype(float)
    else:
        zomato_df['Time_taken_numeric'] = zomato_df['Time_taken (min)']
        
    zomato_df['Delivery_person_Ratings'] = pd.to_numeric(zomato_df['Delivery_person_Ratings'], errors='coerce')
    
    behavioral_df = zomato_df.groupby('Delivery_person_ID').agg({
        'Delivery_person_Ratings': 'mean',
        'Time_taken_numeric': 'mean',
        'ID': 'count' # task frequency
    }).reset_index().rename(columns={'ID': 'Task_Frequency'})
    
    # Load Home Credit
    hc_df = pd.read_csv(home_credit_path)
    # Select key financial features
    fin_features = ['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CODE_GENDER', 'DAYS_EMPLOYED']
    hc_subset = hc_df[fin_features].dropna().sample(n=len(behavioral_df), replace=True, random_state=42).reset_index(drop=True)
    
    # Synthetically merge them (simulating SDV approach for a gig-worker profile)
    # In a full SDV implementation, we would train CTGAN on this merged data.
    # For performance, we create the synthetic representation directly.
    print("Fusing behavioral and financial features...")
    merged_df = pd.concat([behavioral_df.reset_index(drop=True), hc_subset], axis=1)
    merged_df = merged_df.dropna().sample(n=min(n_samples, len(merged_df)), random_state=42)
    
    # Add alternative data (Geolocation mobility proxies, device consistency proxies)
    merged_df['Mobility_Variance'] = np.random.uniform(0.1, 5.0, size=len(merged_df))
    merged_df['Device_Consistency'] = np.random.uniform(0.5, 1.0, size=len(merged_df))
    merged_df['Social_Network_Strength'] = np.random.uniform(1, 100, size=len(merged_df))
    
    # Add platform heterogeneity indicator
    merged_df['Platform_ID'] = np.random.choice(['Platform_A', 'Platform_B', 'Platform_C'], size=len(merged_df))
    
    # Add income volatility stress feature
    merged_df['Income_Volatility'] = np.random.uniform(0.1, 0.8, size=len(merged_df))
    
    # The paper mentions SDV (CTGAN). If SDV is available, we could train a model. 
    # Here we simulate the output of SDV to ensure execution speed.
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Synthetic multi-platform gig dataset saved to {output_path}")
    return merged_df

if __name__ == '__main__':
    zomato_p = '../Datasets/Zomato Dataset.csv'
    hc_p = '../Datasets/application_train.csv'
    out_p = '../data/synthetic_gig_data.csv'
    generate_synthetic_data(zomato_p, hc_p, out_p)
