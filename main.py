import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation import generate_synthetic_data
from src.train import run_pipeline

def main():
    print("="*60)
    print("FinGig-CreditNet: Credit Scoring in the Gig Economy")
    print("="*60)
    
    zomato_path = 'Datasets/Zomato Dataset.csv'
    hc_path = 'Datasets/application_train.csv'
    synthetic_out_path = 'data/synthetic_gig_data.csv'
    
    if not os.path.exists(synthetic_out_path):
        print("\nStep 1: Generating Synthetic Multi-Platform Gig Dataset...")
        generate_synthetic_data(zomato_path, hc_path, synthetic_out_path, n_samples=20000)
    else:
        print(f"\nStep 1: Synthetic dataset already exists at {synthetic_out_path}")
        
    print("\nStep 2: Training Models and Evaluating Performance...")
    run_pipeline(synthetic_out_path)
    
    print("\nPipeline completed successfully.")

if __name__ == '__main__':
    main()
