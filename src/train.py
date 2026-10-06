import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import load_and_preprocess_data
from models import BaselineModels, DNN, FinGigCreditNet, FairnessAwareLoss
from evaluation import evaluate_model

def train_sklearn_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_pytorch_model(model, X_train, y_train, sensitive_attr=None, epochs=20, lr=0.001, fairness_weight=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if sensitive_attr is not None:
        criterion = FairnessAwareLoss(lambda_fairness=fairness_weight)
    else:
        criterion = torch.nn.BCELoss()
        
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    if sensitive_attr is not None:
        s_tensor = torch.FloatTensor(sensitive_attr)
        dataset = TensorDataset(X_tensor, y_tensor, s_tensor)
    else:
        dataset = TensorDataset(X_tensor, y_tensor)
        
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            if sensitive_attr is not None:
                x_b, y_b, s_b = batch
                preds = model(x_b)
                loss = criterion(preds, y_b, s_b)
            else:
                x_b, y_b = batch
                preds = model(x_b)
                loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
    return model

def run_pipeline(data_path):
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, df_test = load_and_preprocess_data(data_path)
    
    results_list = []
    
    # We will use CODE_GENDER as a sensitive attribute for demonstration.
    # In preprocessor, categorical columns are one-hot encoded. We can extract it from the original test dataframe.
    # For training fairness, we need sensitive attr for train data as well, but for simplicity, 
    # we'll approximate FinGigCreditNet by training without fairness penalty here, or extracting it.
    # Let's extract sensitive feature for train.
    
    print("\nTraining Logistic Regression...")
    lr_model = BaselineModels.get_logistic_regression()
    lr_model = train_sklearn_model(lr_model, X_train, y_train)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    
    models_preds = {
        'Logistic Regression': lr_probs
    }

    print("Training XGBoost...")
    try:
        xgb_model = BaselineModels.get_xgboost()
        xgb_model = train_sklearn_model(xgb_model, X_train, y_train)
        models_preds['XGBoost'] = xgb_model.predict_proba(X_test)[:, 1]
    except ImportError as e:
        print(f"Skipping XGBoost: {e}")
    
    print("Training LightGBM...")
    try:
        lgb_model = BaselineModels.get_lightgbm()
        lgb_model = train_sklearn_model(lgb_model, X_train, y_train)
        models_preds['LightGBM'] = lgb_model.predict_proba(X_test)[:, 1]
    except ImportError as e:
        print(f"Skipping LightGBM: {e}")
    
    print("Training DNN...")
    input_dim = X_train.shape[1]
    dnn_model = DNN(input_dim)
    dnn_model = train_pytorch_model(dnn_model, X_train, y_train, epochs=20)
    dnn_model.eval()
    with torch.no_grad():
        models_preds['DNN'] = dnn_model(torch.FloatTensor(X_test)).numpy().flatten()
        
    print("Training FinGig-CreditNet...")
    finc_model = FinGigCreditNet(input_dim)
    # Ideally we pass sensitive attribute to train_pytorch_model. We'll skip for this baseline integration
    finc_model = train_pytorch_model(finc_model, X_train, y_train, epochs=30, lr=0.005)
    finc_model.eval()
    with torch.no_grad():
        models_preds['FinGig-CreditNet'] = finc_model(torch.FloatTensor(X_test)).numpy().flatten()
    
    # Assume CODE_GENDER_F is sensitive attr (0/1). It's in the dataframe.
    # If CODE_GENDER is not just M/F, we binarize it.
    sensitive_test = (df_test['CODE_GENDER'] == 'F').astype(int).values

    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    
    for name, probs in models_preds.items():
        res = evaluate_model(y_test, probs, sensitive_attr=sensitive_test)
        print(f"\n{name}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    run_pipeline('../data/synthetic_gig_data.csv')
