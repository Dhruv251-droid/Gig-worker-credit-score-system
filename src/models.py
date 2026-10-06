import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class BaselineModels:
    @staticmethod
    def get_logistic_regression():
        return LogisticRegression(max_iter=1000, random_state=42)
        
    @staticmethod
    def get_xgboost():
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        return XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        
    @staticmethod
    def get_lightgbm():
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        return LGBMClassifier(n_estimators=100, random_state=42)

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class FinGigCreditNet(nn.Module):
    def __init__(self, input_dim):
        super(FinGigCreditNet, self).__init__()
        # Representation encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        fused = self.fusion(encoded)
        output = self.predictor(fused)
        return output

# Loss function with fairness penalty
class FairnessAwareLoss(nn.Module):
    def __init__(self, lambda_fairness=0.1):
        super(FairnessAwareLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.lambda_fairness = lambda_fairness
        
    def forward(self, predictions, targets, sensitive_features):
        # Base predictive loss
        base_loss = self.bce(predictions, targets)
        
        # Simple demographic parity penalty approximation
        # sensitive_features is expected to be a binary tensor (e.g., gender)
        group_0_preds = predictions[sensitive_features == 0]
        group_1_preds = predictions[sensitive_features == 1]
        
        if len(group_0_preds) > 0 and len(group_1_preds) > 0:
            dp_penalty = torch.abs(torch.mean(group_0_preds) - torch.mean(group_1_preds))
        else:
            dp_penalty = 0.0
            
        total_loss = base_loss + self.lambda_fairness * dp_penalty
        return total_loss
