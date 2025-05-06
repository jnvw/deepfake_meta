# MetaOptNet implementation with SVM classifier
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class MetaOptNetSVM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, support_features, support_labels, query_features):
        # Detach from autograd graph and move to CPU before converting to NumPy
        X_train = support_features.detach().cpu().numpy()
        y_train = support_labels.detach().cpu().numpy()
        X_test = query_features.detach().cpu().numpy()

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM classifier
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)

        # Predict probabilities for query set
        preds = clf.predict_proba(X_test)

        # Convert to torch tensor
        return torch.tensor(preds, dtype=torch.float)


