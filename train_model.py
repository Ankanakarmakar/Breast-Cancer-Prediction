
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_scaled, y)

# Save the scaler and model to joblib files
import os
joblib.dump(scaler, r'C:\Users\ankan\OneDrive\Documents\Breast_cancer\PracticeSessionForJIS\scaler.joblib')
joblib.dump(model, r'C:\Users\ankan\OneDrive\Documents\Breast_cancer\PracticeSessionForJIS\model.joblib')
                                        