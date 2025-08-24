# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load the data
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# 2. Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 3. Train and Evaluate SVM with a Linear Kernel
print("--- SVM with Linear Kernel ---")
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}\n")


# 4. Train and Evaluate SVM with an RBF Kernel
print("--- SVM with RBF Kernel ---")
rbf_svm = SVC(kernel='rbf', random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")
                                