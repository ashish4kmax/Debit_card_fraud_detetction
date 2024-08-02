import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load data
data = pd.read_csv('data/credit_card_transactions.csv')

# Print column names for debugging
print("Column names:", data.columns)

# Strip any leading or trailing spaces from column names
data.columns = data.columns.str.strip()

# Define feature and target columns
features = ['transaction_amount', 'transaction_time']
target = 'is_fraud'  # Updated to match your dataset

# Ensure the target column exists
if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in the dataset.")

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
knn = KNeighborsClassifier()
nb = GaussianNB()
kmeans = KMeans(n_clusters=2, random_state=42)
gmm = GaussianMixture(n_components=2, random_state=42)

# Train models
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
kmeans.fit(X_train)
gmm.fit(X_train)

# Save models
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(nb, 'models/nb_model.pkl')
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(gmm, 'models/gmm_model.pkl')

# Predict and calculate metrics
knn_preds = knn.predict(X_test)
nb_preds = nb.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_preds)
knn_precision = precision_score(y_test, knn_preds)
knn_recall = recall_score(y_test, knn_preds)
knn_f1 = f1_score(y_test, knn_preds)

nb_accuracy = accuracy_score(y_test, nb_preds)
nb_precision = precision_score(y_test, nb_preds)
nb_recall = recall_score(y_test, nb_preds)
nb_f1 = f1_score(y_test, nb_preds)

# Save metrics
metrics_df = pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes'],
    'Accuracy': [knn_accuracy, nb_accuracy],
    'Precision': [knn_precision, nb_precision],
    'Recall': [knn_recall, nb_recall],
    'F1 Score': [knn_f1, nb_f1]
})

metrics_df.to_csv('data/model_metrics.csv', index=False)
print("Metrics saved to data/model_metrics.csv")
