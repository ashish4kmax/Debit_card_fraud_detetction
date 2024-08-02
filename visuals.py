import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def generate_knn_plot():
    try:
        # Load the CSV data
        df = pd.read_csv('./data/credit_card_transactions.csv')  # Adjust path if necessary

        # Check if necessary columns exist
        required_columns = ['transaction_amount', 'transaction_time', 'is_fraud']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Extract features and labels
        X = df[['transaction_amount', 'transaction_time']].values
        y = df['is_fraud'].values

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)

        # Define the range for the plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Reduce the resolution
        step_size = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

        # Flatten the mesh grid and make predictions
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.RdYlBu)
        plt.xlabel('Transaction Amount')
        plt.ylabel('Transaction Time')
        plt.title('KNN Decision Boundary')
        plt.colorbar(label='Prediction')

        # Save plot as an image file
        plt.savefig('./app/static/knn_plot.png')
        print("Plot successfully saved as 'knn_plot.png'")

    except Exception as e:
        print(f"An error occurred: {e}")

generate_knn_plot()
