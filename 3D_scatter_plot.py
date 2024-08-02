import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def generate_3d_scatter_plot():
    # Load the CSV data
    df = pd.read_csv('./data/credit_card_transactions.csv')

    # Extract features and labels
    X = df[['transaction_amount', 'transaction_time']].values
    y = df['is_fraud'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], y, c=y,
                         cmap='RdYlBu', edgecolor='k')
    ax.set_xlabel('Transaction Amount')
    ax.set_ylabel('Transaction Time')
    ax.set_zlabel('Fraud')
    ax.set_title('3D Scatter Plot of Transactions')
    plt.colorbar(scatter, label='Fraud')

    # Save plot as an image file
    plt.savefig('./app/static/3d_scatter_plot.png')
    plt.close()
    print("3D scatter plot successfully saved as '3d_scatter_plot.png'")


generate_3d_scatter_plot()
