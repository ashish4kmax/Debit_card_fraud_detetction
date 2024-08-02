import os
import pandas as pd
import numpy as np

# Create the data directory if it does not exist
if not os.path.exists('./data'):
    os.makedirs('./data')

# Parameters
num_samples = 1000

# Generate synthetic data
np.random.seed(0)
transaction_id = np.arange(1, num_samples + 1)  # Sequential IDs
transaction_amount = np.random.uniform(10, 1000, num_samples)
transaction_time = np.random.uniform(0, 24, num_samples)  # in hours
# 0 for non-fraudulent, 1 for fraudulent
is_fraud = np.random.choice([0, 1], num_samples)

# Generate synthetic names and places
names = np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eve'], num_samples)
places = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], num_samples)

# Create DataFrame
df = pd.DataFrame({
    'transaction_id': transaction_id,
    'name': names,
    'place': places,
    'transaction_amount': transaction_amount,
    'transaction_time': transaction_time,
    'is_fraud': is_fraud
})

# Save to CSV
df.to_csv('./data/credit_card_transactions.csv', index=False)

print("Data generated and saved to './data/credit_card_transactions.csv'")
