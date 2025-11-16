# data_gen.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_synthetic_fraud(n_samples=20000, n_features=10, fraud_ratio=0.02, random_state=42):
# Use make_classification to create an imbalanced binary classification dataset
n_informative = max(2, int(n_features * 0.4))
X, y = make_classification(
n_samples=n_samples,
n_features=n_features,
n_informative=n_informative,
n_redundant=1,
n_repeated=0,
n_classes=2,
weights=[1 - fraud_ratio, fraud_ratio],
flip_y=0.01,
class_sep=1.0,
random_state=random_state,
)


df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df['is_fraud'] = y


# Add a few synthetic categorical columns to show feature engineering
rng = np.random.RandomState(random_state)
df['merchant_id'] = rng.randint(0, 50, size=n_samples).astype(str)
df['device_type'] = rng.choice(['mobile', 'desktop', 'tablet'], size=n_samples, p=[0.6, 0.3, 0.1])


# Create a transaction_amount feature with a heavy tail
df['transaction_amount'] = np.exp(df['feat_0'] + rng.normal(0, 1, size=n_samples)) * 10


# A simple timestamp-like column (days since epoch)
df['day'] = rng.randint(0, 365, size=n_samples)


return df


if __name__ == '__main__':
df = generate_synthetic_fraud()
df.to_csv('synthetic_fraud.csv', index=False)
print('Saved synthetic_fraud.csv — shape:', df.shape)