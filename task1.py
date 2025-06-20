import pandas as pd
from sklearn.preprocessing import LabelEncoder
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)
print("Initial data shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, drop_first=True)
print("\nFinal data shape after encoding:", df.shape)
print("First 5 rows of processed data:")
print(df.head())
