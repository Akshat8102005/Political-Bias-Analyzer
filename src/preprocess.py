import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/processed_dataset.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("../data/processed/train.csv", index=False)
test_df.to_csv("../data/processed/test.csv", index=False)

print("Train:", train_df.shape)
print("Test:", test_df.shape)