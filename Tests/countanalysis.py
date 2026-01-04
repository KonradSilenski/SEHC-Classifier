import pandas as pd
from langdetect import detect, DetectorFactory

df = pd.read_csv('../Datasets/recombined_data_cleaned_eng.csv', header=None)

label_counts = df[1].value_counts()

DetectorFactory.seed = 0

label_1 = 'Not Hate'
label_2 = 'Offensive or Hate Speech'

count_label_1 = df[df[1] == label_1].shape[0]
count_label_2 = df[df[1] == label_2].shape[0]

min_count = min(count_label_1, count_label_2)

if count_label_1 > min_count:
    df_label_1 = df[df[1] == label_1].sample(min_count)
else:
    df_label_1 = df[df[1] == label_1]

if count_label_2 > min_count:
    df_label_2 = df[df[1] == label_2].sample(min_count)
else:
    df_label_2 = df[df[1] == label_2]

df_balanced = pd.concat([df_label_1, df_label_2])

other_labels = df[~df[1].isin([label_1, label_2])]
df_balanced = pd.concat([df_balanced, other_labels])

df_balanced = df_balanced[df_balanced[1] != 'Neutral or Ambiguous']

df_balanced = df_balanced[df_balanced[1] != '1']

print("Unique labels and their counts:")
print(label_counts)

print("\nLabels and counts after deletion:")
print(df_balanced[1].value_counts())


df_balanced.to_csv('../Datasets/recombined_data_cleaned.csv', index=False, header=False)
