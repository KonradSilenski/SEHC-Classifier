from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


input_file = '../Datasets/recombined_data_cleaned_eng.csv'
df = pd.read_csv(input_file)

X = df.iloc[:, 0]
y = df.iloc[:, 1]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)

sentence_lengths = [len(seq) for seq in sequences]

plt.hist(sentence_lengths, bins=10, color='blue', alpha=0.7)
plt.title('Sentence Length Distribution')
plt.xlabel('Sentence Length (number of tokens)')
plt.ylabel('Frequency')
plt.show()

max_len_90 = np.percentile(sentence_lengths, 90)
max_len_95 = np.percentile(sentence_lengths, 95)

print(f"90th percentile max_len: {int(max_len_90)}")
print(f"95th percentile max_len: {int(max_len_95)}")

