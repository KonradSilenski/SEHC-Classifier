import re

import pandas as pd

contractions_dict = {
    "ain&#8217;t": "am not",
    "aren&#8217;t": "are not",
    "can&#8217;t": "cannot",
    "couldn&#8217;t": "could not",
    "didn&#8217;t": "did not",
    "doesn&#8217;t": "does not",
    "don&#8217;t": "do not",
    "hadn&#8217;t": "had not",
    "hasn&#8217;t": "has not",
    "haven&#8217;t": "have not",
    "he&#8217;s": "he is",
    "he&#8217;d": "he would",
    "he&#8217;ll": "he will",
    "how&#8217;s": "how is",
    "i&#8217;d": "i would",
    "i&#8217;ll": "i will",
    "i&#8217;m": "i am",
    "i&#8217;ve": "i have",
    "isn&#8217;t": "is not",
    "it&#8217;s": "it is",
    "let&#8217;s": "let us",
    "mightn&#8217;t": "might not",
    "mustn&#8217;t": "must not",
    "she&#8217;s": "she is",
    "she&#8217;d": "she would",
    "she&#8217;ll": "she will",
    "shouldn&#8217;t": "should not",
    "that&#8217;s": "that is",
    "there&#8217;s": "there is",
    "they&#8217;d": "they would",
    "they&#8217;ll": "they will",
    "they&#8217;re": "they are",
    "they&#8217;ve": "they have",
    "we&#8217;d": "we would",
    "we&#8217;re": "we are",
    "we&#8217;ve": "we have",
    "weren&#8217;t": "were not",
    "what&#8217;s": "what is",
    "where&#8217;s": "where is",
    "who&#8217;s": "who is",
    "won&#8217;t": "will not",
    "wouldn&#8217;t": "would not",
    "you&#8217;d": "you would",
    "you&#8217;ll": "you will",
    "you&#8217;re": "you are",
    "you&#8217;ve": "you have",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "he'd": "he would",
    "he'll": "he will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "she's": "she is",
    "she'd": "she would",
    "she'll": "she will",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


def expand_contractions(text):
    contractions_pattern = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_pattern.sub(replace, text)


def clean_text(text):
    text = text.lower()

    text = expand_contractions(text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'"""', '', text)

    text = re.sub(r'\d{3,}', '', text)

    return text


input_file = '../Datasets/recombined_data.csv'
df = pd.read_csv(input_file)

df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: clean_text(str(x)))

#columns_to_remove = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]

#columns_to_remove_names = df.columns[columns_to_remove]

#df = df.drop(columns=columns_to_remove_names, axis=1)

output_file = '../Datasets/recombined_data_cleaned.csv'
df.to_csv(output_file, index=False)

print(f"Done and saved to {output_file}")
