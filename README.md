# Emoji

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the emojifier can automatically turn this into "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"

Two models are implemented. They input a sentence (such as "Let's go see the baseball game tonight!") and find the most appropriate emoji to be used with this sentence (⚾️).

## Global Vectors

For better results on little training sets pretrained word embeddings should be used. Word embeddings could be dowloaded for example here: https://www.kaggle.com/devjyotichandra/glove6b50dtxt/downloads/glove.6B.50d.txt

## Model_1

The first model is a simple softmax regression. Each string is represented as vector average of all words from this string.

### Model_2

The second model is two-layers LSTM with Dropout and Softmax output.
