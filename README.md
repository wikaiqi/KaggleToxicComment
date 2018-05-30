# Kaggle Toxic Comment classification challenge
## Task
There are 6 type of toxic comment: toxic, severve_toxic, obscene, threat, insult, identity_hate. The task is to classifiy the type of a comment.

## Data Cleaning
The data includes foreign language. I used the google translate API to transform all foreign languages into English. 

## Word Embedding
I used pretrained embeddings: Glove and FastText.

## Models
Two stacks of GRUs and LSTMs were used.

