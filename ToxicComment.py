# coding: utf-8
# # LSTM GloVe Version 1.0
import pandas   as pd
import numpy    as np
import argparse

#keras
from keras                   import optimizers
from keras.callbacks         import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split

#my own modules
from pylib.CleanTextData  import CleanDataText
from pylib.ReSampling     import Random_DownSampling
from pylib.KerasAUC       import RocAucEvaluation
from pylib.WordToken      import Text2Token
from pylib.GolveWord2Vec  import WordEmbedding
from pylib.NNmodels       import GRU_model

np.random.seed(1)

parser = argparse.ArgumentParser(description='LSTM model')
parser.add_argument('-g', '--glove', type=int, default=6, help='glove name')
arg = parser.parse_args()

max_features    = 180000   # how many unique words to use (i.e num rows in embedding vector)
maxlen          = 900      # max number of words in a comment to use
nEpochs         = 10
embed_nv        = arg.glove
if embed_nv==27:
    embed_size      = 200  # how big is each word vector
else:
    embed_size      = 300

EMBEDDING_FILE  = 'glove/glove.'+str(embed_nv)+'B.'+str(embed_size)+'d.txt'

train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')

# text cleaning
train = CleanDataText(train, "comment_text")
test  = CleanDataText(test,  "comment_text")

#save data
train.to_csv('train_after.csv', index=False)
test.to_csv('test_after.csv', index=False)

print(train.head())

cols_target = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train     = train[cols_target].values

print("--------------------------")
print("train size:", train.shape)
print("test size:", test.shape)

# Tokenizer
X_train, X_test, word_index, nb_words = Text2Token(train, test, max_features, maxlen)
print("--------------------------")
print("X_train size: ",X_train.shape)
print("X_test size : ",X_test.shape)
print("nb of words : ",nb_words)
print("--------------------------")

# Word embedding
embedding_matrix, embeddings_index = WordEmbedding(EMBEDDING_FILE,
                                                   nb_words, word_index,
                                                   embed_size, max_features)

max_features = nb_words

# model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=0)
model = GRU_model(maxlen, nb_words, embed_size, embedding_matrix)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#split train and dev sets
X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train, test_size=0.25)

wts_file = 'mdl_wts_'+str(embed_nv)+'B.hdf5'
mcp_save = ModelCheckpoint(wts_file, save_best_only=True, monitor='val_loss', mode='min')

ra_val = RocAucEvaluation(training_data =(X_train_cv, y_train_cv), validation_data=(X_val_cv, y_val_cv), interval=1)
model.fit(X_train_cv, y_train_cv, batch_size=64, verbose=0, epochs=nEpochs, shuffle=True, validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping,ra_val, mcp_save])
    
print("model training is done. ")
    
#predict test set
y_test_pred = model.predict([X_test], batch_size=1024, verbose=0)
sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission[cols_target] = y_test_pred
filename='sub_BidLSTM_'+str(embed_nv)+'B'+str(embed_size)+'d.csv'
sample_submission.to_csv(filename, index=False)




