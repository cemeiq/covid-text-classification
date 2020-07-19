
# coding: utf-8

# **Text classification using convolution neural networks**
import numpy as np
import pandas as pd
import random
import torch
import re
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

nltk.download('punkt')


# Text classification is one of the most common applications of machine learning in the field of natural language processing. 
# For example, text classification is used for sentiment analysis, hate-speech detection etc. In this notebook, we will be using text classification for classifying 
# a collection of articles into two classes, relevant and irrelvant to COVID-19

""" We use Python and Jupyter Notebook to develop our system, the libraries we will use include Pytorch, Gensim, Numpy, Pandas and NLTK.  """

# First, we have a look at our dataset of COVID-19 articles. As the data file is a csv file, we will read it using pandas and extract the necessary columns and create a dataframe from them.

# Importing necessary libraries

class CnnText(torch.nn.Module):
       def __init__(self,vocab_size, num_classes, window_size,w2vmodel,embedding_dim):
            super(CnnText, self).__init__()
            weights = w2vmodel.wv
            # With pretrained embeddings
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
            self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [w, EMBEDDING_SIZE], padding=(w - 1, 0))
                                   for w in window_size
        ])
            self.fc = nn.Linear(10 * len(window_size), num_classes)


            
       def forward(self,x):
            x = self.embedding(x) # [B, T, E]

            # Apply a convolution + max_pool layer for each window size
            x = torch.unsqueeze(x, 1)
            xs = []
            for conv in self.convs:
              x2 = torch.tanh(conv(x))
              x2 = torch.squeeze(x2, -1)
              x2 = F.max_pool1d(x2, x2.size(2))
              xs.append(x2)
            x = torch.cat(xs, 2)

            # FC
            x = x.view(x.size(0), -1)
            logits = self.fc(x)

            probs = F.softmax(logits, dim = 1)




            return probs

def prepare_data(dataframe_path):
            # Loading the csv file using Pandas and creating a dataframe
            df = pd.read_csv(dataframe_path)
            #df.head(3)
            # One-hot encoding of Y (labels column) and getting to know the number of abstracts per class(0,1,2)
            X = df['Title']+' .'+df['Abstract']
            y = df['DistillerSR_SystematicReview'].values

            for i in range(len(y)):
                        if y[i]== 'yes':
                                y[i] = 1
                        elif y[i] == 'no':
                                y[i] = 0
                        else:
                            y[i] = 2   

            df['relevance'] = y 
            df['unprocessed_text'] = X
            #print("Number of abstracts per class:")
            #print(df['relevance'].value_counts())

            # Getting a balanced dataset

            top_n = 100
            df_positive = df[df['relevance'] == 1].head(top_n)
            df_negative = df[df['relevance'] == 0].head(top_n)
            df_neutral = df[df['relevance'] == 2].head(top_n)
            df_updated = pd.concat([df_positive, df_negative, df_neutral])
               

            #balanced_df= (get_top_data(top_n=200))

            # One of the initial steps in text classification is data cleaning. The dataset is pre-processed before it can be fed into a supervised
            # machine learning model.

            return df_updated

def preprocess_texts(texts, lower=True, filters=r"[!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~]"):
                preprocessed_texts = []
                for text in texts:
                    # lower
                    if lower:
                        text = str(text).lower()

                    # remove items text in () ex. (Reuters)
                    # may want to refine to only remove if at end of text
                    text = re.sub(r'\([^)]*\)', '', text)

                    # spacing and filters
                    text = re.sub(r"([.,!?])", r" \1 ", text)
                    text = re.sub(filters, r"", text)
                    text = re.sub(' +', ' ', text)  # remove multiple spaces
                    text = text.strip()

                    preprocessed_texts.append(text)
                return preprocessed_texts


def split_train_test(balanced_df):                

            # Preprocesssing the text column X
            lower = True
            X = balanced_df['unprocessed_text']
            X = preprocess_texts(texts=X, lower=lower)





            balanced_df['processed_text'] = X
            tokens1 = [word_tokenize(sen) for sen in balanced_df['processed_text']]
            balanced_df['tokens'] = tokens1
            len(balanced_df['tokens'])


            # Dividing the dataset into train and test datasets



            test_size = 0.25
            shuffle = True

            data = balanced_df[['processed_text', 'tokens']]
            labels = balanced_df['relevance']
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(data ,labels , test_size=test_size, stratify=labels)

            return X_train, X_test, y_train, y_test





       
    
def train_model(X_train):
    
        EMBEDDING_SIZE = 500
        NUM_FILTERS = 10


        # The next step is training the convolution neural network model to get probabilities of each of the documents belonging to a particular class
        NUM_CLASSES = 3
        VOCAB_SIZE = len(w2vmodel.wv.vocab)

        model = CnnText(VOCAB_SIZE, NUM_CLASSES,[2],w2vmodel,EMBEDDING_SIZE)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5

                
        losses = []
        model.train()
        for epoch in range(num_epochs):
            print("Epoch" + str(epoch + 1))
            train_loss = 0
            for index, row in X_train.iterrows():
                # Clearing the accumulated gradients
                model.zero_grad()

                # Make the bag of words vector for stemmed tokens 
                bow_vec = make_word2vec_vector_cnn(row['tokens'])
            
                # Forward pass to get output
                probs = model(bow_vec)

                # Get the target label
                target = make_target(y_train[index])

                # Calculate Loss: softmax --> cross entropy loss
                loss = loss_function(probs, target)
                train_loss += loss.item()

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()
        return loss, probs      



def test_model(probs):
        EMBEDDING_SIZE = 500
        NUM_FILTERS = 10
        # The next step invovles testing the model and generating a classification report that will help us show how well the model performed in classifying the articles



        bow_cnn_predictions = []
        original_lables_cnn_bow = []
        model.eval()
        target_names = ['0', '1', '2']

        with torch.no_grad():
            for index, row in X_test.iterrows():
                bow_vec = make_word2vec_vector_cnn(row['tokens'])
                probs = model(bow_vec)
                print(probs)
                _, predicted = torch.max(probs.data, 1)
                bow_cnn_predictions.append(predicted.cpu().numpy()[0])
                original_lables_cnn_bow.append(make_target(y_test[index]).cpu().numpy()[0])
    
        return original_lables_cnn_bow,bow_cnn_predictions

  

def classification_results(original_lables_cnn_bow,bow_cnn_predictions):
    print(classification_report(original_lables_cnn_bow,bow_cnn_predictions))           