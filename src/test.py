import os, sys
from covid.src.scripts.modeling import cnn
from covid.src.scripts.modeling.cnn import CnnText


# The first step is getting the dataset from the data folder
data_path  = '/Users/iqra/opt/anaconda3/lib/python3.7/covid-text-classification/data/raw/distillersr-COVID-19_June_2_2020_2020-06-18-13-59-30_All Refs.csv'

# The second step is pre-processing and preparing the dataset by getting a balanced dataset
balanced_df  = cnn.prepare_data(data_path)

# The third step is dividing the dataset into train and test splits
X_train, X_test, y_train, y_test, data = cnn.split_train_test(balanced_df)


# This step involves creating a word2vec model that represents our word embeddings
w2v_model = cnn.make_word2vec_model(balanced_df)


# This step involves training the CNN model
train_loss, train_predicted_probs, cnn_model  = cnn.train_model(X_train,y_train,w2v_model,data)

# This step involves testing the CNN model
original_label , test_predicted_probs = cnn.test_model(cnn_model,X_test,w2v_model,y_test,data)

# This step involves obtaining the classification report
cnn.classification_results(original_label , test_predicted_probs)
