# Text classification of covid-19 articles

_This prject uses convolution neural networks to classify covid-19 articles._


#### [Blog Post associated with this project](https://iqra.hashnode.dev/text-classification-of-covid-19-articles-using-convolution-neural-networks-ckcgax7ph0091jhs1aub58ccy)



### Requirements to use the covid-19 text classification :
-----------
 - Python 2.7 or 3.5
 - Numpy
 - Pandas
 - Torch
 - nltk
 - genism
 - sk-learn
 
### The covid dataset is too large to be uploaded on Github so it can be downloaded from the following link: https://www.evidencepartners.com/resources/covid-19-resources/
### place the downloaded dataset in the raw folder in data module and change the path of the dataset in test.py in src folder
### The main script to run the model and test the model on the covid articles dataset, is in the src folder in test.py 
``` bash
$ pip install nameoflibrary
```



### The resulting directory structure
------------

The directory structure of this project looks like this: 

```

├── main.py           <-  A mainpython scipt containing functions relevant to fetching the path of a repository
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├
│
├-- dist               <- Contains files related to the local installed covid package
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── src                <- Source code for use in this project.
│   │
│   │
│   ├── notebooks      <- Jupyter notebooks
│   │
│   ├── modelling         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
|   |   |__ scripts
|   |   |   ├── cnn.py        <- Script to train  the cnn model and then use trained cnn model to make
│   │   │                 predictions
│   │-- test.py        <- contains the steps to import the dataset, create the cnn model, train and test the model
│

```
 
