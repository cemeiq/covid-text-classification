# Text classification of covid-19 articles

_This prject uses convolution neural networks to classify covid-19 articles._


#### [Project homepage](http://drivendata.github.io/cookiecutter-data-science/)

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


### Requirements to use the covid-19 text classification :
-----------
 - Python 2.7 or 3.5
 - Numpy
 - Pandas
 - Torch
 - nltk
 - genism
 - sk-learn
 


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
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── notebooks      <- Jupyter notebooks
│   │
│   ├── modelling         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── cnn.py        <- Script to train  the cnn model and then use trained cnn model to make
│   │   │                 predictions
│   │-- test.py        <- contains the steps to import the dataset, create the cnn model, train and test the model
│

```
 
