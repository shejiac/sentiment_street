# General DS libs
import numpy as np
import pandas as pd

# Data cleaning utils
from data_cleaning_utils import clean_text, is_junk_comment

# transfomers & classification libs
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
