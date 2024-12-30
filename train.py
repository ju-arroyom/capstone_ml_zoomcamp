import os
import pickle
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

def read_dataset(remove_feature=True):
   """
   Read Kaggle dataset

   Returns:
       Dataframe: Pandas df of corresponding id
   """
   df =  pd.read_csv("./data/GRAPE_QUALITY.csv")
   # To make this problem more interesting, we are going to delete the quality score feature.
   #  This feature allows us to perfectly predict the quality category
   if remove_feature:
      del df['quality_score']
   return df


def prepare_dataset():
   """
   Read dataset and update target variable

   Returns:
       Dataframe: Updated dataframe
   """
   df = read_dataset()
   # Convert target to categorical
   df['quality_category'] = pd.Categorical(df['quality_category'])
   df['harvest_date'] = pd.to_datetime(df['harvest_date'])
   # Create numerical time features
   df['harvest_month'] = df['harvest_date'].dt.month
   df['harvest_day'] = df['harvest_date'].dt.day
   df = df.drop(columns=['sample_id', 'harvest_date'], axis=1)
   return df

def split_dataset(df):
   """
   Train vs. test split

   Args:
       df (Dataframe): Dataset

   Returns:
       Tuple(Df, Df): Tuple containing train vs. test dataframes
   """
   df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
   return df_full_train, df_test

def transform_data(df):
   """
   Transform Data using DictVectorizer

   Args:
       df (Dataframe): Dataframe to be transformed

   Returns:
       Tuple(Df, dv): Tuple containing transformed df and fitted vectorizer.
   """
   dicts = df.to_dict(orient='records')
   dv = DictVectorizer(sparse=True)
   X = dv.fit_transform(dicts)
   return X, dv

def get_feats_and_target(df):
   """
   Split features and target

   Args:
       df (Dataframe): Complete dataset feats + target

   Returns:
       Tuple(Array, Df): Tuple containing numpy array with target and features df.
   """
   target = "quality_category"
   y_train = df[target].values
   del df[target]
   return y_train, df

def write_artifacts(dv, model):
   """
   Write model artifacts to binary file

   Args:
       dv (Vectorizer): Fitted dict vectorizer
       model (model objects): Trained model object
   """
   directory = "./artifacts"
   if not os.path.exists(directory):
      os.makedirs(directory)
   with open(f'{directory}/model.bin', 'wb') as f_out:
       pickle.dump((dv, model), f_out)

def train_model():
   """
   Main function to execute model training.
   """
   logger.info("Preparing raw data")
   df = prepare_dataset()
   logger.info("Splitting train vs. test")
   df_full_train, df_test = split_dataset(df)
   y_train, X_train = get_feats_and_target(df_full_train)
   logger.info("Transform data")
   X_train, dv = transform_data(X_train)
   logger.info("Model training")

   params = {"n_estimators":186,
             "max_depth":11,
             "min_samples_leaf":4,
             "random_state":42, 
             "n_jobs":-1}

   model = RandomForestClassifier(**params)

   model.fit(X_train, y_train)
   logger.info("Writting Model")
   write_artifacts(dv, model)
   logger.info("Writing test data")
   df_test.to_csv("./artifacts/df_test.csv", index=False)

if __name__ == '__main__':
   train_model()