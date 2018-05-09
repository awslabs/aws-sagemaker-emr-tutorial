# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

import flask

import pandas as pd
import json

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    spark = None
    hashes = None

    @classmethod
    def get_spark(cls):
        if cls.spark is None:
            cls.spark = SparkSession \
                .builder \
                .master("local[*]") \
                .appName("SageMaker Hosting") \
                .getOrCreate()

        return cls.spark

    @classmethod
    def get_hashes(cls):
        
        if cls.hashes is None:
            cls.hashes = cls.get_spark().read.format("parquet").load("file:///opt/ml/model/artifacts/hashes")

        return cls.hashes


    @classmethod
    def get_model(cls):

        cls.get_spark()

        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = PipelineModel.load("file:///opt/ml/model/artifacts/SAGEMAKER_MODEL_NAME.model")

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        hash = cls.get_hashes()

        input_df = clf.transform(input) # word2vec transform
        return input_df

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


from pyspark.sql.functions import split, col
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

def getRecs(df):
    key = df.select('features').collect()[0][0]

    udfSquaredDistance = udf(lambda x: float(key.squared_distance(x)), FloatType())
    distances = ScoringService.get_hashes().withColumn("distance", udfSquaredDistance("features"))
    
    recs = distances.sort(col('distance')).select('content').limit(3)
    
    return recs

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=0)

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    df = ScoringService.get_spark().createDataFrame(data)

    # Do the prediction
    predictions = ScoringService.predict(df.withColumn('bow', split(col('content'), ' ')))

    recommendations = getRecs(predictions)

    pd_recs = recommendations.toPandas()
    

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd_recs.to_csv(out, header=['content'], index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
