# Databricks notebook source
# Function used to randomly assign each user a VS Endpoint
def get_fixed_integer(string_input):
    # Calculate the sum of ASCII values of the characters in the input string
    ascii_sum = sum(ord(char) for char in string_input)
    
    # Map the sum to a fixed integer between 1 and 9
    fixed_integer = (ascii_sum % 9) + 1
    
    return fixed_integer

# COMMAND ----------

import time
import re
import io

def pprint(obj):
  import pprint
  pprint.pprint(obj, compact=True, indent=1, width=100)

def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

from pyspark.sql.functions import col, udf, length, pandas_udf, explode
import pandas as pd 
import mlflow.deployments
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
from databricks.vector_search.client import VectorSearchClient

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

def create_vs_endpoint(vs_endpoint_name):
    vsc = VectorSearchClient()

    # check if the endpoint exists
    if vs_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
        vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

    # check the status of the endpoint
    wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
    print(f"Endpoint named {vs_endpoint_name} is ready.")

def create_vs_index(vs_endpoint_name, vs_index_fullname, source_table_fullname, source_col):
    #create compute endpoint
    vsc = VectorSearchClient()
    create_vs_endpoint(vs_endpoint_name)
    
    # create or sync the index
    if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
        print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
        
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=vs_index_fullname,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED", #Sync needs to be manually triggered
            primary_key="id",
            embedding_source_column=source_col,
            embedding_model_endpoint_name="databricks-bge-large-en"
        )
        
        # vsc.create_delta_sync_index(
        #     endpoint_name=vs_endpoint_name,
        #     index_name=vs_index_fullname,
        #     source_table_name=source_table_fullname,
        #     pipeline_type="TRIGGERED", #Sync needs to be manually triggered
        #     primary_key="id",
        #     embedding_dimension=1024, #Match your model embedding size (bge)
        #     embedding_vector_column="embedding"
        # )

    else:
        #Trigger a sync to update our vs content with the new data saved in the table
        vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

    #Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)