# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Model (RAG Chain)
# MAGIC
# MAGIC This notebook should be run before the module to create and register a model that will be used for the rest of the module.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup

# COMMAND ----------

# MAGIC %pip install --quiet transformers==4.30.2 databricks-vectorsearch==0.22 "unstructured[pdf,docx]==0.10.30" langchain==0.1.16 llama-index==0.9.3 pydantic==1.10.9 databricks-feature-engineering==0.2.0 databricks-sdk==0.28.0
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-00

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Shared Catalog

# COMMAND ----------

catalog_name = 'genai_shared_catalog'
schema_name = 'default'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS `genai_shared_catalog`;
# MAGIC USE CATALOG `genai_shared_catalog`;
# MAGIC USE SCHEMA `default`;
# MAGIC GRANT USE CATALOG, USE SCHEMA, EXECUTE ON CATALOG `genai_shared_catalog` TO `account users`;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

import time
import re
import io
import os
import pandas as pd 

from unstructured.partition.auto import partition
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from pyspark.sql.functions import col, udf, length, pandas_udf, explode
from unstructured.partition.auto import partition

# COMMAND ----------

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document
  # concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 


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

# MAGIC %md
# MAGIC ## Prepare Data
# MAGIC
# MAGIC 1. Read raw `pdf` into `pdf_raw_text` delta table
# MAGIC 2. Chunk using llama-index & pandas_udf
# MAGIC 3. Materialize chunked documents into `pdf_text_chunks` delta table

# COMMAND ----------

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

articles_path = f"{DA.paths.datasets}/arxiv-articles/"
table_name = f"{catalog_name}.{schema_name}.pdf_raw_text"

# read pdf files
df = (
        spark.read.format("binaryfile")
        .option("recursiveFileLookup", "true")
        .load(articles_path)
        )

# save list of the files to table
df.write.mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

df_chunks = (df
                .withColumn("content", explode(read_as_chunk("content")))
                .selectExpr('path as pdf_name', 'content')
                )

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS pdf_text_chunks (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   pdf_name STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- Note: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

chunks_table_name = f"{catalog_name}.{schema_name}.pdf_text_chunks"
df_chunks.write.mode("append").saveAsTable(chunks_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Databricks Vector Search Endpoint
# MAGIC
# MAGIC 1. Spin-up VS endpoint
# MAGIC 2. Create index from `pdf_text_chunks` delta table

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


vs_endpoint_name = "genai_vs_endpoint"
source_table_fullname = f"{catalog_name}.{schema_name}.pdf_text_chunks"
vs_index_fullname = f"{catalog_name}.{schema_name}.pdf_text_vs_index"

vsc = VectorSearchClient(disable_notice=True)


if vs_endpoint_name not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(
        name=vs_endpoint_name, 
        endpoint_type="STANDARD"
        )

wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# create or sync the index
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant `SELECT` access to both catalog and created index to current users & `Account Users`
# MAGIC For demo purposes only

# COMMAND ----------

spark.sql(f"GRANT `USE SCHEMA` ON SCHEMA {catalog_name}.{schema_name} TO `account users`;")
spark.sql(f"GRANT SELECT ON TABLE {vs_index_fullname} TO `account users`;")

spark.sql(f"GRANT `USE SCHEMA` ON SCHEMA {catalog_name}.{schema_name} TO `{DA.username}`;")
spark.sql(f"GRANT SELECT ON TABLE {vs_index_fullname} TO `{DA.username}`;")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Chain
# MAGIC
# MAGIC Using LangChain:
# MAGIC 1. Create retriever
# MAGIC 2. Create using pre-defined `RetrievalQA` chain

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Define embedding model
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test your retriever
vectorstore = get_retriever()


# Define foundation model for generating responses
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300)


# Define template for prompt
TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# Merge retriever and foundation model into Langchain chain
chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

question = {"query": "How Generative AI impacts humans?"}
answer = chain.invoke(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Model

# COMMAND ----------

import mlflow
import langchain
from mlflow.models import infer_signature


# Set Model Registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.rag_app"

# Register the assembled RAG model in Model Registry with Unity Catalog
with mlflow.start_run(run_name="rag_app_demo4") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>