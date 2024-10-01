# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Building Multi-stage AI System
# MAGIC
# MAGIC In this lab, you will construct a multi-stage reasoning system using Databricks' features and LangChain.
# MAGIC
# MAGIC You will start by building the first chain, which performs a web search using a dataset containing product descriptions from the Red Dot Design Award. Following that, you will create the second chain, which performs an image search using the same dataset. Finally, you will integrate these chains to form a complete multi-stage AI system.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1:** Create a Vector Store
# MAGIC
# MAGIC * **Task 2:** Build the First Chain (Vector Store Search)
# MAGIC
# MAGIC * **Task 3:** Build the Second Chain (Product Image)
# MAGIC
# MAGIC * **Task 4:**  Integrate Chains into a Multi-chain System
# MAGIC
# MAGIC **📝 Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet langchain==0.1.16 langchain_community==0.0.36 databricks-vectorsearch==0.33 langchain-openai==0.1.6
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02LAB

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Before you start building the AI chain, you will need to load dataset and save to a Delta table. The dataset contains information about the products that have won the Red Dot Design Award. This table will be used for creating embeddings.

# COMMAND ----------

from datasets import load_dataset
from pyspark.sql import functions as F

# Define a persistent cache directory
cache_dir = "/dbfs/cache/"

# Load dataset from Hugging Face, limit to 50%
dataset = load_dataset("xiyuez/red-dot-design-award-product-description", split="train[:50%]", cache_dir=cache_dir)

# The dataset has product, category, and text columns
product = dataset['product']
category = dataset['category']
text = dataset['text']
vs_source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.product_text"

# Create DataFrame
df = spark.createDataFrame(zip(product, category, text), ["product", "category", "text"])
# Save DataFrame as a Delta table
df.write.format("delta").mode("overwrite").saveAsTable(vs_source_table_fullname)

# add id column
df = df.withColumn("id", F.monotonically_increasing_id())

# Save DataFrame as a Delta table with the new schema
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(vs_source_table_fullname)

# Enable Change Data Feed for Delta table
spark.sql(f"ALTER TABLE {vs_source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC %md 
# MAGIC ## Task 1: Create a Vector Store
# MAGIC
# MAGIC In this task, you will compute embeddings for the dataset containing information about the products that have won the Red Dot Design Award and store them in a Vector Search index using Databricks Vector Search.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC * **Store in Vector Store:**
# MAGIC    - Create a Vector Search index. 
# MAGIC    - Create the index using **managed embeddings**. Use the **`text`** field of the dataset for indexing.
# MAGIC

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
# Endpoint and table names
vs_index_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.product_embeddings"
vs_source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.product_text"

# Create compute endpoint
vsc = VectorSearchClient()
create_vs_endpoint(vs_endpoint_name)
    
# Create or sync the index
if not index_exists(vsc, vs_endpoint_name, vs_index_table_fullname):
    print(f"Creating index {vs_index_table_fullname} on endpoint {vs_endpoint_name}...")
        
    vsc.create_delta_sync_index(
        <FILL_IN>
        )

else:
    # Trigger a sync to update our vs content with the new data saved in the table
    vsc.get_index(<FILL_IN>).sync()

# Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Build the First Chain (Vector Store Search)
# MAGIC
# MAGIC In this task, you will create first chain that will search for product details from the Vector Store using a dataset containing product descriptions from the Red Dot Design Award.
# MAGIC
# MAGIC **Instructions:**
# MAGIC    - Configure components for the first chain to perform a search using the Vector Store.
# MAGIC    - Utilize the loaded dataset to generate prompts for Vector Store search queries.
# MAGIC    - Set up retrieval to extract relevant product details based on the generated prompts and search results.
# MAGIC

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.vectorstores import DatabricksVectorSearch

# Define the Databricks Chat model: DBRX
llm_dbrx = <FILL_IN>

Define the prompt template for generating search queries
prompt_template_vs = PromptTemplate.from_template(
   """
   You are a product design expert and your task is to create creative products that are very good and can receive design awards.
#
   Write a product description that is similar to the following product description.
#
   Use following product descriptions as example;
#
   <context>
   {context}
   </context>
#
   Question: {input}
#
   Answer:
   """
)

# Construct the RetrievalQA chain for Vector Store search
product_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
vectorstore = <FILL_IN>
retriever = <FILL_IN>

# Construct the chain for question-answering
question_answer_chain = create_stuff_documents_chain(<FILL_IN>)
chain1 = <FILL_IN>

# Invoke the chain with an example query   
response = chain1.<FILL_IN>
print(response['answer'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Build the Second Chain (Product Image)
# MAGIC
# MAGIC Construct the second chain to search for images related to the product descriptions obtained from the dataset using the DALL-E API.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC   - Set the OpenAI API key to access the DALL-E API.
# MAGIC   - Define a **`GetProductImageRunnable`** class that implements the **`Runnable`** interface.
# MAGIC   - Generate the HTML content to display the product image retrieved from the **DALL-E API**.
# MAGIC   - Use the **`display`** function from IPython's **`display`** module to render the HTML content and display the product image.
# MAGIC

# COMMAND ----------

import os
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.runnables.base import Runnable
from IPython.display import display, HTML

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("llm_scope", "openai_token")

# Generate image using first chain
class GetProductImageRunnable(Runnable):
    def invoke(self, input, config=None):
        <FILL_IN>

# Instantiate the runnable
chain2 = <FILL_IN>

# Get the image URL with query
image_url = chain2.<FILL_IN>

# Construct HTML content with the retrieved image URL
html_product_content = f"<img src='{image_url}'/>"

# Display the HTML content
display(HTML(html_product_content))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Integrate Chains into a Multi-chain System
# MAGIC
# MAGIC In this task, you will link the individual chains created in Task 2 and Task 3 together to form a multi-chain system that can handle multi-stage reasoning.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC    - Define the Databricks **`DBRX Chat model`** for processing text inputs.
# MAGIC    - Create a prompt template to generate an **`HTML page`** for displaying product details, including both the product description and image.
# MAGIC    - Construct the **`Multi-Chain System`**  by combining the outputs of the previous chains, including the product description and image.
# MAGIC    - Invoke the multi-chain system with the input data to generate the HTML page for the specified product.
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template for generating the HTML page
prompt_template_1 = PromptTemplate.from_template(
    """Create an HTML page for the following product details:
   
    Product Description: {description}
   
    Product Image: {product_image}
#
   Return a valid HTML code.
    """
)


# Construct the multi-chain system
chain3 = <FILL_IN>

# Invoke the chain with question and query
output_html = chain3.<FILL_IN>

# Display the generated HTML output
display(HTML(output_html))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you've learned how to build a multi-stage AI system using Databricks and LangChain. By integrating multiple chains, you can perform complex reasoning tasks such as searching for product details and retrieving related images. This approach enables the development of sophisticated AI systems capable of handling diverse tasks efficiently.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>