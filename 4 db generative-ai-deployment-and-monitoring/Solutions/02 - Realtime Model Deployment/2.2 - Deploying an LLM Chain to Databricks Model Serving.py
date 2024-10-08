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
# MAGIC # Deploying an LLM Chain to Databricks Model Serving
# MAGIC
# MAGIC **In this demo, we will focus on deploying and querying GenAI models in realtime.**
# MAGIC
# MAGIC Deployment is a key part of operationalizing our LLM-based applications. We will explore deployment options within Databricks and demonstrate how to achieve each one.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Determine the right deployment strategy for your use case.
# MAGIC * Deploy a custom RAG chain to a Databricks Model Serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, **open & run the provided [00-Build-Model]($../00-Build-Model)** script at begining of class.
# MAGIC
# MAGIC **Note:** This script sets up a RAG applications, including a Databricks Vector Search and Vector Search Index, and it can take ~1 hour to complete at the moment. If the Vector Search and accompanying index are already created in your workspace, they will be used.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 mlflow==2.12.1
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demo, we will walk through basic real-time deployment capabilities in Databricks. Model Serving allows us to deploy models and query it using various methods. 
# MAGIC
# MAGIC In this demo, we'll discuss this in the following steps:
# MAGIC
# MAGIC 1. Prepare a model to be deployed.
# MAGIC
# MAGIC 1. Deploy the registered model to a Databricks Model Serving endpoint.
# MAGIC
# MAGIC 1. Query the endpoint using various methods such as `python sdk` and `mlflow deployments`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Preparation
# MAGIC
# MAGIC When we do this, we need to first to create our model.
# MAGIC
# MAGIC We have created a RAG model as a part of the set up of this lesson and have logged it in Unity Catalog for governance purposes and ease of deployment to Model Serving.
# MAGIC
# MAGIC It's here: **`genai_shared_catalog.default.rag_app`**.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

schema_name = f'ws_{spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")}'

model_name = f"genai_shared_catalog.{schema_name}.rag_app"

# Point to UC registry
mlflow.set_registry_uri("databricks-uc")

def get_latest_model_version(model_name_in:str = None):
    """
    Get latest version of registered model
    """
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name_in)
    if model_version_infos:
      return max([model_version_info.version for model_version_info in model_version_infos])
    else:
      return None

# COMMAND ----------

latest_model_version = get_latest_model_version(model_name)

if latest_model_version:
  print(f"Model created and logged to: {model_name}/{latest_model_version}")
else:
  raise(BaseException("Error: Model not created, verify if 00-Build-Model script ran successfully!"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy a Custom Model to Model Serving
# MAGIC
# MAGIC Deploying custom models once they're in Unity Catalog is similar to the workflow we demonstrated for external models once they're in Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-Requisite: Set up Secrets
# MAGIC
# MAGIC To secure access to the serving endpoint, we need set up a couple of secrets for the host (workspace URL) and a personal access token.
# MAGIC
# MAGIC Secrets can be set up using the Databricks CLI with the following commands:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope <scope-name>
# MAGIC databricks secrets put-secret --json '{
# MAGIC   "scope": "<scope-name>",
# MAGIC   "key": "<key-name>",    
# MAGIC   "string_value": "<value>"
# MAGIC }' 
# MAGIC ```
# MAGIC
# MAGIC So in our case, we've run:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope <scope-name>
# MAGIC databricks secrets put-secret --json '{
# MAGIC   "scope": "genai_training",
# MAGIC   "key": "depl_demo_host",    
# MAGIC   "string_value": "<host-name>"
# MAGIC }'
# MAGIC databricks secrets put-secret --json '{
# MAGIC   "scope": "genai_training",
# MAGIC   "key": "depl_demo_token",    
# MAGIC   "string_value": "<token_value>"
# MAGIC }' 
# MAGIC ```
# MAGIC
# MAGIC Once this is set up, we will load the values into variables for the notebook using the Secrets utility in Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy model using `databricks-sdk` API
# MAGIC
# MAGIC In the notebook we will use the API to create the endpoint and serving the model.
# MAGIC
# MAGIC **Note :** You could also simply use the UI for this task.
# MAGIC
# MAGIC **⏰ Expected deployment time: ~10 mins**

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput

# Configure the endpoint
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": latest_model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
            "environment_vars": {
                # "DATABRICKS_TOKEN": "{{secrets/genai_training/depl_demo_token}}",
                # "DATABRICKS_HOST": "{{secrets/genai_training/depl_demo_host}}", 
                "DATABRICKS_TOKEN": "{{{{secrets/{0}/depl_demo_token}}}}".format(DA.scope_name),
                "DATABRICKS_HOST": "{{{{secrets/{0}/depl_demo_host}}}}".format(DA.scope_name),
                #"DATABRICKS_TOKEN": token,
                #"DATABRICKS_HOST": host,  
            },
        },
    ],
    "auto_capture_config":{
        "catalog_name": DA.catalog_name,
        "schema_name": DA.schema_name,
        "table_name_prefix": "rag_app_realtime"
    }
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC **Important:** Please note the syntax setup for the authentication above. Rather than passing the secret variables directly, we follow syntax requirements **&lcub;&lcub;secrets/&lt;scope&gt;/&lt;key-name&gt;&rcub;&rcub;** so that the endpoint will look up the secrets in real-time rather than automatically configure and expose static values.

# COMMAND ----------

from databricks.sdk import WorkspaceClient


# Initiate the workspace client
w = WorkspaceClient()
serving_endpoint_name = f"{DA.unique_name('_')}_endpoint"

# Get endpoint if it exists
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").value()
serving_endpoint_url = f"{db_host}/ml/endpoints/{serving_endpoint_name}"

# If endpoint doesn't exist, create it
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

# If endpoint does exist, update it to serve the new version
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC The model could have also been deployed programmatically using mlflow's [deploy_client](add ref)
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC
# MAGIC
# MAGIC deploy_client = get_deploy_client("databricks")
# MAGIC endpoint = deploy_client.create_endpoint(
# MAGIC     name=serving_endpoint_name,
# MAGIC     config=endpoint_config
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Method 2) - Create Inference Table via Model Serving UI
# MAGIC
# MAGIC
# MAGIC If an endpoint is already up and running, we can tell if an inference table is not already set up by navigating to the Model Serving endpoint page and view the inference table field.
# MAGIC
# MAGIC To set up this inference table manually, we'll follow the below steps:
# MAGIC
# MAGIC 1. Go to [Serving](/ml/endpoints).
# MAGIC
# MAGIC 1. Find the endpoint you created and click the **Edit endpoint** button on the endpoint page. 
# MAGIC
# MAGIC 1. Expand the **Inference tables** section.
# MAGIC
# MAGIC 1. Check the **Enable inference tables** box.
# MAGIC
# MAGIC 1. Enter the catalog, schema and table information for the inference table.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-realtime-inference.png"  width="=100%">
# MAGIC
# MAGIC **Note:** To set up an inference table, you must configure your endpoint using a Databricks Secret.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Inference on the Model
# MAGIC
# MAGIC Next, we will want to perform inference using the model – that is, provide input and return output.
# MAGIC
# MAGIC We'll start with a simple example of a single input:

# COMMAND ----------

question = "What is PPO?"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference width SDK

# COMMAND ----------

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference with MLflow Deployments

# COMMAND ----------

from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")
response = deploy_client.predict(
  endpoint=serving_endpoint_name,
  inputs={"inputs" : [{"query": question}]}
)
print(response.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View the Inference Table
# MAGIC
# MAGIC Once the table is created and the endpoint is hit couple times, we can view the table in the Catalog Explorer to inspect the saved query data.
# MAGIC
# MAGIC To view the inference table:
# MAGIC
# MAGIC 1. Go to **[Catalog](explore/data)**.
# MAGIC
# MAGIC 1. Select the catalog and schema you entered while configuring the inference table in previous step.
# MAGIC
# MAGIC 1. Select the inference table and view the sample data. 
# MAGIC
# MAGIC **🚨 Note:** It might take couple minutes to see the monitoring data.
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-realtime-inference-table.png" width="100%">
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC **💡 Note:** We can also view the data directly by querying the table. This can be useful if we're wanting to work the data into our application in some way (e.g. using human feedback to inform testing strategy, etc.).
# MAGIC
# MAGIC **Question:** What data do you see in the inference table?

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
# MAGIC In this demo, we demonstrated how to deploy an RAG pipeline in real-time using Databricks Model Serving. The model was created and registered in the Model Registry, making it ready for use. First, we deployed the model to a Model Serving endpoint using the SDK. Then, we configured the endpoint and enabled the inference table. Finally, we showed how to query the endpoint in real-time using the SDK and MLflow deployments.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>