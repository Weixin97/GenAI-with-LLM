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
# MAGIC # LAB: Custom Model Deployment and A/B Testing
# MAGIC
# MAGIC In this lab, you will learn how to deploy and serve a custom model using Databricks Model Serving. You will understand the steps involved in preparing, deploying, and querying a model endpoint in Databricks. This lab will focus on the practical aspects of deploying custom models and querying them for real-time inference.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will need to complete the following tasks:*
# MAGIC
# MAGIC 1. **Task 1:** Get Model Version
# MAGIC 1. **Task 2:** Deploy Model with SDK
# MAGIC 1. **Task 3:** Configure A/B Testing Using the UI
# MAGIC 1. **Task 4:** Query the Endpoint
# MAGIC 1. **Task 5:** Inspect Inference Table

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow==2.12.1 databricks-sdk
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Model Details
# MAGIC The model is created with the **00-Model-Build** notebook. It's registered in Unity Catalog for governance purposes and ease of deployment to Model Serving.
# MAGIC
# MAGIC Model location: `genai_shared_catalog.default.rag_app`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 1: Get Model Version
# MAGIC
# MAGIC In this task, you will retrieve model details and version from the model registry. This will help you identify the latest version of your model for deployment.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

schema_name = f'ws_{spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")}'

# Define the model name
model_name = f"genai_shared_catalog.{schema_name}.rag_app"
# Set the registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = MlflowClient()

# Get the latest version number of the specified model
model_version_infos = client.<FILL_IN>
latest_model_version = <FILL_IN>

# Print the latest model version
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Deploy Model with SDK
# MAGIC
# MAGIC In this task, you will deploy the model using the SDK and enable the inference table. This involves defining environment variables, configuring the endpoint, and setting up the inference table.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.1: Set Up Secrets
# MAGIC
# MAGIC
# MAGIC To secure access to the serving endpoint, set up secrets for the host (workspace URL) and a personal access token. This can be done using the Databricks CLI:
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC databricks secrets create-scope <scope-name>
# MAGIC databricks secrets put-secret --json '{
# MAGIC   "scope": "<scope-name>",
# MAGIC   "key": "<key-name>",
# MAGIC   "string_value": "<value>"
# MAGIC }'
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC **Important:** Please note the syntax setup for the authentication above. Rather than passing the secret variables directly, we follow syntax requirements **&lcub;&lcub;secrets/&lt;scope&gt;/&lt;key-name&gt;&rcub;&rcub;** so that the endpoint will look up the secrets in real-time rather than automatically configure and expose static values.
# MAGIC
# MAGIC **To print the secret values:**

# COMMAND ----------

# Print the value of scope and key
print("Scope: ", DA.scope_name)
print("Key for Token: depl_demo_token")
print("Key for Host: depl_demo_host")

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.2: Configure and Deploy Endpoint
# MAGIC
# MAGIC Configure the endpoint and deploy the model using the SDK, ensuring proper setup of environment variables.

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput
from databricks.sdk import WorkspaceClient

# Define endpoint configuration
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": <FILL_IN>,
            "model_version": <FILL_IN>,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
            "environment_vars": {
               "DATABRICKS_TOKEN": "{{{{secrets/{0}/depl_demo_token}}}}".format(DA.scope_name),
               "DATABRICKS_HOST": "{{{{secrets/{0}/depl_demo_host}}}}".format(DA.scope_name),
            },
        },
    ]
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Initiate the workspace client
w = WorkspaceClient()
serving_endpoint_name = f"{DA.unique_name('_')}_endpoint"

# Get endpoint if it exists
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)

# Get the Databricks host
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").value()
serving_endpoint_url = f"{db_host}/ml/endpoints/{serving_endpoint_name}"

# If endpoint doesn't exist, create it
if existing_endpoint is None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.<FILL_IN>
# If endpoint does exist, update it to serve the new version
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.<FILL_IN>

# Display the endpoint URL
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3: Create Inference Table via Model Serving UI
# MAGIC
# MAGIC Set up an inference table through the Model Serving UI:
# MAGIC
# MAGIC 1. Click the link printed above in the output of **step 2.2**.
# MAGIC 2. Click the **Edit endpoint** button.
# MAGIC 3. Expand the **Inference tables** section.
# MAGIC 4. Check the **Enable inference tables** box.
# MAGIC 5. Enter the catalog, schema, and table information for the inference table:
# MAGIC    - **Catalog Name:** `<Your Catalog Name>`
# MAGIC    - **Schema Name:** `<Your Schema Name>`
# MAGIC    - **Table Name Prefix:** `<Your Table Name>` (e.g.: `rag_app_realtime`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Configure A/B Testing Using the UI
# MAGIC
# MAGIC In this task, you will configure traffic splitting between the same version of the model for A/B testing using the Databricks UI. This will ensure that both configurations are available for inference, and you can direct a percentage of traffic to each configuration for A/B testing or gradual rollouts.
# MAGIC
# MAGIC
# MAGIC 🚨 **Note:** Normally, you would register an improved version of the model. However, due to time constraints, you will deploy the same model.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. **Go to [Serving](/ml/endpoints)**
# MAGIC
# MAGIC 2. Locate the endpoint you created earlier.
# MAGIC 3. Click on the **Edit endpoint** button next to the endpoint name.
# MAGIC
# MAGIC 4. **Add a New Served Entity**
# MAGIC     - In the **Served entities** section, click on **+ Add served entity**.
# MAGIC     - Select the entity name that matches your model name: **`genai_shared_catalog.default.rag_app`**.
# MAGIC     - Choose **Version 1** for the new served entity.
# MAGIC
# MAGIC 5. **Configure Traffic Splitting**
# MAGIC     - In the **Traffic Splitting** section, divide the traffic between the two configurations.
# MAGIC     - Set the traffic percentage to 60% for the new configuration and 40% for the old configuration.
# MAGIC
# MAGIC 6. **Set Compute Scale-out**
# MAGIC     - For **Compute scale-out**, select **Small**.
# MAGIC
# MAGIC 7. **Advanced Configuration**
# MAGIC     - Fill in the environment variables as follows:
# MAGIC       - **DATABRICKS_HOST** : **&lcub;&lcub;secrets/genai_training/depl_demo_host&rcub;&rcub;**
# MAGIC       - **DATABRICKS_TOKEN** : **&lcub;&lcub;secrets/genai_training/depl_demo_token&rcub;&rcub;**
# MAGIC
# MAGIC 8. **Check Inference Table Details**
# MAGIC     - Ensure that the inference table settings are correct.
# MAGIC     - The table should capture inference results for analysis.
# MAGIC
# MAGIC 9. **Update and Wait**
# MAGIC     - Click on the **Update** button to save your changes.
# MAGIC     - Wait for the serving endpoint to update. This may take a few minutes.
# MAGIC
# MAGIC By following these steps, you will successfully configure A/B testing for your model using the same version, allowing you to evaluate different configurations and monitor their performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Query the Endpoint 
# MAGIC
# MAGIC In this task, you will query the model using MLflow deployments.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
# Initialize the deployment client
deploy_client = get_deploy_client("databricks")
# Define the question to be sent to the model for inference
question = "What is PPO?"
# Send the query to the specified serving endpoint and receive the response
response = deploy_client.<FILL_IN>

# Print the model's prediction from the response received
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Inspect Inference Table
# MAGIC
# MAGIC In this task, you will view and inspect the inference table created during the deployment process. The inference table stores data about the inferences made by your model, which can be useful for monitoring and analyzing model performance.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. **Go to [Catalog](explore/data).**
# MAGIC
# MAGIC 2. **Select the Catalog and Schema:**
# MAGIC    - In the Catalog Explorer, find and select the catalog that you entered while configuring the inference table.
# MAGIC    - Within the selected catalog, navigate to the schema that contains your inference table.
# MAGIC
# MAGIC 3. **View the Inference Table:**
# MAGIC    - Locate the inference table within the selected schema. The table name is prefixed as specified during the deployment configuration.
# MAGIC    - Click on the inference table to open and view the sample data stored in it.
# MAGIC
# MAGIC By following these steps, you will be able to access and inspect the inference data stored in the table, allowing you to analyze how your model is performing and what kind of predictions it is making.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cleanup Classroom
# MAGIC Run the following cell to remove lesson-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you successfully deployed a custom model using Databricks Model Serving. You learned how to retrieve model versions, deploy models using the SDK, create and deploy a second version using the UI, query the model endpoint, and inspect inference results stored in the table.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>