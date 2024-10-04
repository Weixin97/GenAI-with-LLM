# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #Lab 2: Implementing AI Guardrails
# MAGIC
# MAGIC In this lab, you will implement guardrails for a simple generative AI application to secure it against malicious behavior and harmful generated output.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks:
# MAGIC
# MAGIC * **Task 1:** Implement LLM-based guardrails with Llama Guard model.
# MAGIC   1. Set Up Llama Guard Model and Configuration Variables
# MAGIC   2. Implement the query_llamaguard Function
# MAGIC   3. Test the Implementation
# MAGIC * **Task 2:** Customize Llama Guard Guardrails
# MAGIC   1. Define Custom Unsafe Categories
# MAGIC   2. Test the Implementation
# MAGIC * **Task 3:** Integrate Llama Guard with Chat Model
# MAGIC   1. Set up an non-Llama Guard query function
# MAGIC   2. Set up a Llama Guard query function
# MAGIC   3. Test the Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, execute the provided classroom setup script to install the required libraries and configure necessary variables.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk mlflow==2.11.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1: Implement LLM-based Guardrails with `Llama Guard`
# MAGIC First, To set up the safety measures of your application, you will integrate Llama Guard, a specialized model available on the Databricks Marketplace. This will enable you to classify chat content as safe or unsafe, allowing for more effective management of potentially harmful interactions.
# MAGIC
# MAGIC
# MAGIC **Llama Guard in Databricks**
# MAGIC
# MAGIC To streamline the integration process and leverage the benefits of Llama Guard, a deployment of this model is readily available on the Databricks Marketplace.
# MAGIC
# MAGIC **Instructions (To be performed by the instructor only):**
# MAGIC
# MAGIC 1. Find the "Lllama Guard Model" in **Databricks Marketplace**.
# MAGIC 2. Click on **Get Instant Access** to load it to a location in Unity Catalog.
# MAGIC 3. **Deploy the model** to a Model Serving endpoint.
# MAGIC
# MAGIC By integrating the Model Serving endpoint into your own application, you gain the flexibility to specify your own policies for detecting and preventing various types of content. This ensures that your application maintains a safe and secure environment for users.
# MAGIC
# MAGIC **🚨 Warning:** Please avoid deploying the model yourself, as it may take time and might not be practical in a classroom setting. Instead, utilize the model that has been provided for this lab. For self-paced learners, you can follow the steps above to create the endpoint.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Instructions:**
# MAGIC 1. **Set Up Llama Guard Model and Configuration Variables**
# MAGIC
# MAGIC     Set up a starting point description of your **Unsafe** category:
# MAGIC
# MAGIC     * **To do:**
# MAGIC     
# MAGIC         You have been provided with 2 categories. Your task is to write descriptions for 3 additional categories based on common themes or topics. 
# MAGIC
# MAGIC     
# MAGIC         **Hint:** You can use AI Playground to generate descriptions for these categories, ensuring they are relevant and distinct from the provided examples.

# COMMAND ----------

DEFAULT_UNSAFE_CATEGORY = """O1: Unsafe Content.
    - Promoting or assisting in violence or hate.
    - Engaging in or encouraging sexually explicit content.
    - Assisting or encouraging criminal activities, including those related to firearms or illegal weapons or robbery.
    - Encouraging or assisting in the use or production of illegal or controlled substances.
    - Encouraging or detailing self-harm or suicide."""

# COMMAND ----------

# MAGIC %md
# MAGIC Define the necessary variables and set up the **Llama Guard** model.

# COMMAND ----------

# llama-guard endpoint name. Change this to your own endpoint name, if you manually create one!
LLAMAGUARD_ENDPOINT_NAME="llama-guard"

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Implement the `query_llamaguard` Function**
# MAGIC
# MAGIC Develop a function to query the Llama Guard model and classify chat content as safe or unsafe.

# COMMAND ----------

import mlflow
import mlflow.deployments
import re

def query_llamaguard(chats, unsafe_categories = DEFAULT_UNSAFE_CATEGORY):
    """
    Queries LlamaGuard model to check if the chat contains unsafe content based on predefined categories.
    
    Args:
        chats (list): List of chat inputs.
        unsafe_categories (str): String defining predefined unsafe content categories.

    Returns:
        list: List of tuples containing the safety assessment results for each chat input.
               Each tuple contains two values: (is_safe, reason)
                   - is_safe (bool): Indicates whether the chat content is safe or not.
                   - reason (str): Description of the reason if the content is classified as unsafe.
    """
    try:
        # Get the MLflow deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        results = []

        # Iterate over each chat input
        for idx, chat in enumerate(chats):
            # Construct the query payload for LlamaGuard
            query_payload = [
                f"""[INST] Task: Check for unsafe content in 'Agent' messages in conversations according to our safety policy.
                <BEGIN UNSAFE CONTENT CATEGORIES>
                {unsafe_categories}
                <END UNSAFE CONTENT CATEGORIES>
                <BEGIN CONVERSATION>
                {chat['content']}
                <END CONVERSATION>
                Provide your safety assessment for Agent in the above conversation:
                - First line must read 'safe' or 'unsafe'.
                - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
            ]

            # Query LlamaGuard model
            response = client.predict(
                endpoint=LLAMAGUARD_ENDPOINT_NAME,
                inputs={"dataframe_split": {
                        "index": [0],
                        "columns": ["prompt"],
                        "data": [query_payload]
                        }
                    })

            # Extract prediction and reason
            prediction = response["predictions"][0]["candidates"][0]["text"].strip()
            is_safe = prediction.split("\n")[0].lower() == 'safe'
            reason = prediction.split("\n")[1].strip() if not is_safe else None
            results.append((is_safe, reason))  # Ensure the return is a tuple with two values

        return results
    
    except Exception as e:
        # Raise exception if there's an error in querying LlamaGuard model
        raise Exception(f"Error in querying LlamaGuard model: {str(e)}")
#

# COMMAND ----------

# MAGIC %md
# MAGIC 3. **Test the Implementation**
# MAGIC
# MAGIC Run the code and test the `query_llamaguard` function to see if it correctly identifies unsafe content.

# COMMAND ----------

# Define the chat inputs
chats = [
  {
      "role": "user",
      "content": "How to bake chocolate chip cookies"
  },
  {
      "role": "user",
      "content": "Discussing strategies for committing fraud"
  },
  {
      "role": "user",
      "content": "I want to hate"
  }
]

# Query Llama Guard for multiple chats
results = query_llamaguard(chats, DEFAULT_UNSAFE_CATEGORY)

# Print the results
for result in results:
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Customize Llama Guard Guardrails
# MAGIC
# MAGIC In this task, you will customize the guardrails provided by Llama Guard by defining custom unsafe categories. This will enable you to tailor the safety assessment criteria to meet specific requirements.
# MAGIC
# MAGIC **Instructions:**
# MAGIC 1. **Define Custom Unsafe Categories**
# MAGIC
# MAGIC Define custom unsafe categories that reflect the specific safety concerns of your application.

# COMMAND ----------

# Define custom unsafe categories
unsafe_categories = """
O1: Violence and Hate.
Should not
- Promote or incite physical harm or violence against individuals or groups.
- Advocate for or glorify acts of terrorism, war crimes, or violent extremism.
- Use or promote language that denigrates individuals based on race, ethnicity, religion, gender, sexual orientation, or other protected characteristics.
Can
- Provide educational content on the history and impact of violence and hate.
- Discuss non-violent conflict resolution strategies and the importance of tolerance and inclusivity.

O2: Sexual Content.
Should not
- Share explicit, graphic, or sexually suggestive content.
- Encourage or solicit participation in sexual activities or online sexual exploitation.
Can
- Offer medically accurate information on sexual health and reproductive education.
- Discuss issues related to consent, relationships, and safe sexual practices.

O3: Criminal Planning.
Should not
- Give instructions or advice on committing crimes such as theft, hacking, or financial fraud.
- Provide resources or support for organizing criminal activities.
Can
- Explain the legal and social consequences of criminal behavior.
- Offer insights into crime prevention and law enforcement practices.

O4: Weapons and Firearms.
Should not
- Instruct on the creation, modification, or use of firearms, explosives, or other dangerous weapons.
- Promote the illegal trade or misuse of weapons.
Can
- Discuss the history and regulations surrounding weapon ownership and use.
- Provide safety information related to responsible weapon handling and storage.

O5: Controlled Substances.
Should not
- Endorse or facilitate the illegal production, sale, or consumption of drugs, alcohol, or tobacco.
- Offer recipes or detailed instructions for creating illegal substances.
Can
- Share facts on the health impacts of substance use and abuse.
- Provide information on treatment options and resources for addiction recovery.

O6: Self-Harm and Suicide.
Should not
- Promote or glorify self-harm, suicide, or other forms of self-injury.
- Describe methods or provide encouragement to engage in self-destructive behaviors.
Can
- Discuss the importance of mental health and strategies for coping with stress and emotional challenges.
- Offer information on where to find professional help and support services.

O7: Personal and Financial Information.
Should not
- Request or disseminate sensitive personal information such as Social Security numbers, bank account details, or passwords.
- Facilitate identity theft or financial fraud.
Can
- Educate on the importance of protecting personal information and online privacy.
- Share tips on recognizing and avoiding scams and fraudulent activities.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Test the Implementation**
# MAGIC
# MAGIC Query the Llama Guard model with your custom unsafe categories to see if it correctly identifies unsafe content.
# MAGIC
# MAGIC

# COMMAND ----------

# Query Llama Guard model with custom unsafe categories
query_llamaguard(chats, unsafe_categories)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Integrate Llama Guard with Chat Model
# MAGIC
# MAGIC Integrate Llama Guard with the chat model to ensure safe interactions between users and the AI system. You'll define two functions: `query_chat` and `query_chat_safely`.
# MAGIC
# MAGIC First, let's set up the endpoint name configuration variable.
# MAGIC
# MAGIC **Note:** The chatbot leverages the **Mixtral 8x7B foundation model** to deliver responses. This model is accessible through the built-in foundation endpoint, available at [/ml/endpoints](/ml/endpoints) and specifically via the `/serving-endpoints/databricks-mixtral-8x7b-instruct/invocations` API.

# COMMAND ----------

CHAT_ENDPOINT_NAME = "databricks-mixtral-8x7b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC **Instructions:**
# MAGIC
# MAGIC 1. **Set up an non-Llama Guard query function** 
# MAGIC - **1.1 - Function: `query_chat`**
# MAGIC
# MAGIC     The `query_chat` function queries the chat model directly without applying Llama Guard guardrails.

# COMMAND ----------

def query_chat(chats):
    """
    Queries the chat model directly without applying Llama Guard guardrails.
    """
    try:
        # Get the MLflow deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        # Query the chat model
        response = client.predict(
            endpoint=CHAT_ENDPOINT_NAME,
            inputs={
                "messages": chats,
                "temperature": 0.1,
                "max_tokens": 512
            }
        )
        # Extract and return the response content
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        # Raise exception if there's an error in querying the chat model
        raise Exception(f"Error in querying chat model: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Set up a Llama Guard query function**
# MAGIC
# MAGIC
# MAGIC   - **2.1 - Function: `query_chat_safely`**
# MAGIC
# MAGIC     The `query_chat_safely` function ensures the application of Llama Guard guardrails both before and after querying the chat model. It evaluates both the user's input and the model's response for safety before processing further.

# COMMAND ----------

def query_chat_safely(chats, unsafe_categories):
    """
    Queries the chat model with Llama Guard guardrails to ensure safe interactions.

    Args:
        chats (list): List of user chat inputs.
        unsafe_categories (str): String defining categories used to determine the safety of the chat content.

    Returns:
        list: List of chat responses with safety classifications.
        
    Raises:
        Exception: If there are issues in querying the chat model, processing the response, 
                    or assessing the safety of the chat.
    """
    results = []
    try:
        # Iterate over each chat input
        for idx, chat in enumerate(chats):
            # Pre-process input with Llama Guard
            unsafe_check = query_llamaguard([chat], unsafe_categories)
            is_safe, reason = unsafe_check[0]
            
            # If input is classified as unsafe, append the reason and category to the results list
            if not is_safe:
                category = parse_category(reason, unsafe_categories)
                results.append(f"Input {idx + 1}: User's prompt classified as unsafe. Fails safety measures. Reason: {reason} - {category}")
                continue

            # Query the chat model
            model_response = query_chat([chat])
            full_chat = [chat] + [{"role": "assistant", "content": model_response}]

            # Post-process output with Llama Guard
            unsafe_check = query_llamaguard([{"role": "user", "content": model_response}], unsafe_categories)
            is_safe, reason = unsafe_check[0]
            
            # If model response is classified as unsafe, append the reason and category to the results list
            if not is_safe:
                category = parse_category(reason, unsafe_categories)
                results.append(f"Input {idx + 1}: Model's response classified as unsafe; fails safety measures. Reason: {reason} - {category}")
                continue

            # Append the model response to the results list
            results.append(f"Input {idx + 1}: {model_response}")
        return results
    except Exception as e:
        # Raise exception if there's an error in the safe query
        raise Exception(f"Error in safe query: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC   - **2.2 - Function: `parse_category`**
# MAGIC
# MAGIC     This function extracts the first sentence of a category description from a taxonomy based on its code. It's used within the `query_chat_safely` function to provide a more understandable reason for unsafe classifications.
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

def parse_category(code, taxonomy):
    """
    Extracts the first sentence of a category description from a taxonomy based on its code.

    Args:
        code (str): Category code in the taxonomy (e.g., 'O1').
        taxonomy (str): Full taxonomy string with categories and descriptions.

    Returns:
         str: First sentence of the description or a default message for unknown codes.
    """
    # Define pattern to match category codes and descriptions
    pattern = r"(O\d+): ([\s\S]*?)(?=\nO\d+:|\Z)"
    
    # Create a dictionary mapping category codes to their descriptions
    taxonomy_mapping = {
        match[0]: re.split(r'(?<=[.!?])\s+', match[1].strip(), 1)[0]
        for match in re.findall(pattern, taxonomy)
    }

    # Return the description for the provided code, or a default message if the code is not found
    return taxonomy_mapping.get(code, "Unknown category: code not in taxonomy.")

# COMMAND ----------

# MAGIC %md
# MAGIC 3. **Test the Implementation**
# MAGIC
# MAGIC Define the test chat inputs and test the `query_chat_safely` function with these inputs and the provided unsafe categories to verify its behavior.
# MAGIC

# COMMAND ----------

# Define the chat inputs
chats = [
  {
      "role": "user",
      "content": "How to bake chocolate chip cookies"
  },
  {
      "role": "user",
      "content": "Discussing strategies for committing fraud"
  },
  {
      "role": "user",
      "content": "I want to hate"
  }
]
# Print the results
results = query_chat_safely(chats, unsafe_categories)
for result in results:
    print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC #Conclusion
# MAGIC In this lab, you have successfully customized the guardrails of our AI application using Llama Guard by defining custom unsafe categories. These customizations allow us to tailor the safety assessment criteria to the specific requirements of our application, thus enhancing the effectiveness of our AI guardrails in identifying and mitigating potential risks associated with generating harmful content.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>