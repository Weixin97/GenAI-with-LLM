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
# MAGIC # LAB - Planning an AI System for Product Quality Complaints
# MAGIC
# MAGIC In this lab, you will deconstruct a use case and define possible components of the AI system. In this scenario, let's consider **a compound AI system designed to handle a customer complaint about a product's quality**. The customer contacts the service center via a chat interface, expressing dissatisfaction with a recently purchased item. The AI system will utilize various components to address and resolve the complaint effectively.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Define system components
# MAGIC
# MAGIC * **Task 2 :** Draw architectural diagram
# MAGIC
# MAGIC * **Task 3 :** Define possible input and output parameters for each component
# MAGIC
# MAGIC * **Task 4:** Define libraries or frameworks for each component
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI System Details
# MAGIC
# MAGIC **Product Quality Complaints:** Let's consider **a compound AI system designed to handle a customer complaint about a product's quality**. The customer contacts the service center via a chat interface, expressing dissatisfaction with a recently purchased item. 
# MAGIC
# MAGIC **Example Agent-User Interaction:** 
# MAGIC
# MAGIC **👱‍♀️ Customer:** "I'm unhappy with the quality of the product I received. It seems defective."
# MAGIC
# MAGIC **🤖 AI Agent:** "I'm sorry to hear that you're not satisfied with your purchase. Let me look into this for you."
# MAGIC
# MAGIC **🤖 AI Agent:** (after processing): "We've reviewed similar feedback and checked the shipping details. It appears there have been a few similar complaints. We can offer a replacement or a full refund. Which would you prefer?"
# MAGIC
# MAGIC **👱‍♀️ Customer:** "I would like a replacement, please."
# MAGIC
# MAGIC **🤖 AI Agent:**  "I've arranged for a replacement to be sent to you immediately. We apologize for the inconvenience and thank you for your understanding."
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Define Components
# MAGIC
# MAGIC Based on the scenario, identify all necessary components that will interact within the system. Also, provide a brief description of the role and functionality of each component. 
# MAGIC
# MAGIC An example component and description could be;
# MAGIC
# MAGIC * **Data Retrieval Component:** Retrieves customer and product data from the company’s database.
# MAGIC
# MAGIC * **Search Past Customer Reviews Component:** Analyzes customer reviews to find similar complaints using natural language processing.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Draw Architectural Diagram
# MAGIC
# MAGIC Begin by selecting a **diagramming tool** such as [draw.io](https://draw.io), or any other tool that you feel comfortable with. Next, **arrange the components identified in Task 1** on the diagram canvas in a logical sequence based on their data interactions. Connect these components with directional arrows to depict the flow of data and interactions clearly. Each component and connection should be clearly labeled, possibly with brief descriptions if necessary to enhance clarity. 
# MAGIC
# MAGIC Finally, review the diagram to ensure it is easy to understand and accurately represents all components and their interactions within the system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Define Possible Input and Output Parameters for Each Component
# MAGIC
# MAGIC For each component, specify what data it receives (input) and what it sends out (output).
# MAGIC
# MAGIC Example for the Data Retrieval Component:
# MAGIC * Input: Customer ID, Product ID
# MAGIC * Output: Customer purchase history, Product details, Previous complaints

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Define Libraries or Frameworks for Each Component
# MAGIC
# MAGIC For this task, you will need to select appropriate libraries or frameworks that will be utilized to build each component of the system. For retrival and generation tasks, identify the type of the language model that need to be used.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab activity, you designed a system architecture to handle customer complaints, focusing on defining components, creating an architectural diagram, specifying data parameters, and selecting appropriate technologies. The architectural diagram helped visualize component interactions, while the choice of technologies ensured efficient processing of customer complaints.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>