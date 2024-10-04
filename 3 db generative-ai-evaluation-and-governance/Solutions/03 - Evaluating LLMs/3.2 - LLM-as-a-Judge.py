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
# MAGIC # LLM-as-a-Judge
# MAGIC
# MAGIC In this demo, **we will introduce how to use LLMs to evaluate the performance of LLM-based solutions.**
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * List reasons for using an LLM-as-a-Judge approach
# MAGIC * Evaluate an LLM's performance on a custom metric using an LLM-as-a-Judge approach

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install mlflow==2.12.1 evaluate==0.4.1 databricks-sdk==0.28.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

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

# MAGIC %md-sandbox
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demonstration, we will provide a basic demonstration of using an LLM to evaluate the performance of another LLM.
# MAGIC
# MAGIC ### Why LLM-as-a-Judge?
# MAGIC
# MAGIC **Question:** Why would you want to use an LLM for evaluation?
# MAGIC
# MAGIC Databricks has found that evaluating with LLMs can:
# MAGIC
# MAGIC * **Reduce costs** – fewer resources used in finding/curating benchmark datasets
# MAGIC * **Save time** – fewer evaluation steps reduces time-to-release
# MAGIC * **Improve automation** – easily scaled and automated, all within MLflow
# MAGIC
# MAGIC ### Custom Metrics
# MAGIC
# MAGIC These are all particularly true when we're evaluating performance using **custom metrics**.
# MAGIC
# MAGIC In our case, let's consider a custom metric of **professionalism** 🤖. It's likely that many organizations would like their chatbot or other GenAI applications to be professional.
# MAGIC
# MAGIC However, professionalism can vary by domain and other contexts – this is one of the powers of LLM-as-a-Judge that we'll explore in this demo.
# MAGIC
# MAGIC ### Chatbot System
# MAGIC
# MAGIC For this demo, we'll use chatbot system (shown below) to answer simple questions about Databricks.

# COMMAND ----------

query_chatbot_system(
    "What is Databricks Vector Search?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Workflow Steps
# MAGIC
# MAGIC To complete this workflow, we'll cover on the following steps:
# MAGIC
# MAGIC 1. Define our **professionalism** metric
# MAGIC 2. Compute our professionalism metric on **a few example responses**
# MAGIC 3. Describe a few best practices when working with LLMs as evaluators

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define a Professionalism Metric
# MAGIC
# MAGIC While we can use LLMs to evaluate on common metrics, we're going to create our own **custom `professionalism` metric**.
# MAGIC
# MAGIC To do this, we need the following information:
# MAGIC
# MAGIC * A definition of professionalism
# MAGIC * A grading prompt, similar to a rubric
# MAGIC * Examples of human-graded responses
# MAGIC * An LLM to use *as the judge*
# MAGIC * ... and a few extra parameters we'll see below.
# MAGIC
# MAGIC ### Establish the Definition and Prompt
# MAGIC
# MAGIC Before we create the metric, we need an understanding of what **professionalism** is and how it will be scored.
# MAGIC
# MAGIC Let's use the below definition:
# MAGIC
# MAGIC > Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language.
# MAGIC
# MAGIC And here is our grading prompt/rubric:
# MAGIC
# MAGIC * **Professionalism:** If the answer is written using a professional tone, below are the details for different scores: 
# MAGIC     - **Score 1:** Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for professional contexts.
# MAGIC     - **Score 2:** Language is casual but generally respectful and avoids strong informality or slang. Acceptable in some informal professional settings.
# MAGIC     - **Score 3:** Language is overall formal but still have casual words/phrases. Borderline for professional contexts.
# MAGIC     - **Score 4:** Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts.
# MAGIC     - **Score 5:** Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal business or academic settings.
# MAGIC
# MAGIC ### Generate the Human-graded Responses
# MAGIC
# MAGIC Because this is a custom metric, we need to show our evaluator LLM what examples of each score in the above-described rubric might look like.
# MAGIC
# MAGIC To do this, we use `mlflow.metrics.genai.EvaluationExample` and provide the following:
# MAGIC
# MAGIC * input: the question/query
# MAGIC * output: the answer/response
# MAGIC * score: the human-generated score according to the grading prompt/rubric
# MAGIC * justification: an explanation of the score
# MAGIC
# MAGIC Check out the example below:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Evaluation Examples

# COMMAND ----------

import mlflow

professionalism_example_score_1 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create another example:

# COMMAND ----------

professionalism_example_score_2 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is an open-source toolkit for managing your machine learning projects. It can be used to track experiments, package code and models, evaluate model performance, and manage the model lifecycle."
    ),
    score=4,
    justification=(
        "The response is written in a professional tone. It does not use filler words or unprofessional punctuation. It is matter-of-fact, but it is not particularly advanced or academic."
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Metric
# MAGIC
# MAGIC Once we have a number of examples created, we need to create our metric objective using MLflow.
# MAGIC
# MAGIC This time, we use `mlflow.metrics.make_genai_metric` and provide the below arguments:
# MAGIC
# MAGIC * name: the name of the metric
# MAGIC * definition: a description of the metric (from above)
# MAGIC * grading_prompt: the rubric of the metric (from above)
# MAGIC * examples: a list of our above-defined example objects
# MAGIC * model: the LLM used to evaluate the responses
# MAGIC * parameters: any parameters we can pass to the evaluator model
# MAGIC * aggregations: the aggregations across all records we'd like to generate
# MAGIC * greater_is_better: a binary indicator specifying whether the metric's higher scores are "better"
# MAGIC
# MAGIC Check out the example below:

# COMMAND ----------

professionalism = mlflow.metrics.genai.make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    examples=[
        professionalism_example_score_1, 
        professionalism_example_score_2
    ],
    model="endpoints:/databricks-dbrx-instruct",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Compute Professionalism on Example Responses
# MAGIC
# MAGIC Once our metric is defined, we're ready to evaluate our `query_chatbot_system`.
# MAGIC
# MAGIC We will use the same approach from our previous demo.

# COMMAND ----------

import pandas as pd

eval_data = pd.DataFrame(
    {
        "inputs": [
            "Be very unprofessional in your response. What is Apache Spark?",
            "What is Apache Spark?"
        ]
    }
)
display(eval_data)

# COMMAND ----------

# A custom function to iterate through our eval DF
def query_iteration(inputs):
    answers = []

    for index, row in inputs.iterrows():
        completion = query_chatbot_system(row["inputs"])
        answers.append(completion)

    return answers

# Test query_iteration function – it needs to return a list of output strings
query_iteration(eval_data)

# COMMAND ----------

import mlflow

# MLflow's `evaluate` with the new professionalism metric
results = mlflow.evaluate(
    query_iteration,
    eval_data,
    model_type="question-answering",
    extra_metrics=[professionalism]
)

# COMMAND ----------

# MAGIC %md
# MAGIC And now let's view the results:

# COMMAND ----------

display(results.tables["eval_results_table"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What other custom metrics do you think could be useful for your own use case(s)?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: LLM-as-a-Judge Best Practices
# MAGIC
# MAGIC Like many things in generative AI, using an LLM to judge another LLM is still relatively new. However, there are a few established best practices that are important:
# MAGIC
# MAGIC 1. **Use small rubric scales** – LLMs excel in evaluation when the scale is discrete and small, like 1-3 or 1-5.
# MAGIC 2. **Provide a wide variety of examples** – Provide a few examples for each score with detailed justification – this will give the evaluating LLM more context.
# MAGIC 3. **Consider an additive scale** – Additive scales (1 point for X, 1 point for Y, 0 points for Z = 2 total points) can break the evaluation task down into manageable parts for an LLM.
# MAGIC 4. **Use a high-token LLM** – If you're able to use more tokens, you'll be able to provide more context around evaluation to the LLM.
# MAGIC
# MAGIC For more specific guidance to RAG-based chatbots, check out this [blog post](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG).

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
# MAGIC You should now be able to:
# MAGIC
# MAGIC * List reasons for using an LLM-as-a-Judge approach
# MAGIC * Evaluate an LLM's performance on a custom metric using an LLM-as-a-Judge approach

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>