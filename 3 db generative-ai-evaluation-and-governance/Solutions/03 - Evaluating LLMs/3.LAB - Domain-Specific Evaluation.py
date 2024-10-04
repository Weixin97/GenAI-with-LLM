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
# MAGIC # LAB - Domain-Specific Evaluation 
# MAGIC
# MAGIC
# MAGIC In this lab, you will have the opportunity to evaluate a large language model on a specific task **using a dataset designed for this exact evaluation.**
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will need to complete the following tasks:*
# MAGIC
# MAGIC - **Task 1:** Create a Benchmark Dataset
# MAGIC - **Task 2:** Compute ROUGE on Custom Benchmark Data
# MAGIC - **Task 3:** Use an LLM-as-a-Judge approach to evaluate custom metrics

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

# MAGIC %pip install mlflow==2.12.1 evaluate==0.4.1 databricks-sdk==0.28.0 rouge_score
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

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

# MAGIC %md-sandbox
# MAGIC # Lab Overview
# MAGIC
# MAGIC In this lab, you will again be evaluating the performance of an AI system designed to summarize text.

# COMMAND ----------

query_product_summary_system(
    "This is the best frozen pizza I've ever had! Sure, it's not the healthiest, but it tasted just like it was delivery from our favorite pizzeria down the street. The cheese browned nicely and fresh tomatoes are a nice touch, too! I would buy it again despite it's high price. If I could change one thing, I'd made it a little healthier – could we get a gluten-free crust option? My son would love that."
)

# COMMAND ----------

# MAGIC %md
# MAGIC However, you will evaluate the LLM using a curated benchmark set specific to our evaluation.
# MAGIC
# MAGIC This lab will follow the below steps:
# MAGIC
# MAGIC 1. Create a custom benchmark dataset specific to the use case
# MAGIC 2. Compute summarization-specific evaluation metrics using the custom benchmark data set
# MAGIC 3. Use an LLM-as-a-Judge approach to evaluate custom metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Create a Benchmark Dataset
# MAGIC
# MAGIC Recall that ROUGE requires reference sets to compute scores. In our demo, we used a large, generic benchmark set.
# MAGIC
# MAGIC In this lab, you have to use a domain-specific benchmark set specific to the use case.
# MAGIC
# MAGIC ### Case-Specific Benchmark Set
# MAGIC
# MAGIC While the base-specific data set likely won't be as large, it does have the advantage of being **more representative of the task we're actually asking the LLM to perform.**
# MAGIC
# MAGIC Below, we've started to create a dataset for grocery product review summaries. It's your task to create **two more** product summaries to this dataset.
# MAGIC
# MAGIC **Hint:** Try opening up another tab and using AI Playground to generate some examples! Just be sure to manually check them since this is our *ground-truth evaluation data*.
# MAGIC
# MAGIC **Note:** For this task, we're creating an *extremely small* reference set. In practice, you'll want to create one with far more example records.

# COMMAND ----------


import pandas as pd

eval_data = pd.DataFrame(
    {
        "inputs": [
            "This coffee is exceptional. Its intensely bold flavor profile is both nutty and fruity – especially with notes of plum and citrus. While the price is relatively good, I find myself needing to purchase bags too often. If this came in 16oz bags instead of just 12oz bags, I'd purchase it all the time. I highly recommend they start scaling up their bag size.",
            "The moment I opened the tub of Chocolate-Covered Strawberry Delight ice cream, I was greeted by the enticing aroma of fresh strawberries and rich chocolate. The appearance of the ice cream was equally appealing, with a swirl of pink strawberry ice cream and chunks of chocolate-covered strawberries scattered throughout. The first bite did not disappoint. The strawberry ice cream was creamy and flavorful, with a natural sweetness that was not overpowering. The chocolate-covered strawberries added a satisfying crunch fruity bite.",
            "Arroz Delicioso is a must-try for Mexican cuisine enthusiasts! This authentic Mexican rice, infused with a blend of tomatoes, onions, and garlic, brings a burst of flavor to any meal. Its vibrant color and delightful aroma will transport you straight to the heart of Mexico. The rice cooks evenly, resulting in separate, fluffy grains that hold their shape, making it perfect for dishes like arroz con pollo or as a side for tacos. With a cook time of just 20 minutes, Arroz Delicioso is a convenient and delicious addition to your pantry. Give it a try and elevate your Mexican food game!",
            "FreshCrunch salad mixes are revolutionizing the way we think about packaged salads! Each bag is packed with a vibrant blend of crisp, nutrient-rich greens, including baby spinach, arugula, and kale. The veggies are pre-washed and ready to eat, making meal prep a breeze. FreshCrunch sets itself apart with its innovative packaging that keeps the greens fresh for up to 10 days, reducing food waste and ensuring you always have a healthy option on hand. The salad mixes are versatile and pair well with various dressings and toppings. Try FreshCrunch for a convenient, delicious, and nutritious meal solution that doesn't compromise on quality or taste!",
            "If you're a grill enthusiast like me, you know the importance of having the right tools for the job. That's why I was thrilled to get my hands on the new Click-Clack Grill Tongs. These tongs are not just any ordinary grilling utensil; they're a game-changer. First impressions matter, and the Click-Clack Grill Tongs certainly deliver. The sleek, stainless steel design exudes a professional feel, and the ergonomic handle ensures a comfortable grip even during those long grilling sessions. But what truly sets these tongs apart is their innovative 'Click-Clack' mechanism. With a simple press of a button, the tongs automatically open and close, allowing for precise control when flipping or turning your food. No more struggling with stiff, unwieldy tongs that can ruin your carefully prepared meals. The tongs also feature a scalloped edge, which provides a secure grip on everything from juicy steaks to delicate vegetables. And with their generous length, you can keep your hands safely away from the heat while still maintaining optimal control. Cleanup is a breeze thanks to the dishwasher-safe construction, and the integrated hanging loop makes storage a snap. In conclusion, the Click-Clack Grill Tongs have earned a permanent spot in my grilling arsenal. They've made my grilling experience more enjoyable and efficient, and I'm confident they'll do the same for you. So, if you're looking to up your grilling game, I highly recommend giving these tongs a try. Happy grilling!",
            "As a parent, I understand the importance of providing my child with nutritious, wholesome food. That's why I was thrilled to discover Fresh 'n' Quik Baby Food, a new product that promises to deliver fresh, homemade baby food in minutes. The concept behind Fresh 'n' Quik is simple yet ingenious. The system consists of pre-portioned, organic fruit and vegetable purees that can be quickly and easily blended with breast milk, formula, or water to create a nutritious meal for your little one. The purees are made with high-quality ingredients, free from additives, preservatives, and artificial flavors, ensuring that your baby receives only the best. One of the standout features of Fresh 'n' Quik is the convenience it offers. The purees come in individual, resealable pouches that can be stored in the freezer until you're ready to use them. When it's time to feed your baby, simply pop a pouch into the Fresh 'n' Quik blender, add your liquid of choice, and blend. In less than a minute, you have a fresh, homemade meal that's ready to serve. The blender itself is compact, easy to use, and even easier to clean. The blades are removable, making it a breeze to rinse off any leftover puree. And the best part? The blender is whisper-quiet, so you don't have to worry about waking your sleeping baby while preparing their meal. But what truly sets Fresh 'n' Quik apart is the variety of flavors available. From classic combinations like apple and banana to more adventurous options like mango and kale, there's something for every palate. And because the purees are made with real fruits and vegetables, your baby is exposed to a wide range of flavors and textures, helping to cultivate a diverse and adventurous palate from an early age. In conclusion, Fresh 'n' Quik Baby Food is a game-changer for parents seeking a convenient, nutritious, and delicious option for their little ones. The system is easy to use, quick to clean, and offers a wide variety of flavors to keep your baby's taste buds excited. I highly recommend giving Fresh 'n' Quik a try – your baby (and your schedule) will thank you!"
        ],
        "ground_truth": [
            "This bold, nutty, and fruity coffee is delicious, and they need to start selling it in larger bags.",
            "Chocolate-Covered Strawberry Delight ice cream looks delicious with its aroma of strawberry and chocolate, and its creamy, naturally sweet taste did not disappoint.",
            "Arroz Delicioso offers authentic, flavorful Mexican rice with a blend of tomatoes, onions, and garlic, cooking evenly into separate, fluffy grains in just 20 minutes, making it a convenient and delicious choice for dishes like arroz con pollo or as a side for tacos.",
            "FreshCrunch salad mixes offer convenient, pre-washed, nutrient-rich greens in an innovative packaging that keeps them fresh for up to 10 days, providing a versatile, tasty, and waste-reducing healthy meal solution.",
            "The Click-Clack Grill Tongs are a high-quality, innovative grilling tool with a sleek design, comfortable grip, and an automatic opening/closing mechanism for precise control. These tongs have made grilling more enjoyable and efficient, and are highly recommended for anyone looking to improve their grilling experience.",
            "Fresh 'n' Quik Baby Food is a revolutionary product that delivers fresh, homemade baby food in minutes. With pre-portioned, organic fruit and vegetable purees, the system offers convenience, high-quality ingredients, and a wide range of flavors to cultivate a diverse palate in your little one. The blender is compact, easy to use, and whisper-quiet, making mealtime a breeze. Fresh 'n' Quik Baby Food is a must-try for parents seeking a nutritious and delicious option for their babies."
        ],
    }
)

display(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What are some strategies for evaluating your custom-generated benchmark data set? For example:
# MAGIC * How can you scale the curation?
# MAGIC * How do you know if the ground truth is correct?
# MAGIC * Who should have input?
# MAGIC * Should it remain static over time?
# MAGIC
# MAGIC Next, we're saving this reference data set for future use.

# COMMAND ----------

spark_df = spark.createDataFrame(eval_data)
spark_df.write.mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.case_spec_summ_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Compute ROUGE on Custom Benchmark Data
# MAGIC
# MAGIC Next, we will want to compute our ROUGE-N metric to understand how well our system summarizes grocery product reviews based on the reference of reviews that was just created.
# MAGIC
# MAGIC Remember that the `mlflow.evaluate` function accepts the following parameters for this use case:
# MAGIC
# MAGIC * An LLM model
# MAGIC * Reference data for evaluation
# MAGIC * Column with ground truth data
# MAGIC * The model/task type (e.g. `"text-summarization"`)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2.1: Run the Evaluation
# MAGIC
# MAGIC Instead of using the generic benchmark dataset like in the demo, your task is to **compute ROUGE metrics using the case-specific benchmark data that we just created.**
# MAGIC
# MAGIC **Note:** If needed, refer back to the demo to complete the below code blocks.
# MAGIC
# MAGIC First, the function that you can use to iterate through rows for `mlflow.evaluate`.

# COMMAND ----------


# A custom function to iterate through our eval DF
def query_iteration(inputs):
    answers = []

    for index, row in inputs.iterrows():
        completion = query_product_summary_system(row["inputs"])
        answers.append(completion)

    return answers

# Test query_iteration function – it needs to return a list of output strings
query_iteration(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, use the above function and `mlflow.evaluate` to perform the `text-summarization` evaluation.

# COMMAND ----------


import mlflow

# MLflow's `evaluate` with a custom function
results = mlflow.evaluate(
    query_iteration,                      # iterative function from above
    eval_data,                            # eval DF
    targets="ground_truth",               # column with expected or "good" output
    model_type="text-summarization"       # type of model or task
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2.2: Evaluate the Results
# MAGIC
# MAGIC Next, take a look at the results.

# COMMAND ----------


display(results.tables["eval_results_table"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** How do we interpret these results? What does it tell us about the summarization quality? About our LLM?
# MAGIC
# MAGIC Next, compute the summarized metrics to view the performance of the LLM on the entire dataset.

# COMMAND ----------

results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC **Bonus:** Take a look at the results in the Experiment Tracking UI.
# MAGIC
# MAGIC Do you see any summaries that you think are particularly good or problematic?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Use an LLM-as-a-Judge Approach to Evaluate Custom Metrics
# MAGIC
# MAGIC In this task, you will define and evaluate a custom metric called "professionalism" using an LLM-as-a-Judge approach. The goal is to assess how professionally written the summaries generated by the language model are, based on a set of predefined criteria.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3.1: Define a Humor Metric
# MAGIC
# MAGIC - Define humor and create a grading prompt.
# MAGIC
# MAGIC   **To Do:**
# MAGIC
# MAGIC   For this task, you are provided with an initial example of humor (humor_example_score_1). Your task is to generate another evaluation example (humor_example_score_2). 
# MAGIC
# MAGIC   **Hint:** You can use AI Playground for this. Ensure that the generated example is relevant to the prompt and reflects a different humor score. Manually verify the generated example to ensure its correctness.
# MAGIC

# COMMAND ----------

# Define an evaluation example for humor with a score of 2
humor_example_score_1 = mlflow.metrics.genai.EvaluationExample(
    input="Tell me a joke!",  
    output=(
        "Why don't scientists trust atoms? Because they make up everything!"  
    ),
    score=2,  # Humor score assigned to the output
    justification=(
        "The joke uses a common pun and is somewhat humorous, but it may not elicit strong laughter or amusement from everyone."  # Justification for the assigned score
    ),
)

# Define another evaluation example for humor with a score of 4
humor_example_score_2 = mlflow.metrics.genai.EvaluationExample(
    input="Tell me a joke!",  
    output=(
        "I told my wife she was drawing her eyebrows too high. She looked surprised!"  
    ),
    score=4,  # Humor score assigned to the output
    justification=(
        "The joke is clever and unexpected, resulting in genuine amusement and laughter. It demonstrates wit and creativity, making it highly enjoyable."  # Justification for the assigned score
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3.2: LLM-as-a-Judge to Compare Metric
# MAGIC
# MAGIC * **3.2.1:  Create a metric for comparing the responses for humor**
# MAGIC
# MAGIC   Define a custom metric to evaluate the humor in generated responses. This metric will assess the level of humor present in the responses generated by the language model.

# COMMAND ----------

# Define the metric for the evaluation
comparison_humor_metric = mlflow.metrics.genai.make_genai_metric(
    name="comparison_humor",
    definition=(
        "Humor refers to the ability to evoke laughter, amusement, or enjoyment through cleverness, wit, or unexpected twists."
    ),
    grading_prompt=(
        "Humor: If the response is funny and induces laughter or amusement, below are the details for different scores: "
        "- Score 1: The response attempts humor but falls flat, eliciting little to no laughter or amusement."
        "- Score 2: The response is somewhat humorous, eliciting mild laughter or amusement from some individuals."
        "- Score 3: The response is moderately funny, eliciting genuine laughter or amusement from most individuals."
        "- Score 4: The response is highly humorous, eliciting strong laughter or amusement from nearly everyone."
        "- Score 5: The response is exceptionally funny, resulting in uncontrollable laughter or intense enjoyment."
    ),
    # Examples for humor
    examples=[
        humor_example_score_1, 
        humor_example_score_2
    ],
    model="endpoints:/databricks-dbrx-instruct",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC * **3.2.3: Generate data with varying humor levels**
# MAGIC
# MAGIC   Add input prompts and corresponding output responses with different levels of humor. 
# MAGIC   
# MAGIC   **Hint:** You can utilize AI playgrounds to generate these values.

# COMMAND ----------

# Define testing data with different humor scores for comparison
humor_data = pd.DataFrame(
    {
        "inputs": [
            "Tell me a joke about pandas.",
            "What's a programmer's favorite place to hang out?",
            "Why don't scientists trust atoms?",
            "Why did the scarecrow win an award?"
        ],
        "ground_truth": [
            "Why did the pandas break up? Because they couldn't bamboo-zle their problems away!",
            "The Foo Bar!",
            "Because they make up everything!",
            "Because he was outstanding in his field!"
        ],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC * **3.2.4: Evaluate the Comparison**
# MAGIC
# MAGIC   Next, evaluate the comparison between the responses generated by the language model. This evaluation will provide you with a metric for assessing the professionalism of the generated summaries based on predefined criteria.
# MAGIC

# COMMAND ----------

benchmark_comparison_results = mlflow.evaluate(
    model="endpoints:/databricks-dbrx-instruct",  # Model used for evaluation
    data=humor_data,                               # Data for evaluation
    targets="ground_truth",                       # Column with the ground truth data
    model_type="text-summarization",              # Type of model or task
    custom_metrics=[comparison_humor_metric],  # Custom metric for evaluating professionalism
)

# COMMAND ----------

# MAGIC %md
# MAGIC * **3.2.5: View Comparison Results**
# MAGIC
# MAGIC   Now, let's take a look at the results of the comparison between the responses generated by the language model. This comparison provides insights into the professionalism of the generated summaries based on the predefined criteria.
# MAGIC

# COMMAND ----------

display(benchmark_comparison_results.tables["eval_results_table"])

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
# MAGIC This lab provided a hands-on experience in creating and evaluating a custom benchmark dataset, computing task-specific evaluation metrics, and leveraging an LLM-as-a-Judge approach to assess custom metrics. These techniques are essential for evaluating the performance of AI systems in domain-specific tasks and ensuring their effectiveness in real-world applications.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>