import json

notebook_str = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# INT315: Cluster Computing - Academic Task 2 Project\\n",
    "## Project Title: Predicting Employee Salaries Using Apache Spark MLlib & Graph Analytics\\n",
    "\\n",
    "**Course Code:** INT315  \\n",
    "**Course Name:** Cluster Computing  \\n",
    "**Evaluation Rubric:** CO4, CO5, CO6 (Bloom's Level L3, L4 - 30 Marks)  \\n",
    "\\n",
    "---\\n",
    "### Project Objective & Scope\\n",
    "This project implements an end-to-end distributed big-data architecture using **Apache Spark**, **Spark MLlib**, and **Graph Analytics** to model and predict employee salaries. Key requirements covered:\\n",
    "1. **System Architecture Design:** Cluster computing topology using Spark Core, Spark SQL DataFrames, and MLlib Pipeline.\\n",
    "2. **Distributed Data Ingestion & Preprocessing:** Distributed CSV reading, schema inference, data validation, and memory caching.\\n",
    "3. **Graph Analytics:** Enterprise hierarchy modeling, managerial span-of-control, and centrality vs. compensation correlation.\\n",
    "4. **Distributed Feature Engineering & Regression:** Multi-stage Spark MLlib Pipeline using `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`, followed by `LinearRegression`.\\n",
    "5. **Model Evaluation & Visualizations:** Evaluation with Mean Squared Error (MSE), RMSE, MAE, R², and plotting **Years of Experience vs. Predicted Salary**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Environment Setup & SparkSession Initialization\\n",
    "Configure Java 17 and initialize local distributed SparkSession."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "import sys\\n",
    "import time\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "# Set Java 17 environment path\\n",
    "jdk_path = os.path.abspath(r'.\\\\jdk17_extracted\\\\jdk-17.0.14+7')\\n",
    "if os.path.exists(jdk_path):\\n",
    "    os.environ['JAVA_HOME'] = jdk_path\\n",
    "    os.environ['PATH'] = os.path.join(jdk_path, 'bin') + ';' + os.environ.get('PATH', '')\\n",
    "\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import networkx as nx\\n",
    "\\n",
    "from pyspark.sql import SparkSession\\n",
    "from pyspark.sql import functions as F\\n",
    "from pyspark.ml import Pipeline\\n",
    "from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler\\n",
    "from pyspark.ml.regression import LinearRegression\\n",
    "from pyspark.ml.evaluation import RegressionEvaluator\\n",
    "\\n",
    "# Initialize Spark Session\\n",
    "spark = SparkSession.builder \\\\\\n",
    "    .appName('EmployeeSalaryPrediction_INT315') \\\\\\n",
    "    .master('local[*]') \\\\\\n",
    "    .config('spark.driver.memory', '2g') \\\\\\n",
    "    .getOrCreate()\\n",
    "\\n",
    "spark.sparkContext.setLogLevel('ERROR')\\n",
    "print(f'Apache Spark initialized successfully! Version: {spark.version}')\\n",
    "print(f'Master Node: {spark.sparkContext.master} | Total Parallel Cores: {spark.sparkContext.defaultParallelism}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Distributed Data Ingestion & Preprocessing\\n",
    "Load the employee dataset into a distributed PySpark DataFrame, verify schema, and inspect summary statistics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = 'employee_salaries.csv'\\n",
    "df = spark.read.csv(data_path, header=True, inferSchema=True)\\n",
    "\\n",
    "print(f'Total Distributed Records: {df.count()}')\\n",
    "print(f'Number of Partitions: {df.rdd.getNumPartitions()}')\\n",
    "\\n",
    "print('\\\\n--- Dataset Schema ---')\\n",
    "df.printSchema()\\n",
    "\\n",
    "print('\\\\n--- First 5 Records ---')\\n",
    "df.show(5, truncate=False)\\n",
    "\\n",
    "# Check for missing/null values across columns\\n",
    "null_counts = {c: df.filter(F.col(c).isNull()).count() for c in df.columns}\\n",
    "print(f'Null Check: {null_counts}')\\n",
    "\\n",
    "# Statistical distribution\\n",
    "df.select('years_experience', 'salary').describe().show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Graph Analytics (Enterprise Hierarchy & Collaboration Modeling)\\n",
    "We model organizational relationships (employee-manager reporting lines and project collaboration networks) to analyze topological centrality and its relationship with compensation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pdf = df.toPandas()\\n",
    "\\n",
    "# Construct Directed Organizational Graph\\n",
    "G_org = nx.DiGraph()\\n",
    "for _, row in pdf.iterrows():\\n",
    "    G_org.add_node(row['emp_id'], title=row['job_title'], salary=row['salary'], exp=row['years_experience'])\\n",
    "    if row['manager_id'] and row['manager_id'] != 'EMP0000':\\n",
    "        G_org.add_edge(row['manager_id'], row['emp_id'])\\n",
    "\\n",
    "out_degrees = dict(G_org.out_degree())\\n",
    "pdf['direct_reports'] = pdf['emp_id'].map(out_degrees).fillna(0)\\n",
    "\\n",
    "corr = pdf['direct_reports'].corr(pdf['salary'])\\n",
    "print(f'Correlation between Managerial Centrality (Reports Managed) and Salary: {corr:.4f}')\\n",
    "\\n",
    "# Visualize Subgraph\\n",
    "sample_nodes = ['EMP0001']\\n",
    "for n1 in list(G_org.successors('EMP0001'))[:6]:\\n",
    "    sample_nodes.append(n1)\\n",
    "    sample_nodes.extend(list(G_org.successors(n1))[:3])\\n",
    "\\n",
    "subgraph = G_org.subgraph(sample_nodes)\\n",
    "pos = nx.spring_layout(subgraph, seed=42)\\n",
    "node_salaries = [subgraph.nodes[n].get('salary', 60000) for n in subgraph.nodes()]\\n",
    "\\n",
    "plt.figure(figsize=(9, 6), dpi=150)\\n",
    "sc = nx.draw_networkx_nodes(subgraph, pos, node_size=350, node_color=node_salaries, cmap='viridis', alpha=0.9)\\n",
    "nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowsize=12, edge_color='#64748b', alpha=0.6)\\n",
    "nx.draw_networkx_labels(subgraph, pos, font_size=7)\\n",
    "plt.colorbar(sc, label='Salary ($USD)')\\n",
    "plt.title('Enterprise Reporting Structure (Graph Analytics)')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Machine Learning Pipeline (Spark MLlib)\\n",
    "Encode categorical attributes (`job_title`, `department`, `education`) using `StringIndexer` and `OneHotEncoder`, assemble features with `VectorAssembler`, and train `LinearRegression`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Index categorical columns\\n",
    "indexer_title = StringIndexer(inputCol='job_title', outputCol='title_idx', handleInvalid='keep')\\n",
    "indexer_dept = StringIndexer(inputCol='department', outputCol='dept_idx', handleInvalid='keep')\\n",
    "indexer_edu = StringIndexer(inputCol='education', outputCol='edu_idx', handleInvalid='keep')\\n",
    "\\n",
    "# One-Hot Encode indexed columns\\n",
    "encoder = OneHotEncoder(\\n",
    "    inputCols=['title_idx', 'dept_idx', 'edu_idx'],\\n",
    "    outputCols=['title_vec', 'dept_vec', 'edu_vec'],\\n",
    "    dropLast=False\\n",
    ")\\n",
    "\\n",
    "# Combine into feature vector\\n",
    "assembler = VectorAssembler(\\n",
    "    inputCols=['years_experience', 'title_vec', 'dept_vec', 'edu_vec'],\\n",
    "    outputCol='features'\\n",
    ")\\n",
    "\\n",
    "# Define Linear Regression\\n",
    "lr = LinearRegression(featuresCol='features', labelCol='salary', maxIter=100, regParam=0.01)\\n",
    "\\n",
    "# Build Spark Pipeline\\n",
    "pipeline = Pipeline(stages=[indexer_title, indexer_dept, indexer_edu, encoder, assembler, lr])\\n",
    "\\n",
    "# 80/20 Train/Test Split\\n",
    "train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)\\n",
    "print(f'Training Set: {train_df.count()} | Testing Set: {test_df.count()}')\\n",
    "\\n",
    "# Train Pipeline\\n",
    "pipeline_model = pipeline.fit(train_df)\\n",
    "print('Spark MLlib Pipeline Training Complete!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Model Evaluation (MSE, RMSE, MAE, R²)\\n",
    "Evaluate the model on unseen test data using `RegressionEvaluator`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = pipeline_model.transform(test_df).cache()\\n",
    "\\n",
    "eval_mse = RegressionEvaluator(labelCol='salary', predictionCol='prediction', metricName='mse')\\n",
    "eval_rmse = RegressionEvaluator(labelCol='salary', predictionCol='prediction', metricName='rmse')\\n",
    "eval_mae = RegressionEvaluator(labelCol='salary', predictionCol='prediction', metricName='mae')\\n",
    "eval_r2 = RegressionEvaluator(labelCol='salary', predictionCol='prediction', metricName='r2')\\n",
    "\\n",
    "mse = eval_mse.evaluate(predictions)\\n",
    "rmse = eval_rmse.evaluate(predictions)\\n",
    "mae = eval_mae.evaluate(predictions)\\n",
    "r2 = eval_r2.evaluate(predictions)\\n",
    "\\n",
    "print('=' * 50)\\n",
    "print('           MODEL PERFORMANCE EVALUATION')\\n",
    "print('=' * 50)\\n",
    "print(f'Mean Squared Error (MSE)       : {mse:,.2f}')\\n",
    "print(f'Root Mean Squared Error (RMSE) : ${rmse:,.2f}')\\n",
    "print(f'Mean Absolute Error (MAE)      : ${mae:,.2f}')\\n",
    "print(f'Coefficient of Determination R²: {r2:.4f} ({r2*100:.2f}%)')\\n",
    "print('=' * 50)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Plotting Experience vs. Predicted Salary\\n",
    "Visualize the relationship between employee experience and both actual and predicted salaries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_pdf = predictions.select('years_experience', 'salary', 'prediction', 'job_title').toPandas()\\n",
    "\\n",
    "plt.figure(figsize=(9, 6), dpi=150)\\n",
    "plt.scatter(pred_pdf['years_experience'], pred_pdf['salary'], color='#2563eb', alpha=0.55, label='Actual Salary', s=35)\\n",
    "plt.scatter(pred_pdf['years_experience'], pred_pdf['prediction'], color='#dc2626', alpha=0.65, label='Predicted Salary (Spark MLlib)', s=25, marker='x')\\n",
    "\\n",
    "# Fit trend line\\n",
    "z = np.polyfit(pred_pdf['years_experience'], pred_pdf['prediction'], 1)\\n",
    "p = np.poly1d(z)\\n",
    "exp_range = np.linspace(pred_pdf['years_experience'].min(), pred_pdf['years_experience'].max(), 100)\\n",
    "plt.plot(exp_range, p(exp_range), color='#991b1b', linestyle='--', linewidth=2, label='Fitted Regression Trend')\\n",
    "\\n",
    "plt.xlabel('Years of Experience', fontsize=12, fontweight='bold')\\n",
    "plt.ylabel('Salary ($USD)', fontsize=12, fontweight='bold')\\n",
    "plt.title('Experience vs. Salary: Actual vs. Spark MLlib Linear Regression Prediction', fontsize=13, fontweight='bold')\\n",
    "plt.legend(frameon=True, facecolor='white')\\n",
    "plt.grid(True, linestyle=':', alpha=0.6)\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Parity Plot & Residual Error Distribution\\n",
    "Examine prediction deviations and confirm Gaussian normal distribution of errors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_pdf['residual'] = pred_pdf['salary'] - pred_pdf['prediction']\\n",
    "\\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)\\n",
    "\\n",
    "# Parity plot\\n",
    "ax1.scatter(pred_pdf['salary'], pred_pdf['prediction'], color='#059669', alpha=0.55, s=25)\\n",
    "min_val = min(pred_pdf['salary'].min(), pred_pdf['prediction'].min())\\n",
    "max_val = max(pred_pdf['salary'].max(), pred_pdf['prediction'].max())\\n",
    "ax1.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='y = x Line')\\n",
    "ax1.set_xlabel('Actual Salary ($USD)', fontweight='bold')\\n",
    "ax1.set_ylabel('Predicted Salary ($USD)', fontweight='bold')\\n",
    "ax1.set_title(f'Actual vs. Predicted Parity (R² = {r2:.4f})', fontweight='bold')\\n",
    "ax1.legend()\\n",
    "ax1.grid(True, linestyle=':', alpha=0.5)\\n",
    "\\n",
    "# Residual distribution\\n",
    "sns.histplot(pred_pdf['residual'], kde=True, color='#6366f1', ax=ax2, bins=30)\\n",
    "ax2.axvline(0, color='red', linestyle='--')\\n",
    "ax2.set_xlabel('Residual Error ($)', fontweight='bold')\\n",
    "ax2.set_ylabel('Count', fontweight='bold')\\n",
    "ax2.set_title('Residual Error Distribution (Normality Check)', fontweight='bold')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 8: Clean Up Resources\\n",
    "Stop SparkSession."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "spark.stop()\\n",
    "print('SparkSession stopped successfully!')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}"""

with open("Employee_Salary_Prediction.ipynb", "w", encoding="utf-8") as f:
    f.write(notebook_str)

print("Jupyter Notebook 'Employee_Salary_Prediction.ipynb' generated successfully!")
