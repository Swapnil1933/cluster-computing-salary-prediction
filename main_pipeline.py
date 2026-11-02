import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

# 1. Configure Java 17 for PySpark
jdk_path = os.path.abspath(r".\jdk17_extracted\jdk-17.0.14+7")
if os.path.exists(jdk_path):
    os.environ['JAVA_HOME'] = jdk_path
    os.environ['PATH'] = os.path.join(jdk_path, 'bin') + ';' + os.environ.get('PATH', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def run_project():
    print("=" * 80)
    print("  INT315: CLUSTER COMPUTING PROJECT")
    print("  Topic: Predicting Employee Salaries Using Apache Spark MLlib & Graph Analytics")
    print("=" * 80)

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Initialize Distributed SparkSession
    # ---------------------------------------------------------
    print("\n[Step 1] Initializing Apache Spark Distributed Environment...")
    t0 = time.time()
    spark = SparkSession.builder \
        .appName("EmployeeSalaryPrediction_INT315") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print(f"SparkSession created successfully! Version: {spark.version} (Time: {time.time()-t0:.2f}s)")
    print(f"Master: {spark.sparkContext.master} | Default Parallelism: {spark.sparkContext.defaultParallelism}")

    # ---------------------------------------------------------
    # Step 2: Data Ingestion & Preprocessing
    # ---------------------------------------------------------
    dataset_path = "employee_salaries.csv"
    print(f"\n[Step 2] Ingesting distributed dataset from '{dataset_path}'...")
    df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    
    total_records = df.count()
    num_partitions = df.rdd.getNumPartitions()
    print(f"Loaded {total_records} records across {num_partitions} distributed partitions.")
    
    print("\nDataset Schema:")
    df.printSchema()

    print("\nSample Data (First 5 Rows):")
    df.show(5, truncate=False)

    # Preprocessing: Check null values
    print("Data Preprocessing & Validation:")
    null_counts = {col: df.filter(F.col(col).isNull()).count() for col in df.columns}
    print(f"Null values check: {null_counts}")
    
    # Cache DataFrame in distributed memory
    df = df.cache()
    df.count()

    # Summary Statistics for numerical attributes
    print("\nStatistical Summary (Experience & Salary):")
    df.select("years_experience", "salary").describe().show()

    # ---------------------------------------------------------
    # Step 3: Graph Analytics (GraphX / Relationship Analysis)
    # ---------------------------------------------------------
    print("\n[Step 3] Graph Analytics: Enterprise Hierarchy & Collaboration Modeling...")
    # Analyzing organizational reporting hierarchy: employee -> manager
    # And project collaboration edges: employees working on the same project
    pdf = df.toPandas()

    # Build Organizational Hierarchy Directed Graph
    G_org = nx.DiGraph()
    for _, row in pdf.iterrows():
        G_org.add_node(row['emp_id'], title=row['job_title'], salary=row['salary'], exp=row['years_experience'])
        if row['manager_id'] and row['manager_id'] != 'EMP0000':
            # Edge from manager to subordinate
            G_org.add_edge(row['manager_id'], row['emp_id'])

    in_degrees = dict(G_org.in_degree()) # incoming edges (for subordinates)
    out_degrees = dict(G_org.out_degree()) # outgoing edges = direct reports managed!
    
    top_managers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Managers by Span of Control (Number of Direct Reports):")
    for mgr, cnt in top_managers:
        mgr_title = pdf[pdf['emp_id'] == mgr]['job_title'].values
        mgr_title_str = mgr_title[0] if len(mgr_title) > 0 else "Executive"
        print(f"  - Manager {mgr} ({mgr_title_str}): manages {cnt} team members")

    # Graph Metric: Correlation between managerial span and salary
    pdf['direct_reports'] = pdf['emp_id'].map(out_degrees).fillna(0)
    corr_reports_salary = pdf['direct_reports'].corr(pdf['salary'])
    print(f"Correlation between Organizational Centrality (Direct Reports) and Salary: {corr_reports_salary:.4f}")

    # Plot Organizational Subgraph
    plt.figure(figsize=(10, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Sample a manageable 40-node ego network of top director for visual clarity
    sample_nodes = ['EMP0001']
    for n1 in list(G_org.successors('EMP0001'))[:8]:
        sample_nodes.append(n1)
        sample_nodes.extend(list(G_org.successors(n1))[:3])
    
    subgraph = G_org.subgraph(sample_nodes)
    pos = nx.spring_layout(subgraph, seed=42)
    node_salaries = [subgraph.nodes[n].get('salary', 60000) for n in subgraph.nodes()]
    
    sc = nx.draw_networkx_nodes(subgraph, pos, node_size=350, node_color=node_salaries,
                               cmap='viridis', alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, arrows=True, arrowsize=12, edge_color='#64748b', alpha=0.6, width=1.2)
    nx.draw_networkx_labels(subgraph, pos, font_size=7, font_family='sans-serif', font_color='black')
    cbar = plt.colorbar(sc, label="Annual Salary ($USD)")
    plt.title("Graph Analysis: Organizational Hierarchy Subgraph (Reporting Structure)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("plots/graph_hierarchy.png")
    plt.close()
    print("Saved Graph Analytics Visualization -> plots/graph_hierarchy.png")

    # ---------------------------------------------------------
    # Step 4: Machine Learning Pipeline (Spark MLlib)
    # ---------------------------------------------------------
    print("\n[Step 4] Building Spark MLlib Regression Pipeline...")
    print("Features to encode:")
    print("  - Categorical: 'job_title', 'department', 'education'")
    print("  - Continuous:  'years_experience'")
    print("  - Target Label: 'salary'")

    # 4.1 StringIndexer for categorical features
    indexer_title = StringIndexer(inputCol="job_title", outputCol="title_idx", handleInvalid="keep")
    indexer_dept = StringIndexer(inputCol="department", outputCol="dept_idx", handleInvalid="keep")
    indexer_edu = StringIndexer(inputCol="education", outputCol="edu_idx", handleInvalid="keep")

    # 4.2 OneHotEncoder for indexed categorical variables
    encoder = OneHotEncoder(
        inputCols=["title_idx", "dept_idx", "edu_idx"],
        outputCols=["title_vec", "dept_vec", "edu_vec"],
        dropLast=False
    )

    # 4.3 VectorAssembler to combine all feature vectors
    assembler = VectorAssembler(
        inputCols=["years_experience", "title_vec", "dept_vec", "edu_vec"],
        outputCol="features"
    )

    # 4.4 Linear Regression Model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="salary",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )

    # 4.5 Assemble into a Spark ML Pipeline
    pipeline = Pipeline(stages=[indexer_title, indexer_dept, indexer_edu, encoder, assembler, lr])

    # 4.6 Train/Test Split (80% Train, 20% Test)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"Data Split: Training set = {train_count} rows | Testing set = {test_count} rows")

    # 4.7 Train Model
    t_start_train = time.time()
    pipeline_model = pipeline.fit(train_df)
    train_time = time.time() - t_start_train
    print(f"Spark MLlib Pipeline trained successfully in {train_time:.3f} seconds!")

    # Extract Model Parameters
    lr_model = pipeline_model.stages[-1]
    print(f"Model Intercept: ${lr_model.intercept:,.2f}")
    print(f"Number of Feature Weights: {len(lr_model.coefficients)}")

    # ---------------------------------------------------------
    # Step 5: Model Evaluation on Test Set
    # ---------------------------------------------------------
    print("\n[Step 5] Evaluating Model Performance on Test Data...")
    predictions = pipeline_model.transform(test_df)
    predictions = predictions.cache()

    # Evaluators
    eval_mse = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="mse")
    eval_rmse = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="r2")

    mse_val = eval_mse.evaluate(predictions)
    rmse_val = eval_rmse.evaluate(predictions)
    mae_val = eval_mae.evaluate(predictions)
    r2_val = eval_r2.evaluate(predictions)

    print("=" * 60)
    print("  MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 60)
    print(f"  Mean Squared Error (MSE)      : {mse_val:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse_val:,.2f}")
    print(f"  Mean Absolute Error (MAE)     : ${mae_val:,.2f}")
    print(f"  Coefficient of Determination (R²): {r2_val:.4f} ({r2_val*100:.2f}%)")
    print("=" * 60)

    # Save metrics
    metrics = {
        "mse": float(mse_val),
        "rmse": float(rmse_val),
        "mae": float(mae_val),
        "r2": float(r2_val),
        "train_time_sec": float(train_time),
        "train_count": int(train_count),
        "test_count": int(test_count),
        "intercept": float(lr_model.intercept)
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ---------------------------------------------------------
    # Step 6: System Performance & Scalability Experiment
    # ---------------------------------------------------------
    print("\n[Step 6] Running Distributed Scalability Benchmark (Varying Partitions)...")
    partition_configs = [1, 2, 4, 8, 16]
    benchmark_results = []

    for parts in partition_configs:
        repart_df = df.repartition(parts)
        t_bench_start = time.time()
        # Perform distributed aggregation and pipeline fit
        sample_train, sample_test = repart_df.randomSplit([0.8, 0.2], seed=42)
        bench_model = pipeline.fit(sample_train)
        bench_preds = bench_model.transform(sample_test)
        # Action to trigger computation
        bench_count = bench_preds.count()
        t_elapsed = time.time() - t_bench_start
        throughput = bench_count / t_elapsed
        benchmark_results.append({
            "partitions": parts,
            "latency_sec": round(t_elapsed, 3),
            "throughput_rows_sec": round(throughput, 1)
        })
        print(f"  Partitions: {parts:2d} | Latency: {t_elapsed:.3f}s | Throughput: {throughput:.1f} records/sec")

    # ---------------------------------------------------------
    # Step 7: Visualizations & Plotting
    # ---------------------------------------------------------
    print("\n[Step 7] Generating High-Resolution Evaluation Plots...")
    pred_pdf = predictions.select("years_experience", "salary", "prediction", "job_title", "education").toPandas()
    pred_pdf['residual'] = pred_pdf['salary'] - pred_pdf['prediction']

    # 7.1 Plot 1: Experience vs. Salary (Actual vs. Predicted) - REQUIRED TASK
    plt.figure(figsize=(9, 6), dpi=300)
    plt.scatter(pred_pdf['years_experience'], pred_pdf['salary'], color='#2563eb', alpha=0.55, label='Actual Salary', s=35, edgecolors='none')
    plt.scatter(pred_pdf['years_experience'], pred_pdf['prediction'], color='#dc2626', alpha=0.65, label='Predicted Salary (Spark MLlib)', s=25, marker='x')
    
    # Trend line for predicted
    z = np.polyfit(pred_pdf['years_experience'], pred_pdf['prediction'], 1)
    p = np.poly1d(z)
    exp_range = np.linspace(pred_pdf['years_experience'].min(), pred_pdf['years_experience'].max(), 100)
    plt.plot(exp_range, p(exp_range), color='#b91c1c', linestyle='--', linewidth=2, label='Fitted Regression Trend')

    plt.xlabel('Years of Experience', fontsize=12, fontweight='bold', labelpad=8)
    plt.ylabel('Salary ($USD)', fontsize=12, fontweight='bold', labelpad=8)
    plt.title('Experience vs. Salary: Actual vs. Spark MLlib Linear Regression Prediction', fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/experience_vs_predicted_salary.png')
    plt.close()
    print("Saved -> plots/experience_vs_predicted_salary.png")

    # 7.2 Plot 2: Actual vs. Predicted Parity Plot ($y = x$)
    plt.figure(figsize=(8, 7), dpi=300)
    plt.scatter(pred_pdf['salary'], pred_pdf['prediction'], color='#059669', alpha=0.6, s=30, edgecolors='none')
    min_val = min(pred_pdf['salary'].min(), pred_pdf['prediction'].min())
    max_val = max(pred_pdf['salary'].max(), pred_pdf['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d97706', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    plt.xlabel('Actual Salary ($USD)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Salary ($USD)', fontsize=12, fontweight='bold')
    plt.title(f'Prediction Accuracy Parity Plot (R² = {r2_val:.4f})', fontsize=13, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted_parity.png')
    plt.close()
    print("Saved -> plots/actual_vs_predicted_parity.png")

    # 7.3 Plot 3: Residuals Distribution
    plt.figure(figsize=(9, 5), dpi=300)
    sns.histplot(pred_pdf['residual'], kde=True, color='#6366f1', bins=30, edgecolor='white')
    plt.axvline(0, color='#ef4444', linestyle='--', linewidth=1.5, label='Zero Error Line')
    plt.xlabel('Residual Error (Actual - Predicted Salary)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Residual Distribution Analysis (Normal Error Distribution)', fontsize=13, fontweight='bold', pad=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/residual_distribution.png')
    plt.close()
    print("Saved -> plots/residual_distribution.png")

    # 7.4 Plot 4: Salary Distribution across Job Titles
    plt.figure(figsize=(11, 6), dpi=300)
    order = pdf.groupby('job_title')['salary'].median().sort_values(ascending=False).index
    sns.boxplot(data=pdf, x='salary', y='job_title', order=order, palette='crest', orient='h')
    plt.xlabel('Salary ($USD)', fontsize=12, fontweight='bold')
    plt.ylabel('Job Title', fontsize=12, fontweight='bold')
    plt.title('Salary Variance across Job Titles (Categorical Distribution)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots/job_title_salary_distribution.png')
    plt.close()
    print("Saved -> plots/job_title_salary_distribution.png")

    # 7.5 Plot 5: Scalability Benchmark (Execution Time vs Partitions)
    bench_df = pd.DataFrame(benchmark_results)
    plt.figure(figsize=(9, 5), dpi=300)
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    
    color = '#2563eb'
    ax1.set_xlabel('Number of Distributed Partitions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Latency (seconds)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(bench_df['partitions'], bench_df['latency_sec'], marker='o', linewidth=2.5, color=color, label='Latency (s)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color = '#059669'
    ax2.set_ylabel('Throughput (records/sec)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(bench_df['partitions'], bench_df['throughput_rows_sec'], marker='s', linewidth=2.5, linestyle='--', color=color, label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Apache Spark Scalability Benchmark across Partition Configurations', fontsize=13, fontweight='bold', pad=14)
    fig.tight_layout()
    plt.savefig('plots/spark_scalability_benchmark.png')
    plt.close()
    print("Saved -> plots/spark_scalability_benchmark.png")

    spark.stop()
    print("\n[SUCCESS] Pipeline execution, model training, evaluation, and visualizations completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_project()
