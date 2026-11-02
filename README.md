# Predicting Employee Salaries Using Apache Spark MLlib & Graph Analytics

**Course Code:** INT315 - Cluster Computing  
**Academic Task:** AT2 Project  
**Author:** Swapnil Nikhij  

---

## 📌 Project Overview
This project develops an end-to-end distributed big-data architecture using **Apache Spark**, **Spark MLlib**, and **Graph Analytics** to model, analyze, and predict employee salaries based on organizational structure, experience, department, job role, and education.

### Key Highlights
- **Distributed Big Data Ingestion**: Schema inference, distributed data loading, and caching with PySpark DataFrames.
- **Graph Analytics**: Modeled organizational hierarchy using NetworkX to determine managerial span-of-control and centrality correlation with compensation ($r = 0.6621$).
- **Distributed Feature Engineering**: Multi-stage ML pipeline with `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`.
- **Machine Learning**: `LinearRegression` model trained across 16 parallel local cluster cores.
- **Model Evaluation**:
  - **$R^2$ Score**: **98.89%** ($0.9889$)
  - **RMSE**: **$3,961.68**
  - **MAE**: **$3,149.60**
- **Interactive Visual Dashboard**: Self-contained web dashboard (`index.html`) featuring KPI cards, SVG charts, salary distribution histograms, and role compensation breakdowns.

---

## 📂 Repository Structure
```
├── employee_salaries.csv                  # Distributed enterprise salary dataset (2,500 records)
├── Employee_Salary_Prediction.ipynb       # Fully executed PySpark Jupyter Notebook with plots
├── index.html                             # Interactive Web Dashboard (GitHub Pages ready)
├── dashboard.html                         # Dashboard source file
├── main_pipeline.py                       # Standalone end-to-end Python Spark ML pipeline
├── generate_dataset.py                    # Synthetic dataset generation script
├── generate_notebook.py                   # Notebook generator script
├── generate_docx_report.py                # Academic report generator
├── plots/                                 # Generated evaluation and graph figures
│   ├── exp_vs_salary.png
│   ├── parity_residuals.png
│   └── org_hierarchy_subgraph.png
└── results/                               # Exported evaluation metrics (JSON)
```

---

## 🚀 Quick Start

### 1. View Dashboard Locally
Open `index.html` in your web browser:
```powershell
python -m http.server 8000
```
Then navigate to `http://localhost:8000`.

### 2. Run the Spark Pipeline
Ensure Python 3.10+ and Java 17 are installed:
```powershell
pip install pyspark matplotlib seaborn networkx pandas
python main_pipeline.py
```

### 3. Open Jupyter Notebook
```powershell
jupyter notebook Employee_Salary_Prediction.ipynb
```

---

## 📊 Live GitHub Pages Deployment
To host the interactive dashboard for free:
1. Go to your repository **Settings** → **Pages**.
2. Under **Build and deployment**, select `Deploy from a branch`.
3. Choose `main` branch and `/ (root)` folder.
4. Click **Save** — your dashboard will be live at:
   `https://<username>.github.io/<repo-name>/`
