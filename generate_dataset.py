import csv
import random
import math

random.seed(42)

job_titles = {
    "Junior Software Engineer": {"base": 45000, "rate": 4200, "dept": "Engineering"},
    "Software Engineer": {"base": 65000, "rate": 4800, "dept": "Engineering"},
    "Senior Software Engineer": {"base": 95000, "rate": 5500, "dept": "Engineering"},
    "Lead Engineer": {"base": 120000, "rate": 6000, "dept": "Engineering"},
    "Data Analyst": {"base": 50000, "rate": 4300, "dept": "Data & AI"},
    "Data Scientist": {"base": 75000, "rate": 5600, "dept": "Data & AI"},
    "Senior Data Scientist": {"base": 105000, "rate": 6200, "dept": "Data & AI"},
    "Machine Learning Engineer": {"base": 85000, "rate": 6000, "dept": "Data & AI"},
    "DevOps Engineer": {"base": 70000, "rate": 5000, "dept": "Operations"},
    "Cloud Architect": {"base": 115000, "rate": 6300, "dept": "Operations"},
    "QA Engineer": {"base": 48000, "rate": 3900, "dept": "Quality"},
    "Project Manager": {"base": 80000, "rate": 5200, "dept": "Management"},
    "Engineering Director": {"base": 150000, "rate": 7000, "dept": "Management"}
}

education_premiums = {
    "Bachelor": 0,
    "Master": 12000,
    "PhD": 25000
}

first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
               "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
               "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
               "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
               "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Aditya", "Pooja", "Arjun", "Neha"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
              "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
              "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
              "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
              "Carter", "Roberts", "Sharma", "Verma", "Patel", "Reddy", "Mehta", "Iyer", "Nair"]

num_employees = 2500
records = []

# Generate organizational hierarchy (Directors first, then managers/leads, then individual contributors)
for i in range(1, num_employees + 1):
    emp_id = f"EMP{i:04d}"
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    
    if i <= 15:
        # Top level Directors
        title = "Engineering Director"
        mgr_id = "EMP0000" # Top of organization
        exp = round(random.uniform(14.0, 25.0), 1)
        edu = random.choice(["Master", "PhD", "Master", "PhD"])
    elif i <= 80:
        # Leads and Managers
        title = random.choice(["Project Manager", "Lead Engineer", "Cloud Architect", "Senior Data Scientist"])
        mgr_id = f"EMP{random.randint(1, 15):04d}"
        exp = round(random.uniform(8.0, 16.0), 1)
        edu = random.choice(["Bachelor", "Master", "PhD", "Master"])
    elif i <= 400:
        # Senior Roles
        title = random.choice(["Senior Software Engineer", "Senior Data Scientist", "DevOps Engineer", "Machine Learning Engineer"])
        mgr_id = f"EMP{random.randint(16, 80):04d}"
        exp = round(random.uniform(4.5, 12.0), 1)
        edu = random.choice(["Bachelor", "Bachelor", "Master", "PhD"])
    else:
        # Mid & Entry level roles
        title = random.choice(["Software Engineer", "Junior Software Engineer", "Data Analyst", "Data Scientist", "QA Engineer"])
        mgr_id = f"EMP{random.randint(81, 400):04d}"
        if "Junior" in title:
            exp = round(random.uniform(0.5, 3.5), 1)
        else:
            exp = round(random.uniform(1.5, 7.5), 1)
        edu = random.choice(["Bachelor", "Bachelor", "Bachelor", "Master"])
    
    dept = job_titles[title]["dept"]
    base = job_titles[title]["base"]
    rate = job_titles[title]["rate"]
    edu_bonus = education_premiums[edu]
    
    # Calculate realistic salary with experience factor + noise
    noise = random.gauss(0, 3500)
    salary = round(base + (rate * exp) + edu_bonus + noise, 2)
    salary = max(35000.0, salary) # Floor at minimum realistic salary
    
    # Also add a collaboration / team project attribute for graph edges
    project_id = f"PROJ_{(i % 45) + 1:02d}"
    
    records.append({
        "emp_id": emp_id,
        "name": name,
        "job_title": title,
        "department": dept,
        "education": edu,
        "years_experience": exp,
        "project_id": project_id,
        "manager_id": mgr_id,
        "salary": salary
    })

csv_file = "employee_salaries.csv"
fieldnames = ["emp_id", "name", "job_title", "department", "education", "years_experience", "project_id", "manager_id", "salary"]

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"Successfully generated {len(records)} employee records in {csv_file}")
