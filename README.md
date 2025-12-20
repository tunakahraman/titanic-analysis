<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1280px-RMS_Titanic_3.jpg" alt="Titanic" width="600"/>
</p>

<h1 align="center">🚢 Titanic Data Analysis Project</h1>

<p align="center">
  <b>A comprehensive exploratory data analysis of the Titanic dataset</b><br>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-2.0+-green?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Seaborn-0.13-red?style=for-the-badge" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset Overview](#-dataset-overview)
- [Project Workflow](#-project-workflow)
- [Data Cleaning Steps](#-data-cleaning-steps)
- [Feature Engineering](#-feature-engineering)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Lessons Learned](#-lessons-learned)

---

## 🎯 About the Project

In this project, I explore the famous Titanic dataset. The goal is to understand the factors that influenced passenger survival rates through data cleaning, exploration, and analysis.

### **Project Goals:**
- ✅ Learn data cleaning techniques
- ✅ Handle missing values professionally
- ✅ Perform feature engineering
- ✅ Analyze survival patterns
- ✅ Detect and understand outliers

---

## 📊 Dataset Overview

The Titanic dataset contains information about **891 passengers** who were aboard the RMS Titanic.

### **Original Dataset Structure:**

| Column | Description | Type |
|--------|-------------|------|
| `survived` | Survival status (0 = No, 1 = Yes) | int |
| `pclass` | Passenger class (1st, 2nd, 3rd) | int |
| `sex` | Gender | object |
| `age` | Age in years | float |
| `sibsp` | # of siblings/spouses aboard | int |
| `parch` | # of parents/children aboard | int |
| `fare` | Ticket fare | float |
| `embarked` | Port of embarkation | object |
| `deck` | Deck level | object |
| `embark_town` | Port name | object |
| `alone` | Whether traveling alone | bool |

### **Missing Values Found:**

```
Column          Missing    Percentage
──────────────  ─────────  ──────────
deck            688        77.10%
age             177        19.87%
embarked        2          0.22%
embark_town     2          0.22%
```

### **Final Cleaned Dataset Structure (14 columns):**

| Column | Description | Type |
|--------|-------------|------|
| `survived` | Survival status (0 = No, 1 = Yes) | int |
| `pclass` | Passenger class (1st, 2nd, 3rd) | int |
| `sex` | Gender (male/female) | object |
| `age` | Age in years (filled with grouped median) | float |
| `sibsp` | # of siblings/spouses aboard | int |
| `parch` | # of parents/children aboard | int |
| `fare` | Ticket fare | float |
| `class` | Passenger class (First/Second/Third) | category |
| `who` | Person type (man/woman/child) | object |
| `adult_male` | Whether adult male | bool |
| `embark_town` | Port name (Southampton/Cherbourg/Queenstown) | object |
| `family_size` | **NEW** Total family size (sibsp + parch + 1) | int |
| `family_type` | **NEW** Family category (Alone/Small_Family/Large_Family) | object |

---

## 🔄 Project Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Loading   │───▶│  Data Cleaning  │───▶│    Analysis     │
│  (Seaborn)      │    │  (Pandas)       │    │  (Statistics)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Export Data    │◀───│  Visualization  │◀───│    Feature      │
│  (CSV)          │    │  (Matplotlib)   │    │   Engineering   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🧹 Data Cleaning Steps

### **Step 1: Dropping Redundant/Problematic Columns**

```python
# Dropped columns due to redundancy or too many missing values
df.drop('deck', axis=1, inplace=True)           # 77% missing - too many
df.drop('alive', axis=1, inplace=True)          # Redundant with 'survived'
df.drop('has_deck_info', axis=1, inplace=True)  # Derived column, no longer needed
df.drop('embarked', axis=1, inplace=True)       # Redundant with 'embark_town'
df.drop('alone', axis=1, inplace=True)          # Redundant with 'family_type'
```

**Reasoning:**
- `deck`: 77% missing values - unreliable for analysis
- `alive`: Same information as `survived` (just different format: "yes/no" vs 0/1)
- `has_deck_info`: We created this to analyze deck importance, then removed it
- `embarked`: Short code version, kept `embark_town` (full name) instead
- `alone`: Redundant with `family_type` (Alone category provides the same info)

### **Step 2: Handling Missing Ages**

```python
# Fill age with median grouped by passenger class and sex
# This is a better approach than using the overall median because:
# - 1st class passengers tend to be older
# - 3rd class passengers tend to be younger
df['age'] = df['age'].fillna(
    df.groupby(['pclass', 'sex'])['age'].transform('median')
)
```

**Why group by pclass and sex?**
- 1st class passengers tend to be older (wealthier, established)
- 3rd class passengers tend to be younger (immigrants, families)
- This provides more accurate estimates than the overall median

### **Step 3: Handling Missing Embark Town**

```python
# Check distribution first
print(df['embark_town'].value_counts())
# Southampton: 644, Cherbourg: 168, Queenstown: 77

# Fill with the most common port (mode)
mode_embark_town = df['embark_town'].mode()[0]
df['embark_town'].fillna(mode_embark_town, inplace=True)
```

**Result:** Only 2 values missing, filled with Southampton (most common port - 72% of passengers)

### **Step 4: Duplicate Check**

```python
# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"--- Duplicate Rows: {duplicates} ---")  # Found 118 duplicates
```

**Note:** 118 duplicate rows found. These are passengers with identical data - could be family members with the same ticket or data entry duplicates. We kept them for analysis but noted their presence.

---

## ⚙️ Feature Engineering

### **Family Size Feature**

Created a new feature by combining `sibsp` and `parch`:

```python
# Total family members = siblings/spouse + parents/children + self
df['family_size'] = df['sibsp'] + df['parch'] + 1
```

### **Family Type Categorization**

```python
def categorize_family(size):
    if size == 1:
        return 'Alone'           # Traveling solo
    elif 2 <= size <= 4:
        return 'Small_Family'    # 2-4 family members
    else:
        return 'Large_Family'    # 5+ family members

df['family_type'] = df['family_size'].apply(categorize_family)
```

---

## 📈 Key Findings

### **1. Survival by Family Size**

| Family Size | Survival Rate | Observation |
|-------------|---------------|-------------|
| 1 (Alone) | 30% | Lower survival - no one to help |
| 2 | 55% | Better survival with a companion |
| 3 | 58% | Optimal family size |
| 4 | 72% | **Highest survival rate** 🏆 |
| 5+ | 14-20% | Too difficult to evacuate together |

### **2. Survival by Family Type**

```
Family Type     Count    Survival Rate
─────────────   ─────    ─────────────
Alone           537      30%
Small_Family    292      58%
Large_Family    62       16%
```

**Insight:** Small families (2-4 people) had the **best survival rates**! They could help each other while still moving quickly during evacuation.

### **3. Statistical Summary**

```
Statistic       Age       Fare
─────────────   ─────     ─────
Mean            29.70     32.20
Median          28.00     14.45
Std Dev         14.53     49.69
Min             0.42      0.00
Max             80.00     512.33
```

### **4. Outlier Detection with Scatter Plots**

We used the **IQR (Interquartile Range) method** to detect outliers in numerical columns:

```python
# Outlier detection function using IQR method
def get_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

# Visualize outliers with scatter plots
age_outliers = get_outliers(df['age'])
fare_outliers = get_outliers(df['fare'])

# Color coding: Red = Outlier, Blue/Green = Normal
colors_age = ['red' if x else 'blue' for x in age_outliers]
colors_fare = ['red' if x else 'green' for x in fare_outliers]
```

**Outlier Analysis Results:**

| Column | Outliers Found | Percentage | Range |
|--------|---------------|------------|-------|
| `age` | ~66 passengers | 7.4% | Outside 2.5-54.5 years |
| `fare` | ~116 passengers | 13% | Above $65 (VIP tickets) |

**Visualization:** Scatter plots with outliers highlighted in **red** for easy identification.

**Decision:** We kept the outliers because:
- Age outliers are real passengers (elderly travelers)
- Fare outliers represent VIP/first-class passengers with expensive tickets
- Removing them would lose valuable survival data

---

## 🛠️ Technologies Used

<table>
<tr>
<td align="center" width="120">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="48" height="48" alt="Python" />
<br><b>Python 3.11</b>
</td>
<td align="center" width="120">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="48" height="48" alt="Pandas" />
<br><b>Pandas</b>
</td>
<td align="center" width="120">
<img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" width="48" height="48" alt="Seaborn" />
<br><b>Seaborn</b>
</td>
<td align="center" width="120">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" width="48" height="48" alt="Matplotlib" />
<br><b>Matplotlib</b>
</td>
<td align="center" width="120">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg" width="48" height="48" alt="Jupyter" />
<br><b>Jupyter</b>
</td>
</tr>
</table>

---

## 🚀 How to Run

### **1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/titanic-analysis.git
cd titanic-analysis
```

### **2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install pandas seaborn matplotlib jupyter
```

### **4. Run the notebook**

```bash
jupyter notebook titanic-project.ipynb
```

---

## 📁 Project Structure

```
titanic-analysis/
│
├── 📓 titanic-project.ipynb    # Main analysis notebook
├── 📊 titanic.csv              # Original raw dataset (891 rows, 15 columns)
├── 📊 titanic_cleaned.csv      # Cleaned dataset (891 rows, 14 columns)
├── 📄 README.md                # This file
└── 📋 requirements.txt         # Dependencies
```

### **Dataset Comparison:**

| File | Description | Columns | Missing Values |
|------|-------------|---------|----------------|
| `titanic.csv` | Original raw data | 15 | ✗ Has missing values |
| `titanic_cleaned.csv` | Cleaned & feature engineered | 14 | ✓ No missing values |

**New features added in cleaned dataset:**
- `family_size` - Total family members (sibsp + parch + 1)
- `family_type` - Categorized as Alone, Small_Family, or Large_Family

---

## 💡 Lessons Learned

### **Data Cleaning Skills:**
- ✅ How to identify and handle missing values
- ✅ When to drop vs. fill missing data
- ✅ How to detect redundant columns
- ✅ Using `fillna()` with grouped medians

### **Feature Engineering Skills:**
- ✅ Creating meaningful features from raw data
- ✅ Categorizing continuous variables
- ✅ Using `apply()` with custom functions

---
