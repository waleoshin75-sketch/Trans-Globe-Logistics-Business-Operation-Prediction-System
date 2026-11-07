# Trans Globe Logistics Business Operation Prediction System

**TransGlobe Logistics | Logistics Operations Analytics**

**Python | pandas | scikit-learn | Streamlit** | An Intelligent Multi-Model System for Optimizing Freight Costs, Delivery Timelines, and Operational Risks

[Live Streamlit App](https://trans-globe-logistics-business-operation-prediction-system-9yr.streamlit.app/) | [GitHub Repository](https://github.com/waleoshin75-sketch/Trans-Globe-Logistics-Business-Operation-Prediction-System)

---

## At a glance

I took five messy logistics tables (customers, carriers, shipments, invoices, claims), cleaned them, joined them into one shipment-level table and trained candidate models for each business problem. The saved models run inside a Streamlit app: fill in a shipment, get a risk score or a price, and a suggested action.

| Business objective | Model that led on unseen data | Result |
|---|---|---|
| Raise on-time delivery and SLA adherence | Logistic Regression | Caught 39 of 44 late shipments (89% recall). ROC AUC 0.886 |
| Cut freight cost and protect margin | Random Forest Regressor | Average miss of $146 per shipment, R² 0.83 |
| Close revenue leakage and speed up cash collection | Logistic Regression | Near chance (ROC AUC 0.50). Reported honestly |
| Reduce claims, damage and loss | Logistic Regression | Near chance (ROC AUC 0.50). Reported honestly |
| Rank carriers on cost and reliability | Built from the models above, plus a carrier scorecard | See section 18 and the app's Carrier Ranking page |
| Shipment and billing anomalies | Isolation Forest, LOF, One-Class SVM | Review flags, no labels to score accuracy |

The data is fictional and was built with realistic quality problems, so every step of a real workflow is shown. Two of the models score close to chance because the outcomes in this dataset carry little signal. I kept them in and said so, rather than hide it.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table](#9-analytical-base-table)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Impact](#23-impact)
24. [Conclusion](#24-conclusion)

---

## 1. Project Overview

TransGlobe Logistics is a fictional multinational freight company. Its customers, carriers, shipments, invoices and claims sit in five separate, untidy tables. Five business goals depend on that data:

| Business objective | How it is answered | Target |
|---|---|---|
| Raise on-time delivery and SLA adherence | Classification | Shipment delivered late (12.5% of shipments with a known outcome) |
| Cut freight cost and protect margin | Regression | Total freight cost in USD (freight plus fuel surcharge) |
| Close revenue leakage and speed up cash collection | Classification, plus anomaly review of billing | Invoice paid late, overdue or disputed (24% of invoices) |
| Reduce claims, damage and loss | Imbalanced classification | Shipment has a claim (9.2% of shipments) |
| Rank carriers on cost and reliability | Carrier scorecard, plus a ranking page that scores every carrier with the freight cost and late-delivery models | Cost per kg, on-time rate, claims |

**Note on numbering:** in sections 14 to 17 the objectives follow the notebook: 1 late delivery, 2 freight cost, 3 payment risk, 4 claim risk, 5 anomaly review. Carrier ranking has no model of its own; it combines objectives 1 and 2.

A fifth model type, **unsupervised anomaly detection** (Isolation Forest, Local Outlier Factor, One-Class SVM), flags unusual shipments and billing for review. It has no target column.

**The workflow:** data quality audit, cleaning, feature engineering, relationship table, EDA, then for each objective an analytical base table, encoding, X/y split, chronological train/test split, scaling, four candidate models, prediction, evaluation, business analysis, saving the model, scaler and feature list, and a Streamlit app.

**Rules I held myself to**
- Four candidate models per objective, compared on unseen data.
- Chronological split: the model trains on the earliest 80% of shipments and is tested on the latest 20%, so it never sees the future.
- No leakage: columns that reveal the answer (actual delivery date, days late, paid date, claim columns) never enter a model.
- Weak results are reported as weak.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Streamlit.

### Repository structure

```
├── README.md
├── app.py                          the Streamlit app (section 20)
├── requirements.txt
├── images/                         every chart shown in this README
├── TransGlobe_Logistics_Consolidated.xlsx   the raw workbook (5 data sheets + plan sheets)
├── model/                          the saved models, scalers and feature lists (compressed to fit GitHub)
└── notebook                        all the code shown below
```

---

## 2. Dataset Overview

Five tables arrive dirty. After cleaning, 76 duplicate rows are gone.

| Table | Role | Rows (raw) | Rows (clean) | What it holds |
|---|---|---|---|---|
| `dim_customers` | Dimension | 63 | 60 | Customer, HQ country, industry, tier, SLA days, credit limit |
| `dim_carriers` | Dimension | 14 | 14 | Carrier, mode, region, contract rate per kg, fleet size |
| `fact_shipments` | Fact | 2,540 | 2,500 | Route, mode, dates, weight, freight cost, fuel surcharge, status |
| `fact_invoices` | Fact | 2,292 | 2,267 | Billed amount, payment status, paid date |
| `fact_claims` | Fact | 238 | 230 | Claim type, amount, status, days to resolve, root cause |

The workbook also holds plan sheets (business objectives, ML objectives, derived columns, leakage rules, cleaning plan, data splitting) that guided the build.

### Code: load the five tables

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Define the master local Excel spreadsheet file name
file_path = "TransGlobe_Logistics_Consolidated.xlsx"

# %%
pd.ExcelFile("TransGlobe_Logistics_Consolidated.xlsx").sheet_names

# %%
# Load and display Customers dimension data
df_customers = pd.read_excel(file_path, sheet_name='dim_customers')
df_customers.head(5)

# %%
# Load and display Carriers dimension data
df_carriers = pd.read_excel(file_path, sheet_name='dim_carriers')
df_carriers.head(5)

# %%
# Load and display Shipments fact data
df_shipments = pd.read_excel(file_path, sheet_name='fact_shipments')
df_shipments.head(5)

# %%
# Load and display Invoices fact data
df_invoices = pd.read_excel(file_path, sheet_name='fact_invoices')
df_invoices.head(5)

# %%
# Load and display Claims fact data
df_claims = pd.read_excel(file_path, sheet_name='fact_claims')
df_claims.head(5)
```

**Output**

```text
['Project_Overview', 'Business_Objectives', 'ML_Objectives', 'Data_Dictionary', 'Derived_Columns', 'Leakage_Rules', 'Cleaning_Plan', 'Data_Splitting', 'dim_customers', 'dim_carriers', 'fact_shipments', 'fact_invoices', 'fact_claims']
```

| customer_id | company_name | hq_country | industry | customer_tier | contract_sla_days | credit_limit_usd |
|---|---|---|---|---|---|---|
| CUS-1000 | Nova Trading Co | india | nan | gold | 5 | $394,000.00 |
| CUS-1001 | Pinnacle PLC | Netherlands | Automotive | Silver | 5 | 270000 |
| CUS-1002 | SUMMIT LTD | Nigeria | Electronics | Silver | 7d | 397000 |
| CUS-1003 | Solace Inc | China | Agriculture | Gold | 2 | 354000 |
| CUS-1004 | Solace PLC | france | Automotive | Gold | 10 | 302000 |

| carrier_id | carrier_name | primary_mode | base_region | contract_rate_per_kg_usd | fleet_size |
|---|---|---|---|---|---|
| CAR-200 |   SkyBridge Air Cargo  | AIR | AMER | 5.98 | 109 |
| CAR-201 | oceanlink shipping | sea | EMEA | 2.34 | 151 |
| CAR-202 | RoadRunner Freight | Road | Africa | 3.97 | 22 |
| CAR-203 |   RailCore Express  | Rail Freight | APAC | 1.86 | 99999 |
| CAR-204 | AeroSwift Cargo | AIR | EMEA | 4.47 | -12 |

| shipment_id | customer_id | carrier_id | origin_country | … | freight_cost_usd | fuel_surcharge_usd | shipment_status |
|---|---|---|---|---|---|---|---|
| SHP-50612 | CUS-1004 | CAR-202 | Germany | … | 347.9 | 42.36 | cancelled |
| SHP-50821 | CUS-1044 | CAR-207 | china | … | 108.5 | 5 | delivered |
| SHP-51381 | CUS-1035 | CAR-200 | China | … | 5,376 | 663.71 | Delivered |
| SHP-50964 | CUS-1040 | CAR-212 | south africa | … | 24.82 | 3.21 | Delivered |
| SHP-52393 | CUS-1030 | CAR-204 | France | … | 177.6 | 28.35 | Delivered |

| invoice_id | shipment_id | invoice_date | billed_amount_usd | payment_status | paid_date |
|---|---|---|---|---|---|
| INV-700000 | SHP-50547 | 06-07-2025 | 347.1 | Paid | June 23, 2025 |
| INV-700001 | SHP-50521 | 12 Feb 2025 | 343.9 | Overdue | nan |
| INV-700002 | SHP-52404 | 2025-10-13 | USD 213.02 | Paid | 11-05-2025 |
| INV-700003 | SHP-52244 | 2025-07-06 | 560 | Disputed | TBD |
| INV-700004 | SHP-51951 | 11 Sep 2025 | 590 | Disputed | TBD |

| claim_id | shipment_id | claim_type | claim_amount_usd | claim_status | days_to_resolve | root_cause |
|---|---|---|---|---|---|---|
| CLM-3000 | SHP-52181 | Damage   | 967.1 | Settled | nan | Packaging |
| CLM-3001 | SHP-51776 | loss | 1,176 | Settled | 78 | Customs |
| CLM-3002 | SHP-51449 |   Delay Penalty  | 132.3 | Settled | 70 | Carrier Delay |
| CLM-3003 | SHP-51943 | Damage | 878 | Under Review | 40 | Carrier Delay |
| CLM-3004 | SHP-50534 | Temperature Breach | 956.9 | Settled | 28 | Packaging |


---

## 3. Data Dictionary

The workbook's `Data_Dictionary` sheet documents every column with its meaning and known data problems. All 40 rows are below.

| Table | Column | Type | Meaning | Known issues |
|---|---|---|---|---|
| `dim_customers` | `customer_id` | Text (key) | Unique customer code, format CUS-#### | Duplicate rows |
| `dim_customers` | `company_name` | Text | Registered customer name | Extra spaces, mixed case |
| `dim_customers` | `hq_country` | Text | Customer headquarters country | Variants: UK / U.K. / United Kingdom |
| `dim_customers` | `industry` | Text | Customer industry vertical | Missing values |
| `dim_customers` | `customer_tier` | Text | Platinum, Gold, Silver or Bronze | Mixed case, spaces |
| `dim_customers` | `contract_sla_days` | Integer | Contracted delivery window in days (on top of transit time) | Text such as '5 days', 'three', 'N/A' |
| `dim_customers` | `credit_limit_usd` | Currency | Approved credit limit in USD | Stored as text with $, USD, spaces |
| `dim_carriers` | `carrier_id` | Text (key) | Unique carrier code, format CAR-### | nan |
| `dim_carriers` | `carrier_name` | Text | Carrier trading name | Extra spaces, mixed case |
| `dim_carriers` | `primary_mode` | Text | Air, Sea, Road or Rail | 'Air Freight', upper/lower case |
| `dim_carriers` | `base_region` | Text | EMEA, APAC, AMER or Africa | nan |
| `dim_carriers` | `contract_rate_per_kg_usd` | Currency | Agreed carrier rate per kg in USD | Text with symbols |
| `dim_carriers` | `fleet_size` | Integer | Number of vehicles / vessels / aircraft | Negative and impossible values |
| `fact_shipments` | `shipment_id` | Text (key) | Unique shipment code, format SHP-##### | Duplicate rows |
| `fact_shipments` | `customer_id` | Text (FK) | Links to dim_customers | Orphan codes, lower case, spaces |
| `fact_shipments` | `carrier_id` | Text (FK) | Links to dim_carriers | Missing values |
| `fact_shipments` | `origin_country` | Text | Country the shipment leaves | Spelling variants |
| `fact_shipments` | `destination_country` | Text | Country the shipment arrives | Spelling variants |
| `fact_shipments` | `transport_mode` | Text | Air, Sea, Road or Rail | Case, spaces, 'Air Freight' |
| `fact_shipments` | `ship_date` | Date | Date the shipment was dispatched | Six different date formats, junk values |
| `fact_shipments` | `promised_delivery_date` | Date | Delivery date promised to the customer | Mixed date formats |
| `fact_shipments` | `actual_delivery_date` | Date | Date actually delivered (blank if not delivered) | Mixed formats, blanks, 'TBD' |
| `fact_shipments` | `weight_kg` | Decimal | Chargeable weight in kilograms | Negative, x1000 outliers, blanks |
| `fact_shipments` | `freight_cost_usd` | Currency | Cost paid to the carrier in USD | Text with symbols |
| `fact_shipments` | `fuel_surcharge_usd` | Currency | Fuel surcharge paid to the carrier in USD | Text with symbols, blanks |
| `fact_shipments` | `shipment_status` | Text | Delivered, In Transit, Delayed, Returned, Cancelled | Typos: 'Delaied', 'Tranist', case |
| `fact_invoices` | `invoice_id` | Text (key) | Unique invoice code, format INV-###### | Duplicate rows |
| `fact_invoices` | `shipment_id` | Text (FK) | Links to fact_shipments | Orphan codes |
| `fact_invoices` | `invoice_date` | Date | Date the customer was invoiced | Mixed date formats |
| `fact_invoices` | `billed_amount_usd` | Currency | Amount invoiced to the customer in USD | Text with symbols |
| `fact_invoices` | `payment_status` | Text | Paid, Pending, Overdue or Disputed | Case and spaces |
| `fact_invoices` | `paid_date` | Date | Date payment was received (blank unless Paid) | Mixed formats, blanks |
| `fact_claims` | `claim_id` | Text (key) | Unique claim code, format CLM-#### | Duplicate rows |
| `fact_claims` | `shipment_id` | Text (FK) | Links to fact_shipments | nan |
| `fact_claims` | `claim_type` | Text | Damage, Loss, Delay Penalty, Customs Hold, Temperature Breach | Case and spaces |
| `fact_claims` | `claim_amount_usd` | Currency | Value of the claim in USD | Text with symbols |
| `fact_claims` | `claim_status` | Text | Open, Settled, Rejected or Under Review | Case and spaces |
| `fact_claims` | `days_to_resolve` | Integer | Days from claim opened to closed | Blanks and negative values |
| `fact_claims` | `root_cause` | Text | Handling, Packaging, Carrier Delay, Weather, Customs, Unknown | Missing values |

**Leakage rules** (from the `Leakage_Rules` sheet). These columns exist on purpose and are kept out of the models:

| Model | Never use as features | Why |
|---|---|---|
| 1. Late delivery | actual_delivery_date, transit_days, days_late, delivery_outcome | Contain the answer |
| 2. Freight cost | fuel_surcharge_usd, freight_cost_usd, cost_per_kg | Are the target or a component of it |
| 3. Payment risk | paid_date, days_to_pay, open_balance, ageing_bucket | Only known after payment |
| 4. Claim risk | claim_amount_usd, claim_type and any claim column | Only exist if a claim happened |

---

## 4. Data Quality Audit

The audit changes nothing. It counts every problem first, table by table.

| Problem | What the audit found |
|---|---|
| Duplicate rows | 76 exact duplicates (shipments 40, invoices 25, claims 8, customers 3) |
| Duplicate IDs | 80 shipment-ID rows, 50 invoice-ID rows, 16 claim-ID rows, 6 customer-ID rows |
| Missing values | 157 shipments with no actual delivery date, 30 with no weight, 18 with no carrier, 631 invoices with no paid date, 6 customers with no industry |
| Broken links | 126 shipment rows point to customers that are not in the customer table; 10 invoices point to unknown shipments |
| Text spacing and case | Transport mode has 501 values with stray spaces, plus `AIR`, `air`, `Air Freight` for the same mode |
| Numbers stored as text | Credit limits as `$394,000.00`, SLA as `7d`, rates as `USD 2.98`, billed amounts as `USD 213.02` |
| Dates | Several formats (`06-07-2025`, `12 Feb 2025`, `2025-10-13`, `June 23, 2025`) and `TBD` |
| Impossible values | Fleet size of 99999 and -12, negative weights |

### Charts

<p align="center">
  <img src="images/audit_01_issues_found.png" alt="What the audit found" width="100%">
</p>
<p align="center"><em>What the audit found</em></p>

### Code: data quality audit

```python
import pandas as pd
from pathlib import Path

# Workbook path (keep the Excel file next to the notebook)
file_path = Path(
    "TransGlobe_Logistics_Consolidated.xlsx"
)

if not file_path.exists():
    raise FileNotFoundError(
        f"Workbook not found at: {file_path}\n"
        "Put the workbook next to the notebook."
    )

table_names = [
    "dim_customers",
    "dim_carriers",
    "fact_shipments",
    "fact_invoices",
    "fact_claims",
]

# Load the data tables from the workbook
tables = {
    name: pd.read_excel(file_path, sheet_name=name)
    for name in table_names
}

# Profile each table
for table_name, df in tables.items():
    print("=" * 80)
    print(f"{table_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nColumn types:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    print(missing.to_string() if not missing.empty else "None")

    print(f"\nFully duplicated rows: {df.duplicated().sum():,}")

    id_col = df.columns[0]
    duplicate_id_rows = df[id_col].duplicated(keep=False).sum()

    print(f"\nID column: {id_col}")
    print(f"  Null IDs: {df[id_col].isna().sum():,}")
    print(f"  Duplicate ID rows: {duplicate_id_rows:,}")
    print(f"  Distinct non-null IDs: {df[id_col].nunique(dropna=True):,}")

    # Identify text columns without select_dtypes.
    text_columns = [
        col for col in df.columns
        if df[col].map(lambda value: isinstance(value, str)).any()
    ]

    for col in text_columns:
        values = df[col].dropna().map(str)
        whitespace_count = values.ne(values.str.strip()).sum()

        if whitespace_count:
            print(
                f"  Leading/trailing whitespace in {col}: "
                f"{whitespace_count:,} values"
            )

# Check foreign key relationships
relationships = [
    ("fact_shipments", "customer_id", "dim_customers", "customer_id"),
    ("fact_shipments", "carrier_id", "dim_carriers", "carrier_id"),
    ("fact_invoices", "shipment_id", "fact_shipments", "shipment_id"),
    ("fact_claims", "shipment_id", "fact_shipments", "shipment_id"),
]

print("\n" + "=" * 80)
print("RELATIONSHIP CHECKS")

for child_table, child_col, parent_table, parent_col in relationships:
    child_keys = tables[child_table][child_col]
    parent_keys = tables[parent_table][parent_col]
    unmatched = child_keys.notna() & ~child_keys.isin(parent_keys)

    print(
        f"{child_table}.{child_col} → {parent_table}.{parent_col}: "
        f"{unmatched.sum():,} unmatched rows; "
        f"{child_keys.isna().sum():,} null foreign keys"
    )

# Check whether invoices or claims can have multiple rows per shipment
for table_name in ["fact_invoices", "fact_claims"]:
    df = tables[table_name]
    counts = df.groupby("shipment_id", dropna=True).size()

    print(
        f"{table_name}: "
        f"{(counts > 1).sum():,} shipment IDs have multiple rows; "
        f"{df['shipment_id'].isna().sum():,} null shipment IDs"
    )
```

**Output**

```text
================================================================================
dim_customers: 63 rows × 7 columns

Column types:
customer_id             str
company_name            str
hq_country              str
industry                str
customer_tier           str
contract_sla_days    object
credit_limit_usd     object

Missing values:
industry             6
contract_sla_days    1

Fully duplicated rows: 3

ID column: customer_id
  Null IDs: 0
  Duplicate ID rows: 6
  Distinct non-null IDs: 60
  Leading/trailing whitespace in company_name: 5 values
  Leading/trailing whitespace in customer_tier: 5 values
================================================================================
dim_carriers: 14 rows × 6 columns

Column types:
carrier_id                     str
carrier_name                   str
primary_mode                   str
base_region                    str
contract_rate_per_kg_usd    object
fleet_size                   int64

Missing values:
None

Fully duplicated rows: 0

... (output shortened here)
```

### Code: charts

```python
# ======================================================================
# CHARTS FOR THE README - every chart below is drawn from the notebook's tables
# Run after the notebook: it needs relationship_table, cleaned_tables, scaled_X_test,
# y_test, models_by_objective, date_for_objective and anomaly_results in memory.
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix, mean_absolute_error)

BLUE, ORANGE, INK, SOFT, GRID, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#c9c7c1"
OUT = "charts_out"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999", "axes.labelcolor": SOFT, "xtick.color": SOFT,
                     "ytick.color": SOFT, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "figure.facecolor": "white"})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_mode(s):
    """For charts only: merge the spelling variants (AIR, air, Air Freight) into one mode."""
    s = s.astype("string").str.strip().str.lower()
    return s.replace({"air freight": "Air", "air": "Air", "sea": "Sea", "road": "Road", "rail": "Rail"})


def late_rate(frame, by):
    known = frame[frame["late_flag"].notna()].copy()
    known["late_flag"] = known["late_flag"].astype(float)
    return known.groupby(by)["late_flag"].agg(["mean", "count"])



def chart_audit():
    issues = {"Shipments: missing actual delivery date": 157, "Invoices: missing paid date": 631,
              "Shipments: customer not in customer table": 126, "Shipments: exact duplicate rows": 40,
              "Shipments: missing weight": 30, "Invoices: exact duplicate rows": 25,
              "Shipments: missing carrier": 18, "Claims: exact duplicate rows": 8}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    s = pd.Series(issues).sort_values()
    ax.barh(s.index, s.values, color=BLUE, height=.6)
    for i, v in enumerate(s.values):
        ax.text(v + 8, i, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_title("What the data quality audit found (row counts)")
    ax.set_xlim(0, 720); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    save(fig, "audit_01_issues_found")
```

---

## 5. Data Cleaning and Error Checking

Cleaning is applied table by table: strip spaces, parse every date style, turn money and numbers stored as text into real numbers, and drop exact duplicate rows. Duplicate IDs that are not exact copies are kept and flagged rather than guessed at.

| Table | Rows before | Rows after | Exact duplicates removed |
|---|---|---|---|
| `dim_customers` | 63 | 60 | 3 |
| `dim_carriers` | 14 | 14 | 0 |
| `fact_shipments` | 2,540 | 2,500 | 40 |
| `fact_invoices` | 2,292 | 2,267 | 25 |
| `fact_claims` | 238 | 230 | 8 |

I did not merge spelling variants such as `AIR`, `Air` and `air`. That decision matters later: it is why the encoded tables in section 10 carry 172 dummy columns instead of about 50. Standardising categories first is the first improvement I would make.

### Charts

<p align="center">
  <img src="images/cleaning_01_rows_before_after.png" alt="Rows before and after cleaning" width="100%">
</p>
<p align="center"><em>Rows before and after cleaning</em></p>

### Code: data cleaning and error checking

```python
import pandas as pd

date_columns = {
    "fact_shipments": [
        "ship_date", "promised_delivery_date", "actual_delivery_date"
    ],
    "fact_invoices": ["invoice_date", "paid_date"],
}

numeric_columns = {
    "dim_customers": ["contract_sla_days", "credit_limit_usd"],
    "dim_carriers": ["contract_rate_per_kg_usd", "fleet_size"],
    "fact_shipments": [
        "weight_kg", "freight_cost_usd", "fuel_surcharge_usd"
    ],
    "fact_invoices": ["billed_amount_usd"],
    "fact_claims": ["claim_amount_usd", "days_to_resolve"],
}

key_columns = {
    "dim_customers": ["customer_id"],
    "dim_carriers": ["carrier_id"],
    "fact_shipments": ["shipment_id", "customer_id", "carrier_id"],
    "fact_invoices": ["invoice_id", "shipment_id"],
    "fact_claims": ["claim_id", "shipment_id"],
}

cleaned_tables = {}
summary = []

for table_name, original_df in tables.items():
    df = original_df.copy()
    rows_before = len(df)

    # Trim text values and convert empty strings to missing values.
    text_columns = [
        col for col in df.columns
        if df[col].map(lambda value: isinstance(value, str)).any()
    ]
    for col in text_columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[col] = df[col].replace("", pd.NA)

    # Normalize ID/key formatting.
    for col in key_columns.get(table_name, []):
        df[col] = df[col].astype("string").str.strip().str.upper()

    # Parse dates; invalid or placeholder values become NaT.
    for col in date_columns.get(table_name, []):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    # Convert numeric columns, including values containing currency/unit text.
    for col in numeric_columns.get(table_name, []):
        values = df[col].astype("string").str.replace(
            r"[^0-9.\-]", "", regex=True
        )
        df[col] = pd.to_numeric(values, errors="coerce")

    # Remove exact duplicate rows only.
    exact_duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    id_col = df.columns[0]
    duplicate_id_rows = int(df[id_col].duplicated(keep=False).sum())

    cleaned_tables[table_name] = df
    summary.append({
        "table": table_name,
        "rows_before": rows_before,
        "rows_after": len(df),
        "exact_duplicates_removed": exact_duplicates_removed,
        "duplicate_ID_rows_retained": duplicate_id_rows,
        "missing_cells_after": int(df.isna().sum().sum()),
    })

cleaning_summary = pd.DataFrame(summary).set_index("table")
display(cleaning_summary)
```

**Output**

| table | rows_before | rows_after | exact_duplicates_removed | duplicate_ID_rows_retained | missing_cells_after |
|---|---|---|---|---|---|
| dim_customers | 63 | 60 | 3 | 0 | 7 |
| dim_carriers | 14 | 14 | 0 | 0 | 0 |
| fact_shipments | 2540 | 2500 | 40 | 0 | 324 |
| fact_invoices | 2292 | 2267 | 25 | 0 | 859 |
| fact_claims | 238 | 230 | 8 | 0 | 49 |

### Code: charts

```python
def chart_cleaning():
    tables = ["Customers", "Carriers", "Shipments", "Invoices", "Claims"]
    before, after = [63, 14, 2540, 2292, 238], [60, 14, 2500, 2267, 230]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(5)
    ax.bar(x - .2, before, .4, color=GREY, label="Before cleaning")
    ax.bar(x + .2, after, .4, color=BLUE, label="After cleaning")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i + .2, a, f"-{b - a}" if b != a else "0", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(tables); ax.set_yscale("log")
    ax.set_title("Rows before and after cleaning (labels show duplicates removed)")
    ax.legend(frameon=False); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    save(fig, "cleaning_01_rows_before_after")
```

---

## 6. Feature Engineering

New columns were built from the clean tables. Everything that reveals an outcome is kept for analysis and later removed from the model inputs.

| Table | New columns |
|---|---|
| `fact_shipments` | `total_freight_cost` (freight plus fuel surcharge), `cost_per_kg`, `transit_days`, `days_late`, `ship_year`, `ship_month`, `ship_day_of_week`, `delivery_outcome`, `late_flag` |
| `fact_invoices` | `days_to_pay`, `payment_late` (1 for Overdue or Disputed, 0 for Paid, blank for Pending) |

The step also lists what to review: 55 shipments with a missing or non-positive weight, 55 with a negative transit time, 190 with an unknown delivery outcome, 368 invoices with a pending or unmapped payment status.

### Code: feature engineering

```python
import pandas as pd

if "cleaned_tables" not in globals():
    raise NameError(
        "cleaned_tables is not defined. Run the data cleaning cell first."
    )

feature_tables = {
    name: df.copy()
    for name, df in cleaned_tables.items()
}

feature_warnings = []
engineered_columns = {}

def add_warning(table, issue, count, detail):
    if count:
        feature_warnings.append({
            "table": table,
            "issue": issue,
            "count": int(count),
            "detail": detail,
        })

# -------------------------------------------------------------------
# Shipment features
# -------------------------------------------------------------------
shipments = feature_tables["fact_shipments"]

# Make sure dates are datetime values before calculating date differences.
shipment_date_cols = [
    "ship_date",
    "promised_delivery_date",
    "actual_delivery_date",
]
for col in shipment_date_cols:
    shipments[col] = pd.to_datetime(shipments[col], errors="coerce")

shipments["total_freight_cost"] = (
    shipments["freight_cost_usd"] + shipments["fuel_surcharge_usd"]
)

shipments["cost_per_kg"] = (
    shipments["total_freight_cost"]
    .div(shipments["weight_kg"].where(shipments["weight_kg"] > 0))
)

shipments["transit_days"] = (
    shipments["actual_delivery_date"] - shipments["ship_date"]
).dt.days

shipments["days_late"] = (
    shipments["actual_delivery_date"] - shipments["promised_delivery_date"]
).dt.days

shipments["ship_year"] = shipments["ship_date"].dt.year
shipments["ship_month"] = shipments["ship_date"].dt.month
shipments["ship_day_of_week"] = shipments["ship_date"].dt.dayofweek

status = shipments["shipment_status"].astype("string").str.strip().str.casefold()
not_delivered = status.isin(["cancelled", "canceled", "not delivered"])
has_actual_date = shipments["actual_delivery_date"].notna()
has_promised_date = shipments["promised_delivery_date"].notna()

shipments["delivery_outcome"] = "Unknown"
shipments.loc[not_delivered, "delivery_outcome"] = "Not Delivered"

classified = (
    ~not_delivered
    & has_actual_date
    & has_promised_date
)
shipments.loc[
    classified & (shipments["days_late"] > 0), "delivery_outcome"
] = "Late"
shipments.loc[
    classified & (shipments["days_late"] <= 0), "delivery_outcome"
] = "On Time"

# Late flag is missing when delivery outcome cannot be determined.
shipments["late_flag"] = pd.Series(
    pd.NA, index=shipments.index, dtype="Int64"
)
shipments.loc[
    shipments["delivery_outcome"].eq("Late"), "late_flag"
] = 1
shipments.loc[
    shipments["delivery_outcome"].eq("On Time"), "late_flag"
] = 0

engineered_columns["fact_shipments"] = [
    "total_freight_cost",
    "cost_per_kg",
    "transit_days",
    "days_late",
    "ship_year",
    "ship_month",
    "ship_day_of_week",
    "delivery_outcome",
    "late_flag",
]

add_warning(
    "fact_shipments",
    "Non-positive or missing weight",
    shipments["weight_kg"].isna().sum()
    + shipments["weight_kg"].le(0).sum(),
    "cost_per_kg is missing when weight_kg is missing or not positive.",
)
add_warning(
    "fact_shipments",
    "Delivery outcome unknown",
    shipments["delivery_outcome"].eq("Unknown").sum(),
    "Check shipment status and delivery dates before using late_flag as a target.",
)
add_warning(
    "fact_shipments",
    "Negative transit duration",
    shipments["transit_days"].lt(0).sum(),
    "Actual delivery date is earlier than ship date; review these records.",
)

# -------------------------------------------------------------------
# Invoice features
# -------------------------------------------------------------------
invoices = feature_tables["fact_invoices"]

for col in ["invoice_date", "paid_date"]:
    invoices[col] = pd.to_datetime(invoices[col], errors="coerce")

invoices["days_to_pay"] = (
    invoices["paid_date"] - invoices["invoice_date"]
).dt.days

payment_status = (
    invoices["payment_status"].astype("string").str.strip().str.casefold()
)

# Target candidate: paid = 0, overdue/disputed = 1, pending/other = missing.
invoices["payment_late"] = pd.Series(
    pd.NA, index=invoices.index, dtype="Int64"
)
invoices.loc[payment_status.eq("paid"), "payment_late"] = 0
invoices.loc[
    payment_status.isin(["overdue", "disputed"]), "payment_late"
] = 1

engineered_columns["fact_invoices"] = [
    "days_to_pay",
    "payment_late",
]

add_warning(
    "fact_invoices",
    "Payment status not mapped",
    invoices["payment_late"].isna().sum(),
    "Rows with Pending or unrecognized statuses have no payment_late value.",
)
add_warning(
    "fact_invoices",
    "Negative payment duration",
    invoices["days_to_pay"].lt(0).sum(),
    "paid_date is earlier than invoice_date; review these records.",
)

# -------------------------------------------------------------------
# Summary of generated features and surfaced issues
# -------------------------------------------------------------------
print("FEATURE ENGINEERING SUMMARY")
for table_name, columns in engineered_columns.items():
    print(f"\n{table_name}: added {len(columns)} columns")
    print(", ".join(columns))

feature_warnings_df = pd.DataFrame(
    feature_warnings,
    columns=["table", "issue", "count", "detail"],
)

print("\nFEATURE ENGINEERING WARNINGS / REVIEW ITEMS")
if feature_warnings_df.empty:
    print("None found.")
else:
    display(feature_warnings_df)

print("\nFeature table shapes:")
display(
    pd.DataFrame([
        {
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
        }
        for name, df in feature_tables.items()
    ]).set_index("table")
)
```

**Output**

```text
FEATURE ENGINEERING SUMMARY

fact_shipments: added 9 columns
total_freight_cost, cost_per_kg, transit_days, days_late, ship_year, ship_month, ship_day_of_week, delivery_outcome, late_flag

fact_invoices: added 2 columns
days_to_pay, payment_late

FEATURE ENGINEERING WARNINGS / REVIEW ITEMS
```

| table | issue | count | detail |
|---|---|---|---|
| fact_shipments | Non-positive or missing weight | 55 | cost_per_kg is missing when wei... |
| fact_shipments | Delivery outcome unknown | 190 | Check shipment status and deliv... |
| fact_shipments | Negative transit duration | 55 | Actual delivery date is earlier... |
| fact_invoices | Payment status not mapped | 368 | Rows with Pending or unrecogniz... |
| fact_invoices | Negative payment duration | 26 | paid_date is earlier than invoi... |

```text
Feature table shapes:
```

| table | rows | columns |
|---|---|---|
| dim_customers | 60 | 7 |
| dim_carriers | 14 | 6 |
| fact_shipments | 2500 | 22 |
| fact_invoices | 2267 | 8 |
| fact_claims | 230 | 7 |


---

## 7. Create Relationship Table

The five tables are joined into one **shipment-level relationship table** (2,500 rows by 41 columns, one row per shipment). Invoice and claim figures are summed to shipment level first so the join never multiplies rows.

| Check | Count |
|---|---|
| Shipments without a matching customer | 14 |
| Shipments without a matching carrier | 18 |
| Shipments without an invoice | 243 |
| Shipments without a claim | 2,270 |

All stay in the table, with blanks, rather than being dropped.

### Charts

<p align="center">
  <img src="images/star_schema.png" alt="How the five tables connect" width="100%">
</p>
<p align="center"><em>How the five tables connect</em></p>

### Code: relationship table (star schema)

```python
import pandas as pd

if "feature_tables" not in globals():
    raise NameError(
        "feature_tables is not defined. Run the feature engineering cell first."
    )

# Keep the star schema's fact and dimension tables available separately.
star_schema_tables = {
    name: df.copy()
    for name, df in feature_tables.items()
}

shipments = star_schema_tables["fact_shipments"].copy()
customers = star_schema_tables["dim_customers"].copy()
carriers = star_schema_tables["dim_carriers"].copy()
invoices = star_schema_tables["fact_invoices"].copy()
claims = star_schema_tables["fact_claims"].copy()

relationship_warnings = []

# Confirm dimension keys are unique before joining.
for table_name, df, key in [
    ("dim_customers", customers, "customer_id"),
    ("dim_carriers", carriers, "carrier_id"),
]:
    duplicate_keys = int(df[key].duplicated(keep=False).sum())
    if duplicate_keys:
        relationship_warnings.append({
            "check": f"{table_name} key uniqueness",
            "count": duplicate_keys,
            "detail": f"Duplicate {key} values found; resolve before joining.",
        })

# Summarize invoice rows to shipment grain to avoid multiplying shipment rows.
invoice_summary = invoices.copy()
invoice_summary["payment_status"] = (
    invoice_summary["payment_status"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

invoice_by_shipment = (
    invoice_summary.groupby("shipment_id", dropna=False)
    .agg(
        invoice_count=("invoice_id", "count"),
        total_billed_amount_usd=("billed_amount_usd", "sum"),
        paid_invoice_count=("payment_status", lambda s: s.eq("paid").sum()),
        overdue_invoice_count=("payment_status", lambda s: s.eq("overdue").sum()),
        disputed_invoice_count=("payment_status", lambda s: s.eq("disputed").sum()),
        pending_invoice_count=("payment_status", lambda s: s.eq("pending").sum()),
    )
    .reset_index()
)

# Summarize claims to shipment grain.
claims_by_shipment = (
    claims.groupby("shipment_id", dropna=False)
    .agg(
        claim_count=("claim_id", "count"),
        total_claim_amount_usd=("claim_amount_usd", "sum"),
    )
    .reset_index()
)

# Start from shipment fact so the resulting table has one row per shipment.
relationship_table = (
    shipments
    .merge(
        customers,
        on="customer_id",
        how="left",
        suffixes=("", "_customer"),
        validate="many_to_one",
        indicator="_customer_match",
    )
    .merge(
        carriers,
        on="carrier_id",
        how="left",
        suffixes=("", "_carrier"),
        validate="many_to_one",
        indicator="_carrier_match",
    )
    .merge(
        invoice_by_shipment,
        on="shipment_id",
        how="left",
        validate="one_to_one",
        indicator="_invoice_match",
    )
    .merge(
        claims_by_shipment,
        on="shipment_id",
        how="left",
        validate="one_to_one",
        indicator="_claim_match",
    )
)

# Record unmatched dimension and fact keys for review.
for match_col, label in [
    ("_customer_match", "Shipments without a matching customer"),
    ("_carrier_match", "Shipments without a matching carrier"),
    ("_invoice_match", "Shipments without an invoice"),
    ("_claim_match", "Shipments without a claim"),
]:
    count = int(relationship_table[match_col].eq("left_only").sum())
    if count:
        relationship_warnings.append({
            "check": label,
            "count": count,
            "detail": "These shipment rows remain in the relationship table.",
        })

# Missing invoice/claim summaries mean no related record was found.
for col in [
    "invoice_count",
    "paid_invoice_count",
    "overdue_invoice_count",
    "disputed_invoice_count",
    "pending_invoice_count",
    "claim_count",
]:
    relationship_table[col] = relationship_table[col].fillna(0).astype("Int64")

for col in ["total_billed_amount_usd", "total_claim_amount_usd"]:
    relationship_table[col] = relationship_table[col].fillna(0)

# Remove merge indicators from the delivered relationship table.
relationship_table = relationship_table.drop(
    columns=[
        "_customer_match",
        "_carrier_match",
        "_invoice_match",
        "_claim_match",
    ]
)

# Confirm the table grain.
duplicate_shipment_rows = int(
    relationship_table["shipment_id"].duplicated(keep=False).sum()
)

if duplicate_shipment_rows:
    relationship_warnings.append({
        "check": "Shipment grain",
        "count": duplicate_shipment_rows,
        "detail": "The relationship table contains repeated shipment IDs.",
    })

print("STAR SCHEMA TABLES")
for name, df in star_schema_tables.items():
    print(f"{name}: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\nSHIPMENT RELATIONSHIP TABLE")
print(f"Shape: {relationship_table.shape[0]:,} rows × "
      f"{relationship_table.shape[1]} columns")
print(f"Distinct shipments: "
      f"{relationship_table['shipment_id'].nunique(dropna=True):,}")
print(f"Duplicate shipment ID rows: {duplicate_shipment_rows:,}")

print("\nRELATIONSHIP WARNINGS")
if relationship_warnings:
    display(pd.DataFrame(relationship_warnings))
else:
    print("None found.")

display(relationship_table.head())
```

**Output**

```text
STAR SCHEMA TABLES
dim_customers: 60 rows × 7 columns
dim_carriers: 14 rows × 6 columns
fact_shipments: 2,500 rows × 22 columns
fact_invoices: 2,267 rows × 8 columns
fact_claims: 230 rows × 7 columns

SHIPMENT RELATIONSHIP TABLE
Shape: 2,500 rows × 41 columns
Distinct shipments: 2,500
Duplicate shipment ID rows: 0

RELATIONSHIP WARNINGS
```

| check | count | detail |
|---|---|---|
| Shipments without a matching cu... | 14 | These shipment rows remain in t... |
| Shipments without a matching ca... | 18 | These shipment rows remain in t... |
| Shipments without an invoice | 243 | These shipment rows remain in t... |
| Shipments without a claim | 2270 | These shipment rows remain in t... |

| shipment_id | customer_id | carrier_id | origin_country | … | pending_invoice_count | claim_count | total_claim_amount_usd |
|---|---|---|---|---|---|---|---|
| SHP-50612 | CUS-1004 | CAR-202 | Germany | … | 0 | 0 | 0 |
| SHP-50821 | CUS-1044 | CAR-207 | china | … | 0 | 0 | 0 |
| SHP-51381 | CUS-1035 | CAR-200 | China | … | 0 | 0 | 0 |
| SHP-50964 | CUS-1040 | CAR-212 | south africa | … | 0 | 1 | 422.1 |
| SHP-52393 | CUS-1030 | CAR-204 | France | … | 0 | 0 | 0 |

### Code: charts

```python
def chart_star_schema():
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.6)
    boxes = {"dim_customers": (0.3, 3.6, ["customer_id (key)", "company_name", "industry, tier", "contract_sla_days", "credit_limit_usd"]),
             "dim_carriers": (0.3, 0.3, ["carrier_id (key)", "carrier_name", "primary_mode", "contract_rate_per_kg_usd", "fleet_size"]),
             "fact_shipments": (4.3, 1.9, ["shipment_id (key)", "customer_id  (FK)", "carrier_id  (FK)", "route, mode, dates", "weight, freight cost"]),
             "fact_invoices": (8.2, 3.6, ["invoice_id (key)", "shipment_id  (FK)", "billed_amount_usd", "payment_status", "paid_date"]),
             "fact_claims": (8.2, 0.3, ["claim_id (key)", "shipment_id  (FK)", "claim_type", "claim_amount_usd", "root_cause"])}
    pos = {}
    for name, (x, y, cols) in boxes.items():
        fact = name.startswith("fact")
        col = BLUE if name == "fact_shipments" else (ORANGE if fact else "#6b6a66")
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.7, boxstyle="round,pad=0.02", fc="white", ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((x, y + 1.3), 2.5, .4, boxstyle="round,pad=0.02", fc=col, ec=col))
        ax.text(x + 1.25, y + 1.5, name, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        for i, c in enumerate(cols):
            ax.text(x + .12, y + 1.12 - i * .24, c, fontsize=8, color=INK, va="center")
        pos[name] = (x, y)
    def arrow(a, b, xa, ya, xb, yb):
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.5))
    arrow(0, 0, 2.8, 4.3, 4.3, 3.2); arrow(0, 0, 2.8, 1.1, 4.3, 2.3)
    arrow(0, 0, 8.2, 4.3, 6.8, 3.2); arrow(0, 0, 8.2, 1.1, 6.8, 2.3)
    ax.text(5.6, 5.4, "One row per shipment: invoices and claims are summed to shipment level before joining",
            ha="center", fontsize=10, color=SOFT)
    ax.set_title("Relationship table: how the five tables connect", pad=18)
    save(fig, "star_schema")
```

---

## 8. Exploratory Data Analysis

Five things stood out. (For the charts and numbers below I merged spelling variants such as `AIR`, `air` and `Air Freight` into one mode, so each mode reads as one bar.)

- **Contract SLA is the story for late deliveries.** Shipments on a 2-day SLA were late 45% of the time and on a 3-day SLA 30%. At 5 days or more it drops to 2.7% or less.
- **Rail is the least reliable mode** (16.7% late) and air the most reliable (9.6%).
- **Cost splits into two worlds.** Air runs about $5.22 per kg at the median. Road, rail and sea sit between $0.85 and $1.56.
- **Carriers differ more on reliability than on price.** JetStream Logistics is late 5.3% of the time, RailCore Express 16.9%.
- **Weight has absurd outliers** (up to 595,000 kg, and some negatives), which squashes the scaled weight to near zero for almost every shipment.

### Charts

<p align="center">
  <img src="images/eda_01_portfolio_overview.png" alt="Portfolio at a glance" width="100%">
</p>
<p align="center"><em>Portfolio at a glance</em></p>

<p align="center">
  <img src="images/eda_02_delivery_outcomes.png" alt="Delivery outcomes and late rate by mode" width="100%">
</p>
<p align="center"><em>Delivery outcomes and late rate by mode</em></p>

<p align="center">
  <img src="images/eda_03_cost_and_weight.png" alt="Cost per kg by mode and weight spread" width="100%">
</p>
<p align="center"><em>Cost per kg by mode and weight spread</em></p>

<p align="center">
  <img src="images/eda_04_shipments_over_time.png" alt="Volume and late rate over time" width="100%">
</p>
<p align="center"><em>Volume and late rate over time</em></p>

<p align="center">
  <img src="images/eda_05_sla_and_carriers.png" alt="Late rate by contract SLA and by carrier" width="100%">
</p>
<p align="center"><em>Late rate by contract SLA and by carrier</em></p>

<p align="center">
  <img src="images/eda_06_invoices_and_margin.png" alt="Invoice status and margin by mode" width="100%">
</p>
<p align="center"><em>Invoice status and margin by mode</em></p>

<p align="center">
  <img src="images/eda_07_claims.png" alt="Claims by type and root cause" width="100%">
</p>
<p align="center"><em>Claims by type and root cause</em></p>

### Code: exploratory data analysis (tables)

```python
import pandas as pd
import matplotlib.pyplot as plt

if "relationship_table" not in globals():
    raise NameError(
        "relationship_table is not defined. Run the relationship-table cell first."
    )

df = relationship_table.copy()

# 1. Overall table profile
print("DATASET OVERVIEW")
display(pd.DataFrame({
    "Rows": [len(df)],
    "Columns": [df.shape[1]],
    "Distinct shipments": [df["shipment_id"].nunique()],
    "Duplicate shipment IDs": [df["shipment_id"].duplicated().sum()],
}))

print("\nMISSING VALUES")
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
display(
    missing.rename("missing_count")
    .to_frame()
    .assign(missing_percent=lambda x: (x["missing_count"] / len(df) * 100).round(2))
    if not missing.empty
    else pd.DataFrame({"result": ["No missing values found"]})
)

# 2. Numeric feature summary
numeric_candidates = [
    "weight_kg",
    "total_freight_cost",
    "cost_per_kg",
    "transit_days",
    "days_late",
    "contract_sla_days",
    "credit_limit_usd",
    "invoice_count",
    "total_billed_amount_usd",
    "claim_count",
    "total_claim_amount_usd",
]
numeric_columns = [col for col in numeric_candidates if col in df.columns]

print("\nNUMERIC SUMMARY")
display(df[numeric_columns].describe().T.round(2))

# 3. Category and outcome counts
for col in [
    "transport_mode",
    "shipment_status",
    "delivery_outcome",
    "payment_status",
    "customer_tier",
    "primary_mode",
]:
    if col in df.columns:
        print(f"\n{col.upper()} COUNTS")
        display(
            df[col].astype("string")
            .fillna("Missing")
            .value_counts(dropna=False)
            .rename("count")
            .to_frame()
        )

# 4. Late delivery rate by transport mode and carrier
if {"transport_mode", "late_flag"}.issubset(df.columns):
    print("\nLATE DELIVERY RATE BY TRANSPORT MODE")
    display(
        df.groupby("transport_mode", dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            known_outcomes=("late_flag", "count"),
            late_rate=("late_flag", "mean"),
        )
        .assign(late_rate=lambda x: (x["late_rate"] * 100).round(2))
        .sort_values("late_rate", ascending=False)
    )

if {"carrier_name", "late_flag"}.issubset(df.columns):
    print("\nLATE DELIVERY RATE BY CARRIER (at least 20 known outcomes)")
    carrier_summary = (
        df.groupby("carrier_name", dropna=False)
        .agg(
            shipments=("shipment_id", "count"),
            known_outcomes=("late_flag", "count"),
            late_rate=("late_flag", "mean"),
            mean_cost_per_kg=("cost_per_kg", "mean"),
            claims=("claim_count", "sum"),
        )
    )
    carrier_summary = carrier_summary[
        carrier_summary["known_outcomes"] >= 20
    ].copy()
    carrier_summary["late_rate"] = (
        carrier_summary["late_rate"] * 100
    ).round(2)
    display(carrier_summary.sort_values("late_rate", ascending=False))

# 5. Correlations among selected numeric columns
correlation_columns = [
    col for col in [
        "weight_kg",
        "total_freight_cost",
        "cost_per_kg",
        "transit_days",
        "days_late",
        "total_billed_amount_usd",
        "claim_count",
        "total_claim_amount_usd",
    ]
    if col in df.columns
]
if len(correlation_columns) >= 2:
    print("\nNUMERIC CORRELATIONS (Pearson)")
    display(df[correlation_columns].corr().round(2))

# 6. Basic distribution plots
plot_columns = [
    col for col in [
        "weight_kg",
        "total_freight_cost",
        "cost_per_kg",
        "transit_days",
        "days_late",
    ]
    if col in df.columns
]

if plot_columns:
    df[plot_columns].hist(
        bins=30,
        figsize=(14, 9),
        edgecolor="black",
    )
    plt.suptitle("Shipment Metric Distributions", y=1.02)
    plt.tight_layout()
    plt.show()

# Late flag is available only for shipments with a determinable delivery outcome.
if "late_flag" in df.columns:
    late_counts = (
        df["late_flag"]
        .map({0: "On Time", 1: "Late"})
        .fillna("Unknown")
        .value_counts()
    )
    late_counts.plot(
        kind="bar",
        figsize=(7, 4),
        color=["steelblue", "tomato", "gray"][:len(late_counts)],
        edgecolor="black",
    )
    plt.title("Shipment Delivery Outcomes")
    plt.xlabel("Outcome")
    plt.ylabel("Shipment count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
```

**Output**

```text
DATASET OVERVIEW
```

| Rows | Columns | Distinct shipments | Duplicate shipment IDs |
|---|---|---|---|
| 2500 | 41 | 2500 | 0 |

```text
MISSING VALUES
```

| index | missing_count | missing_percent |
|---|---|---|
| days_late | 252 | 10.08 |
| late_flag | 252 | 10.08 |
| transit_days | 249 | 9.96 |
| industry | 230 | 9.2 |
| actual_delivery_date | 227 | 9.08 |
| contract_sla_days | 85 | 3.4 |
| … | … | … |
| company_name | 14 | 0.56 |
| customer_tier | 14 | 0.56 |
| hq_country | 14 | 0.56 |

```text
NUMERIC SUMMARY
```

| index | count | mean | std | … | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| weight_kg | 2,470 | 1,504 | 2.178e+04 | … | 185.6 | 364.8 | 5.95e+05 |
| total_freight_cost | 2,500 | 696.6 | 1,103 | … | 312.9 | 787.3 | 1.106e+04 |
| cost_per_kg | 2,445 | 2.374 | 1.913 | … | 1.389 | 4.258 | 6.881 |
| transit_days | 2,251 | 8.342 | 33.86 | … | 7 | 14 | 338 |
| days_late | 2,248 | -3.761 | 30.78 | … | -3 | -1 | 319 |
| contract_sla_days | 2,415 | 5.616 | 2.583 | … | 5 | 7 | 10 |
| credit_limit_usd | 2,486 | 2.176e+05 | 1.081e+05 | … | 2.21e+05 | 3.03e+05 | 3.97e+05 |
| invoice_count | 2,500 | 0.9028 | 0.2963 | … | 1 | 1 | 1 |
| total_billed_amount_usd | 2,500 | 783.3 | 1,360 | … | 312.2 | 836.3 | 1.444e+04 |
| claim_count | 2,500 | 0.092 | 0.2891 | … | 0 | 0 | 1 |
| total_claim_amount_usd | 2,500 | 74.18 | 363.9 | … | 0 | 0 | 6,249 |

```text
TRANSPORT_MODE COUNTS
```

| transport_mode | count |
|---|---|
| Sea | 419 |
| Road | 385 |
| Air | 297 |
| Rail | 208 |
| ROAD | 160 |
| Air Freight | 157 |
| … | … |
| sea | 134 |
| rail | 82 |
| RAIL | 72 |

```text
SHIPMENT_STATUS COUNTS
```

| shipment_status | count |
|---|---|
| Delivered | 1783 |
| Delayed | 166 |
| delivered | 118 |
| In Transit | 114 |
| DELIVERED | 103 |
| Returned | 89 |
| … | … |
| returned | 6 |
| cancelled | 4 |
| RETURNED | 3 |

```text
DELIVERY_OUTCOME COUNTS
```

| delivery_outcome | count |
|---|---|
| On Time | 1968 |
| Late | 280 |
| Unknown | 190 |
| Not Delivered | 62 |

```text
CUSTOMER_TIER COUNTS
```

| customer_tier | count |
|---|---|
| Silver | 1028 |
| Gold | 545 |
| Bronze | 441 |
| silver | 165 |
| gold | 120 |
| GOLD | 96 |
| Platinum | 91 |
| Missing | 14 |

### Code: charts

```python
def chart_eda(rt, claims):
    rt = rt.copy(); rt["mode"] = clean_mode(rt["transport_mode"])
    # 1 overview tiles
    fig, ax = plt.subplots(figsize=(11, 3)); ax.axis("off")
    known = rt["late_flag"].dropna().astype(float)
    tiles = [("Shipments", f"{len(rt):,}"), ("Freight cost", f"${rt['total_freight_cost'].astype(float).sum()/1e6:.2f}M"),
             ("Billed", f"${rt['total_billed_amount_usd'].astype(float).sum()/1e6:.2f}M"),
             ("Late (known outcomes)", f"{known.mean():.1%}"), ("Shipments with a claim", f"{(rt['claim_count'].astype(float)>0).mean():.1%}"),
             ("Carriers / customers", "14 / 60")]
    for i, (lab, val) in enumerate(tiles):
        x = i / len(tiles)
        ax.add_patch(FancyBboxPatch((x + .008, .1), .152, .8, boxstyle="round,pad=0.005", fc="#f4f3ef", ec="none", transform=ax.transAxes))
        ax.text(x + .084, .6, val, ha="center", fontsize=17, fontweight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(x + .084, .27, lab, ha="center", fontsize=7.5, color=SOFT, transform=ax.transAxes)
    ax.set_title("Portfolio at a glance (cleaned data)", pad=6)
    save(fig, "eda_01_portfolio_overview")
    # 2 delivery outcome + late by mode
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    o = rt["delivery_outcome"].fillna("Unknown").value_counts()
    axs[0].bar(o.index, o.values, color=[BLUE, ORANGE, GREY][:len(o)])
    for i, v in enumerate(o.values): axs[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=9)
    axs[0].set_title("Delivery outcomes")
    lm = late_rate(rt, "mode")["mean"].sort_values(ascending=False) * 100
    axs[1].barh(lm.index[::-1], lm.values[::-1], color=ORANGE, height=.55)
    for i, v in enumerate(lm.values[::-1]): axs[1].text(v + .3, i, f"{v:.1f}%", va="center", fontsize=9)
    axs[1].set_title("Late-delivery rate by transport mode"); axs[1].set_xlim(0, lm.max() * 1.25)
    save(fig, "eda_02_delivery_outcomes")
    # 3 cost per kg and weight
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    cpk = rt.groupby("mode")["cost_per_kg"].apply(lambda s: s.astype(float).median()).sort_values()
    axs[0].barh(cpk.index, cpk.values, color=BLUE, height=.55)
    for i, v in enumerate(cpk.values): axs[0].text(v + .05, i, f"${v:.2f}", va="center", fontsize=9)
    axs[0].set_title("Median freight cost per kg by mode"); axs[0].set_xlim(0, cpk.max() * 1.2)
    w = rt["weight_kg"].astype(float); w = w[w > 0]
    axs[1].hist(np.log10(w), bins=40, color=BLUE)
    axs[1].set_title("Shipment weight (log scale, kg)")
    axs[1].set_xticks([0, 1, 2, 3, 4, 5]); axs[1].set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
    axs[1].text(.98, .92, f"Max {w.max():,.0f} kg", transform=axs[1].transAxes, ha="right", fontsize=9, color=ORANGE)
    save(fig, "eda_03_cost_and_weight")
    # 4 over time
    t = rt.copy(); t["month"] = t["ship_date"].dt.to_period("M").dt.to_timestamp()
    m = t.groupby("month").size(); m = m[m >= 30]; lr = late_rate(t, "month"); lr = lr[lr.index.isin(m.index)]["mean"] * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(m.index, m.values, width=20, color=GREY, label="Shipments per month")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.plot(lr.index, lr.values, color=ORANGE, lw=2, marker="o", ms=3, label="Late rate (%)")
    ax.set_title("Shipments per month and late rate (months with 30+ shipments)"); ax.set_ylabel("Shipments"); ax2.set_ylabel("Late rate (%)")
    save(fig, "eda_04_shipments_over_time")
    # 5 SLA and carriers
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    s = rt.copy(); s["sla"] = s["contract_sla_days"].astype(float)
    ls = late_rate(s.dropna(subset=["sla"]), "sla"); ls = ls[ls["count"] >= 20]
    axs[0].bar(ls.index.astype(int).astype(str), ls["mean"] * 100, color=ORANGE)
    axs[0].set_title("Late rate by contract SLA (days)"); axs[0].set_xlabel("SLA days"); axs[0].set_ylabel("% late")
    c = rt.copy(); c["carrier"] = c["carrier_name"].astype("string").str.strip().str.title()
    lc = late_rate(c.dropna(subset=["carrier"]), "carrier"); lc = lc[lc["count"] >= 50]["mean"].sort_values() * 100
    axs[1].barh(lc.index, lc.values, color=BLUE, height=.6)
    for i, v in enumerate(lc.values): axs[1].text(v + .2, i, f"{v:.1f}%", va="center", fontsize=8)
    axs[1].set_title("Late rate by carrier"); axs[1].set_xlim(0, lc.max() * 1.2)
    save(fig, "eda_05_sla_and_carriers")
    # 6 cash
    inv = rt.copy()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    pay = pd.Series({"Paid": (inv["paid_invoice_count"].astype(float) > 0).sum(),
                     "Overdue": (inv["overdue_invoice_count"].astype(float) > 0).sum(),
                     "Disputed": (inv["disputed_invoice_count"].astype(float) > 0).sum(),
                     "Pending": (inv["pending_invoice_count"].astype(float) > 0).sum(),
                     "No invoice": (inv["invoice_count"].astype(float).fillna(0) == 0).sum()})
    axs[0].bar(pay.index, pay.values, color=[BLUE, ORANGE, ORANGE, GREY, "#b43c2a"])
    for i, v in enumerate(pay.values): axs[0].text(i, v + 15, f"{v:,}", ha="center", fontsize=9)
    axs[0].set_title("Shipments by invoice status")
    mg = inv.assign(margin=inv["total_billed_amount_usd"].astype(float) - inv["total_freight_cost"].astype(float)).dropna(subset=["margin"])
    mm = mg.groupby("mode")["margin"].mean().sort_values()
    axs[1].barh(mm.index, mm.values, color=BLUE, height=.55)
    for i, v in enumerate(mm.values): axs[1].text(v + 1, i, f"${v:,.0f}", va="center", fontsize=9)
    axs[1].set_title("Average margin proxy per shipment (billed minus freight)")
    save(fig, "eda_06_invoices_and_margin")
    # 7 claims
    cl = claims.copy()
    cl["type"] = cl["claim_type"].astype("string").str.strip().str.lower().str.title()
    cl["cause"] = cl["root_cause"].astype("string").str.strip().str.title().fillna("Unknown")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    a = cl.groupby("type")["claim_amount_usd"].sum().sort_values()
    axs[0].barh(a.index, a.values, color=ORANGE, height=.6); axs[0].set_title("Claim amount by type (USD)")
    b = cl.groupby("cause")["claim_amount_usd"].sum().sort_values()
    axs[1].barh(b.index, b.values, color=BLUE, height=.6); axs[1].set_title("Claim amount by root cause (USD)")
    save(fig, "eda_07_claims")
```

---

## 9. Analytical Base Table

One modelling table per objective, each with only the columns that are known **before** the outcome happens.

| Objective | Rows | Columns | Target |
|---|---|---|---|
| Late delivery | 2,248 | 17 | `late_flag` (1,968 on time, 280 late) |
| Freight cost | 2,500 | 15 | `total_freight_cost` |
| Payment risk | 1,899 | 12 | `payment_late` (1,440 paid, 459 late or disputed) |
| Claim risk | 2,500 | 17 | `has_claim` (2,270 no claim, 230 claim) |
| Anomaly review | 2,500 | 10 | None |

Late delivery and payment risk have fewer rows because shipments with an unknown delivery outcome and invoices still pending have no label.

### Code: analytical base tables

```python
import pandas as pd

if "relationship_table" not in globals() or "feature_tables" not in globals():
    raise NameError(
        "Run the feature engineering and relationship-table cells first."
    )

shipment_data = relationship_table.copy()
invoice_data = feature_tables["fact_invoices"].copy()

# Helper: keep only columns available in the source table.
def available_columns(df, columns):
    return [col for col in columns if col in df.columns]

# Candidate predictors for shipment-level objectives.
booking_features = [
    "customer_id",
    "customer_tier",
    "industry",
    "hq_country",
    "contract_sla_days",
    "credit_limit_usd",
    "carrier_id",
    "carrier_name",
    "contract_rate_per_kg_usd",
    "origin_country",
    "destination_country",
    "transport_mode",
    "weight_kg",
    "ship_date",
    "ship_month",
]

# 1. Late delivery prediction.
# Excludes actual delivery information and other post-shipment outcomes.
late_target = "late_flag"
late_columns = available_columns(
    shipment_data,
    ["shipment_id"] + booking_features + [late_target]
)
late_delivery_abt = shipment_data[late_columns].copy()
late_delivery_abt = late_delivery_abt.dropna(subset=[late_target])
late_delivery_abt[late_target] = late_delivery_abt[late_target].astype("int64")

# 2. Freight cost prediction.
# Excludes the freight and fuel components used to calculate the target.
freight_features = [
    col for col in booking_features
    if col not in {"customer_id", "credit_limit_usd"}
]
freight_target = "total_freight_cost"
freight_columns = available_columns(
    shipment_data,
    ["shipment_id"] + freight_features + [freight_target]
)
freight_cost_abt = shipment_data[freight_columns].dropna(
    subset=[freight_target]
).copy()

# 3. Invoice payment risk.
# Join each invoice to shipment/customer context at invoice grain.
invoice_context_columns = available_columns(
    shipment_data,
    [
        "shipment_id",
        "customer_id",
        "customer_tier",
        "credit_limit_usd",
        "industry",
        "transport_mode",
        "total_freight_cost",
    ]
)
invoice_context = shipment_data[invoice_context_columns].drop_duplicates(
    subset=["shipment_id"]
)

payment_risk_abt = invoice_data.merge(
    invoice_context,
    on="shipment_id",
    how="left",
    validate="many_to_one",
)

# Use freight cost to derive margin when invoice amount and freight cost exist.
if {"billed_amount_usd", "total_freight_cost"}.issubset(payment_risk_abt.columns):
    payment_risk_abt["margin_usd"] = (
        payment_risk_abt["billed_amount_usd"]
        - payment_risk_abt["total_freight_cost"]
    )

payment_target = "payment_late"
if payment_target in payment_risk_abt.columns:
    payment_risk_abt = payment_risk_abt.dropna(subset=[payment_target]).copy()
    payment_risk_abt[payment_target] = (
        payment_risk_abt[payment_target].astype("int64")
    )

# Remove post-payment fields and the status used to create the target.
payment_exclusions = [
    "paid_date",
    "days_to_pay",
    "payment_status",
    "open_balance",
    "ageing_bucket",
]
payment_risk_abt = payment_risk_abt.drop(
    columns=[col for col in payment_exclusions if col in payment_risk_abt.columns]
)

# 4. Claim risk.
# Target is whether one or more claims exist for the shipment.
claim_risk_abt = shipment_data[
    available_columns(
        shipment_data,
        ["shipment_id"] + booking_features + ["claim_count"]
    )
].copy()

if "claim_count" in claim_risk_abt.columns:
    claim_risk_abt["has_claim"] = (
        claim_risk_abt["claim_count"].fillna(0).gt(0).astype("int64")
    )
    claim_risk_abt = claim_risk_abt.drop(columns=["claim_count"])

# 5. Anomaly review dataset (unsupervised; no target column).
anomaly_features = available_columns(
    shipment_data,
    [
        "shipment_id",
        "ship_date",
        "transport_mode",
        "origin_country",
        "destination_country",
        "weight_kg",
        "cost_per_kg",
        "total_freight_cost",
        "total_billed_amount_usd",
    ]
)
anomaly_review_abt = shipment_data[anomaly_features].copy()

if {"total_billed_amount_usd", "total_freight_cost"}.issubset(
    anomaly_review_abt.columns
):
    anomaly_review_abt["margin_usd"] = (
        anomaly_review_abt["total_billed_amount_usd"]
        - anomaly_review_abt["total_freight_cost"]
    )

# Store the ABTs separately for their later, separate modeling workflows.
analytical_base_tables = {
    "late_delivery": late_delivery_abt,
    "freight_cost": freight_cost_abt,
    "payment_risk": payment_risk_abt,
    "claim_risk": claim_risk_abt,
    "anomaly_review": anomaly_review_abt,
}

# Display summary and any review items.
abt_summary = pd.DataFrame([
    {
        "objective": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_ID_rows": (
            df["shipment_id"].duplicated(keep=False).sum()
            if "shipment_id" in df.columns
            else df["invoice_id"].duplicated(keep=False).sum()
            if "invoice_id" in df.columns
            else 0
        ),
    }
    for name, df in analytical_base_tables.items()
]).set_index("objective")

print("ANALYTICAL BASE TABLE SUMMARY")
display(abt_summary)

print("\nTARGET DISTRIBUTIONS")
for name, target in [
    ("late_delivery", "late_flag"),
    ("payment_risk", "payment_late"),
    ("claim_risk", "has_claim"),
]:
    abt = analytical_base_tables[name]
    if target in abt.columns:
        print(f"\n{name}:")
        display(
            abt[target]
            .value_counts(dropna=False)
            .sort_index()
            .rename("count")
            .to_frame()
        )

print("\nFEATURE / TARGET REVIEW")
review_items = []

for name, target in [
    ("late_delivery", "late_flag"),
    ("freight_cost", "total_freight_cost"),
    ("payment_risk", "payment_late"),
    ("claim_risk", "has_claim"),
]:
    abt = analytical_base_tables[name]
    if target not in abt.columns:
        review_items.append({
            "table": name,
            "issue": f"Target column '{target}' is missing",
            "count": len(abt),
        })
    elif abt[target].isna().any():
        review_items.append({
            "table": name,
            "issue": f"Missing values in target '{target}'",
            "count": int(abt[target].isna().sum()),
        })

if review_items:
    display(pd.DataFrame(review_items))
else:
    print("No missing target columns or target values found.")

print("\nSAMPLE: LATE DELIVERY ABT")
display(late_delivery_abt.head())
```

**Output**

```text
ANALYTICAL BASE TABLE SUMMARY
```

| objective | rows | columns | duplicate_ID_rows |
|---|---|---|---|
| late_delivery | 2248 | 17 | 0 |
| freight_cost | 2500 | 15 | 0 |
| payment_risk | 1899 | 12 | 0 |
| claim_risk | 2500 | 17 | 0 |
| anomaly_review | 2500 | 10 | 0 |

```text
TARGET DISTRIBUTIONS

late_delivery:
```

| late_flag | count |
|---|---|
| 0 | 1968 |
| 1 | 280 |

```text
payment_risk:
```

| payment_late | count |
|---|---|
| 0 | 1440 |
| 1 | 459 |

```text
claim_risk:
```

| has_claim | count |
|---|---|
| 0 | 2270 |
| 1 | 230 |

```text
FEATURE / TARGET REVIEW
No missing target columns or target values found.

SAMPLE: LATE DELIVERY ABT
```

| index | shipment_id | customer_id | customer_tier | … | ship_date | ship_month | late_flag |
|---|---|---|---|---|---|---|---|
| 1 | SHP-50821 | CUS-1044 | Gold | … | 2025-01-16 00:00:00 | 1 | 0 |
| 2 | SHP-51381 | CUS-1035 | Bronze | … | 2025-12-06 00:00:00 | 12 | 0 |
| 4 | SHP-52393 | CUS-1030 | Gold | … | 2025-03-05 00:00:00 | 3 | 0 |
| 5 | SHP-50710 | CUS-1041 | Gold | … | 2025-08-28 00:00:00 | 8 | 0 |
| 6 | SHP-51823 | CUS-1050 | silver | … | 2025-10-24 00:00:00 | 10 | 1 |


---

## 10. Encode Categorical Columns

Text columns (customer tier, industry, countries, carrier, transport mode) become 0/1 dummy columns with `pd.get_dummies`, with a `_nan` column for missing values. Because the spelling variants were never merged, the late-delivery table grows from 17 to 189 columns.

| Objective | Text columns encoded | Dummy columns added | Columns after |
|---|---|---|---|
| Late delivery | 7 | 172 | 189 |
| Freight cost | 7 | 172 | 187 |
| Payment risk | 3 | 28 | 40 |
| Claim risk | 7 | 172 | 189 |
| Anomaly review | 3 | 113 | 123 |

### Charts

<p align="center">
  <img src="images/encode_01_columns.png" alt="Columns before and after encoding" width="100%">
</p>
<p align="center"><em>Columns before and after encoding</em></p>

### Code: encode categorical columns

```python
import pandas as pd

if "analytical_base_tables" not in globals():
    raise NameError(
        "analytical_base_tables is not defined. Run the ABT cell first."
    )

target_columns = {
    "late_delivery": "late_flag",
    "freight_cost": "total_freight_cost",
    "payment_risk": "payment_late",
    "claim_risk": "has_claim",
    "anomaly_review": None,
}

# IDs are retained for tracking, but should not become model predictors.
identifier_columns = {
    "shipment_id",
    "invoice_id",
    "customer_id",
    "carrier_id",
}

# Dates remain unchanged for chronological splitting or later feature extraction.
date_columns = {
    "ship_date",
    "invoice_date",
    "promised_delivery_date",
    "actual_delivery_date",
    "paid_date",
}

encoded_base_tables = {}
encoding_summary = []

for name, original_df in analytical_base_tables.items():
    df = original_df.copy()
    target = target_columns.get(name)

    # Select text-valued columns, excluding IDs, targets, and dates.
    categorical_columns = [
        col for col in df.columns
        if col not in identifier_columns
        and col not in date_columns
        and col != target
        and df[col].map(lambda value: isinstance(value, str)).any()
    ]

    if categorical_columns:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            dummy_na=True,
            dtype="int8",
        )

    encoded_base_tables[name] = df

    encoding_summary.append({
        "objective": name,
        "categorical_columns_encoded": len(categorical_columns),
        "encoded_columns_added": (
            df.shape[1] - original_df.shape[1]
        ),
        "rows": df.shape[0],
        "columns_after_encoding": df.shape[1],
    })

print("CATEGORICAL ENCODING SUMMARY")
display(pd.DataFrame(encoding_summary).set_index("objective"))

print("\nENCODED TABLE SAMPLE: LATE DELIVERY")
display(encoded_base_tables["late_delivery"].head())
```

**Output**

```text
CATEGORICAL ENCODING SUMMARY
```

| objective | categorical_columns_encoded | encoded_columns_added | rows | columns_after_encoding |
|---|---|---|---|---|
| late_delivery | 7 | 172 | 2248 | 189 |
| freight_cost | 7 | 172 | 2500 | 187 |
| payment_risk | 3 | 28 | 1899 | 40 |
| claim_risk | 7 | 172 | 2500 | 189 |
| anomaly_review | 3 | 113 | 2500 | 123 |

```text
ENCODED TABLE SAMPLE: LATE DELIVERY
```

| index | shipment_id | customer_id | contract_sla_days | … | transport_mode_road | transport_mode_sea | transport_mode_nan |
|---|---|---|---|---|---|---|---|
| 1 | SHP-50821 | CUS-1044 | 7 | … | 0 | 0 | 0 |
| 2 | SHP-51381 | CUS-1035 | 7 | … | 0 | 0 | 0 |
| 4 | SHP-52393 | CUS-1030 | 10 | … | 0 | 0 | 0 |
| 5 | SHP-50710 | CUS-1041 | <NA> | … | 0 | 0 | 0 |
| 6 | SHP-51823 | CUS-1050 | 7 | … | 0 | 0 | 0 |

### Code: charts

```python
def chart_encoding():
    names = ["Late delivery", "Freight cost", "Payment risk", "Claim risk", "Anomaly review"]
    before, after = [17, 15, 12, 17, 10], [189, 187, 40, 189, 123]
    fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(5)
    ax.bar(x - .2, before, .4, color=GREY, label="Before encoding"); ax.bar(x + .2, after, .4, color=BLUE, label="After encoding")
    for i, v in enumerate(after): ax.text(i + .2, v + 3, str(v), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(frameon=False)
    ax.set_title("Columns before and after one-hot encoding")
    save(fig, "encode_01_columns")
```

---

## 11. Separate X and y

Predictors (X) and the target (y) are split apart. IDs and dates are held back as metadata so predictions can be traced to a shipment, but they never enter the model.

| Objective | Rows | Predictor columns | Target |
|---|---|---|---|
| Late delivery | 2,248 | 184 | `late_flag` |
| Freight cost | 2,500 | 183 | `total_freight_cost` |
| Payment risk | 1,899 | 35 | `payment_late` |
| Claim risk | 2,500 | 184 | `has_claim` |
| Anomaly review | 2,500 | 121 | none |

### Code: separate X and y

```python
import pandas as pd

if "encoded_base_tables" not in globals():
    raise NameError(
        "encoded_base_tables is not defined. Run the encoding cell first."
    )

target_columns = {
    "late_delivery": "late_flag",
    "freight_cost": "total_freight_cost",
    "payment_risk": "payment_late",
    "claim_risk": "has_claim",
    "anomaly_review": None,
}

identifier_columns = {
    "shipment_id",
    "invoice_id",
    "customer_id",
    "carrier_id",
}

date_columns = {
    "ship_date",
    "invoice_date",
    "promised_delivery_date",
    "actual_delivery_date",
    "paid_date",
}

X_tables = {}
y_tables = {}
metadata_tables = {}
separation_summary = []
non_numeric_review = []

for objective, df in encoded_base_tables.items():
    target = target_columns.get(objective)

    # Keep row identifiers and dates outside of X for traceability and time splitting.
    metadata_cols = [
        col for col in df.columns
        if col in identifier_columns or col in date_columns
    ]
    metadata_tables[objective] = df[metadata_cols].copy()

    excluded = set(metadata_cols)
    if target is not None:
        if target not in df.columns:
            raise KeyError(
                f"Target '{target}' is missing from the {objective} table."
            )
        excluded.add(target)

    X = df.drop(columns=list(excluded), errors="ignore").copy()
    X_tables[objective] = X

    if target is not None:
        y_tables[objective] = df[target].copy()

    # Report any nonnumeric predictor columns that still need attention.
    remaining_text = [
        col for col in X.columns
        if X[col].map(lambda value: isinstance(value, str)).any()
    ]
    if remaining_text:
        non_numeric_review.append({
            "objective": objective,
            "columns": ", ".join(remaining_text),
        })

    separation_summary.append({
        "objective": objective,
        "rows": len(df),
        "predictor_columns": X.shape[1],
        "target": target if target is not None else "None (unsupervised)",
        "metadata_columns": len(metadata_cols),
    })

print("X / y SEPARATION SUMMARY")
display(pd.DataFrame(separation_summary).set_index("objective"))

print("\nNON-NUMERIC PREDICTOR REVIEW")
if non_numeric_review:
    display(pd.DataFrame(non_numeric_review))
else:
    print("No text-valued predictors found.")

print("\nEXAMPLE: LATE DELIVERY")
print("X:")
display(X_tables["late_delivery"].head())
print("y:")
display(y_tables["late_delivery"].head())
```

**Output**

```text
X / y SEPARATION SUMMARY
```

| objective | rows | predictor_columns | target | metadata_columns |
|---|---|---|---|---|
| late_delivery | 2248 | 184 | late_flag | 4 |
| freight_cost | 2500 | 183 | total_freight_cost | 3 |
| payment_risk | 1899 | 35 | payment_late | 4 |
| claim_risk | 2500 | 184 | has_claim | 4 |
| anomaly_review | 2500 | 121 | None (unsupervised) | 2 |

```text
NON-NUMERIC PREDICTOR REVIEW
No text-valued predictors found.

EXAMPLE: LATE DELIVERY
X:
```

| index | contract_sla_days | credit_limit_usd | contract_rate_per_kg_usd | … | transport_mode_road | transport_mode_sea | transport_mode_nan |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 7.2e+04 | 2.98 | … | 0 | 0 | 0 |
| 2 | 7 | 3.4e+05 | 5.98 | … | 0 | 0 | 0 |
| 4 | 10 | 1.93e+05 | 4.47 | … | 0 | 0 | 0 |
| 5 | <NA> | 2.49e+05 | 2.34 | … | 0 | 0 | 0 |
| 6 | 7 | 1.12e+05 | 5.22 | … | 0 | 0 | 0 |

```text
y:
```

| index | late_flag |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 1 |


---

## 12. Train Test Split

The split is **chronological**: the earliest 80% of rows train the model and the latest 20% test it, so the model is judged on shipments it could not have seen. Payment risk is split by invoice date, the rest by ship date. Anomaly review is unsupervised, so it uses all 2,500 rows.

| Objective | Split on | Cut-off date | Train rows | Test rows |
|---|---|---|---|---|
| Late delivery | `ship_date` | 2026-03-15 | 1,781 | 446 |
| Freight cost | `ship_date` | 2026-03-14 | 1,980 | 496 |
| Claim risk | `ship_date` | 2026-03-14 | 1,980 | 496 |
| Payment risk | `invoice_date` | 2026-03-20 | 1,504 | 378 |

### Charts

<p align="center">
  <img src="images/split_01_train_test_timeline.png" alt="Train and test periods" width="100%">
</p>
<p align="center"><em>Train and test periods</em></p>

### Code: train test split

```python
import pandas as pd

required = ["X_tables", "y_tables", "metadata_tables"]
missing = [name for name in required if name not in globals()]
if missing:
    raise NameError(
        f"Missing {missing}. Run the X/y separation cell first."
    )

date_for_objective = {
    "late_delivery": "ship_date",
    "freight_cost": "ship_date",
    "claim_risk": "ship_date",
    "payment_risk": "invoice_date",
}

X_train = {}
X_test = {}
y_train = {}
y_test = {}
split_summary = []

# Keep the unsupervised table intact; it has no target or train/test split.
X_unsupervised = X_tables.get("anomaly_review")

for objective, date_column in date_for_objective.items():
    X = X_tables[objective]
    y = y_tables[objective]
    metadata = metadata_tables[objective]

    if date_column not in metadata.columns:
        raise KeyError(
            f"{date_column} is missing from {objective} metadata."
        )

    dates = pd.to_datetime(metadata[date_column], errors="coerce")
    valid_date = dates.notna()

    # Exclude rows with unusable dates from this chronological split.
    X_valid = X.loc[valid_date].copy()
    y_valid = y.loc[valid_date].copy()
    dates_valid = dates.loc[valid_date]

    if len(X_valid) < 2:
        raise ValueError(
            f"Not enough dated rows to split {objective}."
        )

    # Use the date at the 80% point as cutoff.
    # Rows on the same date stay together in one split.
    sorted_dates = dates_valid.sort_values()
    cutoff_position = min(
        int(len(sorted_dates) * 0.8),
        len(sorted_dates) - 1,
    )
    cutoff_date = sorted_dates.iloc[cutoff_position]

    train_mask = dates_valid < cutoff_date
    test_mask = dates_valid >= cutoff_date

    # If ties at the cutoff leave no training rows, fall back to <= / >.
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        train_mask = dates_valid <= cutoff_date
        test_mask = dates_valid > cutoff_date

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Could not create non-empty chronological splits for {objective}. "
            "There may be too few distinct dates."
        )

    X_train[objective] = X_valid.loc[train_mask].copy()
    X_test[objective] = X_valid.loc[test_mask].copy()
    y_train[objective] = y_valid.loc[train_mask].copy()
    y_test[objective] = y_valid.loc[test_mask].copy()

    split_summary.append({
        "objective": objective,
        "split_date_column": date_column,
        "cutoff_date": cutoff_date.date(),
        "train_rows": len(X_train[objective]),
        "test_rows": len(X_test[objective]),
        "rows_excluded_missing_date": int((~valid_date).sum()),
    })

print("CHRONOLOGICAL SPLIT SUMMARY")
display(pd.DataFrame(split_summary).set_index("objective"))

print("\nANOMALY REVIEW")
print(
    f"No train/test split applied. "
    f"{len(X_unsupervised):,} rows retained for unsupervised review."
)
```

**Output**

```text
CHRONOLOGICAL SPLIT SUMMARY
```

| objective | split_date_column | cutoff_date | train_rows | test_rows | rows_excluded_missing_date |
|---|---|---|---|---|---|
| late_delivery | ship_date | 2026-03-15 | 1781 | 446 | 21 |
| freight_cost | ship_date | 2026-03-14 | 1980 | 496 | 24 |
| claim_risk | ship_date | 2026-03-14 | 1980 | 496 | 24 |
| payment_risk | invoice_date | 2026-03-20 | 1504 | 378 | 17 |

```text
ANOMALY REVIEW
No train/test split applied. 2,500 rows retained for unsupervised review.
```

### Code: charts

```python
def chart_split(metadata_tables, date_for_objective, y_train, y_test):
    names = {"late_delivery": "Late delivery", "freight_cost": "Freight cost", "payment_risk": "Payment risk", "claim_risk": "Claim risk"}
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 6.4), sharex=True)
    for ax, (k, nm) in zip(axs, names.items()):
        d = pd.to_datetime(metadata_tables[k][date_for_objective[k]])
        tr, te = d.loc[y_train[k].index], d.loc[y_test[k].index]
        for s, col, lab in ((tr, BLUE, "Train (earliest 80%)"), (te, ORANGE, "Test (latest 20%)")):
            m = s.dt.to_period("M").dt.to_timestamp().value_counts().sort_index()
            ax.bar(m.index, m.values, width=20, color=col, label=lab)
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=9)
        ax.axvline(te.min(), color=INK, lw=1, ls="--")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(.98, .99))
    fig.suptitle("Chronological split: the model never sees the future", x=.01, ha="left", fontweight="bold", fontsize=11)
    save(fig, "split_01_train_test_timeline")
```

---

## 13. Feature Scaling of Numerical Columns

A `StandardScaler` is fitted on the **training rows only** and applied to the continuous columns (4 or 5 per model). The 0/1 dummy columns are left alone. Fitting on training data only prevents test information leaking into the scaler. The scaler is saved and reused by the app.

### Code: feature scaling

```python
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

if not all(
    name in globals()
    for name in ["X_train", "X_test", "y_train", "y_test"]
):
    raise NameError(
        "Train/test variables are missing. Run the split cell first."
    )

feature_scalers = {}
scaled_X_train = {}
scaled_X_test = {}
scaling_summary = []

for objective in X_train:
    train_df = X_train[objective].copy()
    test_df = X_test[objective].copy()

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            f"Train/test feature columns do not match for {objective}."
        )

    # Select numeric predictors.
    numeric_columns = [
        col for col in train_df.columns
        if is_numeric_dtype(train_df[col])
    ]

    # Leave binary 0/1 columns (including one-hot columns) unchanged.
    continuous_columns = []
    binary_columns = []

    for col in numeric_columns:
        values = set(train_df[col].dropna().unique())
        if values and values.issubset({0, 1}):
            binary_columns.append(col)
        else:
            continuous_columns.append(col)

    # StandardScaler is fit only on the training rows.
    scaler = StandardScaler()

    if continuous_columns:
        train_df.loc[:, continuous_columns] = scaler.fit_transform(
            train_df[continuous_columns]
        )
        test_df.loc[:, continuous_columns] = scaler.transform(
            test_df[continuous_columns]
        )

    feature_scalers[objective] = scaler
    scaled_X_train[objective] = train_df
    scaled_X_test[objective] = test_df

    missing_continuous = int(
        train_df[continuous_columns].isna().sum().sum()
    ) if continuous_columns else 0

    scaling_summary.append({
        "objective": objective,
        "continuous_columns_scaled": len(continuous_columns),
        "binary_columns_left_unchanged": len(binary_columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "missing_values_in_scaled_columns": missing_continuous,
    })

print("FEATURE SCALING SUMMARY")
display(pd.DataFrame(scaling_summary).set_index("objective"))

print("\nExample scaled training data: late delivery")
display(scaled_X_train["late_delivery"].head())
```

**Output**

```text
FEATURE SCALING SUMMARY
```

| objective | continuous_columns_scaled | binary_columns_left_unchanged | train_rows | test_rows | missing_values_in_scaled_columns |
|---|---|---|---|---|---|
| late_delivery | 5 | 179 | 1781 | 446 | 108 |
| freight_cost | 4 | 179 | 1980 | 496 | 106 |
| claim_risk | 5 | 179 | 1980 | 496 | 117 |
| payment_risk | 4 | 31 | 1504 | 378 | 27 |

```text
Example scaled training data: late delivery
```

| index | contract_sla_days | credit_limit_usd | contract_rate_per_kg_usd | … | transport_mode_road | transport_mode_sea | transport_mode_nan |
|---|---|---|---|---|---|---|---|
| 1 | 0.5591 | -1.347 | -0.3253 | … | 0 | 0 | 0 |
| 2 | 0.5591 | 1.131 | 1.633 | … | 0 | 0 | 0 |
| 4 | 1.717 | -0.2282 | 0.6475 | … | 0 | 0 | 0 |
| 5 | <NA> | 0.2896 | -0.7431 | … | 0 | 0 | 0 |
| 6 | 0.5591 | -0.9773 | 1.137 | … | 0 | 0 | 0 |


---

## 14. Train Model

Four candidate models per objective, each inside a pipeline with a median imputer for missing values. The anomaly objective trains three detectors instead.

| Objective | Candidates |
|---|---|
| Late delivery, payment risk, claim risk | Logistic Regression, Random Forest, Extra Trees, Gradient Boosting |
| Freight cost | Ridge Regression, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor |
| Anomaly review | Isolation Forest, Local Outlier Factor, One-Class SVM |

### Code: train model

#### Objective 1: Raise on-time delivery and SLA adherence

```python
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

objective = "late_delivery"

X_train_late = scaled_X_train[objective].copy()
y_train_late = y_train[objective].copy()

if not X_train_late.index.equals(y_train_late.index):
    raise ValueError("Training predictors and target rows are misaligned.")
if y_train_late.isna().any():
    raise ValueError("The training target contains missing values.")
if y_train_late.nunique() < 2:
    raise ValueError("Training data must contain both target classes.")

candidate_models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

models = {}

for model_name, classifier in candidate_models.items():
    pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train_late, y_train_late)
    models[model_name] = pipeline

training_summary = pd.DataFrame([
    {
        "model": model_name,
        "status": "trained",
        "training_rows": len(X_train_late),
        "predictor_columns": X_train_late.shape[1],
    }
    for model_name in models
]).set_index("model")

print("BUSINESS OBJECTIVE 1: CANDIDATE MODELS TRAINED")
display(training_summary)

print("\nTraining target counts:")
display(y_train_late.value_counts().sort_index().rename("shipment_count").to_frame())
```

**Output**

```text
BUSINESS OBJECTIVE 1: CANDIDATE MODELS TRAINED
```

| model | status | training_rows | predictor_columns |
|---|---|---|---|
| Logistic Regression | trained | 1781 | 184 |
| Random Forest | trained | 1781 | 184 |
| Extra Trees | trained | 1781 | 184 |
| Gradient Boosting | trained | 1781 | 184 |

```text
Training target counts:
```

| late_flag | shipment_count |
|---|---|
| 0 | 1547 |
| 1 | 234 |

#### Objective 2: Cut freight cost and protect margin

```python
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

objective = "freight_cost"

X_train_freight = scaled_X_train[objective].copy()
y_train_freight = y_train[objective].copy()

if not X_train_freight.index.equals(y_train_freight.index):
    raise ValueError("Training predictors and target rows are misaligned.")
if y_train_freight.isna().any():
    raise ValueError("The freight-cost training target contains missing values.")

candidate_regressors = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

freight_cost_models = {}

for model_name, regressor in candidate_regressors.items():
    pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("regressor", regressor),
    ])

    pipeline.fit(X_train_freight, y_train_freight)
    freight_cost_models[model_name] = pipeline

# Preserve objective 1 models and store objective 2 separately.
if "models_by_objective" not in globals():
    models_by_objective = {}

if "models" in globals() and "late_delivery" not in models_by_objective:
    models_by_objective["late_delivery"] = models.copy()

models_by_objective["freight_cost"] = freight_cost_models

print("BUSINESS OBJECTIVE 2: FREIGHT-COST MODELS TRAINED")
print(f"Training rows: {len(X_train_freight):,}")
print(f"Predictor columns: {X_train_freight.shape[1]:,}")
print(f"Training target mean: ${y_train_freight.mean():,.2f}")
print(f"Training target median: ${y_train_freight.median():,.2f}")
print(f"Candidate models trained: {len(freight_cost_models)}")

display(pd.DataFrame({
    "model": list(freight_cost_models.keys()),
    "status": ["trained"] * len(freight_cost_models),
}).set_index("model"))
```

**Output**

```text
BUSINESS OBJECTIVE 2: FREIGHT-COST MODELS TRAINED
Training rows: 1,980
Predictor columns: 183
Training target mean: $691.39
Training target median: $314.78
Candidate models trained: 4
```

| model | status |
|---|---|
| Ridge Regression | trained |
| Random Forest Regressor | trained |
| Extra Trees Regressor | trained |
| Gradient Boosting Regressor | trained |

#### Objective 3: Close revenue leakage and speed up cash collection

```python
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

objective = "payment_risk"

X_train_payment = scaled_X_train[objective].copy()
y_train_payment = y_train[objective].copy()

if not X_train_payment.index.equals(y_train_payment.index):
    raise ValueError("Training predictors and target rows are misaligned.")
if y_train_payment.isna().any():
    raise ValueError("The training target contains missing values.")
if y_train_payment.nunique() < 2:
    raise ValueError("Training data must contain both target classes.")

candidate_classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

payment_risk_models = {}

for model_name, classifier in candidate_classifiers.items():
    pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train_payment, y_train_payment)
    payment_risk_models[model_name] = pipeline

models_by_objective["payment_risk"] = payment_risk_models

print("BUSINESS OBJECTIVE 3: PAYMENT-RISK MODELS TRAINED")
print(f"Training invoices: {len(X_train_payment):,}")
print(f"Predictor columns: {X_train_payment.shape[1]:,}")
print(f"Candidate models trained: {len(payment_risk_models)}")

print("\nTraining target counts (0 = Paid, 1 = Overdue/Disputed):")
display(
    y_train_payment.value_counts()
    .sort_index()
    .rename("invoice_count")
    .to_frame()
)

display(pd.DataFrame({
    "model": list(payment_risk_models.keys()),
    "status": ["trained"] * len(payment_risk_models),
}).set_index("model"))
```

**Output**

```text
BUSINESS OBJECTIVE 3: PAYMENT-RISK MODELS TRAINED
Training invoices: 1,504
Predictor columns: 35
Candidate models trained: 4

Training target counts (0 = Paid, 1 = Overdue/Disputed):
```

| payment_late | invoice_count |
|---|---|
| 0 | 1137 |
| 1 | 367 |

| model | status |
|---|---|
| Logistic Regression | trained |
| Random Forest | trained |
| Extra Trees | trained |
| Gradient Boosting | trained |

#### Objective 4: Reduce claims, damage and loss

```python
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

objective = "claim_risk"

X_train_claim = scaled_X_train[objective].copy()
y_train_claim = y_train[objective].copy()

if not X_train_claim.index.equals(y_train_claim.index):
    raise ValueError("Training predictors and target rows are misaligned.")
if y_train_claim.isna().any():
    raise ValueError("The claim-risk training target contains missing values.")
if y_train_claim.nunique() < 2:
    raise ValueError("Training data must contain both target classes.")

candidate_classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

claim_risk_models = {}

for model_name, classifier in candidate_classifiers.items():
    pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("classifier", classifier),
    ])

    if model_name == "Gradient Boosting":
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train_claim,
        )
        pipeline.fit(
            X_train_claim,
            y_train_claim,
            classifier__sample_weight=sample_weights,
        )
    else:
        pipeline.fit(X_train_claim, y_train_claim)

    claim_risk_models[model_name] = pipeline

models_by_objective["claim_risk"] = claim_risk_models

print("BUSINESS OBJECTIVE 4: CLAIM-RISK MODELS TRAINED")
print(f"Training shipments: {len(X_train_claim):,}")
print(f"Predictor columns: {X_train_claim.shape[1]:,}")
print(f"Candidate models trained: {len(claim_risk_models)}")

print("\nTraining target counts (0 = No Claim, 1 = Claim):")
display(
    y_train_claim.value_counts()
    .sort_index()
    .rename("shipment_count")
    .to_frame()
)

display(pd.DataFrame({
    "model": list(claim_risk_models.keys()),
    "status": ["trained"] * len(claim_risk_models),
}).set_index("model"))
```

**Output**

```text
BUSINESS OBJECTIVE 4: CLAIM-RISK MODELS TRAINED
Training shipments: 1,980
Predictor columns: 184
Candidate models trained: 4

Training target counts (0 = No Claim, 1 = Claim):
```

| has_claim | shipment_count |
|---|---|
| 0 | 1801 |
| 1 | 179 |

| model | status |
|---|---|
| Logistic Regression | trained |
| Random Forest | trained |
| Extra Trees | trained |
| Gradient Boosting | trained |

#### Objective 5: Shipment and billing anomaly review

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if "X_unsupervised" not in globals() or X_unsupervised is None:
    raise NameError(
        "The anomaly review features are missing. "
        "Run the train/test split cell first."
    )

X_anomaly = X_unsupervised.copy()

if X_anomaly.empty:
    raise ValueError("The anomaly review table contains no rows.")
if not all(pd.api.types.is_numeric_dtype(X_anomaly[c]) for c in X_anomaly.columns):
    raise ValueError(
        "Anomaly predictors must be numeric. Rerun categorical encoding "
        "and X/y separation for the anomaly review table."
    )

anomaly_models = {
    "Isolation Forest": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("detector", IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "Local Outlier Factor": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("detector", LocalOutlierFactor(
            n_neighbors=20,
            contamination="auto",
            novelty=True,
        )),
    ]),
    "One-Class SVM": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("detector", OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
        )),
    ]),
}

for model in anomaly_models.values():
    model.fit(X_anomaly)

models_by_objective["anomaly_review"] = anomaly_models

print("BUSINESS OBJECTIVE 5: ANOMALY DETECTORS TRAINED")
print(f"Shipment rows used: {len(X_anomaly):,}")
print(f"Predictor columns: {X_anomaly.shape[1]:,}")
print(f"Candidate detectors trained: {len(anomaly_models)}")
display(pd.DataFrame({
    "model": list(anomaly_models.keys()),
    "status": ["trained on full anomaly table"] * len(anomaly_models),
}).set_index("model"))
```

**Output**

```text
BUSINESS OBJECTIVE 5: ANOMALY DETECTORS TRAINED
Shipment rows used: 2,500
Predictor columns: 121
Candidate detectors trained: 3
```

| model | status |
|---|---|
| Isolation Forest | trained on full anomaly table |
| Local Outlier Factor | trained on full anomaly table |
| One-Class SVM | trained on full anomaly table |


---

## 15. Predict X_test

Every trained model scores the held-out test rows. The prediction tables keep the shipment or invoice ID and date, so any score can be traced back. For the anomaly detectors, the whole table is scored and flagged.

### Code: predict X_test

#### Objective 1: Raise on-time delivery and SLA adherence

```python
import pandas as pd

objective = "late_delivery"

if objective not in X_test:
    raise KeyError("Late-delivery test features are missing.")
if "models" not in globals() or len(models) == 0:
    raise NameError("No trained models found. Run the model-training cell first.")

X_test_late = X_test[objective].copy()
objective1_predictions = pd.DataFrame(index=X_test_late.index)

# Include shipment IDs and split dates for traceability, if available.
if objective in metadata_tables:
    metadata = metadata_tables[objective].reindex(X_test_late.index)
    display_columns = [
        col for col in ["shipment_id", "ship_date"]
        if col in metadata.columns
    ]
    for col in display_columns:
        objective1_predictions[col] = metadata[col]

for model_name, model in models.items():
    predicted_labels = model.predict(X_test_late)

    classifier = model.named_steps["classifier"]
    class_list = list(classifier.classes_)
    positive_class_position = class_list.index(1)
    late_probabilities = model.predict_proba(X_test_late)[
        :, positive_class_position
    ]

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    objective1_predictions[f"{safe_name}_prediction"] = predicted_labels
    objective1_predictions[f"{safe_name}_late_probability"] = late_probabilities

print(f"Predictions generated for {len(X_test_late):,} test shipments "
      f"using {len(models)} models.")
display(objective1_predictions.head(5))
```

**Output**

```text
Predictions generated for 446 test shipments using 4 models.
```

| index | shipment_id | ship_date | logistic_regression_prediction | … | extra_trees_late_probability | gradient_boosting_prediction | gradient_boosting_late_probability |
|---|---|---|---|---|---|---|---|
| 9 | SHP-51603 | 2026-03-18 00:00:00 | 1 | … | 0.12 | 0 | 0.01699 |
| 10 | SHP-51426 | 2026-03-21 00:00:00 | 1 | … | 0.04333 | 0 | 0.01524 |
| 14 | SHP-51296 | 2026-04-22 00:00:00 | 1 | … | 0.003333 | 0 | 0.01757 |
| 30 | SHP-50843 | 2026-06-14 00:00:00 | 1 | … | 0.02 | 0 | 0.01757 |
| 31 | SHP-51330 | 2026-03-22 00:00:00 | 1 | … | 0.08 | 0 | 0.0163 |

#### Objective 2: Cut freight cost and protect margin

```python
import pandas as pd

objective = "freight_cost"

if objective not in scaled_X_test:
    raise KeyError("Freight-cost test features are missing.")
if (
    "models_by_objective" not in globals()
    or objective not in models_by_objective
):
    raise NameError("Train the objective 2 models first.")

X_test_freight = scaled_X_test[objective].copy()
freight_cost_predictions = pd.DataFrame(index=X_test_freight.index)

# Include shipment identifiers and dates for traceability.
metadata = metadata_tables[objective].reindex(X_test_freight.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        freight_cost_predictions[col] = metadata[col]

for model_name, model in models_by_objective[objective].items():
    predicted_cost = model.predict(X_test_freight)
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    freight_cost_predictions[f"{safe_name}_prediction_usd"] = predicted_cost

print(
    f"Predictions generated for {len(X_test_freight):,} test shipments "
    f"using {len(models_by_objective[objective])} models."
)
display(freight_cost_predictions.head(5))
```

**Output**

```text
Predictions generated for 496 test shipments using 4 models.
```

| index | shipment_id | ship_date | ridge_regression_prediction_usd | random_forest_regressor_predict... | extra_trees_regressor_predictio... | gradient_boosting_regressor_pre... |
|---|---|---|---|---|---|---|
| 0 | SHP-50612 | 2026-05-16 00:00:00 | 764.9 | 344.2 | 307.5 | 340.5 |
| 3 | SHP-50964 | 2026-03-22 00:00:00 | 36.2 | 24.73 | 35.42 | -6.43 |
| 9 | SHP-51603 | 2026-03-18 00:00:00 | 235.1 | 293 | 187.1 | 203.2 |
| 10 | SHP-51426 | 2026-03-21 00:00:00 | 217 | 128.6 | 218.6 | 93.21 |
| 14 | SHP-51296 | 2026-04-22 00:00:00 | 345.1 | 192.1 | 179.1 | 131.1 |

#### Objective 3: Close revenue leakage and speed up cash collection

```python
import pandas as pd

objective = "payment_risk"

X_test_payment = scaled_X_test[objective].copy()
payment_risk_predictions = pd.DataFrame(index=X_test_payment.index)

if "payment_risk" not in models_by_objective:
    raise NameError("Train the objective 3 models first.")

# Include invoice and shipment IDs plus invoice date for traceability.
metadata = metadata_tables[objective].reindex(X_test_payment.index)
for col in ["invoice_id", "shipment_id", "invoice_date"]:
    if col in metadata.columns:
        payment_risk_predictions[col] = metadata[col]

for model_name, model in models_by_objective[objective].items():
    predicted_labels = model.predict(X_test_payment)

    classes = list(model.named_steps["classifier"].classes_)
    positive_position = classes.index(1)
    late_probabilities = model.predict_proba(X_test_payment)[
        :, positive_position
    ]

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    payment_risk_predictions[f"{safe_name}_prediction"] = predicted_labels
    payment_risk_predictions[
        f"{safe_name}_late_probability"
    ] = late_probabilities

print(
    f"Predictions generated for {len(X_test_payment):,} test invoices "
    f"using {len(models_by_objective[objective])} models."
)
display(payment_risk_predictions.head(5))
```

**Output**

```text
Predictions generated for 378 test invoices using 4 models.
```

| index | invoice_id | shipment_id | invoice_date | … | extra_trees_late_probability | gradient_boosting_prediction | gradient_boosting_late_probability |
|---|---|---|---|---|---|---|---|
| 5 | INV-700005 | SHP-51622 | 2026-06-16 00:00:00 | … | 0.4167 | 0 | 0.2175 |
| 8 | INV-700008 | SHP-50165 | 2026-05-21 00:00:00 | … | 0.1367 | 0 | 0.1927 |
| 9 | INV-700009 | SHP-50395 | 2026-06-28 00:00:00 | … | 0.3533 | 0 | 0.2 |
| 16 | INV-700016 | SHP-50887 | 2026-04-15 00:00:00 | … | 0.2833 | 0 | 0.2866 |
| 20 | INV-700020 | SHP-52301 | 2026-05-12 00:00:00 | … | 0.02479 | 0 | 0.2244 |

#### Objective 4: Reduce claims, damage and loss

```python
import pandas as pd

objective = "claim_risk"

if objective not in scaled_X_test:
    raise KeyError("Claim-risk test features are missing.")
if (
    "models_by_objective" not in globals()
    or objective not in models_by_objective
):
    raise NameError("Train the objective 4 models first.")

X_test_claim = scaled_X_test[objective].copy()
claim_risk_predictions = pd.DataFrame(index=X_test_claim.index)

# Include shipment identifiers and dates for traceability.
metadata = metadata_tables[objective].reindex(X_test_claim.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        claim_risk_predictions[col] = metadata[col]

for model_name, model in models_by_objective[objective].items():
    predicted_labels = model.predict(X_test_claim)

    classes = list(model.named_steps["classifier"].classes_)
    positive_position = classes.index(1)
    claim_probabilities = model.predict_proba(X_test_claim)[
        :, positive_position
    ]

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    claim_risk_predictions[f"{safe_name}_prediction"] = predicted_labels
    claim_risk_predictions[f"{safe_name}_claim_probability"] = claim_probabilities

print(
    f"Predictions generated for {len(X_test_claim):,} test shipments "
    f"using {len(models_by_objective[objective])} models."
)
display(claim_risk_predictions.head(10))
```

**Output**

```text
Predictions generated for 496 test shipments using 4 models.
```

| index | shipment_id | ship_date | logistic_regression_prediction | … | extra_trees_claim_probability | gradient_boosting_prediction | gradient_boosting_claim_probabi... |
|---|---|---|---|---|---|---|---|
| 0 | SHP-50612 | 2026-05-16 00:00:00 | 0 | … | 0.006667 | 0 | 0.2588 |
| 3 | SHP-50964 | 2026-03-22 00:00:00 | 0 | … | 0.02333 | 0 | 0.3152 |
| 9 | SHP-51603 | 2026-03-18 00:00:00 | 0 | … | 0.06667 | 1 | 0.506 |
| 10 | SHP-51426 | 2026-03-21 00:00:00 | 0 | … | 0.02667 | 0 | 0.2812 |
| 14 | SHP-51296 | 2026-04-22 00:00:00 | 0 | … | 0.13 | 0 | 0.2823 |
| 30 | SHP-50843 | 2026-06-14 00:00:00 | 0 | … | 0.06333 | 0 | 0.3797 |
| 31 | SHP-51330 | 2026-03-22 00:00:00 | 1 | … | 0.06667 | 0 | 0.4963 |
| 43 | SHP-51974 | 2026-06-17 00:00:00 | 1 | … | 0.1 | 0 | 0.4714 |
| 49 | SHP-50287 | 2026-04-08 00:00:00 | 1 | … | 0.1033 | 0 | 0.403 |
| 53 | SHP-51001 | 2026-03-15 00:00:00 | 1 | … | 0.09 | 1 | 0.5672 |

#### Objective 5: Shipment and billing anomaly review

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

if "X_unsupervised" not in globals() or X_unsupervised is None:
    raise NameError("The anomaly review feature table is missing.")
if (
    "models_by_objective" not in globals()
    or "anomaly_review" not in models_by_objective
):
    raise NameError("Train the objective 5 anomaly detectors first.")

X_anomaly = X_unsupervised.copy()
anomaly_results = pd.DataFrame(index=X_anomaly.index)

# Include identifiers and dates for case review.
metadata = metadata_tables["anomaly_review"].reindex(X_anomaly.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        anomaly_results[col] = metadata[col]

for model_name, pipeline in models_by_objective["anomaly_review"].items():
    detector = pipeline.named_steps["detector"]
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")

    if isinstance(detector, LocalOutlierFactor):
        # LOF's training-set scores are stored during fit.
        raw_scores = detector.negative_outlier_factor_
        anomaly_score = -raw_scores
        flags = np.where(raw_scores < detector.offset_, -1, 1)
    else:
        # Decision values below zero are flagged as anomalies.
        decision_values = pipeline.decision_function(X_anomaly)
        anomaly_score = -decision_values
        flags = np.where(decision_values < 0, -1, 1)

    anomaly_results[f"{safe_name}_flag"] = flags
    anomaly_results[f"{safe_name}_score"] = anomaly_score

# Summarize flags: -1 means anomaly, 1 means normal.
summary_rows = []
for model_name in models_by_objective["anomaly_review"]:
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    flag_col = f"{safe_name}_flag"
    flagged_count = int(anomaly_results[flag_col].eq(-1).sum())

    summary_rows.append({
        "model": model_name,
        "rows_scored": len(anomaly_results),
        "flagged_anomalies": flagged_count,
        "flagged_percent": round(
            flagged_count / len(anomaly_results) * 100, 2
        ),
    })

print("ANOMALY REVIEW SCORING SUMMARY")
display(pd.DataFrame(summary_rows).set_index("model"))

# Show the top ten highest anomaly scores from Isolation Forest.
if "isolation_forest_score" in anomaly_results.columns:
    print("\nTOP 10 ISOLATION FOREST CASES FOR REVIEW")
    display(
        anomaly_results
        .sort_values("isolation_forest_score", ascending=False)
        .head(10)
    )

print("\nAll anomaly flags and scores are stored in `anomaly_results`.")
```

**Output**

```text
ANOMALY REVIEW SCORING SUMMARY
```

| model | rows_scored | flagged_anomalies | flagged_percent |
|---|---|---|---|
| Isolation Forest | 2500 | 0 | 0 |
| Local Outlier Factor | 2500 | 11 | 0.44 |
| One-Class SVM | 2500 | 141 | 5.64 |

```text
TOP 10 ISOLATION FOREST CASES FOR REVIEW
```

| index | shipment_id | ship_date | isolation_forest_flag | … | local_outlier_factor_score | one_class_svm_flag | one_class_svm_score |
|---|---|---|---|---|---|---|---|
| 905 | SHP-50397 | 2025-04-25 00:00:00 | 1 | … | 1.681 | -1 | 1.743 |
| 1170 | SHP-50757 | 2026-04-09 00:00:00 | 1 | … | 1.553 | 1 | -8.06e-05 |
| 842 | SHP-50779 | 2026-04-28 00:00:00 | 1 | … | 1.397 | -1 | 0.3975 |
| 2194 | SHP-50944 | 2026-02-15 00:00:00 | 1 | … | 1.149 | 1 | -0.6104 |
| 520 | SHP-50400 | 2025-06-22 00:00:00 | 1 | … | 1.502 | 1 | -9.214e-05 |
| 2394 | SHP-51180 | 2025-03-25 00:00:00 | 1 | … | 1.412 | -1 | 0.3465 |
| 688 | SHP-50706 | 2025-05-20 00:00:00 | 1 | … | 1.47 | 1 | -0.1164 |
| 1077 | SHP-52466 | 2026-08-06 00:00:00 | 1 | … | 1.543 | -1 | 0.4704 |
| 1050 | SHP-51101 | 2025-12-17 00:00:00 | 1 | … | 1.223 | -1 | 0.2787 |
| 1388 | SHP-50967 | 2026-05-06 00:00:00 | 1 | … | 1.48 | -1 | 0.5673 |

```text
All anomaly flags and scores are stored in `anomaly_results`.
```


---

## 16. Compare X and y

Predictions are set next to the real outcomes, and feature averages are compared between actual groups. One useful read from objective 1: shipments that were truly late had an average contract SLA of 2.8 days against 6.2 for on-time ones, so tight SLAs are where lateness concentrates.

The driver charts use a simple test: shuffle one input at a time (all its dummy columns together, so `carrier_name` counts as one input) and measure how much the model's score falls. A big fall means the model leans on that input. For payment and claim risk the chart is noise, because those models score close to chance.

### Charts

<p align="center">
  <img src="images/obj1_features.png" alt="Objective 1: what drives late deliveries" width="100%">
</p>
<p align="center"><em>Objective 1: what drives late deliveries</em></p>

<p align="center">
  <img src="images/obj2_features.png" alt="Objective 2: what drives freight cost" width="100%">
</p>
<p align="center"><em>Objective 2: what drives freight cost</em></p>

<p align="center">
  <img src="images/obj3_features.png" alt="Objective 3: what the payment-risk model uses (near chance)" width="100%">
</p>
<p align="center"><em>Objective 3: what the payment-risk model uses (near chance)</em></p>

<p align="center">
  <img src="images/obj4_features.png" alt="Objective 4: what the claim-risk model uses (near chance)" width="100%">
</p>
<p align="center"><em>Objective 4: what the claim-risk model uses (near chance)</em></p>

<p align="center">
  <img src="images/obj5_features.png" alt="Objective 5: what makes a shipment look unusual" width="100%">
</p>
<p align="center"><em>Objective 5: what makes a shipment look unusual</em></p>

### Code: compare X and y

#### Objective 1: Raise on-time delivery and SLA adherence

```python
import pandas as pd

objective = "late_delivery"

if objective not in X_test or objective not in y_test:
    raise KeyError("Late-delivery test features or target are missing.")
if "objective1_predictions" not in globals():
    raise NameError("Run the objective 1 prediction cell first.")

X_test_late = X_test[objective]
y_test_late = y_test[objective].rename("actual_late_flag")

# Include a few important features for readable row-by-row comparison.
preferred_features = [
    "weight_kg",
    "ship_month",
    "contract_sla_days",
    "contract_rate_per_kg_usd",
]
selected_features = [
    col for col in preferred_features
    if col in X_test_late.columns
]

# Include one-hot transport mode columns, if present.
selected_features += [
    col for col in X_test_late.columns
    if col.startswith("transport_mode_")
]

comparison = X_test_late[selected_features].copy()
comparison["actual_late_flag"] = y_test_late

# Add shipment identifiers and dates for traceability.
if objective in metadata_tables:
    metadata = metadata_tables[objective].reindex(comparison.index)
    for col in ["shipment_id", "ship_date"]:
        if col in metadata.columns:
            comparison.insert(0, col, metadata[col])

# Add predictions from each trained candidate model.
prediction_columns = [
    col for col in objective1_predictions.columns
    if col.endswith("_prediction")
    or col.endswith("_late_probability")
]
comparison = comparison.join(
    objective1_predictions[prediction_columns],
    how="left",
)

print("TEST FEATURES, ACTUAL TARGET, AND MODEL PREDICTIONS")
display(comparison.head(10))

print("\nFEATURE VALUES GROUPED BY ACTUAL DELIVERY OUTCOME")
if selected_features:
    display(
        comparison.groupby("actual_late_flag")[selected_features]
        .mean(numeric_only=True)
        .round(2)
    )
else:
    print("No selected numeric feature columns were available.")

print("\nACTUAL TARGET COUNTS")
display(
    comparison["actual_late_flag"]
    .value_counts()
    .sort_index()
    .rename("shipment_count")
    .to_frame()
)
```

**Output**

```text
TEST FEATURES, ACTUAL TARGET, AND MODEL PREDICTIONS
```

| index | ship_date | shipment_id | weight_kg | … | extra_trees_late_probability | gradient_boosting_prediction | gradient_boosting_late_probability |
|---|---|---|---|---|---|---|---|
| 9 | 2026-03-18 00:00:00 | SHP-51603 | 163.1 | … | 0.12 | 0 | 0.01699 |
| 10 | 2026-03-21 00:00:00 | SHP-51426 | 139.4 | … | 0.04333 | 0 | 0.01524 |
| 14 | 2026-04-22 00:00:00 | SHP-51296 | 114.5 | … | 0.003333 | 0 | 0.01757 |
| 30 | 2026-06-14 00:00:00 | SHP-50843 | 125.1 | … | 0.02 | 0 | 0.01757 |
| 31 | 2026-03-22 00:00:00 | SHP-51330 | 72.3 | … | 0.08 | 0 | 0.0163 |
| 43 | 2026-06-17 00:00:00 | SHP-51974 | 107.8 | … | 0.3667 | 0 | 0.02227 |
| 49 | 2026-04-08 00:00:00 | SHP-50287 | 69.2 | … | 0.006667 | 0 | 0.01401 |
| 53 | 2026-03-15 00:00:00 | SHP-51001 | 77.8 | … | 0.03 | 0 | 0.0136 |
| 62 | 2026-04-27 00:00:00 | SHP-51349 | 393.2 | … | 0.04 | 0 | 0.01897 |
| 69 | 2026-03-31 00:00:00 | SHP-50579 | 237.9 | … | 0 | 0 | 0.01757 |

```text
FEATURE VALUES GROUPED BY ACTUAL DELIVERY OUTCOME
```

| actual_late_flag | weight_kg | ship_month | contract_sla_days | … | transport_mode_road | transport_mode_sea | transport_mode_nan |
|---|---|---|---|---|---|---|---|
| 0 | 2,196 | 4.68 | 6.16 | … | 0.07 | 0.04 | 0 |
| 1 | 284.7 | 4.73 | 2.82 | … | 0.07 | 0.07 | 0 |

```text
ACTUAL TARGET COUNTS
```

| actual_late_flag | shipment_count |
|---|---|
| 0 | 402 |
| 1 | 44 |

#### Objective 2: Cut freight cost and protect margin

```python
import pandas as pd

objective = "freight_cost"

if objective not in X_test or objective not in y_test:
    raise KeyError("Freight-cost test features or target are missing.")
if "freight_cost_predictions" not in globals():
    raise NameError("Run the objective 2 prediction cell first.")

X_test_freight = X_test[objective]
y_test_freight = y_test[objective].rename("actual_total_freight_cost_usd")

comparison = X_test_freight.copy()
comparison["actual_total_freight_cost_usd"] = y_test_freight

# Add IDs and dates for traceability.
metadata = metadata_tables[objective].reindex(comparison.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        comparison.insert(0, col, metadata[col])

prediction_cols = [
    col for col in freight_cost_predictions.columns
    if col.endswith("_prediction_usd")
]
comparison = comparison.join(
    freight_cost_predictions[prediction_cols],
    how="left",
)

print("TEST FEATURES, ACTUAL FREIGHT COST, AND MODEL PREDICTIONS")
display(comparison.head(10))

print("\nACTUAL FREIGHT COST SUMMARY")
display(y_test_freight.describe().to_frame("actual_cost_usd").round(2))

# Compare average numeric feature values across lower and higher cost shipments.
feature_cols = [
    col for col in X_test_freight.columns
    if pd.api.types.is_numeric_dtype(X_test_freight[col])
]
if feature_cols:
    comparison["actual_cost_group"] = pd.qcut(
        comparison["actual_total_freight_cost_usd"],
        q=2,
        labels=["Lower actual cost", "Higher actual cost"],
        duplicates="drop",
    )
    print("\nMEAN FEATURE VALUES BY ACTUAL COST GROUP")
    display(
        comparison.groupby("actual_cost_group", observed=False)[feature_cols]
        .mean(numeric_only=True)
        .round(2)
    )

print("\nComparison table stored in `comparison`.")
```

**Output**

```text
TEST FEATURES, ACTUAL FREIGHT COST, AND MODEL PREDICTIONS
```

| index | ship_date | shipment_id | contract_sla_days | … | random_forest_regressor_predict... | extra_trees_regressor_predictio... | gradient_boosting_regressor_pre... |
|---|---|---|---|---|---|---|---|
| 0 | 2026-05-16 00:00:00 | SHP-50612 | 10 | … | 344.2 | 307.5 | 340.5 |
| 3 | 2026-03-22 00:00:00 | SHP-50964 | 7 | … | 24.73 | 35.42 | -6.43 |
| 9 | 2026-03-18 00:00:00 | SHP-51603 | 3 | … | 293 | 187.1 | 203.2 |
| 10 | 2026-03-21 00:00:00 | SHP-51426 | 7 | … | 128.6 | 218.6 | 93.21 |
| 14 | 2026-04-22 00:00:00 | SHP-51296 | 10 | … | 192.1 | 179.1 | 131.1 |
| 30 | 2026-06-14 00:00:00 | SHP-50843 | 10 | … | 733.2 | 747.4 | 793.5 |
| 31 | 2026-03-22 00:00:00 | SHP-51330 | 2 | … | 102.7 | 146.6 | 69.19 |
| 43 | 2026-06-17 00:00:00 | SHP-51974 | 2 | … | 175.9 | 894.7 | 119.6 |
| 49 | 2026-04-08 00:00:00 | SHP-50287 | 5 | … | 380.8 | 510.6 | 551.9 |
| 53 | 2026-03-15 00:00:00 | SHP-51001 | 7 | … | 80.8 | 226.9 | 11.27 |

```text
ACTUAL FREIGHT COST SUMMARY
```

| index | actual_cost_usd |
|---|---|
| count | 496 |
| mean | 714.8 |
| std | 1,195 |
| min | 9.63 |
| 25% | 136.2 |
| 50% | 307.4 |
| 75% | 751.5 |
| max | 1.076e+04 |

```text
MEAN FEATURE VALUES BY ACTUAL COST GROUP
```

| actual_cost_group | contract_sla_days | contract_rate_per_kg_usd | weight_kg | … | transport_mode_road | transport_mode_sea | transport_mode_nan |
|---|---|---|---|---|---|---|---|
| Lower actual cost | 5.73 | 3.34 | 159.4 | … | 0.08 | 0.07 | 0 |
| Higher actual cost | 5.94 | 3.55 | 4,894 | … | 0.05 | 0.02 | 0 |

```text
Comparison table stored in `comparison`.
```

#### Objective 3: Close revenue leakage and speed up cash collection

```python
import pandas as pd

objective = "payment_risk"

if objective not in X_test or objective not in y_test:
    raise KeyError("Payment-risk test features or target are missing.")
if "payment_risk_predictions" not in globals():
    raise NameError("Run the objective 3 prediction cell first.")

comparison = X_test[objective].copy()
comparison["actual_payment_late"] = y_test[objective]

# Add IDs and dates for traceability.
metadata = metadata_tables[objective].reindex(comparison.index)
for col in ["invoice_id", "shipment_id", "invoice_date"]:
    if col in metadata.columns:
        comparison.insert(0, col, metadata[col])

# Add predictions and predicted late-payment probabilities.
prediction_cols = [
    col for col in payment_risk_predictions.columns
    if col.endswith("_prediction")
    or col.endswith("_late_probability")
]
comparison = comparison.join(
    payment_risk_predictions[prediction_cols],
    how="left",
)

print("TEST FEATURES, ACTUAL PAYMENT TARGET, AND MODEL PREDICTIONS")
display(comparison.head(10))

print("\nACTUAL PAYMENT-RISK TARGET COUNTS (0 = Paid, 1 = Overdue/Disputed)")
display(
    comparison["actual_payment_late"]
    .value_counts()
    .sort_index()
    .rename("invoice_count")
    .to_frame()
)

# Compare selected interpretable numeric features by actual target class.
preferred_features = [
    "billed_amount_usd",
    "credit_limit_usd",
    "total_freight_cost",
    "margin_usd",
]
available_features = [
    col for col in preferred_features
    if col in X_test[objective].columns
]

if available_features:
    print("\nSELECTED FEATURE MEANS BY ACTUAL PAYMENT-RISK CLASS")
    display(
        comparison.groupby("actual_payment_late")[available_features]
        .mean(numeric_only=True)
        .round(2)
    )

print("\nComparison table stored in `comparison`.")
```

**Output**

```text
TEST FEATURES, ACTUAL PAYMENT TARGET, AND MODEL PREDICTIONS
```

| index | invoice_date | shipment_id | invoice_id | … | extra_trees_late_probability | gradient_boosting_prediction | gradient_boosting_late_probability |
|---|---|---|---|---|---|---|---|
| 5 | 2026-06-16 00:00:00 | SHP-51622 | INV-700005 | … | 0.4167 | 0 | 0.2175 |
| 8 | 2026-05-21 00:00:00 | SHP-50165 | INV-700008 | … | 0.1367 | 0 | 0.1927 |
| 9 | 2026-06-28 00:00:00 | SHP-50395 | INV-700009 | … | 0.3533 | 0 | 0.2 |
| 16 | 2026-04-15 00:00:00 | SHP-50887 | INV-700016 | … | 0.2833 | 0 | 0.2866 |
| 20 | 2026-05-12 00:00:00 | SHP-52301 | INV-700020 | … | 0.02479 | 0 | 0.2244 |
| 24 | 2026-03-23 00:00:00 | SHP-52076 | INV-700024 | … | 0.06 | 0 | 0.2436 |
| 27 | 2026-04-19 00:00:00 | SHP-51919 | INV-700027 | … | 0.71 | 0 | 0.3727 |
| 34 | 2026-05-27 00:00:00 | SHP-50336 | INV-700034 | … | 0.41 | 0 | 0.3115 |
| 62 | 2026-06-17 00:00:00 | SHP-50332 | INV-700062 | … | 0.7167 | 0 | 0.3056 |
| 70 | 2026-06-24 00:00:00 | SHP-51974 | INV-700070 | … | 0.1767 | 0 | 0.2332 |

```text
ACTUAL PAYMENT-RISK TARGET COUNTS (0 = Paid, 1 = Overdue/Disputed)
```

| actual_payment_late | invoice_count |
|---|---|
| 0 | 290 |
| 1 | 88 |

```text
SELECTED FEATURE MEANS BY ACTUAL PAYMENT-RISK CLASS
```

| actual_payment_late | billed_amount_usd | credit_limit_usd | total_freight_cost | margin_usd |
|---|---|---|---|---|
| 0 | 859.9 | 2.205e+05 | 705.6 | 151.7 |
| 1 | 954 | 2.036e+05 | 754.2 | 199.8 |

```text
Comparison table stored in `comparison`.
```

#### Objective 4: Reduce claims, damage and loss

```python
import pandas as pd

objective = "claim_risk"

if objective not in X_test or objective not in y_test:
    raise KeyError("Claim-risk test features or target are missing.")
if "claim_risk_predictions" not in globals():
    raise NameError("Run the objective 4 prediction cell first.")

comparison = X_test[objective].copy()
comparison["actual_has_claim"] = y_test[objective]

# Add shipment identifiers and dates for traceability.
metadata = metadata_tables[objective].reindex(comparison.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        comparison.insert(0, col, metadata[col])

# Add model predictions and claim probabilities.
prediction_cols = [
    col for col in claim_risk_predictions.columns
    if col.endswith("_prediction")
    or col.endswith("_claim_probability")
]
comparison = comparison.join(
    claim_risk_predictions[prediction_cols],
    how="left",
)

print("TEST FEATURES, ACTUAL CLAIM TARGET, AND MODEL PREDICTIONS")
display(comparison.head(10))

print("\nACTUAL CLAIM TARGET COUNTS (0 = No Claim, 1 = Claim)")
display(
    comparison["actual_has_claim"]
    .value_counts()
    .sort_index()
    .rename("shipment_count")
    .to_frame()
)

# Compare selected numeric features across the two actual target classes.
preferred_features = [
    "weight_kg",
    "ship_month",
    "contract_sla_days",
    "contract_rate_per_kg_usd",
]
available_features = [
    col for col in preferred_features
    if col in X_test[objective].columns
]

if available_features:
    print("\nSELECTED FEATURE MEANS BY ACTUAL CLAIM STATUS")
    display(
        comparison.groupby("actual_has_claim")[available_features]
        .mean(numeric_only=True)
        .round(2)
    )

print("\nComparison table stored in `comparison`.")
```

**Output**

```text
TEST FEATURES, ACTUAL CLAIM TARGET, AND MODEL PREDICTIONS
```

| index | ship_date | shipment_id | contract_sla_days | … | extra_trees_claim_probability | gradient_boosting_prediction | gradient_boosting_claim_probabi... |
|---|---|---|---|---|---|---|---|
| 0 | 2026-05-16 00:00:00 | SHP-50612 | 10 | … | 0.006667 | 0 | 0.2588 |
| 3 | 2026-03-22 00:00:00 | SHP-50964 | 7 | … | 0.02333 | 0 | 0.3152 |
| 9 | 2026-03-18 00:00:00 | SHP-51603 | 3 | … | 0.06667 | 1 | 0.506 |
| 10 | 2026-03-21 00:00:00 | SHP-51426 | 7 | … | 0.02667 | 0 | 0.2812 |
| 14 | 2026-04-22 00:00:00 | SHP-51296 | 10 | … | 0.13 | 0 | 0.2823 |
| 30 | 2026-06-14 00:00:00 | SHP-50843 | 10 | … | 0.06333 | 0 | 0.3797 |
| 31 | 2026-03-22 00:00:00 | SHP-51330 | 2 | … | 0.06667 | 0 | 0.4963 |
| 43 | 2026-06-17 00:00:00 | SHP-51974 | 2 | … | 0.1 | 0 | 0.4714 |
| 49 | 2026-04-08 00:00:00 | SHP-50287 | 5 | … | 0.1033 | 0 | 0.403 |
| 53 | 2026-03-15 00:00:00 | SHP-51001 | 7 | … | 0.09 | 1 | 0.5672 |

```text
ACTUAL CLAIM TARGET COUNTS (0 = No Claim, 1 = Claim)
```

| actual_has_claim | shipment_count |
|---|---|
| 0 | 450 |
| 1 | 46 |

```text
SELECTED FEATURE MEANS BY ACTUAL CLAIM STATUS
```

| actual_has_claim | weight_kg | ship_month | contract_sla_days | contract_rate_per_kg_usd |
|---|---|---|---|---|
| 0 | 2,758 | 4.68 | 5.84 | 3.43 |
| 1 | 236.7 | 4.48 | 5.82 | 3.57 |

```text
Comparison table stored in `comparison`.
```

#### Objective 5: Shipment and billing anomaly review

```python
import pandas as pd

if "X_unsupervised" not in globals():
    raise NameError("Anomaly review features are missing.")
if "anomaly_results" not in globals():
    raise NameError("Run the anomaly scoring cell first.")

anomaly_feature_comparison = X_unsupervised.join(
    anomaly_results,
    how="left",
)

print("ANOMALY FLAGS BY MODEL")
flag_columns = [
    col for col in anomaly_results.columns
    if col.endswith("_flag")
]
display(
    anomaly_results[flag_columns]
    .apply(lambda col: col.value_counts(dropna=False))
    .fillna(0)
    .astype("int64")
)

# Compare numeric feature averages for Isolation Forest flagged vs normal cases.
if "isolation_forest_flag" in anomaly_feature_comparison.columns:
    numeric_features = [
        col for col in X_unsupervised.columns
        if pd.api.types.is_numeric_dtype(X_unsupervised[col])
    ]

    if numeric_features:
        print("\nMEAN FEATURE VALUES: ISOLATION FOREST FLAGGED VS NORMAL")
        display(
            anomaly_feature_comparison
            .groupby("isolation_forest_flag")[numeric_features]
            .mean(numeric_only=True)
            .round(2)
            .rename(index={-1: "Flagged anomaly", 1: "Normal"})
        )

print("\nTOP 15 FLAGGED CASES")
if "isolation_forest_flag" in anomaly_feature_comparison.columns:
    flagged = anomaly_feature_comparison[
        anomaly_feature_comparison["isolation_forest_flag"] == -1
    ].sort_values("isolation_forest_score", ascending=False)

    display(flagged.head(15))

print("\nFeature-to-flag comparison stored in `anomaly_feature_comparison`.")
```

**Output**

```text
ANOMALY FLAGS BY MODEL
```

| index | isolation_forest_flag | local_outlier_factor_flag | one_class_svm_flag |
|---|---|---|---|
| -1 | 0 | 11 | 141 |
| 1 | 2500 | 2489 | 2359 |

```text
MEAN FEATURE VALUES: ISOLATION FOREST FLAGGED VS NORMAL
```

| isolation_forest_flag | weight_kg | cost_per_kg | total_freight_cost | … | destination_country_united kingdom | destination_country_united states | destination_country_nan |
|---|---|---|---|---|---|---|---|
| Normal | 1,504 | 2.37 | 696.6 | … | 0.01 | 0.01 | 0 |

```text
TOP 15 FLAGGED CASES
```

| weight_kg | cost_per_kg | total_freight_cost | total_billed_amount_usd | … | local_outlier_factor_score | one_class_svm_flag | one_class_svm_score |
|---|---|---|---|---|---|---|---|

```text
Feature-to-flag comparison stored in `anomaly_feature_comparison`.
```

### Code: charts

```python
PREFIXES = ['customer_tier_', 'industry_', 'hq_country_', 'carrier_name_', 'origin_country_', 'destination_country_', 'transport_mode_']


def feature_groups(columns):
    groups = {}
    for c in columns:
        for p in PREFIXES:
            if c.startswith(p):
                groups.setdefault(p.rstrip("_"), []).append(c); break
        else:
            groups[c] = [c]
    return groups


def grouped_importance(score_fn, X, groups, repeats=5, seed=42):
    rng = np.random.default_rng(seed); base = score_fn(X); out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy(); idx = rng.permutation(len(X)); Xp[cols] = X[cols].values[idx]
            drops.append(base - score_fn(Xp))
        out[g] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)


def chart_features(obj, name, model, X, y, title, note=None, fname=None):
    if obj == "freight_cost":
        score = lambda Z: -mean_absolute_error(y, model.predict(Z)); label = "Increase in average error (USD) when shuffled"
    else:
        cls = list(model.classes_).index(1)
        score = lambda Z: roc_auc_score(y, model.predict_proba(Z)[:, cls]); label = "Drop in ROC AUC when shuffled"
    imp = grouped_importance(score, X, feature_groups(X.columns)).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(imp.index, imp.values, color=BLUE, height=.6)
    ax.set_title(title); ax.set_xlabel(label + ("\n" + note if note else "")); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    save(fig, fname)
    return imp[::-1]


def chart_features_anomaly(X_anomaly, flagged_if):
    num = [c for c in ["weight_kg", "cost_per_kg", "total_freight_cost", "total_billed_amount_usd", "margin_usd"] if c in X_anomaly.columns]
    Z = (X_anomaly[num] - X_anomaly[num].mean()) / X_anomaly[num].std()
    diff = (Z[flagged_if.values].mean() - Z[~flagged_if.values].mean()).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(diff.index, diff.values, color=ORANGE, height=.55)
    ax.set_title("What makes a shipment look unusual (top 5% by Isolation Forest score)")
    ax.set_xlabel("Average (in standard deviations) flagged minus not flagged"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    save(fig, "obj5_features")
```

---

## 17. Evaluate Model

| Objective | Chosen metric | Leading candidate | Value |
|---|---|---|---|
| 1. Late delivery | F1 | Logistic Regression | 0.481 (recall 0.886, ROC AUC 0.886) |
| 2. Freight cost | MAE (lower is better) | Random Forest Regressor | $146.29 (R² 0.832) |
| 3. Payment risk | Recall | Logistic Regression | 0.477 (ROC AUC 0.496, chance level) |
| 4. Claim risk | PR AUC | Logistic Regression | 0.093 (ROC AUC 0.504, chance level) |
| 5. Anomaly review | Flag rate only | Isolation Forest (default) | 0 flags at default threshold. One-Class SVM flags 5.6%, LOF 0.4% |

**How to read these.** Objectives 1 and 2 have real signal. Objectives 3 and 4 do not: with ROC AUC at 0.5 the models cannot separate late payers or claim shipments from the rest, and the high recall of logistic regression there comes from flagging many cases, not from skill. The anomaly detectors have no answer key, so only flag rates are shown, not accuracy.

### Charts

<p align="center">
  <img src="images/compare_obj1_late_delivery.png" alt="Objective 1: four candidate models" width="100%">
</p>
<p align="center"><em>Objective 1: four candidate models</em></p>

<p align="center">
  <img src="images/compare_obj2_freight_cost.png" alt="Objective 2: four candidate models" width="100%">
</p>
<p align="center"><em>Objective 2: four candidate models</em></p>

<p align="center">
  <img src="images/compare_obj3_payment_risk.png" alt="Objective 3: four candidate models" width="100%">
</p>
<p align="center"><em>Objective 3: four candidate models</em></p>

<p align="center">
  <img src="images/compare_obj4_claim_risk.png" alt="Objective 4: four candidate models" width="100%">
</p>
<p align="center"><em>Objective 4: four candidate models</em></p>

<p align="center">
  <img src="images/compare_obj5_anomaly_flags.png" alt="Objective 5: share flagged by each detector" width="100%">
</p>
<p align="center"><em>Objective 5: share flagged by each detector</em></p>

<p align="center">
  <img src="images/obj1_evaluation.png" alt="Objective 1: late delivery, evaluation on unseen shipments" width="100%">
</p>
<p align="center"><em>Objective 1: late delivery, evaluation on unseen shipments</em></p>

<p align="center">
  <img src="images/obj2_evaluation.png" alt="Objective 2: freight cost, evaluation on unseen shipments" width="100%">
</p>
<p align="center"><em>Objective 2: freight cost, evaluation on unseen shipments</em></p>

<p align="center">
  <img src="images/obj3_evaluation.png" alt="Objective 3: payment risk, evaluation on unseen invoices" width="100%">
</p>
<p align="center"><em>Objective 3: payment risk, evaluation on unseen invoices</em></p>

<p align="center">
  <img src="images/obj4_evaluation.png" alt="Objective 4: claim risk, evaluation on unseen shipments" width="100%">
</p>
<p align="center"><em>Objective 4: claim risk, evaluation on unseen shipments</em></p>

<p align="center">
  <img src="images/obj5_evaluation.png" alt="Objective 5: anomaly review" width="100%">
</p>
<p align="center"><em>Objective 5: anomaly review</em></p>

### Code: evaluate model

#### Objective 1: Raise on-time delivery and SLA adherence

```python
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

objective = "late_delivery"

X_test_late = scaled_X_test[objective].copy()
y_test_late = y_test[objective].copy()

if not X_test_late.index.equals(y_test_late.index):
    raise ValueError("Test predictors and target rows are misaligned.")
if y_test_late.isna().any():
    raise ValueError("The test target contains missing values.")
if y_test_late.nunique() < 2:
    raise ValueError(
        "ROC-AUC cannot be calculated because the test set has only one class."
    )

objective1_predictions = pd.DataFrame(index=X_test_late.index)

# Keep IDs and dates for traceability.
metadata = metadata_tables[objective].reindex(X_test_late.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        objective1_predictions[col] = metadata[col]

metric_rows = []
confusion_tables = {}

for model_name, model in models.items():
    predicted_labels = model.predict(X_test_late)

    classes = list(model.named_steps["classifier"].classes_)
    positive_position = classes.index(1)
    late_probabilities = model.predict_proba(X_test_late)[:, positive_position]

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    objective1_predictions[f"{safe_name}_prediction"] = predicted_labels
    objective1_predictions[f"{safe_name}_late_probability"] = late_probabilities

    metric_rows.append({
        "model": model_name,
        "precision": precision_score(
            y_test_late, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            y_test_late, predicted_labels, zero_division=0
        ),
        "f1": f1_score(
            y_test_late, predicted_labels, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test_late, late_probabilities),
        "pr_auc": average_precision_score(
            y_test_late, late_probabilities
        ),
    })

    cm = confusion_matrix(y_test_late, predicted_labels, labels=[0, 1])
    confusion_tables[model_name] = pd.DataFrame(
        cm,
        index=["Actual On Time (0)", "Actual Late (1)"],
        columns=["Predicted On Time (0)", "Predicted Late (1)"],
    )

evaluation_results = (
    pd.DataFrame(metric_rows)
    .set_index("model")
    .sort_values("f1", ascending=False)
)

print("BUSINESS OBJECTIVE 1: TEST-SET EVALUATION")
display(evaluation_results.round(3))

for model_name, matrix in confusion_tables.items():
    print(f"\n{model_name} — confusion matrix")
    display(matrix)

print("\nPrediction results stored in objective1_predictions.")
```

**Output**

```text
BUSINESS OBJECTIVE 1: TEST-SET EVALUATION
```

| model | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.331 | 0.886 | 0.481 | 0.886 | 0.413 |
| Random Forest | 0.342 | 0.614 | 0.439 | 0.857 | 0.311 |
| Extra Trees | 0.317 | 0.295 | 0.306 | 0.862 | 0.315 |
| Gradient Boosting | 0.083 | 0.023 | 0.036 | 0.848 | 0.281 |

```text
Logistic Regression — confusion matrix
```

| index | Predicted On Time (0) | Predicted Late (1) |
|---|---|---|
| Actual On Time (0) | 323 | 79 |
| Actual Late (1) | 5 | 39 |

```text
Random Forest — confusion matrix
```

| index | Predicted On Time (0) | Predicted Late (1) |
|---|---|---|
| Actual On Time (0) | 350 | 52 |
| Actual Late (1) | 17 | 27 |

```text
Extra Trees — confusion matrix
```

| index | Predicted On Time (0) | Predicted Late (1) |
|---|---|---|
| Actual On Time (0) | 374 | 28 |
| Actual Late (1) | 31 | 13 |

```text
Gradient Boosting — confusion matrix
```

| index | Predicted On Time (0) | Predicted Late (1) |
|---|---|---|
| Actual On Time (0) | 391 | 11 |
| Actual Late (1) | 43 | 1 |

```text
Prediction results stored in objective1_predictions.
```

#### Objective 2: Cut freight cost and protect margin

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

objective = "freight_cost"

X_test_freight = scaled_X_test[objective].copy()
y_test_freight = y_test[objective].copy()

if not X_test_freight.index.equals(y_test_freight.index):
    raise ValueError("Test predictors and target rows are misaligned.")

metric_rows = []
freight_cost_predictions = pd.DataFrame(index=X_test_freight.index)

metadata = metadata_tables[objective].reindex(X_test_freight.index)
for col in ["shipment_id", "ship_date"]:
    if col in metadata.columns:
        freight_cost_predictions[col] = metadata[col]

for model_name, model in models_by_objective[objective].items():
    predictions = model.predict(X_test_freight)
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")

    freight_cost_predictions[f"{safe_name}_prediction_usd"] = predictions

    metric_rows.append({
        "model": model_name,
        "MAE_USD": mean_absolute_error(y_test_freight, predictions),
        "RMSE_USD": mean_squared_error(
            y_test_freight, predictions
        ) ** 0.5,
        "R_squared": r2_score(y_test_freight, predictions),
    })

freight_cost_evaluation = (
    pd.DataFrame(metric_rows)
    .set_index("model")
    .sort_values("MAE_USD")
)

print("BUSINESS OBJECTIVE 2: TEST-SET REGRESSION EVALUATION")
display(freight_cost_evaluation.round(3))

print("\nBest model by lowest MAE:")
print(freight_cost_evaluation.index[0])
```

**Output**

```text
BUSINESS OBJECTIVE 2: TEST-SET REGRESSION EVALUATION
```

| model | MAE_USD | RMSE_USD | R_squared |
|---|---|---|---|
| Random Forest Regressor | 146.3 | 488.7 | 0.832 |
| Extra Trees Regressor | 227.1 | 590.6 | 0.755 |
| Gradient Boosting Regressor | 234 | 534.5 | 0.799 |
| Ridge Regression | 628.5 | 1,104 | 0.144 |

```text
Best model by lowest MAE:
Random Forest Regressor
```

#### Objective 3: Close revenue leakage and speed up cash collection

```python
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

objective = "payment_risk"
X_test_payment = scaled_X_test[objective].copy()
y_test_payment = y_test[objective].copy()

if not X_test_payment.index.equals(y_test_payment.index):
    raise ValueError("Test predictors and target rows are misaligned.")
if y_test_payment.isna().any():
    raise ValueError("The payment-risk test target contains missing values.")

if y_test_payment.nunique() < 2:
    raise ValueError(
        "The test set contains only one target class; ROC-AUC is undefined."
    )

metric_rows = []
confusion_tables = {}

for model_name, model in models_by_objective[objective].items():
    predicted_labels = model.predict(X_test_payment)

    classes = list(model.named_steps["classifier"].classes_)
    positive_position = classes.index(1)
    late_probabilities = model.predict_proba(X_test_payment)[
        :, positive_position
    ]

    metric_rows.append({
        "model": model_name,
        "precision": precision_score(
            y_test_payment, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            y_test_payment, predicted_labels, zero_division=0
        ),
        "f1": f1_score(
            y_test_payment, predicted_labels, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test_payment, late_probabilities),
        "pr_auc": average_precision_score(
            y_test_payment, late_probabilities
        ),
    })

    cm = confusion_matrix(
        y_test_payment,
        predicted_labels,
        labels=[0, 1],
    )
    confusion_tables[model_name] = pd.DataFrame(
        cm,
        index=["Actual Paid (0)", "Actual Late/Disputed (1)"],
        columns=["Predicted Paid (0)", "Predicted Late/Disputed (1)"],
    )

payment_risk_evaluation = (
    pd.DataFrame(metric_rows)
    .set_index("model")
    .sort_values("recall", ascending=False)
)

print("BUSINESS OBJECTIVE 3: TEST-SET EVALUATION")
display(payment_risk_evaluation.round(3))

for model_name, matrix in confusion_tables.items():
    print(f"\n{model_name} — confusion matrix")
    display(matrix)
```

**Output**

```text
BUSINESS OBJECTIVE 3: TEST-SET EVALUATION
```

| model | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.219 | 0.477 | 0.3 | 0.496 | 0.227 |
| Random Forest | 0.209 | 0.159 | 0.181 | 0.489 | 0.226 |
| Extra Trees | 0.255 | 0.159 | 0.196 | 0.51 | 0.237 |
| Gradient Boosting | 0.167 | 0.011 | 0.021 | 0.482 | 0.23 |

```text
Logistic Regression — confusion matrix
```

| index | Predicted Paid (0) | Predicted Late/Disputed (1) |
|---|---|---|
| Actual Paid (0) | 140 | 150 |
| Actual Late/Disputed (1) | 46 | 42 |

```text
Random Forest — confusion matrix
```

| index | Predicted Paid (0) | Predicted Late/Disputed (1) |
|---|---|---|
| Actual Paid (0) | 237 | 53 |
| Actual Late/Disputed (1) | 74 | 14 |

```text
Extra Trees — confusion matrix
```

| index | Predicted Paid (0) | Predicted Late/Disputed (1) |
|---|---|---|
| Actual Paid (0) | 249 | 41 |
| Actual Late/Disputed (1) | 74 | 14 |

```text
Gradient Boosting — confusion matrix
```

| index | Predicted Paid (0) | Predicted Late/Disputed (1) |
|---|---|---|
| Actual Paid (0) | 285 | 5 |
| Actual Late/Disputed (1) | 87 | 1 |

#### Objective 4: Reduce claims, damage and loss

```python
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

objective = "claim_risk"
X_test_claim = scaled_X_test[objective].copy()
y_test_claim = y_test[objective].copy()

if not X_test_claim.index.equals(y_test_claim.index):
    raise ValueError("Test predictors and target rows are misaligned.")
if y_test_claim.isna().any():
    raise ValueError("The claim-risk test target contains missing values.")
if y_test_claim.nunique() < 2:
    raise ValueError(
        "The test set contains only one class; ROC-AUC is undefined."
    )

metric_rows = []
confusion_tables = {}

for model_name, model in models_by_objective[objective].items():
    predicted_labels = model.predict(X_test_claim)

    classes = list(model.named_steps["classifier"].classes_)
    positive_position = classes.index(1)
    claim_probabilities = model.predict_proba(X_test_claim)[
        :, positive_position
    ]

    metric_rows.append({
        "model": model_name,
        "precision": precision_score(
            y_test_claim, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            y_test_claim, predicted_labels, zero_division=0
        ),
        "f1": f1_score(
            y_test_claim, predicted_labels, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test_claim, claim_probabilities),
        "pr_auc": average_precision_score(
            y_test_claim, claim_probabilities
        ),
    })

    cm = confusion_matrix(
        y_test_claim,
        predicted_labels,
        labels=[0, 1],
    )
    confusion_tables[model_name] = pd.DataFrame(
        cm,
        index=["Actual No Claim (0)", "Actual Claim (1)"],
        columns=["Predicted No Claim (0)", "Predicted Claim (1)"],
    )

claim_risk_evaluation = (
    pd.DataFrame(metric_rows)
    .set_index("model")
    .sort_values("pr_auc", ascending=False)
)

print("BUSINESS OBJECTIVE 4: TEST-SET EVALUATION")
display(claim_risk_evaluation.round(3))

for model_name, matrix in confusion_tables.items():
    print(f"\n{model_name} — confusion matrix")
    display(matrix)
```

**Output**

```text
BUSINESS OBJECTIVE 4: TEST-SET EVALUATION
```

| model | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.098 | 0.413 | 0.158 | 0.504 | 0.093 |
| Gradient Boosting | 0.075 | 0.283 | 0.118 | 0.476 | 0.091 |
| Extra Trees | 0 | 0 | 0 | 0.486 | 0.089 |
| Random Forest | 0 | 0 | 0 | 0.467 | 0.084 |

```text
Logistic Regression — confusion matrix
```

| index | Predicted No Claim (0) | Predicted Claim (1) |
|---|---|---|
| Actual No Claim (0) | 275 | 175 |
| Actual Claim (1) | 27 | 19 |

```text
Random Forest — confusion matrix
```

| index | Predicted No Claim (0) | Predicted Claim (1) |
|---|---|---|
| Actual No Claim (0) | 450 | 0 |
| Actual Claim (1) | 46 | 0 |

```text
Extra Trees — confusion matrix
```

| index | Predicted No Claim (0) | Predicted Claim (1) |
|---|---|---|
| Actual No Claim (0) | 444 | 6 |
| Actual Claim (1) | 46 | 0 |

```text
Gradient Boosting — confusion matrix
```

| index | Predicted No Claim (0) | Predicted Claim (1) |
|---|---|---|
| Actual No Claim (0) | 289 | 161 |
| Actual Claim (1) | 33 | 13 |

#### Objective 5: Shipment and billing anomaly review

```python
import pandas as pd

if "anomaly_results" not in globals():
    raise NameError("Run the anomaly scoring cell first.")

flag_columns = [
    col for col in anomaly_results.columns
    if col.endswith("_flag")
]
score_columns = [
    col for col in anomaly_results.columns
    if col.endswith("_score")
]

if not flag_columns:
    raise ValueError("No anomaly detector flag columns were found.")

# Convert -1 flags to True (flagged); 1 flags to False (normal).
flagged = anomaly_results[flag_columns].eq(-1)
flagged.columns = [
    col.replace("_flag", "") for col in flagged.columns
]

# Flag rates for each detector.
flag_rate_summary = pd.DataFrame({
    "flagged_count": flagged.sum(),
    "flagged_percent": (flagged.mean() * 100).round(2),
})

print("ANOMALY DETECTOR FLAG RATES")
display(flag_rate_summary)

# Pairwise agreement and Jaccard overlap between flagged cases.
detector_names = list(flagged.columns)
agreement = pd.DataFrame(index=detector_names, columns=detector_names, dtype=float)
jaccard = pd.DataFrame(index=detector_names, columns=detector_names, dtype=float)

for model_a in detector_names:
    for model_b in detector_names:
        a = flagged[model_a]
        b = flagged[model_b]

        agreement.loc[model_a, model_b] = (a == b).mean()

        union_count = (a | b).sum()
        jaccard.loc[model_a, model_b] = (
            (a & b).sum() / union_count if union_count else 1.0
        )

print("\nPAIRWISE FLAG AGREEMENT")
display((agreement * 100).round(1))

print("\nPAIRWISE FLAG OVERLAP (JACCARD)")
display(jaccard.round(3))

# Compare rankings of anomaly scores; higher values indicate more unusual cases.
if score_columns:
    score_frame = anomaly_results[score_columns].copy()
    score_frame.columns = [
        col.replace("_score", "") for col in score_frame.columns
    ]

    print("\nSPEARMAN CORRELATION OF ANOMALY SCORE RANKINGS")
    display(score_frame.corr(method="spearman").round(3))

# Consensus count helps prioritize records flagged by multiple detectors.
consensus = anomaly_results.copy()
consensus["detectors_flagging_case"] = flagged.sum(axis=1)
consensus["flagged_by_all_detectors"] = (
    consensus["detectors_flagging_case"] == len(flag_columns)
)

print("\nTOP 15 CASES FLAGGED BY THE MOST DETECTORS")
sort_columns = ["detectors_flagging_case"]
ascending = [False]

if "isolation_forest_score" in consensus.columns:
    sort_columns.append("isolation_forest_score")
    ascending.append(False)

display(
    consensus
    .sort_values(sort_columns, ascending=ascending)
    .head(15)
)

print(
    "\nInterpretation: agreement and score correlation show detector "
    "consistency only. Review flagged records against operational context "
    "or obtain confirmed anomaly labels for true performance evaluation."
)
```

**Output**

```text
ANOMALY DETECTOR FLAG RATES
```

| index | flagged_count | flagged_percent |
|---|---|---|
| isolation_forest | 0 | 0 |
| local_outlier_factor | 11 | 0.44 |
| one_class_svm | 141 | 5.64 |

```text
PAIRWISE FLAG AGREEMENT
```

| index | isolation_forest | local_outlier_factor | one_class_svm |
|---|---|---|---|
| isolation_forest | 100 | 99.6 | 94.4 |
| local_outlier_factor | 99.6 | 100 | 94.6 |
| one_class_svm | 94.4 | 94.6 | 100 |

```text
PAIRWISE FLAG OVERLAP (JACCARD)
```

| index | isolation_forest | local_outlier_factor | one_class_svm |
|---|---|---|---|
| isolation_forest | 1 | 0 | 0 |
| local_outlier_factor | 0 | 1 | 0.063 |
| one_class_svm | 0 | 0.063 | 1 |

```text
SPEARMAN CORRELATION OF ANOMALY SCORE RANKINGS
```

| index | isolation_forest | local_outlier_factor | one_class_svm |
|---|---|---|---|
| isolation_forest | 1 | 0.35 | 0.427 |
| local_outlier_factor | 0.35 | 1 | 0.527 |
| one_class_svm | 0.427 | 0.527 | 1 |

```text
TOP 15 CASES FLAGGED BY THE MOST DETECTORS
```

| index | shipment_id | ship_date | isolation_forest_flag | … | one_class_svm_score | detectors_flagging_case | flagged_by_all_detectors |
|---|---|---|---|---|---|---|---|
| 905 | SHP-50397 | 2025-04-25 00:00:00 | 1 | … | 1.743 | 2 | False |
| 1077 | SHP-52466 | 2026-08-06 00:00:00 | 1 | … | 0.4704 | 2 | False |
| 850 | SHP-51006 | 2026-02-06 00:00:00 | 1 | … | 2.957e-05 | 2 | False |
| 1914 | SHP-51212 | 2026-06-16 00:00:00 | 1 | … | 2.398 | 2 | False |
| 385 | SHP-50605 | 2026-04-19 00:00:00 | 1 | … | 2.134 | 2 | False |
| 1098 | SHP-52153 | 2025-03-01 00:00:00 | 1 | … | 1.854 | 2 | False |
| … | … | … | … | … | … | … | … |
| 2394 | SHP-51180 | 2025-03-25 00:00:00 | 1 | … | 0.3465 | 1 | False |
| 1050 | SHP-51101 | 2025-12-17 00:00:00 | 1 | … | 0.2787 | 1 | False |
| 1388 | SHP-50967 | 2026-05-06 00:00:00 | 1 | … | 0.5673 | 1 | False |

```text
Interpretation: agreement and score correlation show detector consistency only. Review flagged records against operational context or obtain confirmed anomaly labels for true performance evaluation.
```

### Code: model comparison charts (all five objectives)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Objective definitions from the project workbook.
ml_objectives = pd.DataFrame([
    {
        "Objective": 1,
        "Business question": "Which shipments are likely to be delivered late?",
        "ML task": "Classification",
        "Target": "late_flag",
        "Primary metrics": "F1, ROC-AUC",
    },
    {
        "Objective": 2,
        "Business question": "What will freight cost before booking?",
        "ML task": "Regression",
        "Target": "total_freight_cost",
        "Primary metrics": "MAE, RMSE, R-squared",
    },
    {
        "Objective": 3,
        "Business question": "Which invoices may become overdue or disputed?",
        "ML task": "Classification",
        "Target": "payment_late",
        "Primary metrics": "Recall, ROC-AUC",
    },
    {
        "Objective": 4,
        "Business question": "Which shipments are likely to lead to a claim?",
        "ML task": "Imbalanced classification",
        "Target": "has_claim",
        "Primary metrics": "Precision-recall AUC",
    },
    {
        "Objective": 5,
        "Business question": "Which shipments or bills are anomalous?",
        "ML task": "Unsupervised anomaly detection",
        "Target": "None",
        "Primary metrics": "Review flagged cases",
    },
])

print("MACHINE LEARNING OBJECTIVES")
display(ml_objectives.set_index("Objective"))

# Gather evaluation results already produced in the notebook.
objective_results = {
    1: globals().get("evaluation_results"),
    2: globals().get("freight_cost_evaluation"),
    3: globals().get("payment_risk_evaluation"),
    4: globals().get("claim_risk_evaluation"),
}

required_results = {
    1: ["f1", "roc_auc"],
    2: ["MAE_USD", "RMSE_USD", "R_squared"],
    3: ["recall", "roc_auc"],
    4: ["precision", "recall", "f1", "pr_auc"],
}

missing_results = [
    objective for objective, result in objective_results.items()
    if result is None
]
if missing_results:
    raise NameError(
        f"Evaluation results are missing for objectives {missing_results}. "
        "Run their evaluation cells first."
    )

# Chart 1: Late-delivery model comparison.
late = objective_results[1]
fig, ax = plt.subplots(figsize=(9, 5))
late[["f1", "roc_auc"]].plot(kind="bar", ax=ax, color=["#E76F51", "#457B9D"])
ax.set_title("Objective 1 — Late-delivery classifier comparison")
ax.set_ylabel("Test metric")
ax.set_xlabel("")
ax.set_ylim(0, 1)
ax.legend(title="Metric")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# Chart 2: Freight-cost model comparison; separate panels because units differ.
freight = objective_results[2]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
freight["MAE_USD"].plot(kind="bar", ax=axes[0], color="#2A9D8F")
axes[0].set_title("MAE (lower is better)")
axes[0].set_ylabel("USD")
freight["RMSE_USD"].plot(kind="bar", ax=axes[1], color="#F4A261")
axes[1].set_title("RMSE (lower is better)")
axes[1].set_ylabel("USD")
freight["R_squared"].plot(kind="bar", ax=axes[2], color="#457B9D")
axes[2].set_title("R-squared (higher is better)")
axes[2].set_ylabel("R-squared")
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Objective 2 — Freight-cost regressor comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Chart 3: Payment-risk classification.
payment = objective_results[3]
fig, ax = plt.subplots(figsize=(9, 5))
payment[["recall", "roc_auc"]].plot(
    kind="bar", ax=ax, color=["#E76F51", "#457B9D"]
)
ax.set_title("Objective 3 — Payment-risk classifier comparison")
ax.set_ylabel("Test metric")
ax.set_xlabel("")
ax.set_ylim(0, 1)
ax.legend(title="Metric")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# Chart 4: Claim-risk model comparison.
claim = objective_results[4]
fig, ax = plt.subplots(figsize=(10, 5))
claim[["precision", "recall", "f1", "pr_auc"]].plot(
    kind="bar", ax=ax,
    color=["#457B9D", "#E76F51", "#2A9D8F", "#9B5DE5"]
)
ax.set_title("Objective 4 — Claim-risk classifier comparison")
ax.set_ylabel("Test metric")
ax.set_xlabel("")
ax.set_ylim(0, 1)
ax.legend(title="Metric")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# Chart 5: Unsupervised anomaly detector flag rates.
if "anomaly_results" not in globals():
    raise NameError("Run the objective 5 anomaly scoring cell first.")

flag_columns = [
    col for col in anomaly_results.columns
    if col.endswith("_flag")
]
if not flag_columns:
    raise ValueError("No anomaly flag columns found in anomaly_results.")

anomaly_rates = (
    anomaly_results[flag_columns]
    .eq(-1)
    .mean()
    .mul(100)
    .rename(lambda col: col.replace("_flag", "").replace("_", " ").title())
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 5))
anomaly_rates.plot(kind="barh", ax=ax, color="#6A4C93")
ax.set_title("Objective 5 — Records flagged by each anomaly detector")
ax.set_xlabel("Flagged records (%)")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()

print(
    "The first four charts compare held-out test metrics. "
    "The anomaly chart compares flag rates only; without confirmed anomaly "
    "labels, it does not measure detector accuracy."
)
```

**Output**

```text
MACHINE LEARNING OBJECTIVES
```

| Objective | Business question | ML task | Target | Primary metrics |
|---|---|---|---|---|
| 1 | Which shipments are likely to b... | Classification | late_flag | F1, ROC-AUC |
| 2 | What will freight cost before b... | Regression | total_freight_cost | MAE, RMSE, R-squared |
| 3 | Which invoices may become overd... | Classification | payment_late | Recall, ROC-AUC |
| 4 | Which shipments are likely to l... | Imbalanced classification | has_claim | Precision-recall AUC |
| 5 | Which shipments or bills are an... | Unsupervised anomaly detection | None | Review flagged cases |

```text
The first four charts compare held-out test metrics. The anomaly chart compares flag rates only; without confirmed anomaly labels, it does not measure detector accuracy.
```

### Code: charts

```python
def chart_eval_classifier(obj, fname, title, models, X, y, default):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = {"Logistic Regression": BLUE, "Random Forest": ORANGE, "Extra Trees": "#2a9d8f", "Gradient Boosting": "#7b4fc4"}
    for n, m in models.items():
        p = m.predict_proba(X)[:, list(m.classes_).index(1)]
        f, t, _ = roc_curve(y, p); axs[0].plot(f, t, color=cols[n], lw=2.2 if n == default else 1.2, label=f"{n} ({roc_auc_score(y, p):.2f})")
    axs[0].plot([0, 1], [0, 1], "--", color="#999"); axs[0].set_title("ROC curves (unseen data)")
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    m = models[default]; p = m.predict_proba(X)[:, list(m.classes_).index(1)]
    pr, rc, _ = precision_recall_curve(y, p)
    axs[1].plot(rc, pr, color=BLUE, lw=2); axs[1].axhline(np.mean(y), color=ORANGE, ls="--", label=f"Base rate {np.mean(y):.1%}")
    axs[1].set_title(f"Precision-recall: {default}"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_ylim(0, 1.02); axs[1].legend(frameon=False)
    cm = confusion_matrix(y, (p >= .5).astype(int))
    axs[2].imshow(cm, cmap="Blues"); axs[2].spines[:].set_visible(False)
    for (i, j), v in np.ndenumerate(cm):
        axs[2].text(j, i, f"{v}", ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else INK)
    axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["Predicted no", "Predicted yes"]); axs[2].set_yticks([0, 1]); axs[2].set_yticklabels(["Actual no", "Actual yes"])
    axs[2].set_title("Confusion matrix (cut-off 0.5)")
    fig.suptitle(title, x=.01, ha="left", fontweight="bold", fontsize=12, y=1.03)
    save(fig, fname)


def chart_eval_regressor(models, X, y, rt, default, fname):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    p = models[default].predict(X)
    axs[0].scatter(y, p, s=9, color=BLUE, alpha=.5, edgecolor="none"); lim = [5, 15000]
    axs[0].plot(lim, lim, "--", color="#999"); axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlim(lim); axs[0].set_ylim(lim)
    axs[0].set_title(f"Predicted vs actual: {default}"); axs[0].set_xlabel("Actual (USD)"); axs[0].set_ylabel("Predicted (USD)")
    mae = pd.Series({n: mean_absolute_error(y, m.predict(X)) for n, m in models.items()}).sort_values(ascending=False)
    axs[1].barh([n.replace(" Regressor", "").replace(" Regression", "") for n in mae.index], mae.values, color=[GREY if n != default else BLUE for n in mae.index], height=.55)
    for i, v in enumerate(mae.values): axs[1].text(v + 8, i, f"${v:,.0f}", va="center", fontsize=9)
    axs[1].set_title("Average error by model (lower is better)"); axs[1].set_xlim(0, mae.max() * 1.2)
    err = pd.Series(np.abs(y.values - p), index=y.index)
    band = pd.qcut(y, 4, labels=["Cheapest 25%", "25-50%", "50-75%", "Costliest 25%"])
    e = err.groupby(band, observed=True).mean()
    axs[2].bar(e.index.astype(str), e.values, color=ORANGE); axs[2].set_title("Average error by shipment cost size (USD)")
    axs[2].tick_params(axis="x", labelsize=8)
    fig.subplots_adjust(wspace=.55)
    fig.suptitle("Objective 2: Freight cost, evaluation on unseen shipments", x=.01, ha="left", fontweight="bold", fontsize=12, y=1.03)
    save(fig, fname)


def chart_eval_anomaly(anomaly_results, rt, flag_rate_summary):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    names = {"isolation_forest": "Isolation Forest", "local_outlier_factor": "Local Outlier Factor", "one_class_svm": "One-Class SVM"}
    rates = pd.Series({v: float(flag_rate_summary.loc[k, "flagged_percent"]) for k, v in names.items()})
    axs[0].barh(rates.index[::-1], rates.values[::-1], color=BLUE, height=.55)
    for i, v in enumerate(rates.values[::-1]): axs[0].text(v + .1, i, f"{v:.2f}%", va="center", fontsize=9)
    axs[0].set_title("Share flagged at each detector's default"); axs[0].set_xlim(0, max(rates.max() * 1.3, 1))
    sc = anomaly_results["isolation_forest_score"].astype(float)
    cut = sc.quantile(.95)
    axs[1].hist(sc, bins=50, color=GREY); axs[1].axvline(cut, color=ORANGE, lw=2, label="Top 5% cut-off used in the app")
    axs[1].set_title("Isolation Forest scores (higher = more unusual)"); axs[1].legend(frameon=False, fontsize=8)
    m = anomaly_results.merge(rt[["shipment_id", "transport_mode"]].drop_duplicates("shipment_id"), on="shipment_id", how="left")
    m["mode"] = clean_mode(m["transport_mode"]); top = m[m["isolation_forest_score"] > cut]["mode"].value_counts()
    axs[2].bar(top.index.astype(str), top.values, color=ORANGE); axs[2].set_title("Top 5% unusual shipments by mode")
    fig.suptitle("Objective 5: Anomaly review (no answer key, so we show flag rates, not accuracy)", x=.01, ha="left", fontweight="bold", fontsize=12, y=1.03)
    save(fig, "obj5_evaluation")
```

---

## 18. Business Analysis

The business questions answered directly from the data, before any model. **Margin is billed amount minus freight cost only**, a proxy, not an accounting margin.

| Question | Answer |
|---|---|
| Which mode earns the most margin? | Air: a margin proxy of $146,200 on $1.27M billed (12%), about $196 per shipment. Road and sea earn about $38 per shipment |
| Which mode is most reliable? | Air is on time 90.4% of the time. Rail is the weakest at 83.3% |
| How much cash is at risk? | 459 overdue or disputed invoices worth $404,348 |
| Is any freight unbilled? | 243 shipments have no matched invoice, carrying $156,533 of freight cost |
| Best carriers? | JetStream Logistics is the most reliable (94.7% on time) but costs about $5.21 per kg. Oceanlink Shipping is the standout: 92.4% on time at $0.83 per kg. BlueWave Maritime is similarly cheap ($0.83) but on time only 84% of the time |
| What do claims cost? | $185,458 in total. Damage is the biggest type ($68,045 over 97 claims), then delay penalties ($56,041). Weather is the costliest named root cause at $38,576 |

Mode, carrier and claim-type figures merge spelling variants (for example `AIR`, `air`, `Air Freight`) so each reads as one group.

### Charts

<p align="center">
  <img src="images/business_01_freight_margin.png" alt="Freight cost and margin" width="100%">
</p>
<p align="center"><em>Freight cost and margin</em></p>

<p align="center">
  <img src="images/business_02_on_time_delivery.png" alt="On-time delivery and SLA" width="100%">
</p>
<p align="center"><em>On-time delivery and SLA</em></p>

<p align="center">
  <img src="images/business_03_invoices_cash.png" alt="Invoices and cash collection" width="100%">
</p>
<p align="center"><em>Invoices and cash collection</em></p>

<p align="center">
  <img src="images/business_04_carrier_scorecard.png" alt="Carrier scorecard: cost versus reliability" width="100%">
</p>
<p align="center"><em>Carrier scorecard: cost versus reliability</em></p>

<p align="center">
  <img src="images/business_05_claim_types.png" alt="Claims by type" width="100%">
</p>
<p align="center"><em>Claims by type</em></p>

<p align="center">
  <img src="images/business_06_claim_root_causes.png" alt="Claims by root cause" width="100%">
</p>
<p align="center"><em>Claims by root cause</em></p>

### Code: business analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

if "relationship_table" not in globals() or "feature_tables" not in globals():
    raise NameError(
        "Run the relationship-table and feature-engineering cells first."
    )

shipments = relationship_table.copy()
invoices = feature_tables["fact_invoices"].copy()
claims = feature_tables["fact_claims"].copy()

# Ensure monetary/duration fields are numeric for summaries and plotting.
for df, columns in [
    (shipments, [
        "total_freight_cost", "total_billed_amount_usd",
        "cost_per_kg", "late_flag", "invoice_count",
        "claim_count", "total_claim_amount_usd",
    ]),
    (invoices, ["billed_amount_usd"]),
    (claims, ["claim_amount_usd", "days_to_resolve"]),
]:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# Consistent chart style.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F7F9FC",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Create analytical summaries.
if {"total_billed_amount_usd", "total_freight_cost"}.issubset(shipments.columns):
    shipments["margin_proxy_usd"] = (
        shipments["total_billed_amount_usd"] - shipments["total_freight_cost"]
    )

if "claim_count" in shipments.columns:
    shipments["has_claim_business"] = shipments["claim_count"].fillna(0).gt(0)

if "payment_status" in invoices.columns:
    invoices["payment_status_clean"] = (
        invoices["payment_status"].astype("string").str.strip().str.title()
    )

business_analysis_tables = {}

# -------------------------------------------------------------------
# OBJECTIVE 1: Cut freight cost and protect margin
# -------------------------------------------------------------------
if "transport_mode" in shipments.columns:
    objective1 = shipments.groupby("transport_mode", dropna=False).agg(
        shipments=("shipment_id", "nunique"),
        total_freight_cost_usd=("total_freight_cost", "sum"),
        avg_cost_per_kg_usd=("cost_per_kg", "mean"),
        total_margin_proxy_usd=("margin_proxy_usd", "sum"),
        avg_margin_proxy_usd=("margin_proxy_usd", "mean"),
    ).sort_values("total_freight_cost_usd", ascending=False)

    business_analysis_tables["freight_cost_and_margin"] = objective1
    print("OBJECTIVE 1 — Freight cost and margin by transport mode")
    display(objective1.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    objective1["total_margin_proxy_usd"].sort_values().plot(
        kind="barh", ax=axes[0], color="#2A9D8F"
    )
    axes[0].set_title("Total margin proxy by transport mode")
    axes[0].set_xlabel("Billed amount minus freight cost (USD)")
    axes[0].set_ylabel("")

    objective1["avg_cost_per_kg_usd"].sort_values().plot(
        kind="barh", ax=axes[1], color="#E9C46A"
    )
    axes[1].set_title("Average freight cost per kg by mode")
    axes[1].set_xlabel("USD per kg")
    axes[1].set_ylabel("")
    fig.suptitle("1. Freight cost and margin", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# OBJECTIVE 2: Raise on-time delivery and SLA adherence
# -------------------------------------------------------------------
if "late_flag" in shipments.columns:
    objective2_mode = shipments.groupby("transport_mode", dropna=False).agg(
        known_outcomes=("late_flag", "count"),
        late_rate=("late_flag", "mean"),
    )
    objective2_mode["on_time_rate_percent"] = (
        (1 - objective2_mode["late_rate"]) * 100
    )
    objective2_mode = objective2_mode.drop(columns="late_rate")

    business_analysis_tables["delivery_by_mode"] = objective2_mode
    print("OBJECTIVE 2 — On-time performance by transport mode")
    display(objective2_mode.round(2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    objective2_mode["on_time_rate_percent"].sort_values().plot(
        kind="barh", ax=axes[0], color="#457B9D"
    )
    axes[0].set_title("On-time rate by mode")
    axes[0].set_xlabel("On-time deliveries (%)")
    axes[0].set_ylabel("")

    # Worst-performing lanes, requiring at least 10 known shipment outcomes.
    lane_data = shipments.dropna(
        subset=["origin_country", "destination_country", "late_flag"]
    ).copy()
    lane_data["lane"] = (
        lane_data["origin_country"].astype(str)
        + " → "
        + lane_data["destination_country"].astype(str)
    )
    lane_summary = lane_data.groupby("lane").agg(
        known_outcomes=("late_flag", "count"),
        late_rate=("late_flag", "mean"),
    )
    lane_summary = lane_summary[lane_summary["known_outcomes"] >= 10]
    lane_summary["late_rate_percent"] = lane_summary["late_rate"] * 100
    worst_lanes = lane_summary.nlargest(10, "late_rate_percent")

    if not worst_lanes.empty:
        worst_lanes["late_rate_percent"].sort_values().plot(
            kind="barh", ax=axes[1], color="#E76F51"
        )
        axes[1].set_title("Highest late-rate lanes (10+ outcomes)")
        axes[1].set_xlabel("Late deliveries (%)")
        axes[1].set_ylabel("")

    # Worst-performing customers, requiring at least 5 known outcomes.
    if "company_name" in shipments.columns:
        customer_summary = shipments.dropna(
            subset=["company_name", "late_flag"]
        ).groupby("company_name").agg(
            known_outcomes=("late_flag", "count"),
            late_rate=("late_flag", "mean"),
        )
        customer_summary = customer_summary[
            customer_summary["known_outcomes"] >= 5
        ]
        customer_summary["late_rate_percent"] = (
            customer_summary["late_rate"] * 100
        )
        worst_customers = customer_summary.nlargest(10, "late_rate_percent")

        if not worst_customers.empty:
            worst_customers["late_rate_percent"].sort_values().plot(
                kind="barh", ax=axes[2], color="#F4A261"
            )
            axes[2].set_title("Highest late-rate customers (5+ outcomes)")
            axes[2].set_xlabel("Late deliveries (%)")
            axes[2].set_ylabel("")

    fig.suptitle("2. On-time delivery and SLA adherence",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# OBJECTIVE 3: Close revenue leakage and speed up cash collection
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if "payment_status_clean" in invoices.columns:
    invoice_status_summary = invoices.groupby(
        "payment_status_clean", dropna=False
    ).agg(
        invoice_count=("invoice_id", "count"),
        billed_amount_usd=("billed_amount_usd", "sum"),
    ).sort_values("billed_amount_usd", ascending=False)

    business_analysis_tables["invoice_status"] = invoice_status_summary
    print("OBJECTIVE 3 — Invoice status and billed amount")
    display(invoice_status_summary.round(2))

    invoice_status_summary["billed_amount_usd"].sort_values().plot(
        kind="barh", ax=axes[0], color="#6A4C93"
    )
    axes[0].set_title("Billed amount by payment status")
    axes[0].set_xlabel("Billed amount (USD)")
    axes[0].set_ylabel("")

if {"invoice_count", "total_freight_cost"}.issubset(shipments.columns):
    unbilled = shipments["invoice_count"].fillna(0).eq(0)
    billed = ~unbilled

    coverage = pd.Series({
        "Invoiced shipments": int(billed.sum()),
        "No invoice found": int(unbilled.sum()),
    })
    coverage_exposure = pd.Series({
        "Invoiced freight cost": shipments.loc[
            billed, "total_freight_cost"
        ].sum(),
        "Freight cost with no invoice": shipments.loc[
            unbilled, "total_freight_cost"
        ].sum(),
    })

    business_analysis_tables["invoice_coverage"] = pd.DataFrame({
        "shipment_count": coverage,
        "freight_cost_usd": coverage_exposure,
    })
    print("Invoice coverage and freight cost exposure")
    display(business_analysis_tables["invoice_coverage"].round(2))

    coverage_exposure.plot(
        kind="bar", ax=axes[1], color=["#2A9D8F", "#E63946"]
    )
    axes[1].set_title("Freight cost by invoice coverage")
    axes[1].set_ylabel("Freight cost (USD)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)

fig.suptitle("3. Invoice coverage and cash collection",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# OBJECTIVE 4: Rank carriers on cost, reliability, and claims
# -------------------------------------------------------------------
if "carrier_name" in shipments.columns:
    carrier_summary = shipments.groupby("carrier_name", dropna=False).agg(
        shipments=("shipment_id", "nunique"),
        avg_cost_per_kg_usd=("cost_per_kg", "mean"),
        late_rate=("late_flag", "mean"),
        claim_count=("claim_count", "sum"),
    )
    carrier_summary["on_time_rate_percent"] = (
        (1 - carrier_summary["late_rate"]) * 100
    )
    carrier_summary = carrier_summary.drop(columns="late_rate")

    business_analysis_tables["carrier_scorecard"] = carrier_summary
    print("OBJECTIVE 4 — Carrier scorecard")
    display(carrier_summary.sort_values("shipments", ascending=False).round(2))

    plot_data = carrier_summary.dropna(
        subset=["avg_cost_per_kg_usd", "on_time_rate_percent"]
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    if not plot_data.empty:
        point_sizes = (
            plot_data["claim_count"].fillna(0).clip(lower=0) * 18 + 45
        )
        ax.scatter(
            plot_data["avg_cost_per_kg_usd"],
            plot_data["on_time_rate_percent"],
            s=point_sizes,
            alpha=0.75,
            color="#277DA1",
            edgecolor="white",
            linewidth=0.7,
        )
        for carrier, row in plot_data.iterrows():
            ax.annotate(
                str(carrier),
                (row["avg_cost_per_kg_usd"], row["on_time_rate_percent"]),
                fontsize=8,
                xytext=(4, 3),
                textcoords="offset points",
            )

    ax.set_title("Carrier cost and delivery reliability")
    ax.set_xlabel("Average cost per kg (USD)")
    ax.set_ylabel("On-time delivery rate (%)")
    ax.grid(alpha=0.25)
    fig.suptitle(
        "4. Carrier scorecard (larger markers indicate more claims)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# OBJECTIVE 5: Reduce claims, damage, and loss
# -------------------------------------------------------------------
if {"claim_type", "claim_amount_usd"}.issubset(claims.columns):
    claim_type_summary = claims.groupby("claim_type", dropna=False).agg(
        claim_count=("claim_id", "count"),
        total_claim_amount_usd=("claim_amount_usd", "sum"),
        avg_resolution_days=("days_to_resolve", "mean"),
    ).sort_values("total_claim_amount_usd", ascending=False)

    business_analysis_tables["claims_by_type"] = claim_type_summary
    print("OBJECTIVE 5 — Claim type cost and resolution")
    display(claim_type_summary.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    claim_type_summary["total_claim_amount_usd"].sort_values().plot(
        kind="barh", ax=axes[0], color="#D62828"
    )
    axes[0].set_title("Total claim amount by claim type")
    axes[0].set_xlabel("Claim amount (USD)")
    axes[0].set_ylabel("")

    claim_type_summary["avg_resolution_days"].sort_values().plot(
        kind="barh", ax=axes[1], color="#F77F00"
    )
    axes[1].set_title("Average resolution time by claim type")
    axes[1].set_xlabel("Days to resolve")
    axes[1].set_ylabel("")

    fig.suptitle("5. Claims, damage, and loss",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Root cause cost chart, when available.
if {"root_cause", "claim_amount_usd"}.issubset(claims.columns):
    root_cause_summary = claims.groupby("root_cause", dropna=False).agg(
        claim_count=("claim_id", "count"),
        total_claim_amount_usd=("claim_amount_usd", "sum"),
    ).sort_values("total_claim_amount_usd", ascending=False)

    business_analysis_tables["claims_by_root_cause"] = root_cause_summary
    print("Claim amount by root cause")
    display(root_cause_summary.round(2))

    root_cause_summary["total_claim_amount_usd"].sort_values().plot(
        kind="barh",
        figsize=(9, 5),
        color="#9D4EDD",
        title="Total claim amount by root cause",
    )
    plt.xlabel("Claim amount (USD)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

print(
    "\nBusiness summaries are stored in `business_analysis_tables`. "
    "Margin proxy is billed amount minus freight cost only; it is not "
    "a complete accounting margin. Freight cost without an invoice is "
    "an exposure indicator, not proof of revenue leakage."
)
```

**Output**

```text
OBJECTIVE 1 — Freight cost and margin by transport mode
```

| transport_mode | shipments | total_freight_cost_usd | avg_cost_per_kg_usd | total_margin_proxy_usd | avg_margin_proxy_usd |
|---|---|---|---|---|---|
| Air | 297 | 4.765e+05 | 5.21 | 4.615e+04 | 155.4 |
| Air Freight | 157 | 2.243e+05 | 5.16 | 3.255e+04 | 207.3 |
| AIR | 149 | 2.19e+05 | 5.04 | 3.559e+04 | 238.8 |
| air | 144 | 2.047e+05 | 5.24 | 3.192e+04 | 221.7 |
| Road | 385 | 1.735e+05 | 1.58 | 1.78e+04 | 46.24 |
| Sea | 419 | 1.071e+05 | 0.86 | 1.72e+04 | 41.05 |
| … | … | … | … | … | … |
| SEA | 149 | 3.42e+04 | 0.83 | 4,753 | 31.9 |
| RAIL | 72 | 2.942e+04 | 1.14 | 4,748 | 65.95 |
| rail | 82 | 2.636e+04 | 1.11 | 3,252 | 39.66 |

```text
OBJECTIVE 2 — On-time performance by transport mode
```

| transport_mode | known_outcomes | on_time_rate_percent |
|---|---|---|
| AIR | 140 | 89.29 |
| Air | 269 | 90.33 |
| Air Freight | 139 | 89.93 |
| RAIL | 64 | 76.56 |
| ROAD | 143 | 88.11 |
| Rail | 185 | 87.03 |
| … | … | … |
| rail | 74 | 79.73 |
| road | 125 | 89.6 |
| sea | 120 | 84.17 |

```text
OBJECTIVE 3 — Invoice status and billed amount
```

| payment_status_clean | invoice_count | billed_amount_usd |
|---|---|---|
| Paid | 1440 | 1.214e+06 |
| Pending | 368 | 3.46e+05 |
| Overdue | 341 | 2.972e+05 |
| Disputed | 118 | 1.071e+05 |

```text
Invoice coverage and freight cost exposure
```

| index | shipment_count | freight_cost_usd |
|---|---|---|
| Freight cost with no invoice | nan | 1.565e+05 |
| Invoiced freight cost | nan | 1.585e+06 |
| Invoiced shipments | 2,257 | nan |
| No invoice found | 243 | nan |

```text
OBJECTIVE 4 — Carrier scorecard
```

| carrier_name | shipments | avg_cost_per_kg_usd | claim_count | on_time_rate_percent |
|---|---|---|---|---|
| SkyBridge Air Cargo | 199 | 5.2 | 18 | 91.06 |
| AeroSwift Cargo | 198 | 5.17 | 11 | 88.64 |
| BlueWave Maritime | 191 | 0.84 | 21 | 84.02 |
| EuroLine Transport | 185 | 1.59 | 20 | 87.57 |
| IronTrack Rail | 184 | 1.11 | 19 | 83.23 |
| PortKing Marine | 183 | 0.86 | 20 | 85.63 |
| … | … | … | … | … |
| DeepSea Carriers | 161 | 0.85 | 12 | 84.67 |
| JetStream Logistics | 161 | 5.18 | 16 | 94.67 |
| nan | 18 | 2.7 | 0 | 93.75 |

```text
OBJECTIVE 5 — Claim type cost and resolution
```

| claim_type | claim_count | total_claim_amount_usd | avg_resolution_days |
|---|---|---|---|
| Damage | 84 | 5.831e+04 | 41.29 |
| Delay Penalty | 61 | 4.991e+04 | 45.02 |
| Temperature Breach | 25 | 2.032e+04 | 35.78 |
| Customs Hold | 13 | 1.61e+04 | 49.58 |
| Loss | 16 | 1.285e+04 | 31.17 |
| damage | 10 | 8,421 | 42.56 |
| … | … | … | … |
| DAMAGE | 3 | 1,313 | 30 |
| TEMPERATURE BREACH | 1 | 487.9 | 83 |
| temperature breach | 1 | 128.7 | -5 |

```text
Claim amount by root cause
```

| root_cause | claim_count | total_claim_amount_usd |
|---|---|---|
| Weather | 35 | 3.858e+04 |
| Unknown | 41 | 3.388e+04 |
| Customs | 39 | 3.289e+04 |
| Packaging | 33 | 2.726e+04 |
| nan | 23 | 1.937e+04 |
| Handling | 30 | 1.708e+04 |
| Carrier Delay | 29 | 1.64e+04 |


---

## 19. Save Model, Scaler and Features

For every objective three files are saved with joblib, plus one bundle that contains all three:

- `<objective>_models.joblib`: all candidate pipelines
- `<objective>_scaler.joblib`: the fitted scaler (none for anomaly review)
- `<objective>_features.joblib`: the exact column order, scaled columns and notes
- `<objective>_bundle.joblib`: the three above in one file

That is 19 files. The app rebuilds each input in the saved column order, scales it, and scores it.

### Code: save model, scaler and features

```python
from pathlib import Path
import joblib
import pandas as pd

output_dir = Path("model_artifacts_joblib")
output_dir.mkdir(parents=True, exist_ok=True)

target_by_objective = {
    "late_delivery": "late_flag",
    "freight_cost": "total_freight_cost",
    "payment_risk": "payment_late",
    "claim_risk": "has_claim",
    "anomaly_review": None,
}

saved_files = []

for objective, objective_models in models_by_objective.items():
    if not objective_models:
        continue

    feature_names = (
        list(X_unsupervised.columns)
        if objective == "anomaly_review"
        else list(X_tables[objective].columns)
    )

    scaler = feature_scalers.get(objective)
    scaled_columns = (
        list(scaler.feature_names_in_)
        if scaler is not None and hasattr(scaler, "feature_names_in_")
        else []
    )

    feature_metadata = {
        "objective": objective,
        "target": target_by_objective.get(objective),
        "feature_names_in_order": feature_names,
        "scaled_columns": scaled_columns,
        "candidate_model_names": list(objective_models.keys()),
        "preprocessing_note": (
            "Scaler saved separately; imputation is inside each model pipeline."
            if scaler is not None
            else "Preprocessing is stored inside each anomaly detector pipeline."
        ),
    }

    model_path = output_dir / f"{objective}_models.joblib"
    features_path = output_dir / f"{objective}_features.joblib"
    bundle_path = output_dir / f"{objective}_bundle.joblib"

    joblib.dump(objective_models, model_path)
    joblib.dump(feature_metadata, features_path)

    if scaler is not None and hasattr(scaler, "mean_"):
        scaler_path = output_dir / f"{objective}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        saved_files.append(scaler_path)

    bundle = {
        "objective": objective,
        "target": target_by_objective.get(objective),
        "models": objective_models,
        "scaler": scaler,
        "features": feature_metadata,
    }
    joblib.dump(bundle, bundle_path)

    saved_files.extend([model_path, features_path, bundle_path])

print("JOBLIB-ONLY ARTIFACTS SAVED")
display(pd.DataFrame([
    {
        "file": path.as_posix(),
        "size_kb": round(path.stat().st_size / 1024, 2),
    }
    for path in sorted(set(saved_files))
]).reset_index(drop=True))

print(f"\nArtifact folder: {output_dir.resolve()}")
```

**Output**

```text
JOBLIB-ONLY ARTIFACTS SAVED
```

| file | size_kb |
|---|---|
| model_artifacts_joblib/anomaly_... | 4,961 |
| model_artifacts_joblib/anomaly_... | 3.4 |
| model_artifacts_joblib/anomaly_... | 4,957 |
| model_artifacts_joblib/claim_ri... | 4.618e+04 |
| model_artifacts_joblib/claim_ri... | 4.89 |
| model_artifacts_joblib/claim_ri... | 4.617e+04 |
| … | … |
| model_artifacts_joblib/payment_... | 1.02 |
| model_artifacts_joblib/payment_... | 6.765e+04 |
| model_artifacts_joblib/payment_... | 1.01 |

```text
Artifact folder: /tmp/claude-0/-home-claude/d4834066-0350-5509-8034-a5a343a62845/scratchpad/tgen/model_artifacts_joblib
```


---

## 20. Download Model, Scaler and Features

The 19 files are zipped into `model_artifacts_download.zip` for download.

**Deployment note:** the freight-cost model file was about 140 MB, over GitHub's 100 MB limit. For the repo I re-saved each `_models.joblib` with `joblib.dump(..., compress=3)` (same models, about a quarter of the size) and left out the four `_bundle` files, which the app does not use.

### Code: download model, scaler and features

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display

artifact_dir = Path("model_artifacts_joblib")
zip_path = Path("model_artifacts_download.zip")

if not artifact_dir.exists():
    raise FileNotFoundError(
        "The 'model_artifacts' folder was not found. "
        "Run the save-model cell first."
    )

artifact_files = [
    path for path in artifact_dir.rglob("*")
    if path.is_file()
]

if not artifact_files:
    raise FileNotFoundError("No saved artifact files were found.")

with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
    for path in artifact_files:
        zip_file.write(path, arcname=path.relative_to(artifact_dir))

print(f"Download package created: {zip_path.resolve()}")
display(FileLink(str(zip_path)))

print("\nIndividual artifact files:")
for path in sorted(artifact_files):
    display(FileLink(str(path)))
```

**Output**

```text
Download package created: /tmp/claude-0/-home-claude/d4834066-0350-5509-8034-a5a343a62845/scratchpad/tgen/model_artifacts_download.zip
model_artifacts_download.zip

Individual artifact files:
model_artifacts_joblib/anomaly_review_bundle.joblib
model_artifacts_joblib/anomaly_review_features.joblib
model_artifacts_joblib/anomaly_review_models.joblib
model_artifacts_joblib/claim_risk_bundle.joblib
model_artifacts_joblib/claim_risk_features.joblib
model_artifacts_joblib/claim_risk_models.joblib
model_artifacts_joblib/claim_risk_scaler.joblib
model_artifacts_joblib/freight_cost_bundle.joblib
model_artifacts_joblib/freight_cost_features.joblib
model_artifacts_joblib/freight_cost_models.joblib
model_artifacts_joblib/freight_cost_scaler.joblib
model_artifacts_joblib/late_delivery_bundle.joblib
model_artifacts_joblib/late_delivery_features.joblib
model_artifacts_joblib/late_delivery_models.joblib
model_artifacts_joblib/late_delivery_scaler.joblib
model_artifacts_joblib/payment_risk_bundle.joblib
model_artifacts_joblib/payment_risk_features.joblib
model_artifacts_joblib/payment_risk_models.joblib
model_artifacts_joblib/payment_risk_scaler.joblib
```



### The Streamlit app

The saved files power [`app.py`](app.py), a single file with seven pages:

- **Home:** the five objectives, the page and model behind each, and how reliable each one is.
- **Freight Cost:** predicted cost, cost per kg and expected margin against a billed amount.
- **Late Delivery:** late probability, Low, Medium or High level and a suggested action.
- **Payment Risk:** probability of late payment, and a warning when an invoice bills below freight cost.
- **Claim Risk:** probability of a claim, damage or loss.
- **Carrier Ranking:** scores all 14 carriers for the same shipment on predicted cost and late risk, with a slider for how much each counts.
- **Anomaly Review:** one shipment or a CSV upload. A shipment is flagged when it is more unusual than 95% of the historical ones.

Each page has a model selector, defaulting to the model that led in section 17. The payment and claim pages carry a warning banner because those models score near chance.

**Run it yourself**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live app: [Open the Streamlit app](https://trans-globe-logistics-business-operation-prediction-system-9yr.streamlit.app/)

### Full code: app.py

```python
# =====================================================================
# TransGlobe Logistics - Business Operations Prediction System
# One-file Streamlit app. Run with:   streamlit run app.py
#
# Pages
#   1. Home               - overview + model summary
#   2. Freight Cost       - predict shipment cost and margin
#   3. Late Delivery      - probability a shipment misses its promise date
#   4. Payment Risk       - probability an invoice is paid late
#   5. Claim Risk         - probability a shipment ends in a claim
#   6. Carrier Ranking    - rank all carriers on cost + reliability
#   7. Anomaly Review     - flag unusual shipments / billing (single or CSV)
# =====================================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TransGlobe Prediction System", page_icon="🚚", layout="wide")

# ---------------------------------------------------------------------
# 1. LOAD THE SAVED MODEL FILES
#    Put this app.py next to (or above) your `model_artifacts_joblib` folder.
# ---------------------------------------------------------------------
OBJECTIVES = ["late_delivery", "freight_cost", "payment_risk", "claim_risk", "anomaly_review"]


def find_model_folder():
    """Look for the folder that holds late_delivery_models.joblib (any depth)."""
    here = Path(__file__).parent
    for hit in here.rglob("late_delivery_models.joblib"):
        return hit.parent
    return None


@st.cache_resource
def load_all():
    folder = find_model_folder()
    if folder is None:
        return None
    store = {}
    for obj in OBJECTIVES:
        scaler_path = folder / f"{obj}_scaler.joblib"
        store[obj] = {
            "models": joblib.load(folder / f"{obj}_models.joblib"),      # dict {name: pipeline}
            "features": joblib.load(folder / f"{obj}_features.joblib"),  # dict with column order
            "scaler": joblib.load(scaler_path) if scaler_path.exists() else None,
        }
    return store


STORE = load_all()
if STORE is None:
    st.error("Could not find the model files. Put the `model_artifacts_joblib` folder "
             "(the one with late_delivery_models.joblib etc.) next to app.py.")
    st.stop()

# ---------------------------------------------------------------------
# 2. DROPDOWN CHOICES (these match the cleaned categories used in training)
# ---------------------------------------------------------------------
COUNTRIES = ["Brazil", "China", "France", "Germany", "India", "Netherlands", "Nigeria",
             "Singapore", "South Africa", "UAE", "United Kingdom", "United States"]
MODES = ["Air", "Sea", "Road", "Rail"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
INDUSTRIES = ["Agriculture", "Automotive", "Electronics", "Energy", "FMCG", "Fashion", "Pharma", "Retail"]
# carrier name -> contract rate per kg (USD) from the carrier table
CARRIERS = {
    "AeroSwift Cargo": 4.47, "BlueWave Maritime": 1.18, "Continental Trucking": 3.78,
    "DeepSea Carriers": 5.03, "EuroLine Transport": 5.17, "IronTrack Rail": 2.98,
    "JetStream Logistics": 1.52, "PortKing Marine": 1.87, "RailCore Express": 1.86,
    "RoadRunner Freight": 3.97, "SilverArrow Air": 3.07, "SkyBridge Air Cargo": 5.98,
    "TransContinental Haulage": 5.22, "oceanlink shipping": 2.34,
}
# a column name prefix in the model  ->  the name of the input column we use
CAT_PREFIXES = {
    "customer_tier_": "customer_tier", "industry_": "industry", "hq_country_": "hq_country",
    "carrier_name_": "carrier_name", "origin_country_": "origin_country",
    "destination_country_": "destination_country", "transport_mode_": "transport_mode",
}

# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def make_frame(raw, objective):
    """Turn plain inputs (one or many rows) into the exact table the model was trained on.
    - start with all zeros using the saved column order
    - copy the numeric values in
    - switch on the matching one-hot (dummy) column for each category
    - scale the continuous columns with the saved scaler
    """
    info = STORE[objective]
    columns = info["features"]["feature_names_in_order"]
    X = pd.DataFrame(0.0, index=range(len(raw)), columns=columns)

    for col in raw.columns:                      # numeric inputs
        if col in X.columns:
            X[col] = raw[col].astype(float).values

    for prefix, name in CAT_PREFIXES.items():    # categorical inputs -> dummy columns
        if name in raw.columns:
            for i, value in enumerate(raw[name].astype(str)):
                dummy = prefix + value
                if dummy in X.columns:
                    X.loc[i, dummy] = 1.0

    scaler = info["scaler"]
    if scaler is not None:                       # scale only what the scaler was fitted on
        cols = list(scaler.feature_names_in_)
        X[cols] = scaler.transform(X[cols])
    return X


def predict_probability(raw, objective, model_name):
    """Probability of the 'yes' class (late / claim / late payment)."""
    model = STORE[objective]["models"][model_name]
    X = make_frame(raw, objective)
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def predict_cost(raw, model_name):
    """Predicted total freight cost in USD."""
    model = STORE["freight_cost"]["models"][model_name]
    return model.predict(make_frame(raw, "freight_cost"))


# Isolation Forest never flags anything at its built-in threshold (0 of 2,500 shipments),
# so we flag a shipment when its score is in the top 5% of the scores seen in training.
# Cut-offs below = 95th percentile of the training scores for each detector.
REVIEW_CUTOFF = {"Isolation Forest": -0.1450, "Local Outlier Factor": -0.3424, "One-Class SVM": 0.0}


def anomaly_score(raw, model_name):
    """Higher score = more unusual. Flagged when the score is above the review cut-off."""
    model = STORE["anomaly_review"]["models"][model_name]
    score = -model.decision_function(make_frame(raw, "anomaly_review"))
    return score, score > REVIEW_CUTOFF[model_name]


def risk_level(p):
    if p < 0.35:
        return "Low", "🟢"
    if p < 0.50:
        return "Medium", "🟡"
    return "High", "🔴"


def money(x):
    return f"${x:,.2f}"


def pick_model(objective, default_name, key):
    names = list(STORE[objective]["models"])
    index = names.index(default_name) if default_name in names else 0
    return st.sidebar.selectbox("Model", names, index=index, key=key)


def show_risk(p, what, action_high, action_low):
    level, icon = risk_level(p)
    c1, c2 = st.columns(2)
    c1.metric(f"{what} probability", f"{p:.1%}")
    c2.metric("Risk level", f"{icon} {level}")
    st.progress(float(min(max(p, 0), 1)))
    st.info(action_high if level == "High" else action_low)


def weak_model_warning(what):
    st.warning(f"**Read with care.** On held-out data the {what} models score close to chance "
               "(ROC AUC about 0.5), because this dataset's outcomes were assigned at random. "
               "Use this page to show the workflow, not as a decision tool.")


# --- shared input forms ------------------------------------------------
def customer_inputs(col):
    tier = col.selectbox("Customer tier", TIERS, index=2)
    industry = col.selectbox("Industry", INDUSTRIES)
    hq = col.selectbox("Customer HQ country", COUNTRIES, index=4)
    sla = col.slider("Contract SLA (days)", 1, 14, 5)
    credit = col.number_input("Credit limit (USD)", 0, 1_000_000, 250_000, step=10_000)
    return tier, industry, hq, sla, credit


def route_inputs(col):
    origin = col.selectbox("Origin country", COUNTRIES, index=6)
    dest = col.selectbox("Destination country", COUNTRIES, index=11)
    mode = col.selectbox("Transport mode", MODES)
    weight = col.number_input("Weight (kg)", 1.0, 100_000.0, 200.0, step=10.0)
    month = col.slider("Ship month", 1, 12, 6)
    return origin, dest, mode, weight, month


def shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month):
    return pd.DataFrame([{
        "customer_tier": tier, "industry": industry, "hq_country": hq,
        "contract_sla_days": sla, "credit_limit_usd": credit,
        "carrier_name": carrier, "contract_rate_per_kg_usd": rate,
        "origin_country": origin, "destination_country": dest,
        "transport_mode": mode, "weight_kg": weight, "ship_month": month,
    }])


# ---------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("🚚 TransGlobe")
page = st.sidebar.radio("Go to", [
    "Home", "Freight Cost", "Late Delivery", "Payment Risk",
    "Claim Risk", "Carrier Ranking", "Anomaly Review"])
st.sidebar.markdown("---")

# ---------------------------------------------------------------------
# 5. PAGES
# ---------------------------------------------------------------------
if page == "Home":
    st.title("TransGlobe Logistics: Business Operations Prediction System")
    st.write("Five models that help the operations team cut freight cost, deliver on time, "
             "collect cash faster, choose better carriers and reduce claims.")
    table = pd.DataFrame([
        ["Cut freight cost and protect margin", "Freight Cost", "Regression", "Random Forest Regressor", "Strong"],
        ["Raise on-time delivery and SLA adherence", "Late Delivery", "Classification", "Logistic Regression", "Good"],
        ["Close revenue leakage and speed up cash collection", "Payment Risk + Anomaly Review", "Classification + anomaly", "Logistic Regression / Isolation Forest", "Weak (payment), review tool (anomaly)"],
        ["Rank carriers on cost and reliability", "Carrier Ranking", "Uses the cost and late models", "Combined score", "Follows the two models"],
        ["Reduce claims, damage and loss", "Claim Risk", "Classification", "Logistic Regression", "Weak"],
    ], columns=["Business objective", "Page", "Type", "Default model", "Reliability"])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("**How to use it:** pick a page in the sidebar, fill in the shipment details, "
                "and read the prediction and the suggested action. You can switch the model in the sidebar.")
    st.caption("Models were trained on a chronological 80/20 split. Payment and claim outcomes in the "
               "source data are random, so those two models score near chance and are shown for workflow demonstration.")

# ---------------------------------------------------------------- freight
elif page == "Freight Cost":
    st.title("💰 Freight Cost and Margin")
    model_name = pick_model("freight_cost", "Random Forest Regressor", "m_cost")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)
    billed = st.number_input("Amount you plan to bill the customer (USD, optional)", 0.0, 10_000_000.0, 0.0, step=100.0)

    if st.button("Predict freight cost", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        cost = max(float(predict_cost(raw, model_name)[0]), 0.0)
        c1, c2 = st.columns(2)
        c1.metric("Predicted freight cost", money(cost))
        c2.metric("Cost per kg", money(cost / weight))
        if billed > 0:
            margin = billed - cost
            st.metric("Expected margin", money(margin), f"{margin / billed:.1%} of billed")
            if margin < 0:
                st.error("Billing is below the predicted cost. Raise the price or switch carrier.")
            elif margin / billed < 0.10:
                st.warning("Margin is thin (under 10%). Check the carrier ranking for a cheaper option.")
            else:
                st.success("Margin looks healthy.")
        else:
            st.info("Enter a billed amount to see the expected margin.")

# ---------------------------------------------------------------- late
elif page == "Late Delivery":
    st.title("⏱️ Late Delivery Risk")
    model_name = pick_model("late_delivery", "Logistic Regression", "m_late")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)

    if st.button("Check late risk", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        p = float(predict_probability(raw, "late_delivery", model_name)[0])
        show_risk(p, "Late delivery",
                  "Action: add buffer time, pick a more reliable carrier, or warn the customer early.",
                  "Action: ship as planned and monitor.")

# ---------------------------------------------------------------- payment
elif page == "Payment Risk":
    st.title("🧾 Payment Risk and Revenue Leakage")
    weak_model_warning("payment-risk")
    model_name = pick_model("payment_risk", "Logistic Regression", "m_pay")
    left, right = st.columns(2)
    tier = left.selectbox("Customer tier", TIERS, index=2)
    industry = left.selectbox("Industry", INDUSTRIES)
    credit = left.number_input("Credit limit (USD)", 0, 1_000_000, 250_000, step=10_000)
    mode = right.selectbox("Transport mode", MODES)
    billed = right.number_input("Billed amount (USD)", 0.0, 10_000_000.0, 2_000.0, step=100.0)
    freight = right.number_input("Total freight cost (USD)", 0.0, 10_000_000.0, 1_500.0, step=100.0)

    if st.button("Check payment risk", type="primary"):
        raw = pd.DataFrame([{
            "customer_tier": tier, "industry": industry, "transport_mode": mode,
            "credit_limit_usd": credit, "billed_amount_usd": billed,
            "total_freight_cost": freight, "margin_usd": billed - freight}])
        p = float(predict_probability(raw, "payment_risk", model_name)[0])
        show_risk(p, "Late payment",
                  "Action: ask for a deposit, shorten payment terms or follow up early.",
                  "Action: standard payment terms.")
        if billed < freight:
            st.error("This invoice bills less than the freight cost: a revenue leak.")

# ---------------------------------------------------------------- claims
elif page == "Claim Risk":
    st.title("📦 Claim, Damage and Loss Risk")
    weak_model_warning("claim-risk")
    model_name = pick_model("claim_risk", "Logistic Regression", "m_claim")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    carrier = right.selectbox("Carrier", list(CARRIERS))
    rate = right.number_input("Contract rate per kg (USD)", 0.0, 50.0, CARRIERS[carrier], step=0.1)

    if st.button("Check claim risk", type="primary"):
        raw = shipment_row(carrier, rate, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
        p = float(predict_probability(raw, "claim_risk", model_name)[0])
        show_risk(p, "Claim",
                  "Action: improve packaging, add insurance, or choose a carrier with fewer incidents.",
                  "Action: normal handling.")

# ---------------------------------------------------------------- ranking
elif page == "Carrier Ranking":
    st.title("🏆 Carrier Ranking")
    st.write("Scores every carrier for the same shipment using the freight-cost and late-delivery models.")
    cost_model = pick_model("freight_cost", "Random Forest Regressor", "m_rank_cost")
    late_model = pick_model("late_delivery", "Logistic Regression", "m_rank_late")
    left, right = st.columns(2)
    tier, industry, hq, sla, credit = customer_inputs(left)
    origin, dest, mode, weight, month = route_inputs(right)
    w_cost = st.slider("Importance of cost vs reliability", 0, 100, 50,
                       help="100 = only cost matters, 0 = only on-time reliability matters")

    if st.button("Rank carriers", type="primary"):
        rows = pd.concat([shipment_row(c, r, tier, industry, hq, sla, credit, origin, dest, mode, weight, month)
                          for c, r in CARRIERS.items()], ignore_index=True)
        result = pd.DataFrame({
            "Carrier": list(CARRIERS),
            "Predicted cost (USD)": np.maximum(predict_cost(rows, cost_model), 0),
            "Late probability": predict_probability(rows, "late_delivery", late_model),
        })

        def to_01(s):   # 0 = best, 1 = worst
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

        result["Score"] = (w_cost / 100) * to_01(result["Predicted cost (USD)"]) \
            + (1 - w_cost / 100) * to_01(result["Late probability"])
        result = result.sort_values("Score").reset_index(drop=True)
        result.insert(0, "Rank", result.index + 1)
        best = result.iloc[0]
        st.success(f"Best carrier for this shipment: **{best['Carrier']}** "
                   f"({money(best['Predicted cost (USD)'])}, {best['Late probability']:.1%} late risk)")
        st.dataframe(result.style.format({"Predicted cost (USD)": "${:,.2f}",
                                          "Late probability": "{:.1%}", "Score": "{:.2f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Score: 0 is best, 1 is worst. It blends min-max scaled cost and late probability.")
        st.bar_chart(result.set_index("Carrier")["Score"])

# ---------------------------------------------------------------- anomaly
elif page == "Anomaly Review":
    st.title("🔍 Anomaly Review")
    st.write("Flags shipments whose weight, cost, billing and route look unusual. "
             "This is an unsupervised review list, not an accuracy score. A shipment is flagged when it is "
             "more unusual than 95% of the historical shipments.")
    model_name = pick_model("anomaly_review", "Isolation Forest", "m_anom")
    tab1, tab2 = st.tabs(["Single shipment", "Upload CSV"])

    with tab1:
        left, right = st.columns(2)
        weight = left.number_input("Weight (kg)", 1.0, 100_000.0, 200.0, step=10.0)
        freight = left.number_input("Total freight cost (USD)", 0.0, 10_000_000.0, 500.0, step=50.0)
        billed = left.number_input("Total billed amount (USD)", 0.0, 10_000_000.0, 700.0, step=50.0)
        mode = right.selectbox("Transport mode", MODES, key="a_mode")
        origin = right.selectbox("Origin country", COUNTRIES, index=6, key="a_o")
        dest = right.selectbox("Destination country", COUNTRIES, index=11, key="a_d")
        if st.button("Check shipment", type="primary"):
            raw = pd.DataFrame([{
                "weight_kg": weight, "total_freight_cost": freight, "total_billed_amount_usd": billed,
                "cost_per_kg": freight / weight, "margin_usd": billed - freight,
                "transport_mode": mode, "origin_country": origin, "destination_country": dest}])
            score, flag = anomaly_score(raw, model_name)
            c1, c2 = st.columns(2)
            c1.metric("Anomaly score", f"{score[0]:.3f}")
            c2.metric("Result", "🔴 Unusual" if flag[0] else "🟢 Normal")
            st.info("Send this one for manual review." if flag[0] else "Looks like a typical shipment.")

    with tab2:
        st.write("Columns needed: `weight_kg`, `total_freight_cost`, `total_billed_amount_usd`, "
                 "`transport_mode`, `origin_country`, `destination_country`. Use the same country and mode names as the dropdowns (e.g. United Kingdom, not UK).")
        file = st.file_uploader("Upload CSV", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            need = ["weight_kg", "total_freight_cost", "total_billed_amount_usd",
                    "transport_mode", "origin_country", "destination_country"]
            missing = [c for c in need if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                raw = data.copy()
                raw["cost_per_kg"] = raw["total_freight_cost"] / raw["weight_kg"].replace(0, np.nan)
                raw["margin_usd"] = raw["total_billed_amount_usd"] - raw["total_freight_cost"]
                score, flag = anomaly_score(raw.fillna(0), model_name)
                data["anomaly_score"] = score
                data["flagged"] = flag
                st.metric("Flagged rows", f"{int(flag.sum())} of {len(data)}")
                st.dataframe(data.sort_values("anomaly_score", ascending=False),
                             use_container_width=True, hide_index=True)
                st.download_button("Download results", data.to_csv(index=False), "anomaly_results.csv")
```

---

## 21. Key Findings and Insights

**Objective 1: late delivery**
- This model has real signal. Logistic Regression ranks shipments well (ROC AUC 0.886) and caught 39 of 44 late shipments in the test period.
- The price of that recall is false alarms: 79 on-time shipments were flagged too, so about one flag in three is a true late delivery. It works as a watch-list, not a verdict.
- **Contract SLA is the dominant driver.** Shuffling it drops the model's ROC AUC by 0.22. The next biggest driver (customer HQ country) drops it by only 0.04.
- A 2-day SLA means 45% late, a 3-day SLA 30%, and 5 days or more under 3%. Late shipments averaged 2.8 SLA days against 6.2 for on-time ones.
- Rail is the weakest mode (16.7% late) and JetStream Logistics the strongest carrier (5.3% late against 16.9% for RailCore Express).

**Objective 2: freight cost**
- Random Forest predicts the cost of a shipment to within $146 on average (R² 0.83). Ridge Regression, the simple linear baseline, misses by $628, so the relationship is not linear.
- **Weight, transport mode and carrier drive the price.** Shuffling weight raises the average error by $589, mode by $280 and carrier by $78. Air runs about $5.22 per kg against $0.85 to $1.56 for road, rail and sea.
- The model is most accurate on small and mid-priced shipments. Its average error is about $40 on the cheapest quarter and about $400 on the costliest quarter.

**Objective 3: payment risk**
- No model beat chance (ROC AUC 0.48 to 0.51). Billed amount, credit limit, tier and mode do not tell late payers apart in this data.
- The business problem is still real: 459 overdue or disputed invoices hold $404,348, and 243 shipments carrying $156,533 of freight were never invoiced.

**Objective 4: claims**
- Also at chance level (PR AUC 0.093 against a 9.2% base rate). Two of the four models caught none of the 46 claims in the test period.
- Claims are expensive when they happen: $185,458 in total, led by damage and delay penalties, with weather the costliest root cause.

**Objective 5: anomaly review and carrier ranking**
- Isolation Forest flags nothing at its default threshold (0 of 2,500), LOF flags 11 and One-Class SVM 141. They barely agree (overlap under 7%), so these are review lists, not proof of error.
- In the app I flag the top 5% by Isolation Forest score. Those 125 shipments are mostly air (97 of 125), and what makes them unusual is mainly high freight cost and billing, so "unusual" here largely means "expensive".
- The carrier scorecard shows a clear trade-off: JetStream (94.7% on time) and SkyBridge (91.1%) cost about $5.2 per kg, while BlueWave, DeepSea and PortKing cost $0.84 to $0.86 but run on time only 84% to 86% of the time. oceanlink shipping is the exception: $0.84 per kg and 92.4% on time.

**Across the project**
- **Signal decides everything.** Late delivery and freight cost are predictable. Payment and claims are not, whatever the algorithm.
- **Data cleaning shows up in the model.** Spelling variants were never merged, so the models carry 172 dummy columns where about 50 would do.
- **Weight outliers** (up to 595,000 kg and some negative values) were left in the data, which is worth fixing before the next retrain.
- **Honest reporting.** Where a model fails, the README, the app and the numbers say so.

---

## 22. Recommendations

1. **Use the late-delivery model as a watch-list.** Score shipments at booking, review the high-risk ones first and add buffer time or warn the customer.
2. **Use the freight-cost model to price quotes and check margin.** Compare the predicted cost to the billed amount before a quote goes out.
3. **Look hard at tight-SLA customers and rail lanes.** That is where late deliveries concentrate.
4. **Close the invoice gap.** Build a shipment-to-invoice check so the 243 un-invoiced shipments ($156,533 of freight) are billed.
5. **Chase the 459 overdue and disputed invoices** by size, with or without a model. The score does not beat chance, so do not use it to prioritise.
6. **Choose carriers on a trade-off, not on price.** Use the Carrier Ranking page and set the cost-versus-reliability slider to the shipment's urgency.
7. **Start a root-cause review of weather, customs and packaging claims,** the three biggest claim costs.
8. **Standardise categories before modelling.** Merge `AIR`, `Air` and `air`, and country spellings. It shrinks the feature set and should help every model.
9. **Treat anomaly flags as review lists.** Track how many flagged cases turn out to be real, then decide on a threshold.
10. **Deploy carefully.** Use the same scikit-learn version that trained the files, keep a person in the loop and retrain as new data arrives.

---

## 23. Impact

These are the observed baselines in the data today, not projected savings. They show how big each problem is and what an improvement would be measured against.

| Area | Measure | Observed |
|---|---|---|
| Operations | Shipments | 2,500 |
| Delivery | Known late shipments | 280 (12.5% of known outcomes) |
| Freight | Total freight cost | $1,741,619 |
| Freight | Billed amount | $1,958,152 |
| Freight | Margin proxy (billed minus freight) | $216,534 |
| Invoice coverage | Shipments without a matched invoice | 243, with $156,533 of freight cost |
| Cash collection | Overdue or disputed invoices | 459, worth $404,348 |
| Claims | Shipments with a claim | 230, worth $185,458 |

The margin proxy is about 11% of billed revenue, so small changes in freight cost matter. For example, a model-guided 1% change in freight spend is about $17,000 here. That is an illustration of scale, not a promise.

### Charts

<p align="center">
  <img src="images/impact_baseline.png" alt="Observed impact baseline" width="100%">
</p>
<p align="center"><em>Observed impact baseline</em></p>


---

## 24. Conclusion

I built a full prediction system for a logistics business: five tables cleaned and joined, five modelling tables, four supervised models plus three anomaly detectors compared on a chronological split, everything saved and served in a seven-page Streamlit app.

Two results are usable. Late delivery can be flagged ahead of time (ROC AUC 0.886) and freight cost can be predicted to within about $146 per shipment (R² 0.83). Two are not: payment risk and claim risk sit at chance level, because the outcomes in this dataset carry little signal, and the app says so on those pages. The anomaly detectors and carrier ranking are decision aids for review rather than scored predictions.

**Limits to keep in mind**
- The data is fictional, so the findings demonstrate the method, not the performance of a real carrier.
- Category spellings and weight outliers were not standardised before modelling.
- The margin figure is a proxy: billed amount minus freight cost only.
- Each result comes from a single held-out period.

**Next steps:** standardise the categories and cap weight outliers, retrain, add cost-based thresholds for late delivery, collect richer data for payment and claims, and track how many anomaly flags prove real.

---
