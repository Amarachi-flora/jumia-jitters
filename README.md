
#  Dataverse Africa July Challenge 2025  
**Theme:** E-Commerce Logistics, Seller Risk Profiling, and Review Fraud Detection  
**Project Title:** Jumia Jitters  

---

##  Introduction

This project addresses key operational and trust issues faced by **AfriMarket**, a fictional e-commerce platform operating in Nigeria and Ghana. The analysis focuses on identifying seller fraud, improving delivery performance, detecting suspicious reviews, and recommending strategies to restore customer confidence.

---

##  Dataset Overview

The dataset includes a 3-month log of AfriMarket transactions with the following features:

- **Order metadata:** order ID, dates (order, dispatch, delivery)  
- **Seller & product info:** seller ID, product category, price, quantity  
- **Logistics info:** warehouse zone, delivery method, customer region  
- **Feedback & quality:** customer rating, review text, sentiment score, return flag, complaint code  

---

##  Task 1: Data Cleaning & Feature Engineering

**Goals:**
- Standardize entries and fill missing values
- Engineer features such as:
  - Delivery delay
  - Seller performance metrics
  - Suspicious review flags
  - Seller Risk Score (weighted formula using return rate, complaint rate, rating penalty, delay)

**Outcome:**  
Cleaned dataset with new performance metrics for each seller, stored in `cleaned_jumia_data.csv`.

---

##  Task 2: Pattern Surveillance

**Goals:**
- Explore trends using visual analytics
- Identify high-risk categories, regions, and seller behavior patterns

**Key Insights & Recommendations:**
- Standard and Express delivery methods dominate → optimize for them  
- Electronics and Fashion categories drive both sales and complaints → monitor closely  
- Greater Accra and Lagos are top buyer regions → strengthen logistics in these zones  
- Some sellers rely on short/fake reviews → include in fraud scoring  

**Statistical Test:**  
ANOVA confirmed that delivery method does not significantly affect customer ratings.

---

##  Task 3: Prediction & Risk Modeling

**Goals:**
- Predict likelihood of returns using seller/product features  
- Flag risky sellers using classification model and sentiment-enhanced analysis  

**Model Highlights:**
- **Random Forest Classifier**
- **SMOTE** used for class imbalance
- **AUC Score:** 0.8588
- **Key Risk Drivers:** Complaint rate, return flag, review sentiment, delivery delay

**High-Risk Sellers Flagged:**

S014 – 0.90
S041 – 0.89
S018 – 0.86



---

##  Task 4: Strategy & Resolution

###  Streamlit Interactive Dashboard

A responsive Streamlit dashboard was developed to visualize seller performance and filter data by:

-  **Customer Region**
-  **Product Category**

###  Interactive Visuals Include:

- **Top 5 Sellers to Suspend** (bar chart)  
- **Seller Risk Framework** (interactive weight breakdown)  
- **Complaint Heatmap** (by product category and region)  
- **Fake Review Detection** (short 5-star reviews)

###  Recommendations

#### 1. Suspend Top 5 Risky Sellers  
Based on the Seller Risk Score and performance metrics. Immediate suspension is advised to protect buyer trust and platform integrity.

#### 2. Blacklist Categories with High Complaints  
- Health  
- Electronics  
- Toys  
- Groceries  
- Fashion  
> These should be audited or temporarily delisted until quality improves.

#### 3. Reduce Delivery Delays  
- Reassign underperforming warehouses in delay-heavy regions  
- Enforce strict seller dispatch timelines  
- Implement predictive traffic-based delivery routing  

#### 4. Customer Trust Policy  
- Transparent seller profiles with performance data  
- 100% return guarantee on defective or late items  
- 24/7 multilingual customer support for escalations  

#### 5. Detect Review Fraud  
- Automatically flag sellers with repeated short or suspiciously positive 5-star reviews  
- Include review length and duplication in risk scoring model  

---

##  Business Curveballs Handled

### 1. Lagos Warehouse Delay Issue  
Delivery delay data from Lagos was excluded in regional analysis due to overcapacity and unreliability.

### 2. Fake Reviews Detected  
Sellers using suspiciously short or repetitive high-rated reviews were identified through review length analysis and duplication detection.

---

##  Visual Outputs (Saved as `.png` for backup)

- Seller/Product Risk Dashboard  
- Seller Risk Framework  
- Top Risk Sellers  
- Complaint Heatmap  
- Delivery Delay by Region  
- Suspicious Reviews Bar Chart  

(But all visuals are now fully **interactive in Streamlit**)

---

##  Final Deliverables

- `cleaned_jumia_data.csv`  
- `seller_risk_summary.csv`  
- `sellers_to_investigate.csv`  
- `dashboard.py` (Streamlit app)  
- `requirements.txt` (dependencies)  
- `logs/output_log.txt`  
- Interactive charts via `plotly` in Streamlit  

---

##  How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
streamlit run dashboard.py

Filters and charts will auto-update based on region and product category selections.

##  Live Demo

You can explore the dashboard live here:  
👉 [Click here to view the live Streamlit app](https://jumia-jitters.streamlit.app/)



 Future Work
Refine fraud detection using deep learning techniques (e.g., transformer-based NLP)

Expand real-time monitoring using Streamlit Cloud or Power BI dashboards

Develop a periodic audit system for review and seller activity validation



# THANK YOU. DONE BY AMARACHI FLORENCE
