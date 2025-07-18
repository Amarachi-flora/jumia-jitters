# Dataverse Africa July Challenge 2025  
**Theme:** E-Commerce Logistics, Seller Risk Profiling, and Review Fraud Detection  
**Project Title:** Jumia Jitters  

---

##  Introduction

This project was developed as part of the Dataverse Africa July Challenge. It tackles real-world issues faced by **AfriMarket**, a fictional e-commerce platform operating in Nigeria and Ghana. Over a 3-month transaction period, the platform has faced growing customer dissatisfaction—ranging from delivery delays and high return rates to fake reviews and unreliable sellers.

The goal here was to dive deep into the data and uncover exactly where things are going wrong, then use analytics and machine learning to detect fraud, assess risk, and suggest actionable strategies that could help AfriMarket rebuild customer trust and optimize operations.

---

##  Dataset Overview


The dataset includes a detailed snapshot of marketplace activities, with these major components:

- **Order metadata:** Order ID, order date, dispatch date, delivery date  
- **Seller & product info:** Seller ID, product category, price, quantity  
- **Logistics info:** Warehouse zone, delivery method, customer region  
- **Customer feedback:** Rating, review text, sentiment score, return flag, complaint code  

This rich structure gave a solid base for data-driven storytelling and intervention planning.

##  Task 1: Data Cleaning & Feature Engineering

###  Introduction

Before any meaningful analysis could happen, I cleaned the data and generated important features that would power later tasks. This phase helped standardize formats and fill in gaps.

###  Insights

From the raw dataset, it became clear that sellers’ behaviors weren’t just about sales, they were tied to how often customers returned items, left complaints, or dropped low ratings. Delivery delays were also not uniformly calculated.

So, I created a **Seller Risk Score**,a weighted combination of:

- Return Rate  
- Complaint Rate  
- Average Delivery Delay  
- Rating Penalties  
- Suspicious Review Flags (like short reviews)

This score provided a single number to measure how risky a seller might be.

###  Recommendations

From this phase, **Jumia Jitters should** start tracking and scoring every seller using a similar risk formula. It makes it easier to spot who’s delivering value—and who’s hurting customer trust.

---

##  Task 2: Pattern Surveillance

###  Introduction

This stage used visualizations to uncover hidden patterns in complaints, delivery issues, and fake reviews. I also ran hypothesis tests to validate whether certain logistics choices actually affected customer ratings.

###  Insights

- **Standard and Express** deliveries dominate, while other methods are rarely used  
- **Fashion and Electronics** lead in both order volume and complaint rates  
- **Lagos and Greater Accra** are hot zones for both demand and delays  
- Some sellers repeatedly post suspiciously short 5-star reviews, likely faked  

A hypothesis test (ANOVA) showed **no significant link between delivery method and customer rating**, so factors like delays or seller behavior carry more weight in customer perception.

###  Recommendations

- Jumia Jitters should **focus delivery optimization** efforts on Standard and Express only  
- Sellers in complaint-heavy categories like Fashion should be audited  
- Suspicious review patterns should be flagged early, not just after returns spike  
- Sellers in Lagos may need performance reviews, especially those with consistent delays  

---

##  Task 3: Prediction & Risk Modeling

###  Introduction

In this phase, I trained a machine learning model to predict whether an item would be returned—based on seller behavior, delivery time, customer region, and sentiment analysis.

###  Modeling Details

- **Random Forest Classifier**  
- Handled class imbalance with **SMOTE**  
- **AUC Score:** 0.8588  
- Tracked precision, recall, and confusion matrix  

###  Insights

Some sellers, like `S014`, `S041`, and `S018`, consistently showed extremely high return probabilities (≥ 0.86), even when their reviews seemed perfect. This mismatch hinted at **fake reviews masking poor performance**.

###  Recommendations

- These risky sellers should be **immediately suspended or reviewed**  
- **Review text analytics** (like length and duplication) should become standard parts of fraud detection  
- Machine learning should power **return prediction in real time**, especially for high-volume sellers

---

##  Task 4: Strategy, Streamlit Dashboard & Resolution

###  Introduction

To bring the insights to life, I built a fully interactive dashboard with Streamlit + Plotly. This dashboard empowers AfriMarket’s trust and safety team to filter data by region and product category, track risk scores, and spot fraudulent activity visually.

### [👉 Click here to view the live Streamlit app](https://jumia-jitters.streamlit.app/)

---

###  Dashboard Features

- **Filter by Region & Product Category**  
- **Top 5 Sellers to Suspend (Interactive Bar Chart)**  
- **Seller Risk Framework (Weight Breakdown)**  
- **Complaint Heatmap (By Region & Category)**  
- **Fake Review Detection (Short 5-Star Reviews)**  

---

###  Insights

- A few sellers dominate the risk charts consistently, despite having high ratings  
- Complaint patterns are regional and product-specific  
- Short, duplicated 5-star reviews are a red flag almost every time  

---

###  Strategic Recommendations

####  1. Suspend Top 5 Risky Sellers  
Based on complaint rate, delay history, return rate, and fake review patterns.

####  2. Blacklist High-Complaint Categories  
Health, Electronics, Fashion, Toys, and Groceries, until quality control improves.

####  3. Improve Delivery Operations  
Reassign underperforming warehouses and use predictive routing to minimize delays.

####  4. Build Customer Trust  
Make seller profiles transparent, offer 100% return guarantees, and run 24/7 support.

####  5. Detect Review Fraud Proactively  
Sellers relying on short or duplicated 5-star reviews should be flagged immediately.

---

##  Business Curveballs Handled

### 1. Lagos Warehouse Delay  
Due to inconsistent delay reporting from Lagos, that region’s data was adjusted in analysis.

### 2. Fake Review Signals  
Several sellers were flagged for using the same short review text across many 5-star ratings.

---

##  Visual Outputs (Backups in `.png`)

- Seller/Product Risk Dashboard  
- Seller Risk Framework  
- Top Risk Sellers  
- Complaint Heatmap  
- Delivery Delay by Region  
- Suspicious Reviews Bar Chart  



---

##  Final Deliverables

- `cleaned_jumia_data.csv`  
- `seller_risk_summary.csv`  
- `sellers_to_investigate.csv`  
- `dashboard.py`  
- `requirements.txt`  
- `logs/output_log.txt` 



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



The app includes region/category filters and interactive charts for easy navigation and decision-making.

  Future Work
Refine fraud detection with transformer-based NLP

Build live monitoring dashboards via Streamlit Cloud or Power BI

Automate monthly seller audits using risk score pipelines

🙏 Thank You
Prepared by: Amarachi Florence
For the Dataverse Africa July 2025 Challenge

