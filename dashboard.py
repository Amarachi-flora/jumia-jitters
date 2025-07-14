import streamlit as st
import pandas as pd
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="AfriMarket Seller Risk Dashboard", layout="wide")

# Title
st.title(" Jumia Jitters: Seller Risk & Logistics Analytics")
st.markdown("Prepared by: **Amarachi Florence** | Submitted: **14 July 2025**")

# Load datasets
df = pd.read_csv("cleaned_jumia_data.csv")
sellers = pd.read_csv("sellers_to_investigate.csv")

# ===============================
# Section: Data Preview
# ===============================
st.subheader(" Cleaned Dataset Preview")
st.dataframe(df.head())

# ===============================
# Section: Seller Risk Summary
# ===============================
st.subheader(" Top Sellers to Investigate")
st.dataframe(sellers)

# ===============================
# Section: Key Visual Insights
# ===============================
st.markdown("---")
st.header("Key Visual Insights")

# Visual 1: Seller/Product Risk Dashboard
if os.path.exists("seller_product_risk_dashboard.png"):
    st.image("seller_product_risk_dashboard.png", caption="Seller & Product Risk Dashboard", use_container_width=True)
    st.markdown("""
     Objective: Combine seller risk and product complaint data in one view.  
     Insight: Key sellers and categories emerge as clear outliers in risk.  
     Recommendation: Use dashboard regularly to support trust & safety team decisions.
    """)
else:
    st.warning("seller_product_risk_dashboard.png not found.")

# Visual 2: Seller Risk Framework
if os.path.exists("seller_risk_framework.png"):
    st.image("seller_risk_framework.png", caption="Seller Risk Framework", use_container_width=True)
    st.markdown("""
     Objective: Communicate the weighted components of the seller risk score.  
     Insight: Return rate and complaint rate carry the most weight (40% each).  
     Recommendation: Consider adjusting weights if new fraud indicators emerge (e.g., fake reviews).
    """)
else:
    st.warning("seller_risk_framework.png not found.")

# Visual 3: Suspicious Fake Reviews
if os.path.exists("suspicious_sellers_fake_reviews.png"):
    st.image("suspicious_sellers_fake_reviews.png", caption="Fake Review Detection", use_container_width=True)
    st.markdown("""
     Objective: Flag sellers using suspiciously short and high-rated reviews.  
     Insight: Sellers with unusually short, duplicated 5-star reviews likely engage in fraud.  
     Recommendation: Integrate review length and duplication into future risk scoring models.
    """)
else:
    st.warning("suspicious_sellers_fake_reviews.png not found.")

# ===============================
# Section: Customer Trust Policy
# ===============================
st.markdown("---")
st.header(" Customer Trust Policy")
st.markdown("""
-  **Transparent Seller Profiles:** Show ratings, reviews, and return history.
-  **100% Return Guarantee:** For defective or delayed products.
-  **24/7 Multilingual Support:** To resolve customer concerns fast.
""")

# ===============================
# Section: Recommendations
# ===============================
st.markdown("---")
st.header(" Strategic Recommendations")
st.markdown("""
**Suspend Top 5 Risky Sellers:** Based on poor ratings, high returns, and complaints.

**Blacklist Product Categories:** Health, Electronics, Toys, Groceries, Fashion — pending quality audit.

**Improve Delivery Efficiency:**
- Reassign underperforming warehouses
- Enforce seller dispatch timelines
- Use predictive routing based on traffic

**Detect Review Fraud:** Integrate short/dummy 5-star reviews into the risk score.
""")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.info("Project for Dataverse Africa July Challenge | Powered by Streamlit")
