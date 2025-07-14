import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page configuration
st.set_page_config(page_title="AfriMarket Seller Risk Dashboard", layout="wide")

# Title and author info
st.title(" Jumia Jitters: Seller Risk & Logistics Analytics")
st.markdown("Prepared by: **Amarachi Florence** | Submitted: **14 July 2025**")

# Load data
df = pd.read_csv("cleaned_jumia_data.csv")
sellers = pd.read_csv("sellers_to_investigate.csv")

# Preview cleaned dataset
st.subheader(" Cleaned Dataset Preview")
st.dataframe(df.head())

# Seller Risk Table
st.subheader(" Top Sellers to Investigate")
st.dataframe(sellers)

# ----------------------------------------
#  VISUALIZATION SECTION
# ----------------------------------------
st.markdown("---")
st.header(" Key Visual Insights")

# 1. Seller Risk (Interactive Plotly)
st.subheader(" Top 5 Sellers to Suspend")
if not sellers.empty:
    fig = px.bar(
        sellers.head(5),
        x='Average Return Risk',
        y='Seller ID',
        orientation='h',
        color='Average Return Risk',
        color_continuous_scale='Reds',
        height=400
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:** These sellers show the highest risk based on complaints, return rates, and delays.  
    **Recommendation:** Immediate suspension or audit required.
    """)

# 2. Seller Risk Framework
st.subheader(" Seller Risk Framework")
if os.path.exists("seller_risk_framework.png"):
    st.image("seller_risk_framework.png", use_container_width=True)
    st.markdown("""
    **Insight:** Return rate and complaint rate contribute most to the risk score.  
    **Recommendation:** Consider expanding this framework with fraud signals like review duplication.
    """)
else:
    st.warning("Risk Framework image not found.")

# 3. Complaint Heatmap
st.subheader(" Complaint Heatmap by Region & Product")
if os.path.exists("seller_product_risk_dashboard.png"):
    st.image("seller_product_risk_dashboard.png", use_container_width=True)
    st.markdown("""
    **Insight:** Complaint-heavy categories vary by region (e.g., Fashion in South West).  
    **Recommendation:** Launch targeted seller training in those areas.
    """)
else:
    st.warning("Complaint heatmap image not found.")

# 4. Fake Review Detection
st.subheader(" Suspicious Sellers (Fake Reviews)")
if os.path.exists("suspicious_sellers_fake_reviews.png"):
    st.image("suspicious_sellers_fake_reviews.png", use_container_width=True)
    st.markdown("""
    **Insight:** Some sellers submit unusually short, high-rated reviews.  
    **Recommendation:** Flag such sellers automatically for fraud review.
    """)
else:
    st.warning("Suspicious sellers chart not found.")

# ----------------------------------------
#  Trust Policy & Strategic Actions
# ----------------------------------------
st.markdown("---")
st.header(" Customer Trust Policy")
st.markdown("""
-  **Transparent Seller Profiles:** Show ratings, reviews, and return history.  
-  **100% Return Guarantee:** For defective or delayed products.  
-  **24/7 Multilingual Support:** For escalations and customer queries.
""")

st.header(" Strategic Recommendations")
st.markdown("""
**Suspend Top 5 Risky Sellers**  
Poor ratings, high returns, and complaints — suspend and audit immediately.

**Blacklist Problem Categories**  
Health, Electronics, Toys, Groceries, and Fashion — pending supplier review.

**Improve Delivery**  
- Reassign underperforming warehouses  
- Enforce dispatch deadlines  
- Use predictive traffic routing  

**Detect Review Fraud**  
- Flag dummy 5-star reviews under 10–15 characters  
- Penalize duplicated reviews in seller score
""")

# Footer
st.markdown("---")
st.info("Project for Dataverse Africa July Challenge | Powered by Streamlit + Plotly")
