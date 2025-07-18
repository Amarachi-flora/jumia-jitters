import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(page_title="AfriMarket Seller Risk Dashboard", layout="wide")
st.title(" Jumia Jitters: Seller Risk & Logistics Analytics")
st.markdown("Prepared by: **Amarachi Florence** | Submitted: **14 July 2025**")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("cleaned_jumia_data.csv")
sellers = pd.read_csv("sellers_to_investigate.csv")

# -------------------------
# Filters
# -------------------------
st.sidebar.header(" Filter Dashboard")

# Unique filter options
regions = df['customer_region'].unique().tolist()
categories = df['product_category'].unique().tolist()

# Sidebar selections
selected_region = st.sidebar.selectbox("Select Customer Region", ["All"] + regions)
selected_category = st.sidebar.selectbox("Select Product Category", ["All"] + categories)

# Apply filters
filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["customer_region"] == selected_region]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["product_category"] == selected_category]

# -------------------------
# Dataset Preview
# -------------------------
st.subheader(" Filtered Dataset Preview")
st.dataframe(filtered_df.head())

# -------------------------
# Seller Risk Table
# -------------------------
st.subheader(" Top Sellers to Investigate")
st.dataframe(sellers)

# -------------------------
# VISUALIZATION SECTION
# -------------------------
st.markdown("---")
st.header(" Key Visual Insights")

# 1. Top Sellers to Suspend (Interactive)
st.subheader(" Top 5 Sellers to Suspend")
if not sellers.empty:
    fig1 = px.bar(
        sellers.head(5),
        x='Average Return Risk',
        y='Seller ID',
        orientation='h',
        color='Average Return Risk',
        color_continuous_scale='Reds',
        height=400
    )
    fig1.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    **Insight:** These sellers show the highest risk based on complaints, return rates, and delays.  
    **Recommendation:** Immediate suspension or audit required.
    """)

# 2. Seller Risk Framework (Interactive)
st.subheader(" Seller Risk Framework (Interactive)")
framework_weights = pd.DataFrame({
    "Component": ['Return Rate (40%)', 'Complaint Rate (40%)', 'Rating Penalty (15%)', 'Delay Penalty (15%)'],
    "Weight": [0.4, 0.4, 0.15, 0.15]
})
fig2 = px.bar(
    framework_weights,
    x="Weight",
    y="Component",
    orientation='h',
    color="Weight",
    color_continuous_scale='blues',
    title="Seller Risk Score Weight Breakdown"
)
fig2.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
**Insight:** Return rate and complaint rate are most heavily weighted.  
**Recommendation:** Consider adjusting weights if new fraud signals emerge.
""")

# 3. Complaint Heatmap by Region & Category
st.subheader(" Complaint Heatmap by Region & Product Category")
heatmap_data = filtered_df.pivot_table(index='product_category', columns='customer_region',
                                       values='complaint_code', aggfunc='count', fill_value=0)
heatmap_df = heatmap_data.reset_index().melt(id_vars='product_category', var_name='Region', value_name='Complaint Count')
fig3 = px.density_heatmap(
    heatmap_df,
    x="Region",
    y="product_category",
    z="Complaint Count",
    color_continuous_scale="YlOrRd",
    title="Complaints by Region and Category"
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""
**Insight:** Complaint clusters show strong product-region issues.  
**Recommendation:** Launch seller audits or quality checks for those zones.
""")

# 4. Suspicious Sellers (Fake Reviews)
st.subheader(" Suspicious Sellers (Fake Reviews)")
suspicious = df[df["review_length"] < 15].groupby("seller_id").size().reset_index(name="Short Review Count")
top_suspicious = suspicious.sort_values("Short Review Count", ascending=False).head(5)
fig4 = px.bar(
    top_suspicious,
    x="Short Review Count",
    y="seller_id",
    orientation="h",
    color="Short Review Count",
    color_continuous_scale="reds",
    title="Top 5 Sellers with Suspiciously Short Reviews"
)
fig4.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
**Insight:** These sellers use many very short reviews, a likely fraud indicator.  
**Recommendation:** Integrate review length into your fraud risk scoring.
""")

# -------------------------
# Trust Policy & Strategy
# -------------------------
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
Poor ratings, high returns, and complaints, suspend and audit immediately.

**Blacklist Problem Categories**  
Health, Electronics, Toys, Groceries, and Fashion, pending supplier review.

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
