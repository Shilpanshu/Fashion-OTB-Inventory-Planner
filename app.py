import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Fashion OTB & Inventory Planner")

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.title("Fashion OTB & Inventory Planner")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """
    Loads data from an uploaded CSV or defaults to the local Excel file.
    """
    required_columns = [
        'Category', 'Sub_Category', 'MRP', 'Cost_Price', 
        'Stock_On_Hand', 'Season', 'Price_Status', 'Weekly_Sales'
    ]
    
    df = None
    
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Fallback to local default
            default_path = "data/Retail_Trinity_Data.xlsx"
            # We need to specify the sheet name as per exploration
            df = pd.read_excel(default_path, sheet_name='Sales_Data')
            
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a CSV.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    if df is not None:
        # Normalize column names just in case (optional, but good practice)
        # df.columns = [c.strip() for c in df.columns]
        
        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"The uploaded data is missing required columns: {', '.join(missing_cols)}")
            return None
            
        # Ensure numeric types
        numeric_cols = ['MRP', 'Cost_Price', 'Stock_On_Hand', 'Weekly_Sales']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# Sidebar Data Uploader
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=['csv'])

# Load the data
df = load_data(uploaded_file)

if df is None:
    st.info("Please upload a file or ensure the default data exists to proceed.")
    st.stop()

# Global Data Handling
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Filter out any rows where Date failed parsing (optional, but good for safety)
    df = df[df['Date'].notna()]

# -----------------------------------------------------------------------------
# 3. Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Category Filter
categories = df['Category'].unique().tolist()
selected_categories = st.sidebar.multiselect("Select Category", categories, default=categories)

# Season Filter
seasons = df['Season'].unique().tolist()
selected_season = st.sidebar.selectbox("Select Season", ["All"] + seasons)

# Store Filter (if exists)
if 'Store' in df.columns:
    stores = df['Store'].unique().tolist()
    selected_store = st.sidebar.selectbox("Select Store", ["All"] + [str(s) for s in stores])
else:
    selected_store = "All" # Fallback if no store column

# Apply Filters
df_filtered = df[df['Category'].isin(selected_categories)]

if selected_season != "All":
    df_filtered = df_filtered[df_filtered['Season'] == selected_season]

if selected_store != "All":
    # Ensure type matching for filtering
    # Assuming Store might be int or str in dataframe
    if 'Store' in df.columns:
        df_filtered = df_filtered[df_filtered['Store'].astype(str) == str(selected_store)]

# -----------------------------------------------------------------------------
# 4. Trinity KPI Deck
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main Tabs: Dashboard vs Data
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Dashboard", "Data View"])

with tab1:
    # -----------------------------------------------------------------------------
    # 4. Trinity KPI Deck
    # -----------------------------------------------------------------------------
    st.header("The Trinity KPI Deck")

    # Calculations
    # 1. Total Revenue
    total_revenue = df_filtered['Weekly_Sales'].sum()

    # 2. Gross Margin %
    # First, estimate Units Sold safely
    # Fix: Ensure MRP is never zero to avoid division errors
    # Unit Calculation: Weekly_Sales / MRP
    # NOTE: df_filtered is already copied/modified in global scope above? 
    # Wait, in the previous code, I modified df_filtered IN PLACE in section 4.
    # If I wrap it in a tab, it still runs linearly.
    
    # Let's re-run the logic cleanly.
    # We should ensure df_filtered has the unit columns. 
    # Actually, I added the unit conversion logic inside Section 4 in previous steps.
    # It's better to move that calculation UP to the Filter section so the Data View also sees it!
    
    # Moving Unit Calculations to Global/Filter scope (Lines 144-154 of original)
    # But for now, keeping it here to minimize refactoring risk, just indented.
    
    df_dashboard = df_filtered.copy()
    df_dashboard['Units_Sold'] = df_dashboard.apply(lambda x: x['Weekly_Sales'] / x['MRP'] if x['MRP'] > 0 else 0, axis=1)
    df_dashboard['Total_Cost'] = df_dashboard['Units_Sold'] * df_dashboard['Cost_Price']
    
    total_cost = df_dashboard['Total_Cost'].sum()
    gross_margin_pct = ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0

    # 3. Weeks of Cover (WOC)
    df_dashboard['Stock_Units'] = df_dashboard.apply(lambda x: x['Stock_On_Hand'] / x['MRP'] if x['MRP'] > 0 else 0, axis=1)

    total_stock_units = df_dashboard['Stock_Units'].sum()
    total_units_sold = df_dashboard['Units_Sold'].sum()

    if 'Date' in df_dashboard.columns:
        num_weeks_kpi = df_dashboard['Date'].nunique()
    else:
        num_weeks_kpi = 1
    if num_weeks_kpi == 0: num_weeks_kpi = 1

    avg_weekly_units = total_units_sold / num_weeks_kpi
    woc = (total_stock_units / avg_weekly_units) if avg_weekly_units > 0 else 0

    # 4. Sell-Through %
    if (total_units_sold + total_stock_units) > 0:
        sell_through_pct = (total_units_sold / (total_units_sold + total_stock_units)) * 100
    else:
        sell_through_pct = 0

    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

    with col2:
        st.metric("Gross Margin %", f"{gross_margin_pct:.1f}%")

    with col3:
        woc_val = f"{woc:.1f}"
        color_style = ""
        if woc < 4 or woc > 12:
            color_style = "color: red;"
        
        st.markdown(f"""
            <div style="text-align: left;" class="metric-card">
                <p style="font-size: 14px; margin-bottom: 0px; color: #555;">Weeks of Cover</p>
                <p style="font-size: 32px; font-weight: bold; {color_style}">{woc_val}</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.metric("Sell-Through %", f"{sell_through_pct:.1f}%")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 5. Advanced Visuals
    # -----------------------------------------------------------------------------
    st.header("Deep Dive Analysis")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Category Mix")
        cat_sales = df_dashboard.groupby('Category')['Weekly_Sales'].sum().reset_index()
        fig1 = px.pie(cat_sales, values='Weekly_Sales', names='Category', 
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Profit vs. Speed (Sub-Category)")
        sub_cat_group = df_dashboard.groupby('Sub_Category').agg({
            'Weekly_Sales': 'sum',
            'Stock_On_Hand': 'sum',
            'Stock_Units': 'sum',
            'Units_Sold': 'sum',
            'Total_Cost': 'sum',
            'Category': 'first'
        }).reset_index()
        
        sub_cat_group['Gross_Margin_Pct'] = (
            (sub_cat_group['Weekly_Sales'] - sub_cat_group['Total_Cost']) / sub_cat_group['Weekly_Sales'] * 100
        ).fillna(0)
        
        sub_cat_group['Avg_Weekly_Units'] = sub_cat_group['Units_Sold'] / num_weeks_kpi
        
        sub_cat_group['WOC'] = (
            sub_cat_group['Stock_Units'] / sub_cat_group['Avg_Weekly_Units']
        ).fillna(0)
        
        fig2 = px.scatter(
            sub_cat_group, 
            x='WOC', 
            y='Gross_Margin_Pct', 
            color='Category', 
            size='Weekly_Sales',
            hover_name='Sub_Category',
            labels={'WOC': 'Weeks of Cover', 'Gross_Margin_Pct': 'Gross Margin %'},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig2.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Target Margin")
        fig2.add_vline(x=12, line_dash="dot", line_color="red", annotation_text="Stock Risk")
        
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Seasonality
    st.subheader("Sales Seasonality")
    if 'Date' in df_dashboard.columns:
        date_sales = df_dashboard.groupby(['Date', 'Season'])['Weekly_Sales'].sum().reset_index()
        fig3 = px.line(date_sales, x='Date', y='Weekly_Sales', color='Season', 
                       color_discrete_map={'SS': '#FFC107', 'AW': '#3F51B5'})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Date column missing, cannot display Seasonality over time.")

    # -----------------------------------------------------------------------------
    # 6. Open-to-Buy Calculator
    # -----------------------------------------------------------------------------
    st.header("Open-to-Buy (OTB) Calculator")
    st.markdown("Use this tool to plan your inventory needs for the next Season.")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        planned_growth = st.slider("Planned Growth %", min_value=0, max_value=30, value=10, step=1)

    with col_input2:
        target_woc = st.slider("Target Weeks of Cover", min_value=4, max_value=12, value=8, step=1)

    if 'Date' in df_dashboard.columns:
        num_weeks = df_dashboard['Date'].nunique()
    else:
        num_weeks = 1
    if num_weeks == 0: num_weeks = 1

    df_otb = df_dashboard.copy()
    df_otb['Avg_Weekly_Units'] = df_otb['Units_Sold'] / num_weeks
    df_otb['Projected_Weekly_Units'] = df_otb['Avg_Weekly_Units'] * (1 + planned_growth/100)
    df_otb['Required_Stock'] = df_otb['Projected_Weekly_Units'] * target_woc
    
    # Net Buy based on Units (using Stock_Units calculated earlier in this tab)
    df_otb['Net_Buy_Qty'] = df_otb['Required_Stock'] - df_otb['Stock_Units']
    df_otb['Recommended_Buy_Qty'] = df_otb['Net_Buy_Qty'].apply(lambda x: max(x, 0))
    df_otb['Estimated_Budget'] = df_otb['Recommended_Buy_Qty'] * df_otb['Cost_Price']

    total_buy_qty = df_otb['Recommended_Buy_Qty'].sum()
    total_budget = df_otb['Estimated_Budget'].sum()

    st.subheader(f"Planning Results (Based on {num_weeks} weeks of history)")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Recommended Buy Quantity", f"{int(total_buy_qty):,}")
    with res_col2:
        st.metric("Estimated Budget Required", f"${total_budget:,.0f}")

    with st.expander("View OTB Breakdown by Sub-Category"):
        breakdown = df_otb.groupby('Sub_Category')[['Recommended_Buy_Qty', 'Estimated_Budget']].sum().sort_values('Estimated_Budget', ascending=False)
        st.dataframe(breakdown.style.format({'Recommended_Buy_Qty': '{:,.0f}', 'Estimated_Budget': '${:,.2f}'}))

# -----------------------------------------------------------------------------
# TAB 2: Data View
# -----------------------------------------------------------------------------
with tab2:
    st.header("Filtered Sales Data")
    st.markdown("Detailed view of the data based on current filters.")
    st.dataframe(df_filtered, use_container_width=True)




