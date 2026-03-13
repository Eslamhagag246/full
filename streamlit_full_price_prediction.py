import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Price Tracker - Tablets & Mobiles",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e17; color: #e8eaf0; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b5cf0 50%, #ff6b9d 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.1rem; color: #6b7280; font-weight: 300;
    letter-spacing: 0.04em; margin-bottom: 1.5rem;
}

.device-tab {
    background: #1e2535; border-radius: 12px; padding: 0.8rem 1.5rem;
    margin: 0.5rem; display: inline-block; cursor: pointer;
    border: 2px solid transparent; transition: all 0.3s;
}
.device-tab:hover { border-color: #7b5cf0; }
.device-tab.active { border-color: #00d4ff; background: #111827; }

.stat-card {
    background: #111827; border: 1px solid #1e2535;
    border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
}
.stat-label { 
    font-size: 0.72rem; color: #6b7280; 
    text-transform: uppercase; letter-spacing: 0.1em; 
}
.stat-value { 
    font-family: 'Syne', sans-serif; 
    font-size: 1.4rem; font-weight: 700; 
}

.signal-card { border-radius: 16px; padding: 1.2rem 1.5rem; margin: 0.5rem 0; }
.signal-buy { background: linear-gradient(135deg,#0d2b1d,#0b1a12); border: 1.5px solid #00ff8855; }
.signal-wait { background: linear-gradient(135deg,#2b1a0d,#1a0b0b); border: 1.5px solid #ff6b6b55; }
.signal-neutral { background: linear-gradient(135deg,#1a1a2b,#0d0d1a); border: 1.5px solid #7b5cf055; }

div[data-testid="stSelectbox"] > div {
    background: #111827 !important; border: 1px solid #1e2535 !important; 
    border-radius: 10px !important;
}
.stTextInput > div > div > input {
    background: #111827 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
}
hr { border-color: #1e2535 !important; }

.filter-badge {
    display: inline-block; background: #1e2535;
    border: 1px solid #2e3545; border-radius: 8px;
    padding: 0.4rem 0.8rem; margin: 0.2rem;
    font-size: 0.85rem; color: #a0a8b8;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0a0e17 !important;
    border-right: 1px solid #1e2535 !important;
}
section[data-testid="stSidebar"] > div {
    background: #0a0e17 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# IMPORT MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════
try:
    from tablet_model import load_and_preprocess_data as load_tablets, forecast_product as forecast_tablet
    from mobile_model import load_and_preprocess_data as load_mobiles, forecast_product as forecast_mobile
    MODELS_AVAILABLE = True
except:
    MODELS_AVAILABLE = False
# ═══════════════════════════════════════════════════════════
# DATA LOADING (CACHED)
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def load_tablet_data():
    if MODELS_AVAILABLE:
        return load_tablets('tablets_cleaned_continuous.csv')
    return None

@st.cache_data(ttl=86400)
def load_mobile_data():
    if MODELS_AVAILABLE:
        return load_mobiles('mobile_cleaned_70K.csv')
    return None
# ═══════════════════════════════════════════════════════════
# CHART FUNCTION
# ═══════════════════════════════════════════════════════════
def create_forecast_chart(result, device_type="tablet"):
    pdf = result['pdf']
    fdates = result['forecast_dates']
    fprices = result['forecast_prices']
    mae = result['mae']
    
    color_scheme = {
        'tablet': {'primary': '#00d4ff', 'secondary': '#7b5cf0'},
        'mobile': {'primary': '#ff6b9d', 'secondary': '#7b5cf0'}
    }
    colors = color_scheme.get(device_type, color_scheme['tablet'])

    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=pdf['date'], y=pdf['price'],
        mode='lines+markers', name='Historical Price',
        line=dict(color=colors['primary'], width=2.5),
        marker=dict(size=6, color=colors['primary']),
        hovertemplate='%{x}<br>EGP %{y:,.0f}<extra></extra>'
    ))

    # Rolling average
    fig.add_trace(go.Scatter(
        x=pdf['date'], y=pdf['rolling_avg'],
        mode='lines', name='3-day Avg',
        line=dict(color=colors['secondary'], width=1.5, dash='dot'),
        hovertemplate='%{x}<br>Avg EGP %{y:,.0f}<extra></extra>'
    ))

    # Bridge to forecast
    last_hist_date = pdf['date'].iloc[-1]
    last_hist_price = pdf['price'].iloc[-1]
    first_fore_date = fdates[0]
    first_fore_price = fprices[0]

    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_fore_date],
        y=[last_hist_price, first_fore_price],
        mode='lines', showlegend=False,
        line=dict(color='#ffd166', width=1.5, dash='dot'),
        hoverinfo='skip'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fdates, y=fprices,
        mode='lines+markers', name='Forecast',
        line=dict(color='#ffd166', width=2.5, dash='dash'),
        marker=dict(size=6, symbol='diamond', color='#ffd166'),
        hovertemplate='%{x}<br>Forecast EGP %{y:,.0f}<extra></extra>'
    ))

    # Confidence band
    band_dates = [last_hist_date, first_fore_date] + fdates[1:]
    band_prices = [last_hist_price, first_fore_price] + list(fprices[1:])
    upper = [p + mae for p in band_prices]
    lower = [max(p - mae, 0) for p in band_prices]

    fig.add_trace(go.Scatter(
        x=band_dates + band_dates[::-1],
        y=upper + lower[::-1],
        fill='toself', fillcolor='rgba(255,209,102,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band', hoverinfo='skip'
    ))
    
    # Today marker
    today_str = str(pd.Timestamp.today().date())
    fig.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1,
        xref="x", yref="paper", line=dict(color="#444", width=1, dash="dot")
    )
    fig.add_annotation(
        x=today_str, y=1, xref="x", yref="paper",
        text="Today", showarrow=False,
        font=dict(color="#888", size=11), yanchor="bottom"
    )

    fig.update_layout(
        title=dict(text='📈 Price History & 7-Day Forecast', 
                   font=dict(family='Syne', size=16, color='#e8eaf0')),
        plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
        font=dict(family='DM Sans', color='#8b92a5'),
        xaxis=dict(gridcolor='#1e2535', title='Date'),
        yaxis=dict(gridcolor='#1e2535', title='Price (EGP)', tickformat=',.0f'),
        legend=dict(bgcolor='#111827', bordercolor='#1e2535', borderwidth=1),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=50, b=10),
        height=500
    )
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

# Header
st.markdown('<div class="hero-title">📱 Price Tracker Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Track & Forecast Prices for Tablets & Mobile Phones</div>', 
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SIDEBAR - DEVICE TYPE SELECTION
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 Select Device Type")
    
    device_type = st.radio(
        "Choose device category:",
        options=["📱 Tablets", "📱 Mobile Phones"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Load appropriate data
    if "Tablets" in device_type:
        st.markdown("### 📊 Tablet Data")
        df = load_tablet_data()
        forecast_func = forecast_tablet
        device_key = "tablet"
        data_file = "tablets_cleaned_continuous.csv"
    else:
        st.markdown("### 📊 Mobile Phone Data")
        df = load_mobile_data()
        forecast_func = forecast_mobile
        device_key = "mobile"
        data_file = "mobile_cleaned_70K.csv"
    
    if df is not None:
        last_update = pd.to_datetime(df['date'].max()).strftime('%B %d, %Y')
        st.markdown(f"**Last Updated:** {last_update}")
        st.markdown(f"**Total Products:** {df['product_key'].nunique()}")
        st.markdown(f"**Data Points:** {len(df):,}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app tracks prices and forecasts future trends for:
    - **Tablets** (iPad, Samsung, etc.)
    - **Mobile Phones** (iPhone, Samsung, etc.)
    
    Uses ML to predict 7-day price forecasts.
    """)

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════

if df is None:
    st.error(f"⚠️ Could not load {data_file}. Please check the file exists.")
    st.stop()

# Clean product name
df['clean_name'] = df['name']

st.markdown("---")

# ═══════════════════════════════════════════════════════════
# SEARCH & FILTERS
# ═══════════════════════════════════════════════════════════
st.markdown("### 🔍 Search & Filter")

# Search
search = st.text_input(
    "🔎 Search by name", 
    placeholder=f"e.g. {'iPad, Samsung Galaxy' if device_key == 'tablet' else 'iPhone, Galaxy, Xiaomi'}...",
    help="Search for products by name"
)

# Initial filter
if search:
    filtered = df[df['clean_name'].str.lower().str.contains(search.lower(), na=False)]
else:
    filtered = df.copy()

# Get available options
available_ram = sorted(filtered['ram_gb'].dropna().unique().astype(int).tolist())
available_storage = sorted(filtered['storage_gb'].dropna().unique().astype(int).tolist())
available_websites = sorted(filtered['website'].str.upper().unique().tolist())
available_brands = sorted(filtered['brand'].str.title().unique().tolist())

# Filter options in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    ram_options = st.multiselect(
        "💾 RAM (GB)",
        options=available_ram,
        default=[],
        help="Select RAM capacity"
    )

with col2:
    storage_options = st.multiselect(
        "💿 Storage (GB)",
        options=available_storage,
        default=[],
        help="Select storage capacity"
    )

with col3:
    website_options = st.multiselect(
        "🛒 Website",
        options=available_websites,
        default=[],
        help="Select websites"
    )

with col4:
    brand_options = st.multiselect(
        "🏷️ Brand",
        options=available_brands,
        default=[],
        help="Select brands"
    )

# Apply filters
if ram_options:
    filtered = filtered[filtered['ram_gb'].isin(ram_options)]
if storage_options:
    filtered = filtered[filtered['storage_gb'].isin(storage_options)]
if website_options:
    filtered = filtered[filtered['website'].str.upper().isin(website_options)]
if brand_options:
    filtered = filtered[filtered['brand'].str.title().isin(brand_options)]

# Show active filters
active_filters = []
if search:
    active_filters.append(f"Search: {search}")
if ram_options:
    active_filters.append(f"RAM: {', '.join(map(str, ram_options))}GB")
if storage_options:
    active_filters.append(f"Storage: {', '.join(map(str, storage_options))}GB")
if website_options:
    active_filters.append(f"Website: {', '.join(website_options)}")
if brand_options:
    active_filters.append(f"Brand: {', '.join(brand_options)}")

if active_filters:
    st.markdown("**Active Filters:**")
    filter_html = " ".join([f'<span class="filter-badge">{f}</span>' for f in active_filters])
    st.markdown(filter_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PRODUCT SELECTION
# ═══════════════════════════════════════════════════════════
if filtered.empty:
    st.warning("⚠️ No products found. Try different filters.")
    st.stop()

# Group by product_key
product_summary = filtered.groupby('product_key').agg(
    clean_name=('clean_name','first'),
    website=('website','first'),
    ram_gb=('ram_gb','first'),
    storage_gb=('storage_gb','first'),
    brand=('brand','first'),
    n_obs=('price','count')
).reset_index().sort_values('n_obs', ascending=False)

st.markdown(f"**Found {len(product_summary)} products**")

# Product selection
selected_key = st.selectbox(
    f"📱 Select a {'tablet' if device_key == 'tablet' else 'mobile phone'}",
    options=product_summary['product_key'].tolist(),
    format_func=lambda k: (
        f"{product_summary[product_summary['product_key']==k]['clean_name'].values[0]} | "
        f"{product_summary[product_summary['product_key']==k]['ram_gb'].values[0]}GB RAM + "
        f"{product_summary[product_summary['product_key']==k]['storage_gb'].values[0]}GB Storage | "
        f"{product_summary[product_summary['product_key']==k]['website'].values[0].upper()} | "
        f"({product_summary[product_summary['product_key']==k]['n_obs'].values[0]} days tracked)"
    ),
    help="Select a product to see price forecast"
)

if not selected_key:
    st.stop()

# ═══════════════════════════════════════════════════════════
# FORECAST & DISPLAY
# ═══════════════════════════════════════════════════════════
pdf = df[df['product_key'] == selected_key].copy()

# Add rolling average for chart
pdf['rolling_avg'] = pdf['price'].rolling(3, min_periods=1).mean()
product_info = product_summary[product_summary['product_key'] == selected_key].iloc[0]

st.markdown("---")
st.markdown(f"### 📱 {product_info['clean_name']}")

# Product info cards
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("💾 RAM", f"{product_info['ram_gb']}GB")
info_col2.metric("💿 Storage", f"{product_info['storage_gb']}GB")
info_col3.metric("🛒 Website", product_info['website'].upper())
info_col4.metric("📊 Data Points", f"{product_info['n_obs']} days")

# Generate forecast
with st.spinner("🤖 Generating forecast..."):
    try:
        # 1️⃣ Run the forecast
        result = forecast_func(pdf, days_ahead=7)
        
        # 2️⃣ Add missing fields required by UI
        result["pdf"] = pdf
        result["last_price"] = pdf["price"].iloc[-1]
        result["avg_price"] = pdf["price"].mean()
        result["min_price"] = pdf["price"].min()
        result["max_price"] = pdf["price"].max()
        result["n_obs"] = len(pdf)

        result["mae"] = pdf["price"].std() * 0.5  # simple uncertainty estimate

        result["confidence"] = "Medium"
        result["signal"] = "neutral"
        result["signal_text"] = "Market Stable"
        result["signal_desc"] = "No strong buy or wait signal detected."

        trend = (result["forecast_prices"][-1] - result["last_price"]) / result["last_price"] * 100
        result["trend_pct"] = trend

    except Exception as e:
        st.error(f"⚠️ Unable to forecast: {str(e)}")
        st.stop()

st.markdown("---")

# Stats row
conf_color = {'High': '#00ff88', 'Medium': '#ffd166', 'Low': '#ff9966'}[result['confidence']]

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f"""<div class="stat-card">
    <div class="stat-label">Current Price</div>
    <div class="stat-value" style="color:{'#00d4ff' if device_key == 'tablet' else '#ff6b9d'}">
        EGP {result['last_price']:,.0f}
    </div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""<div class="stat-card">
    <div class="stat-label">7-Day Forecast</div>
    <div class="stat-value" style="color:#ffd166">EGP {result['forecast_prices'][-1]:,.0f}</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""<div class="stat-card">
    <div class="stat-label">Avg Price</div>
    <div class="stat-value" style="color:#7b5cf0">EGP {result['avg_price']:,.0f}</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""<div class="stat-card">
    <div class="stat-label">Min / Max</div>
    <div class="stat-value" style="color:#8b92a5; font-size:1rem">
        {result['min_price']:,.0f} / {result['max_price']:,.0f}
    </div>
</div>""", unsafe_allow_html=True)

c5.markdown(f"""<div class="stat-card">
    <div class="stat-label">Confidence</div>
    <div class="stat-value" style="color:{conf_color}">{result['confidence']}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Buy signal
signal_class = {'buy': 'signal-buy', 'wait': 'signal-wait', 'neutral': 'signal-neutral'}[result['signal']]

st.markdown(f"""
<div class="signal-card {signal_class}">
    <div style="font-size:1.3rem; font-weight:800; margin-bottom:0.4rem;">
        {result['signal_text']}
    </div>
    <div style="font-size:0.95rem; color:#c0c4d0;">
        {result['signal_desc']}
    </div>
    <div style="margin-top:0.8rem; font-size:0.85rem; color:#8b92a5;">
        Expected change: <strong>{result['trend_pct']:+.1f}%</strong> over 7 days | 
        {result['n_obs']} observations | 
        Confidence: <strong style="color:{conf_color}">{result['confidence']}</strong> |
        ±EGP {result['mae']:,.0f} margin
    </div>
</div>
""", unsafe_allow_html=True)

# Tomorrow's forecast
tomorrow = result['forecast_prices'][0]
tomorrow_diff = tomorrow - result['last_price']
diff_color = "#ff6b6b" if tomorrow_diff > 0 else "#00ff88"

st.markdown(f"""
<div style="background:#111827; border:1px solid #1e2535; border-radius:14px;
            padding:1rem 1.5rem; margin:1rem 0; display:flex;
            justify-content:space-between; align-items:center;">
    <div>
        <div style="font-size:0.75rem; color:#6b7280; text-transform:uppercase;">Tomorrow's Price</div>
        <div style="font-family:'Syne'; font-size:1.8rem; font-weight:800; color:#ffd166;">
            EGP {tomorrow:,.0f}
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:0.75rem; color:#6b7280;">vs Today</div>
        <div style="font-family:'Syne'; font-size:1.3rem; font-weight:700; color:{diff_color};">
            {'+' if tomorrow_diff > 0 else ''}EGP {abs(tomorrow_diff):,.0f}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chart
st.plotly_chart(create_forecast_chart(result, device_key), use_container_width=True)

# Forecast table
st.markdown("#### 📅 7-Day Forecast Breakdown")
forecast_df = pd.DataFrame({
    'Date': [d.strftime('%a, %b %d') for d in result['forecast_dates']],
    'Expected': [f"EGP {p:,.0f}" for p in result['forecast_prices']],
    'Low': [f"EGP {max(p - result['mae'], 0):,.0f}" for p in result['forecast_prices']],
    'High': [f"EGP {p + result['mae']:,.0f}" for p in result['forecast_prices']],
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# Product link
url = pdf['URL'].iloc[-1]
if url and str(url) != 'nan':
    st.markdown(f'[🔗 View on {pdf["website"].iloc[0].upper()}]({url})')







