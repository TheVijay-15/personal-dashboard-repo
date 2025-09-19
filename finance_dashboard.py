import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.colors as mcolors

# Set page configuration
st.set_page_config(page_title="Personalized Financial Dashboard", layout="wide")

# Custom CSS with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; 
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid #4a5568;
    }
    .financial-suggestion {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid #4CAF50;
        font-size: 14px;
        color: white;
        border: 1px solid #4a5568;
    }
    .section-header {
        font-size: 1.5rem;
        color: #E2E8F0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4a5568;
    }
    .budget-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid #4a5568;
    }
    /* Dark mode chart styling */
    .js-plotly-plot .plotly, .plotly-container {
        font-family: 'Arial', sans-serif !important;
        background-color: #1a202c !important;
    }
    .js-plotly-plot .plotly .xtitle, .js-plotly-plot .plotly .ytitle {
        font-weight: bold !important;
        font-size: 14px !important;
        color: #E2E8F0 !important;
    }
    .js-plotly-plot .plotly .legend text {
        font-size: 12px !important;
        color: #E2E8F0 !important;
    }
    .js-plotly-plot .plotly .g-gtitle {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to format numbers in lakhs
def format_lakhs(x):
    """Format numbers in lakhs (e.g., 150000 becomes '1.5L')"""
    if abs(x) >= 100000:
        return f"{x/100000:.1f}L"
    else:
        return f"{x:,.0f}"

# Generate comprehensive sample dataset for middle-class family (in lakhs)
def generate_sample_data():
    """Generate a large sample dataset matching the provided structure for middle-class family"""
    # Create date range
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    
    # Define categories and their spending patterns (adjusted for middle-class family in lakhs)
    categories = {
        'Food': {'base': 1500, 'variance': 1000, 'frequency': 0.8, 'seasonality': 0.3},
        'Transport': {'base': 800, 'variance': 500, 'frequency': 0.6, 'seasonality': 0.2},
        'Utilities': {'base': 1200, 'variance': 400, 'frequency': 0.1, 'seasonality': 0.1},
        'Shopping': {'base': 2000, 'variance': 1500, 'frequency': 0.4, 'seasonality': 0.4},
        'Entertainment': {'base': 1500, 'variance': 1000, 'frequency': 0.3, 'seasonality': 0.5},
        'Healthcare': {'base': 1000, 'variance': 800, 'frequency': 0.2, 'seasonality': 0.1},
        'Recharge': {'base': 800, 'variance': 500, 'frequency': 0.2, 'seasonality': 0.1},
        'Travel': {'base': 1000, 'variance': 800, 'frequency': 0.1, 'seasonality': 0.6},
        'Personal': {'base': 1200, 'variance': 800, 'frequency': 0.3, 'seasonality': 0.3},
        'Fees': {'base': 500, 'variance': 400, 'frequency': 0.05, 'seasonality': 0.1},
        'Transfer': {'base': 1500, 'variance': 1000, 'frequency': 0.1, 'seasonality': 0.1},
        'Other': {'base': 1000, 'variance': 800, 'frequency': 0.2, 'seasonality': 0.2}
    }
    
    # Define descriptions for each category
    category_descriptions = {
        'Food': ['Restaurant Bill', 'Grocery Shopping', 'Food Delivery', 'Coffee Shop', 'Lunch'],
        'Transport': ['Uber Ride', 'Bus Fare', 'Train Ticket', 'Fuel Payment', 'Taxi'],
        'Utilities': ['Electricity Bill', 'Water Bill', 'Internet Bill', 'Gas Bill'],
        'Shopping': ['Amazon Purchase', 'Clothing Store', 'Electronics', 'Online Shopping'],
        'Entertainment': ['Movie Tickets', 'Streaming Service', 'Concert', 'Theater'],
        'Healthcare': ['Doctor Visit', 'Pharmacy', 'Hospital Bill', 'Medicines'],
        'Recharge': ['Mobile Recharge', 'DTH Recharge', 'Data Pack'],
        'Travel': ['Flight Booking', 'Hotel Stay', 'Vacation Package', 'Car Rental'],
        'Personal': ['Salon Visit', 'Gym Membership', 'Spa Treatment'],
        'Fees': ['Bank Charges', 'Credit Card Fee', 'Late Fee'],
        'Transfer': ['Bank Transfer', 'UPI Payment', 'IMPS Transfer'],
        'Other': ['Miscellaneous', 'Cash Expense', 'General Purchase']
    }
    
    # Generate transactions
    transactions = []
    
    for date in dates:
        for category, params in categories.items():
            # Determine if a transaction occurs for this category on this date
            if np.random.random() < params['frequency']:
                # Add seasonality effect (higher in certain months)
                month_factor = 1 + (params['seasonality'] * np.sin(2 * np.pi * date.month / 12))
                
                # Generate transaction amount with seasonality
                amount = -abs(np.random.normal(params['base'] * month_factor, params['variance']))
                
                # Select random description
                description = np.random.choice(category_descriptions[category])
                
                transactions.append({
                    'date': date,
                    'description': description,
                    'amount': round(amount, 2),
                    'Category': category
                })
    
    # Add some income transactions (more at the beginning of months)
    for date in dates:
        if date.day in [1, 2, 3]:  # Higher probability at month start
            if np.random.random() < 0.7:
                transactions.append({
                    'date': date,
                    'description': 'Salary Credit',
                    'amount': round(np.random.normal(30000, 3000), 2),  # Adjusted for middle-class
                    'Category': 'Income'
                })
        elif np.random.random() < 0.05:  # Other random income
            transactions.append({
                'date': date,
                'description': np.random.choice(['Freelance Work', 'Investment Return', 'Bonus', 'Gift']),
                'amount': round(np.random.normal(5000, 2000), 2),  # Adjusted for middle-class
                'Category': 'Income'
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

# Initialize data
@st.cache_data
def load_data():
    return generate_sample_data()

# Load the data
df = load_data()

# Generate a dynamic color palette for charts
def generate_color_palette(n_colors):
    """Generate a visually appealing color palette for dark mode"""
    # Colors that work well in dark mode
    base_colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', 
                  '#EC4899', '#06B6D4', '#F97316', '#84CC16', '#14B8A6']
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Generate additional colors if needed
    additional_colors = list(mcolors.CSS4_COLORS.values())
    return base_colors + additional_colors[:n_colors - len(base_colors)]

# RBI Inflation Data (pre-loaded)
rbi_inflation_data = {
    'Food': 6.2, 
    'Transport': 5.1, 
    'Utilities': 4.3, 
    'Healthcare': 7.1, 
    'Entertainment': 3.8,
    'Shopping': 5.4,
    'Recharge': 2.9,
    'Travel': 6.7,
    'Personal': 4.8,
    'Other': 4.5,
    'Fees': 3.2,
    'Transfer': 4.0
}

# Enhanced budget inputs with RBI inflation adjustment
def create_budget_inputs(categories):
    """Create budget input widgets for each category with RBI inflation adjustment"""
    budgets = {}
    
    st.markdown('<div class="section-header">Budget Settings</div>', unsafe_allow_html=True)
    
    # Display inflation overview
    avg_inflation = sum(rbi_inflation_data.values()) / len(rbi_inflation_data)
    st.metric("Average Expected Inflation", f"{avg_inflation:.1f}%", 
              "Based on RBI Inflation Expectations Survey")
    
    cols = st.columns(3)
    for i, category in enumerate(categories):
        if category == 'Income':  # Skip income category for budgets
            continue
            
        with cols[i % 3]:
            # Get inflation rate for this category
            inflation_rate = rbi_inflation_data.get(category, avg_inflation)
            
            # Base budget input (in thousands)
            base_budget = st.number_input(
                f"{category} Budget (₹ '000)", 
                min_value=0, 
                value=30 if category in ['Food', 'Shopping'] else 15,
                step=5,
                key=f"base_{category}",
                help=f"RBI expected inflation for {category}: {inflation_rate}%"
            )
            
            # Convert to actual amount (multiply by 1000) and adjust for inflation
            actual_budget = base_budget * 1000
            adjusted_budget = actual_budget * (1 + inflation_rate/100)
            budgets[category] = adjusted_budget
    
    return budgets

# Enhanced budget analysis with RBI inflation awareness
def analyze_budget_performance(_df, budgets):
    """Analyze budget performance considering RBI inflation data"""
    # Calculate actual spending
    spending = _df[_df['amount'] < 0].groupby('Category')['amount'].sum().abs().reset_index()
    spending.columns = ['Category', 'actual_spending']
    
    # Merge with budgets
    budget_df = pd.DataFrame(list(budgets.items()), columns=['Category', 'budget'])
    analysis_df = spending.merge(budget_df, on='Category', how='right')
    analysis_df['actual_spending'] = analysis_df['actual_spending'].fillna(0)
    
    # Get inflation rates for adjustment
    avg_inflation = sum(rbi_inflation_data.values()) / len(rbi_inflation_data)
    
    # Calculate inflation-adjusted metrics
    analysis_df['inflation_rate'] = analysis_df['Category'].map(
        lambda x: rbi_inflation_data.get(x, avg_inflation)
    )
    analysis_df['inflation_adjusted_budget'] = analysis_df['budget'] / (1 + analysis_df['inflation_rate']/100)
    analysis_df['real_utilization'] = analysis_df['actual_spending'] / analysis_df['inflation_adjusted_budget'] * 100
    analysis_df['nominal_utilization'] = analysis_df['actual_spending'] / analysis_df['budget'] * 100
    
    # Calculate budget variance
    analysis_df['budget_variance'] = analysis_df['actual_spending'] - analysis_df['budget']
    analysis_df['inflation_adjusted_variance'] = analysis_df['actual_spending'] - analysis_df['inflation_adjusted_budget']
    
    return analysis_df

# Enhanced forecasting visualization with better text visibility
def generate_forecast(_df, category):
    """Generate expense forecast using Prophet with enhanced visualization"""
    try:
        # Prepare data for forecasting
        category_data = _df[_df['Category'] == category].copy()
        if len(category_data) < 10:
            return None, None, None, None
            
        category_data['date'] = pd.to_datetime(category_data['date'])
        daily_spending = category_data.groupby('date')['amount'].sum().reset_index()
        daily_spending = daily_spending[daily_spending['amount'] < 0]
        daily_spending['amount'] = daily_spending['amount'].abs()
        
        # Prepare data for Prophet
        prophet_df = daily_spending.rename(columns={'date': 'ds', 'amount': 'y'})
        
        # Fit model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Calculate metrics
        actuals = prophet_df['y'].values
        predicted = forecast['yhat'].head(len(actuals)).values
        mae = mean_absolute_error(actuals, predicted)
        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        
        return forecast, mae, rmse, prophet_df
        
    except Exception as e:
        st.error(f"Forecast failed for {category}: {str(e)}")
        return None, None, None, None

# Enhanced forecast visualization with better text visibility
def plot_forecast(forecast, actuals_data, category):
    """Create enhanced forecast visualization with improved text visibility"""
    fig = go.Figure()
    
    # Add actual data with enhanced styling
    fig.add_trace(go.Scatter(
        x=actuals_data['ds'],
        y=actuals_data['y'],
        mode='markers+lines',
        name='Actual Spending',
        marker=dict(color='#6366F1', size=8, line=dict(width=2, color='white')),
        line=dict(color='#6366F1', width=3),
        hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Amount:</b> ₹%{y:.2f}<extra></extra>'
    ))
    
    # Add forecast with enhanced styling
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#10B981', width=4, dash='solid'),
        hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Forecast:</b> ₹%{y:.2f><extra></extra>'
    ))
    
    # Add uncertainty interval with enhanced styling
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Uncertainty Range',
        hovertemplate='<b>Uncertainty Range</b><extra></extra>'
    ))
    
    # Update layout for dark mode
    fig.update_layout(
        title=f'Expense Forecast for {category}',
        yaxis_title='Amount (₹)',
        xaxis_title='Date',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=14, family='Arial', color='#E2E8F0'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        height=500,
        title_font_size=20,
        title_x=0.5
    )
    
    # Improve axis labels visibility for dark mode
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True,
        title_font=dict(size=14),
        tickfont=dict(size=12),
        zerolinecolor='rgba(255, 255, 255, 0.1)'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True,
        title_font=dict(size=14),
        tickfont=dict(size=12),
        tickprefix='₹',
        zerolinecolor='rgba(255, 255, 255, 0.1)'
    )
    
    return fig

# Enhanced calendar view
def create_calendar_view(_df):
    """Create a calendar heatmap view of spending"""
    if _df is None or _df.empty:
        return go.Figure()
    
    # Prepare data for calendar
    spending_df = _df[_df['amount'] < 0].copy()
    spending_df['amount_abs'] = spending_df['amount'].abs()
    spending_df['date'] = pd.to_datetime(spending_df['date'])
    spending_df['year'] = spending_df['date'].dt.year
    spending_df['month'] = spending_df['date'].dt.month
    spending_df['day'] = spending_df['date'].dt.day
    
    # Create daily spending totals
    daily_spending = spending_df.groupby('date')['amount_abs'].sum().reset_index()
    
    # Create calendar heatmap
    fig = px.density_heatmap(
        spending_df, 
        x=spending_df['date'].dt.day, 
        y=spending_df['date'].dt.month_name(),
        z='amount_abs',
        histfunc='sum',
        title='Calendar View of Spending (Darker = Higher Spending)',
        labels={'x': 'Day of Month', 'y': 'Month', 'z': 'Amount Spent (₹)'},
        color_continuous_scale='Viridis'
    )
    
    # Update layout for dark mode
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12, family='Arial', color='#E2E8F0'),
        title_font_size=18,
        title_x=0.5
    )
    
    # Improve axis visibility
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
    
    return fig

# Enhanced time series visualization with statistical analysis
def create_enhanced_time_series(_df):
    """Create enhanced time series visualization with clear labels and statistical analysis"""
    # Monthly spending trend with enhanced styling
    monthly_trend = _df[_df['amount'] < 0].copy()
    monthly_trend['month'] = pd.to_datetime(monthly_trend['date']).dt.to_period('M')
    monthly_trend = monthly_trend.groupby('month')['amount'].sum().abs().reset_index()
    monthly_trend['month'] = monthly_trend['month'].dt.to_timestamp()
    monthly_trend.columns = ['month', 'spending']
    
    if len(monthly_trend) > 0:
        # Calculate statistical measures
        mean_spending = monthly_trend['spending'].mean()
        max_spending = monthly_trend['spending'].max()
        min_spending = monthly_trend['spending'].min()
        
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=monthly_trend['month'],
            y=monthly_trend['spending'],
            mode='lines+markers',
            name='Monthly Spending',
            line=dict(color='#6366F1', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Spending:</b> ₹%{y:,.2f}<extra></extra>'
        ))
        
        # Add highest and lowest points
        max_month = monthly_trend.loc[monthly_trend['spending'].idxmax()]
        min_month = monthly_trend.loc[monthly_trend['spending'].idxmin()]
        
        fig.add_trace(go.Scatter(
            x=[max_month['month']],
            y=[max_month['spending']],
            mode='markers+text',
            marker=dict(color='#EF4444', size=12),
            text=['Highest'],
            textposition='top center',
            name='Highest Spending',
            hovertemplate='<b>Highest:</b> ₹%{y:,.2f}<br>%{x|%b %Y}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[min_month['month']],
            y=[min_month['spending']],
            mode='markers+text',
            marker=dict(color='#10B981', size=12),
            text=['Lowest'],
            textposition='bottom center',
            name='Lowest Spending',
            hovertemplate='<b>Lowest:</b> ₹%{y:,.2f}<br>%{x|%b %Y><extra></extra>'
        ))
        
        # Add mean line
        fig.add_hline(y=mean_spending, line_dash="dash", line_color="#F59E0B",
                     annotation_text=f"Average: ₹{format_lakhs(mean_spending)}", 
                     annotation_position="top right")
        
        fig.update_layout(
            yaxis_title="Spending (₹)",
            xaxis_title="Month",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(size=12, family='Arial', color='#E2E8F0'),
            title_font_size=18,
            title_x=0.5,
            hovermode='x unified',
            title='Monthly Spending Trend with Highest & Lowest Points'
        )
        
        # Improve axis visibility
        fig.update_xaxes(
            title_font=dict(size=14), 
            tickfont=dict(size=12),
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        )
        fig.update_yaxes(
            title_font=dict(size=14), 
            tickfont=dict(size=12), 
            tickprefix='₹',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        )
        
        return fig, monthly_trend, mean_spending, max_spending, min_spending
    else:
        return None, None, None, None, None

# Enhanced category breakdown visualization with pie chart
def create_enhanced_category_chart(_df):
    """Create enhanced category breakdown visualization with pie chart"""
    category_spending = _df[_df['amount'] < 0].groupby('Category')['amount'].sum().abs().reset_index()
    if len(category_spending) > 0:
        # Sort by amount
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        # Generate dynamic colors
        colors = generate_color_palette(len(category_spending))
        
        # Create pie chart with the highest spending category pulled out
        pull_amounts = [0.1 if i == 0 else 0 for i in range(len(category_spending))]
        
        # Create pie chart
        fig = px.pie(category_spending, values='amount', names='Category', 
                    title='Spending Distribution by Category',
                    color_discrete_sequence=colors)
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont=dict(size=12, family='Arial'),
            hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Percentage: %{percent}<extra></extra>',
            pull=pull_amounts
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(size=12, family='Arial', color='#E2E8F0'),
            title_font_size=18,
            title_x=0.5,
            legend=dict(font=dict(size=12)),
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        
        return fig, category_spending
    else:
        return None, None

# Create category-wise bar chart with frequency line
def create_category_bar_chart(_df):
    """Create category-wise bar chart with frequency line"""
    category_spending = _df[_df['amount'] < 0].groupby('Category')['amount'].sum().abs().reset_index()
    category_frequency = _df[_df['amount'] < 0].groupby('Category').size().reset_index(name='count')
    
    if len(category_spending) > 0:
        # Sort by amount
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add spending bars
        fig.add_trace(go.Bar(
            x=category_spending['Category'],
            y=category_spending['amount'],
            name='Total Spending',
            marker_color='#6366F1',
            width=0.4,
            hovertemplate='<b>%{x}</b><br>Total Spending: ₹%{y:,.2f}<extra></extra>'
        ), secondary_y=False)
        
        # Add frequency line
        fig.add_trace(go.Scatter(
            x=category_frequency['Category'],
            y=category_frequency['count'],
            mode='lines+markers',
            name='Transaction Count',
            line=dict(color='#10B981', width=3),
            marker=dict(size=8, color='#10B981'),
            hovertemplate='<b>%{x}</b><br>Transactions: %{y}<extra></extra>'
        ), secondary_y=True)
        
        fig.update_layout(
            title='Spending by Category with Transaction Frequency',
            yaxis_title="Spending (₹)",
            xaxis_title="Category",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(size=12, family='Arial', color='#E2E8F0'),
            title_font_size=16,
            title_x=0.5,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(
            title_font=dict(size=14), 
            tickfont=dict(size=12),
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        fig.update_yaxes(
            title_text="Spending (₹)",
            title_font=dict(size=14), 
            tickfont=dict(size=12), 
            tickprefix='₹',
            gridcolor='rgba(255, 255, 255, 0.1)',
            secondary_y=False
        )
        
        fig.update_yaxes(
            title_text="Transaction Count",
            title_font=dict(size=14), 
            tickfont=dict(size=12),
            gridcolor='rgba(255, 255, 255, 0.1)',
            secondary_y=True
        )
        
        return fig
    else:
        return None

# Completely redesigned budget visualization
def create_budget_visualization(analysis_df):
    """Create enhanced budget visualization with detailed metrics"""
    # Prepare data for visualization
    viz_df = analysis_df.copy()
    viz_df = viz_df.sort_values('actual_spending', ascending=False)
    
    # Create detailed bar chart for budget vs actual
    bar_fig = go.Figure()
    
    # Add budget bars
    bar_fig.add_trace(go.Bar(
        name='Budget',
        x=viz_df['Category'],
        y=viz_df['budget'],
        marker_color='#6366F1',
        width=0.4,  # Thinner bars
        hovertemplate='<b>%{x}</b><br>Budget: ₹%{y:,.2f}<extra></extra>'
    ))
    
    # Add actual spending bars
    bar_fig.add_trace(go.Bar(
        name='Actual Spending',
        x=viz_df['Category'],
        y=viz_df['actual_spending'],
        marker_color='#10B981',
        width=0.4,  # Thinner bars
        hovertemplate='<b>%{x}</b><br>Actual: ₹%{y:,.2f}<extra></extra>'
    ))
    
    bar_fig.update_layout(
        title='Budget vs Actual Spending by Category',
        yaxis_title='Amount (₹)',
        xaxis_title='Category',
        barmode='group',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hovermode='x unified',
        font=dict(size=12, family='Arial', color='#E2E8F0'),
        title_font_size=16,
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    bar_fig.update_xaxes(
        title_font=dict(size=14), 
        tickfont=dict(size=12),
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    bar_fig.update_yaxes(
        title_font=dict(size=14), 
        tickfont=dict(size=12), 
        tickprefix='₹',
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    # Create utilization chart
    util_fig = go.Figure()
    
    util_fig.add_trace(go.Bar(
        x=viz_df['Category'],
        y=viz_df['nominal_utilization'],
        marker_color=np.where(viz_df['nominal_utilization'] > 100, '#EF4444', '#10B981'),
        width=0.6,
        hovertemplate='<b>%{x}</b><br>Utilization: %{y:.1f}%<extra></extra>'
    ))
    
    util_fig.add_hline(y=100, line_dash="dash", line_color="#EF4444",
                      annotation_text="Budget Limit", 
                      annotation_position="top right")
    
    util_fig.update_layout(
        title='Budget Utilization by Category (%)',
        yaxis_title='Utilization (%)',
        xaxis_title='Category',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hovermode='x unified',
        font=dict(size=12, family='Arial', color='#E2E8F0'),
        title_font_size=16,
        title_x=0.5
    )
    
    return bar_fig, util_fig

# Create simplified weekday spending chart with frequency line
def create_weekday_chart(_df):
    """Create weekday spending chart with frequency line but without frequency curve"""
    daily_data = _df.copy()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['amount_abs'] = daily_data['amount'].abs()
    daily_data['weekday'] = daily_data['date'].dt.day_name()
    
    # Analyze spending by day of week
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_spending = daily_data.groupby('weekday')['amount_abs'].sum().reset_index()
    weekday_spending['weekday'] = pd.Categorical(weekday_spending['weekday'], categories=weekday_order, ordered=True)
    weekday_spending = weekday_spending.sort_values('weekday')
    
    # Calculate frequency by day of week
    weekday_frequency = daily_data.groupby('weekday').size().reset_index(name='count')
    weekday_frequency['weekday'] = pd.Categorical(weekday_frequency['weekday'], categories=weekday_order, ordered=True)
    weekday_frequency = weekday_frequency.sort_values('weekday')
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add spending bars (thin)
    fig.add_trace(go.Bar(
        x=weekday_spending['weekday'],
        y=weekday_spending['amount_abs'],
        name='Total Spending',
        marker_color='#6366F1',
        width=0.3,  # Thin bars
        hovertemplate='<b>%{x}</b><br>Total Spending: ₹%{y:,.2f}<extra></extra>'
    ), secondary_y=False)
    
    # Add frequency line (without curve)
    fig.add_trace(go.Scatter(
        x=weekday_frequency['weekday'],
        y=weekday_frequency['count'],
        mode='lines+markers',
        name='Transaction Count',
        line=dict(color='#10B981', width=3),
        marker=dict(size=8, color='#10B981'),
        hovertemplate='<b>%{x}</b><br>Transactions: %{y}<extra></extra>'
    ), secondary_y=True)
    
    # Calculate statistics for weekdays
    weekday_mean = weekday_spending['amount_abs'].mean()
    max_spending = weekday_spending['amount_abs'].max()
    min_spending = weekday_spending['amount_abs'].min()
    
    # Add average line
    fig.add_hline(y=weekday_mean, line_dash="dash", line_color="#F59E0B",
                 annotation_text=f"Avg: ₹{format_lakhs(weekday_mean)}", 
                 annotation_position="top right",
                 secondary_y=False)
    
    # Add highest and lowest points
    max_day = weekday_spending.loc[weekday_spending['amount_abs'].idxmax()]
    min_day = weekday_spending.loc[weekday_spending['amount_abs'].idxmin()]
    
    fig.add_trace(go.Scatter(
        x=[max_day['weekday']],
        y=[max_day['amount_abs']],
        mode='markers+text',
        marker=dict(color='#EF4444', size=12),
        text=['Highest'],
        textposition='top center',
        name='Highest Spending',
        hovertemplate='<b>Highest:</b> ₹%{y:,.2f}<br>%{x}<extra></extra>',
        showlegend=False
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=[min_day['weekday']],
        y=[min_day['amount_abs']],
        mode='markers+text',
        marker=dict(color='#10B981', size=12),
        text=['Lowest'],
        textposition='bottom center',
        name='Lowest Spending',
        hovertemplate='<b>Lowest:</b> ₹%{y:,.2f}<br>%{x}<extra></extra>',
        showlegend=False
    ), secondary_y=False)
    
    # Set axis titles
    fig.update_layout(
        title='Weekly Spending Analysis with Transaction Frequency',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(size=12, family='Arial', color='#E2E8F0'),
        title_font_size=16,
        title_x=0.5,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        title_text="Day of Week",
        title_font=dict(size=14), 
        tickfont=dict(size=12),
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    fig.update_yaxes(
        title_text="Spending (₹)",
        title_font=dict(size=14), 
        tickfont=dict(size=12), 
        tickprefix='₹',
        gridcolor='rgba(255, 255, 255, 0.1)',
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Transaction Count",
        title_font=dict(size=14), 
        tickfont=dict(size=12),
        gridcolor='rgba(255, 255, 255, 0.1)',
        secondary_y=True
    )
    
    return fig, weekday_mean, max_spending, min_spending

# Function to generate financial suggestions
def generate_financial_suggestions(_df, analysis_df=None, selected_category=None):
    """Generate personalized financial suggestions based on spending data"""
    suggestions = []
    
    # Calculate total spending and income
    total_spending = abs(_df[_df['amount'] < 0]['amount'].sum())
    total_income = _df[_df['amount'] > 0]['amount'].sum()
    savings_rate = (total_income - total_spending) / total_income * 100 if total_income > 0 else 0
    
    # Suggestion based on savings rate
    if savings_rate < 10:
        suggestions.append(f"Your savings rate is {savings_rate:.1f}%, which is below the recommended 20%. Consider reducing discretionary spending.")
    elif savings_rate < 20:
        suggestions.append(f"Your savings rate is {savings_rate:.1f}%. Good job, but try to reach 20% for better financial security.")
    else:
        suggestions.append(f"Excellent! Your savings rate is {savings_rate:.1f}%, which meets the recommended target.")
    
    # Analyze spending patterns by category
    category_spending = _df[_df['amount'] < 0].groupby('Category')['amount'].sum().abs()
    top_category = category_spending.idxmax()
    top_amount = category_spending.max()
    
    if top_amount > total_income * 0.3:  # If top category is more than 30% of income
        suggestions.append(f"Your spending on {top_category} is relatively high (₹{format_lakhs(top_amount)}). Consider reviewing expenses in this category.")
    
    # Check for budget performance if analysis is provided
    if analysis_df is not None:
        over_budget_categories = analysis_df[analysis_df['real_utilization'] > 100]
        if not over_budget_categories.empty:
            worst_category = over_budget_categories.loc[over_budget_categories['real_utilization'].idxmax()]
            suggestions.append(f"You're significantly over budget for {worst_category['Category']} ({(worst_category['real_utilization']-100):.1f}% over limit). Consider adjusting your spending or budget.")
    
    # Check for irregular income patterns
    income_data = _df[_df['amount'] > 0]
    if len(income_data) > 0:
        income_data['date'] = pd.to_datetime(income_data['date'])
        income_by_month = income_data.groupby(income_data['date'].dt.to_period('M'))['amount'].sum()
        income_std = income_by_month.std()
        if income_std > income_by_month.mean() * 0.3:  # High variability
            suggestions.append("Your income shows significant variability month-to-month. Consider building a larger emergency fund.")
    
    # Return only the most relevant suggestion if a category is selected
    if selected_category:
        category_suggestions = [s for s in suggestions if selected_category in s]
        if category_suggestions:
            return [category_suggestions[0]]
    
    # Return the first suggestion if no category is selected
    return [suggestions[0]] if suggestions else ["No specific suggestions at this time."]

def main():
    st.title("Personalized Financial Dashboard")
    st.markdown("Comprehensive analysis of your financial patterns with actionable insights")
    
    # Display dataset info
    st.success(f"Loaded {len(df):,} transactions from enhanced sample dataset")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Category Analysis", 
        "Expense Forecasting", 
        "Budget Tracking", 
        "Calendar View"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Spending Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly spending trend with enhanced styling and statistical analysis
            time_series_fig, monthly_data, mean_spending, max_spending, min_spending = create_enhanced_time_series(df)
            if time_series_fig:
                st.plotly_chart(time_series_fig, use_container_width=True)
            else:
                st.info("No spending data available for trend analysis")

        with col2:
            # Category breakdown with pie chart
            category_fig, category_data = create_enhanced_category_chart(df)
            if category_fig:
                st.plotly_chart(category_fig, use_container_width=True)
            else:
                st.info("No spending data available for category analysis")

        # Display statistics and top categories below both charts
        if monthly_data is not None:
            st.markdown("### Spending Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Average Monthly Spending", f"₹{format_lakhs(mean_spending)}")
            stat_col2.metric("Highest Month", f"₹{format_lakhs(max_spending)}")
            stat_col3.metric("Lowest Month", f"₹{format_lakhs(min_spending)}")
            stat_col4.metric("Total Months", f"{len(monthly_data)}")

        # Display top categories
        if category_data is not None and len(category_data) > 0:
            st.markdown("### Top Spending Categories")
            top_3 = category_data.head(3)
            cat_col1, cat_col2, cat_col3 = st.columns(3)
            for i, row in enumerate(top_3.iterrows()):
                with [cat_col1, cat_col2, cat_col3][i]:
                    st.metric(f"{row[1]['Category']}", f"₹{format_lakhs(row[1]['amount'])}")
            
            # Financial suggestions
            suggestions = generate_financial_suggestions(df)
            if suggestions:
                st.markdown("### Financial Suggestions")
                for suggestion in suggestions[:1]:  # Show only the most relevant suggestion
                    st.markdown(f'<div class="financial-suggestion">{suggestion}</div>', unsafe_allow_html=True)
            
            # Display recent transactions
            st.markdown('<div class="section-header">Recent Transactions</div>', unsafe_allow_html=True)
            st.dataframe(
                df[['date', 'description', 'amount', 'Category']].head(10),
                height=300,
                use_container_width=True
            )
        
    with tab2:
        st.markdown('<div class="section-header">Category-wise Analysis</div>', unsafe_allow_html=True)
        
        category = st.selectbox("Select Category", options=df['Category'].unique(), key='category_analysis')
        
        category_data = df[df['Category'] == category]
        
        # Summary for selected category with enhanced metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">Total Transactions</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: #6366F1;">{len(category_data):,}</div>', unsafe_allow_html=True)
        
        with col2:
            total_spent = abs(category_data[category_data["amount"] < 0]["amount"].sum())
            st.markdown('<div class="metric-card">Total Spent</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: #EF4444;">₹{format_lakhs(total_spent)}</div>', unsafe_allow_html=True)
        
        with col3:
            total_received = category_data[category_data["amount"] > 0]["amount"].sum()
            st.markdown('<div class="metric-card">Total Received</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: #10B981;">₹{format_lakhs(total_received)}</div>', unsafe_allow_html=True)
        
        # Monthly trend for selected category with frequency line
        category_monthly = category_data[category_data['amount'] < 0].copy()
        if len(category_monthly) > 0:
            category_monthly['month'] = pd.to_datetime(category_monthly['date']).dt.to_period('M')
            category_monthly = category_monthly.groupby('month')['amount'].sum().abs().reset_index()
            category_monthly['month'] = category_monthly['month'].dt.to_timestamp()
            
            # Calculate transaction frequency by month
            monthly_frequency = category_data[category_data['amount'] < 0].copy()
            monthly_frequency['month'] = pd.to_datetime(monthly_frequency['date']).dt.to_period('M')
            monthly_frequency = monthly_frequency.groupby('month').size().reset_index(name='count')
            monthly_frequency['month'] = monthly_frequency['month'].dt.to_timestamp()
            
            # Calculate statistical measures
            avg_spending = category_monthly['amount'].mean()
            max_spending = category_monthly['amount'].max()
            min_spending = category_monthly['amount'].min()
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add spending bars
            fig.add_trace(go.Bar(
                x=category_monthly['month'],
                y=category_monthly['amount'],
                name='Monthly Spending',
                marker_color='#6366F1',
                hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Spending:</b> ₹%{y:,.2f}<extra></extra>'
            ), secondary_y=False)
            
            # Add frequency line
            fig.add_trace(go.Scatter(
                x=monthly_frequency['month'],
                y=monthly_frequency['count'],
                mode='lines+markers',
                name='Transaction Count',
                line=dict(color='#10B981', width=3),
                marker=dict(size=8, color='#10B981'),
                hovertemplate='<b>Month:</b> %{x|%b %Y}<br><b>Transactions:</b> %{y}<extra></extra>'
            ), secondary_y=True)
            
            # Add highest and lowest points
            max_month = category_monthly.loc[category_monthly['amount'].idxmax()]
            min_month = category_monthly.loc[category_monthly['amount'].idxmin()]
            
            fig.add_trace(go.Scatter(
                x=[max_month['month']],
                y=[max_month['amount']],
                mode='markers+text',
                marker=dict(color='#EF4444', size=12),
                text=['Highest'],
                textposition='top center',
                name='Highest Spending',
                hovertemplate='<b>Highest:</b> ₹%{y:,.2f}<br>%{x|%b %Y}<extra></extra>',
                showlegend=False
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=[min_month['month']],
                y=[min_month['amount']],
                mode='markers+text',
                marker=dict(color='#10B981', size=12),
                text=['Lowest'],
                textposition='bottom center',
                name='Lowest Spending',
                hovertemplate='<b>Lowest:</b> ₹%{y:,.2f}<br>%{x|%b %Y}<extra></extra>',
                showlegend=False
            ), secondary_y=False)
            
            fig.update_layout(
                title=f'Monthly Spending on {category} with Transaction Frequency',
                yaxis_title="Spending (₹)",
                xaxis_title="Month",
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(size=12, family='Arial', color='#E2E8F0'),
                title_font_size=16,
                title_x=0.5,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(
                title_font=dict(size=14), 
                tickfont=dict(size=12),
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
            
            fig.update_yaxes(
                title_text="Spending (₹)",
                title_font=dict(size=14), 
                tickfont=dict(size=12), 
                tickprefix='₹',
                gridcolor='rgba(255, 255, 255, 0.1)',
                secondary_y=False
            )
            
            fig.update_yaxes(
                title_text="Transaction Count",
                title_font=dict(size=14), 
                tickfont=dict(size=12),
                gridcolor='rgba(255, 255, 255, 0.1)',
                secondary_y=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics for the category
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Average Monthly Spending", f"₹{format_lakhs(avg_spending)}")
            col2.metric(f"Highest Month", f"₹{format_lakhs(max_spending)}")
            col3.metric(f"Lowest Month", f"₹{format_lakhs(min_spending)}")
            
            # Generate category-specific suggestion
            suggestions = generate_financial_suggestions(df, selected_category=category)
            if suggestions:
                st.markdown("### Financial Suggestion")
                for suggestion in suggestions:
                    st.markdown(f'<div class="financial-suggestion">{suggestion}</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header">Expense Forecasting</div>', unsafe_allow_html=True)
        
        forecast_category = st.selectbox("Select Category for Forecasting", 
                                       options=[c for c in df['Category'].unique() if c != 'Income'],
                                       key='forecast_category')
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Training model and generating forecast..."):
                forecast, mae, rmse, prophet_df = generate_forecast(df, forecast_category)
                
                if forecast is not None:
                    # Display forecast metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error", f"₹{format_lakhs(mae)}", 
                                 help="Average prediction error amount")
                    with col2:
                        st.metric("Root Mean Square Error", f"₹{format_lakhs(rmse)}", 
                                 help="Standard deviation of prediction errors")
                    
                    # Plot enhanced forecast
                    fig = plot_forecast(forecast, prophet_df, forecast_category)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast summary
                    last_date = forecast['ds'].max()
                    next_month_forecast = forecast[forecast['ds'] > last_date - pd.DateOffset(months=1)]
                    avg_forecast = next_month_forecast['yhat'].mean()
                    
                    st.subheader("Forecast Summary")
                    st.metric("Predicted Average Monthly Spending", f"₹{format_lakhs(avg_forecast)}", 
                             help="Average expected spending for next month")
                    
                    # Generate forecast-based suggestion
                    current_avg = prophet_df['y'].mean()
                    if avg_forecast > current_avg * 1.2:
                        st.markdown(f'<div class="financial-suggestion">Forecast suggests your {forecast_category} spending may increase significantly. Consider reviewing your upcoming expenses in this category.</div>', unsafe_allow_html=True)
                    elif avg_forecast < current_avg * 0.8:
                        st.markdown(f'<div class="financial-suggestion">Forecast suggests your {forecast_category} spending may decrease. You might be able to reallocate some funds to other areas.</div>', unsafe_allow_html=True)
                    
                else:
                    st.warning(f"Not enough data to generate forecast for {forecast_category}. Need at least 10 transactions.")
    
    with tab4:
        st.markdown('<div class="section-header">Budget Tracking</div>', unsafe_allow_html=True)
        
        # Get unique categories (excluding Income)
        categories = [c for c in df['Category'].unique() if c != 'Income']
        
        # Create budget inputs with RBI inflation adjustment
        budgets = create_budget_inputs(categories)
        
        if st.button("Analyze Budget Performance", type="primary"):
            # Use inflation-aware analysis
            analysis_df = analyze_budget_performance(df, budgets)
            
            # Display budget analysis
            st.subheader("Budget Analysis")
            
            # Create enhanced budget visualization
            bar_fig, util_fig = create_budget_visualization(analysis_df)
            
            st.plotly_chart(util_fig, use_container_width=True)
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Calculate overall performance metrics
            total_budget = analysis_df['budget'].sum()
            total_spending = analysis_df['actual_spending'].sum()
            overall_utilization = (total_spending / total_budget) * 100
            
            # Display budget performance metrics
            st.markdown("### Budget Performance Summary")
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            perf_col1.metric("Total Budget", f"₹{format_lakhs(total_budget)}")
            perf_col2.metric("Total Spending", f"₹{format_lakhs(total_spending)}")
            perf_col3.metric("Budget Utilization", f"{overall_utilization:.1f}%")
            
            # Count over-budget categories
            over_budget_count = len(analysis_df[analysis_df['actual_spending'] > analysis_df['budget']])
            perf_col4.metric("Over-Budget Categories", f"{over_budget_count}")
            
            # Display detailed budget table
            st.markdown("### Detailed Budget Analysis")
            display_df = analysis_df.copy()
            display_df['budget'] = display_df['budget'].round(2)
            display_df['actual_spending'] = display_df['actual_spending'].round(2)
            display_df['nominal_utilization'] = display_df['nominal_utilization'].round(1)
            display_df['budget_variance'] = display_df['budget_variance'].round(2)
            
            st.dataframe(
                display_df[['Category', 'budget', 'actual_spending', 'nominal_utilization', 'budget_variance']].rename(
                    columns={
                        'budget': 'Budget (₹)',
                        'actual_spending': 'Actual Spending (₹)',
                        'nominal_utilization': 'Utilization (%)',
                        'budget_variance': 'Variance (₹)'
                    }
                ),
                use_container_width=True
            )
            
            # Generate budget-specific suggestions
            budget_suggestions = generate_financial_suggestions(df, analysis_df)
            if budget_suggestions:
                st.markdown("### Budget Suggestions")
                for suggestion in budget_suggestions[:1]:  # Show only the most relevant suggestion
                    st.markdown(f'<div class="financial-suggestion">{suggestion}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="section-header">Calendar View of Spending</div>', unsafe_allow_html=True)
        
        # Create calendar view
        calendar_fig = create_calendar_view(df)
        st.plotly_chart(calendar_fig, use_container_width=True)
        
        # Weekly spending analysis
        st.markdown('<div class="section-header">Weekly Spending Analysis</div>', unsafe_allow_html=True)
        
        weekday_fig, weekday_mean, max_spending, min_spending = create_weekday_chart(df)
        if weekday_fig:
            st.plotly_chart(weekday_fig, use_container_width=True)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Daily Spending", f"₹{format_lakhs(weekday_mean)}")
            col2.metric("Highest Spending Day", f"₹{format_lakhs(max_spending)}")
            col3.metric("Lowest Spending Day", f"₹{format_lakhs(min_spending)}")
            
            # Identify highest spending day
            daily_data = df.copy()
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            daily_data['amount_abs'] = daily_data['amount'].abs()
            daily_data['weekday'] = daily_data['date'].dt.day_name()
            
            weekday_spending = daily_data.groupby('weekday')['amount_abs'].sum().reset_index()
            max_day = weekday_spending.loc[weekday_spending['amount_abs'].idxmax()]
            
            st.markdown(f'<div class="financial-suggestion">Your highest spending day is {max_day["weekday"]} (₹{format_lakhs(max_day["amount_abs"])} in total). Consider planning your major purchases on lower spending days.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()