import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
import yfinance as yf
from dotenv import load_dotenv
import os 

load_dotenv()

# Get API key from Streamlit secrets
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Enhanced stock symbol mappings
COMMON_STOCKS = {
    # US Stocks
    'NVIDIA': 'NVDA',
    'APPLE': 'AAPL',
    'GOOGLE': 'GOOGL',
    'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA',
    'AMAZON': 'AMZN',
    'META': 'META',
    'NETFLIX': 'NFLX',
    # Indian Stocks - NSE
    'TCS': 'TCS.NS',
    'RELIANCE': 'RELIANCE.NS',
    'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HDFC': 'HDFCBANK.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'MARUTI': 'MARUTI.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'HCLTECH': 'HCLTECH.NS',
    'ITC': 'ITC.NS',
    'AXISBANK': 'AXISBANK.NS', 
    'ABBOTT': 'ABBOTINDIA.NS',
}

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stock-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 10px;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateX(5px);
    }
    .stButton>button {
        width: 100%;
    }
    .market-indicator {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.watchlist = set()
    st.session_state.analysis_history = []
    st.session_state.last_refresh = None

def initialize_agents():
    """Initialize all agent instances with improved error handling"""
    if not st.session_state.agents_initialized:
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for the information",
                model=Groq(api_key=GROQ_API_KEY, id = 'llama3-groq-70b-8192-tool-use-preview'),
                tools=[
                    GoogleSearch(fixed_language='english', fixed_max_results=5),
                    DuckDuckGo(fixed_max_results=5, search = True, news = True)
                ],
                instructions=['Always include sources and verification'],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY, id = 'llama3-groq-70b-8192-tool-use-preview'),
                tools=[
                    YFinanceTools(
                        stock_price=True,
                        company_info = True, 
                        income_statements = True,
                        key_financial_ratios = True,
                        analyst_recommendations=True,
                        company_news = True,
                        technical_indicators = True,
                        historical_prices = True,
                        stock_fundamentals=True
                    )
                ],
                instructions=["Provide detailed analysis with data visualization"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.multi_ai_agent = Agent(
                name='A Stock Market Agent',
                role='A comprehensive assistant specializing in stock market analysis',
                model=Groq(api_key=GROQ_API_KEY, id = 'llama3-groq-70b-8192-tool-use-preview'),
                team=[st.session_state.web_agent, st.session_state.finance_agent],
                instructions=["Provide comprehensive analysis with multiple data sources"],
                show_tool_calls=True,
                markdown=True
            )

            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return False

def get_symbol_from_name(stock_name):
    """Enhanced function to fetch stock symbol from full stock name"""
    try:
        # Clean up input
        stock_name = stock_name.strip().upper()
        
        # First check if it's in our common stocks dictionary
        if stock_name in COMMON_STOCKS:
            return COMMON_STOCKS[stock_name]
        
        # Check if it's already a valid symbol
        ticker = yf.Ticker(stock_name)
        try:
            info = ticker.info
            if info and 'symbol' in info:
                return stock_name
        except:
            pass
        
        # Try Indian stock market (NSE)
        try:
            indian_symbol = f"{stock_name}.NS"
            ticker = yf.Ticker(indian_symbol)
            info = ticker.info
            if info and 'symbol' in info:
                return indian_symbol
        except:
            # Try BSE
            try:
                bse_symbol = f"{stock_name}.BO"
                ticker = yf.Ticker(bse_symbol)
                info = ticker.info
                if info and 'symbol' in info:
                    return bse_symbol
            except:
                pass
        
        st.error(f"Could not find valid symbol for {stock_name}")
        return None
    except Exception as e:
        st.error(f"Error processing {stock_name}: {str(e)}")
        return None

def calculate_z_score(hist_data):
    """Calculate z-score for the close prices."""
    mean_close = hist_data['Close'].mean()
    std_close = hist_data['Close'].std()
    hist_data['z_score'] = (hist_data['Close'] - mean_close) / std_close
    return hist_data

def calculate_ema(hist_data, period=50):
    """Calculate the Exponential Moving Average (EMA)."""
    hist_data[f'{period}_EMA'] = hist_data['Close'].ewm(span=period, adjust=False).mean()
    return hist_data

def get_stock_data(symbol, period="1y"):
    """Enhanced function to fetch stock data with proper cache handling"""
    try:
        # Create a new ticker instance
        stock = yf.Ticker(symbol)
        
        # Fetch data with error handling
        try:
            info = stock.info
            if not info:
                raise ValueError("No data retrieved for symbol")
        except Exception as info_error:
            # If .NS suffix is missing for Indian stocks, try adding it
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                try:
                    indian_symbol = f"{symbol}.NS"
                    stock = yf.Ticker(indian_symbol)
                    info = stock.info
                    symbol = indian_symbol
                except:
                    # Try Bombay Stock Exchange
                    try:
                        bse_symbol = f"{symbol}.BO"
                        stock = yf.Ticker(bse_symbol)
                        info = stock.info
                        symbol = bse_symbol
                    except:
                        raise info_error
            else:
                raise info_error

        # Fetch historical data
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
        
        if hist.empty:
            raise ValueError("No historical data available")
         
        
        # Calculate z-score and 50 EMA
        hist = calculate_z_score(hist)
        hist = calculate_ema(hist, period=50) 
        hist = calculate_rsi(hist, window=14) 

        return info, hist
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def create_price_chart(hist_data, symbol):
    """Create an interactive price chart using plotly"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))
    # Add 50 EMA
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['50_EMA'],
        mode='lines',
        name='50 EMA',
        line=dict(color='purple')
    ))
    # Add moving averages
    ma20 = hist_data['Close'].rolling(window=20).mean()
    ma50 = hist_data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma20, name='20 Day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, name='50 Day MA', line=dict(color='blue')))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def create_volume_chart(hist_data):
    """Create enhanced volume chart using plotly"""
    # Calculate volume moving average
    volume_ma = hist_data['Volume'].rolling(window=20).mean()
    
    fig = go.Figure()
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
        name='Volume',
        marker_color='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add volume moving average
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=volume_ma,
        name='20 Day Volume MA',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Trading Volume Analysis',
        yaxis_title='Volume',
        template='plotly_white',
        height=400
    )
    
    return fig

def format_large_number(number):
    """Format large numbers into readable format"""
    if number >= 1e12:
        return f"${number/1e12:.2f}T"
    elif number >= 1e9:
        return f"${number/1e9:.2f}B"
    elif number >= 1e6:
        return f"${number/1e6:.2f}M"
    else:
        return f"${number:,.2f}"

def display_metrics(info):
    """Display enhanced key metrics in a grid"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = format_large_number(market_cap)
        st.metric("Market Cap", market_cap)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pe_ratio = info.get('trailingPE', 'N/A')
        if pe_ratio != 'N/A':
            pe_ratio = f"{pe_ratio:.2f}"
        st.metric("P/E Ratio", pe_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        high = info.get('fiftyTwoWeekHigh', 'N/A')
        if high != 'N/A':
            high = f"${high:.2f}"
        st.metric("52 Week High", high)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        low = info.get('fiftyTwoWeekLow', 'N/A')
        if low != 'N/A':
            low = f"${low:.2f}"
        st.metric("52 Week Low", low)
        st.markdown('</div>', unsafe_allow_html=True)

def calculate_rsi(hist_data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = hist_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    hist_data['RSI'] = 100 - (100 / (1 + rs))
    return hist_data

def create_rsi_chart(hist_data, symbol):
    """Create an RSI chart using Plotly."""
    fig = go.Figure()

    # Add RSI line
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='gray')
    ))

    # Add overbought and oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")

    fig.update_layout(
        title=f'{symbol} RSI (14)',
        yaxis_title='RSI',
        template='plotly_white',
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_rsi_chart_with_trendline(hist_data, symbol):
    """Create an RSI chart with a rising trendline."""
    fig = go.Figure()

    # Add RSI line
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='gray')
    ))

    # Add overbought and oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")

    # Identify and add a rising trendline manually (customize points as needed)
    rising_points = hist_data.iloc[[10, 25, 40]]  # Indices of specific rising highs
    fig.add_trace(go.Scatter(
        x=rising_points.index,
        y=rising_points['RSI'],
        mode='lines+markers',
        name='Rising RSI Trendline',
        line=dict(color='orange', dash='dot'),
        marker=dict(size=8, color='orange')
    ))

    # Annotate the trendline
    fig.add_annotation(
        x=rising_points.index[1],
        y=rising_points['RSI'].iloc[1],
        text="Rising RSI highs",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(size=12, color="black"),
        arrowcolor="orange"
    )

    fig.update_layout(
        title=f'{symbol} RSI (14) with Trendline',
        yaxis_title='RSI',
        template='plotly_white',
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def main():
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Options")
        analysis_type = st.selectbox(
            "Choose Analysis Type",
            ["Comprehensive Analysis", "Technical Analysis", "Fundamental Analysis", 
             "News Analysis", "Sentiment Analysis"]
        )
        
        # Add market selection
        market = st.selectbox(
            "Select Market",
            ["US Market", "Indian Market (NSE)", "Indian Market (BSE)"]
        )
        
        st.markdown("---")
        
        # Watchlist section
        st.markdown("### 📋 Watchlist")
        watchlist_symbol = st.text_input("Add to Watchlist", 
                                       help="Enter stock name or symbol (e.g., NVIDIA, TCS)")
        if st.button("Add to Watchlist"):
            symbol = get_symbol_from_name(watchlist_symbol)
            if symbol:
                st.session_state.watchlist.add(symbol)
                st.success(f"Added {symbol} to watchlist")
        
        if st.session_state.watchlist:
            st.markdown("#### Current Watchlist")
            for symbol in st.session_state.watchlist:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(symbol)
                with col2:
                    if st.button("❌", key=f"remove_{symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        st.experimental_rerun()

    # Main content
    st.markdown('<h1 class="stock-header">🤖 Stock Market Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Search and Analysis Section
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_input = st.text_input(
            "Enter Stock Name or Symbol",
            help="Enter company name (e.g., NVIDIA) or symbol (e.g., NVDA)"
        )
    with col2:
        date_range = st.selectbox(
            "Select Time Range",
            ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], key="time_range"
        )
        # Convert selected range to yfinance period format
        period_map = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "5 Years": "5y"
        }
        period = period_map[date_range]

    if st.button("Analyze", type="primary"):
        if not stock_input:
            st.error("Please enter a stock name or symbol.")
            return

        # Convert input to symbol
        stock_symbol = get_symbol_from_name(stock_input)
        if stock_symbol:
            try:
                # Initialize agents
                if initialize_agents():
                    # Show loading spinner
                    with st.spinner(f"Analyzing {stock_symbol}..."):
                        # Fetch fresh stock data
                        info, hist = get_stock_data(stock_symbol, period=period)
                        #st.write(info)
                        
                        if info and hist is not None:
                            # Display market status
                            market_status = "🟢 Market Open" if info.get('regularMarketOpen') else "🔴 Market Closed"
                            st.markdown(f"<div class='market-indicator'>{market_status}</div>", unsafe_allow_html=True)
                            
                            # Create tabs for different sections
                            overview_tab, charts_tab, analysis_tab, news_tab, other_info = st.tabs([
                                "Overview", "Charts", "AI Analysis", "News", "Other Information"
                            ])
                            
                            with other_info:
                                st.markdown("### Company's Entire Information")
                                # # Address and Contact Details
                                # st.markdown("#### Address and Contact")
                                # st.write(f"Address 1: {info.get('address1', 'N/A')}")
                                # st.write(f"Address 2: {info.get('address2', 'N/A')}")
                                # st.write(f"City: {info.get('city', 'N/A')}")
                                # st.write(f"Zip Code: {info.get('zip', 'N/A')}")
                                # st.write(f"Country: {info.get('country', 'N/A')}")
                                # st.write(f"Phone: {info.get('phone', 'N/A')}")
                                # st.write(f"Fax: {info.get('fax', 'N/A')}")
                                # st.write(f"Website: {info.get('website', 'N/A')}")
                                
                                # # Industry and Sector Details
                                # st.markdown("#### Industry and Sector")
                                # st.write(f"Industry: {info.get('industry', 'N/A')} ({info.get('industryDisp', 'N/A')})")
                                # st.write(f"Sector: {info.get('sector', 'N/A')} ({info.get('sectorDisp', 'N/A')})")
                                
                                # # Business Summary
                                # st.markdown("#### Business Summary")
                                # st.write(info.get('longBusinessSummary', 'N/A'))
                                
                                # # Company Officers
                                # st.markdown("#### Company Officers")
                                # for officer in info.get('companyOfficers', []):
                                #     st.write(f"- **Name**: {officer.get('name', 'N/A')}")
                                #     st.write(f"  - Title: {officer.get('title', 'N/A')}")
                                #     st.write(f"  - Age: {officer.get('age', 'N/A')}")
                                #     st.write(f"  - Total Pay: {officer.get('totalPay', 'N/A')}")
                                #     st.write(f"  - Fiscal Year: {officer.get('fiscalYear', 'N/A')}")
                                
                                # Financial Information
                                st.markdown("#### Financial Information")
                                st.write(f"Market Cap: {info.get('marketCap', 'N/A'):,}")
                                st.write(f"Enterprise Value: {info.get('enterpriseValue', 'N/A'):,}")
                                st.write(f"Revenue: {info.get('totalRevenue', 'N/A'):,}")
                                st.write(f"Net Income: {info.get('netIncomeToCommon', 'N/A'):,}")
                                st.write(f"EBITDA: {info.get('ebitda', 'N/A'):,}")
                                st.write(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
                                st.write(f"Beta: {info.get('beta', 'N/A')}")
                                
                                # Dividend Details
                                st.markdown("#### Dividend Information")
                                st.write(f"Dividend Rate: {info.get('dividendRate', 'N/A')}")
                                st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A'):.2%}")
                                st.write(f"Ex-Dividend Date: {datetime.fromtimestamp(info.get('exDividendDate', 0)).strftime('%Y-%m-%d') if info.get('exDividendDate') else 'N/A'}")
                                st.write(f"Five-Year Average Dividend Yield: {info.get('fiveYearAvgDividendYield', 'N/A'):.2%}")
                                
                                # Risk and Governance
                                st.markdown("#### Risk and Governance")
                                st.write(f"Audit Risk: {info.get('auditRisk', 'N/A')}")
                                st.write(f"Board Risk: {info.get('boardRisk', 'N/A')}")
                                st.write(f"Compensation Risk: {info.get('compensationRisk', 'N/A')}")
                                st.write(f"Shareholder Rights Risk: {info.get('shareHolderRightsRisk', 'N/A')}")
                                st.write(f"Overall Risk: {info.get('overallRisk', 'N/A')}")
                                
                                # Stock Performance
                                st.markdown("#### Stock Performance")
                                st.write(f"Current Price: {info.get('currentPrice', 'N/A')}")
                                st.write(f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
                                st.write(f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
                                st.write(f"Average Volume (10 days): {info.get('averageVolume10days', 'N/A'):,}")
                                st.write(f"Target Mean Price: {info.get('targetMeanPrice', 'N/A')}")
                                st.write(f"Recommendation Key: {info.get('recommendationKey', 'N/A')}")

                            with overview_tab:
                                # Display company info
                                st.markdown("### Company Overview")
                                st.write(info.get('longBusinessSummary', 'No description available.'))
                                
                                # Display key metrics
                                st.markdown("### Key Metrics")
                                display_metrics(info)
                                
                                # Additional company information
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### Company Details")
                                    st.write(f"Sector: {info.get('sector', 'N/A')}")
                                    st.write(f"Industry: {info.get('industry', 'N/A')}")
                                    st.write(f"Country: {info.get('country', 'N/A')}")
                                    st.write(f"Employees: {info.get('fullTimeEmployees', 'N/A'):,}")
                                
                                with col2:
                                    st.markdown("### Trading Information")
                                    st.write(f"Exchange: {info.get('exchange', 'N/A')}")
                                    st.write(f"Currency: {info.get('currency', 'N/A')}")
                                    st.write(f"Volume: {info.get('volume', 'N/A'):,}")
                                    st.markdown("### Z-Score Analysis")
                                    latest_z_score = hist['z_score'].iloc[-1]
                                    st.metric("Latest z-Score", f"{latest_z_score:.2f}")

                                    # Provide interpretation
                                    # if abs(latest_z_score) > 2:
                                    #     st.warning("The z-score indicates the price is significantly far from the average.")
                                    # else:
                                    #     st.info("The z-score indicates the price is within the normal range.")
                            
                            with charts_tab:
                                # Price chart
                                st.markdown("### Price & RSI Analysis")
                                price_chart = create_price_chart(hist, stock_symbol)
                                st.plotly_chart(price_chart, use_container_width=True)                              
                                
                                rsi_chart = create_rsi_chart_with_trendline(hist, stock_symbol)
                                st.plotly_chart(rsi_chart, use_container_width=True)

                                # Volume chart
                                volume_chart = create_volume_chart(hist)
                                st.plotly_chart(volume_chart, use_container_width=True)
                                
                                # Technical indicators
                                st.markdown("### Technical Indicators")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    rsi = hist['Close'].diff()
                                    rsi_pos = rsi.copy()
                                    rsi_neg = rsi.copy()
                                    rsi_pos[rsi_pos < 0] = 0
                                    rsi_neg[rsi_neg > 0] = 0
                                    rsi_14_pos = rsi_pos.rolling(window=14).mean()
                                    rsi_14_neg = abs(rsi_neg.rolling(window=14).mean())
                                    rsi_14 = 100 - (100 / (1 + rsi_14_pos / rsi_14_neg))
                                    st.metric("RSI (14)", f"{rsi_14.iloc[-1]:.2f}")
                                    st.metric("Latest z-Score", f"{hist['z_score'].iloc[-1]:.2f}")
                                
                                with col2:
                                    ma20 = hist['Close'].rolling(window=20).mean()
                                    ma50 = hist['Close'].rolling(window=50).mean()
                                    cross_signal = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"
                                    st.metric("MA Cross Signal", cross_signal)
                                    st.metric("50 EMA (Latest)", f"{hist['50_EMA'].iloc[-1]:.2f}")
                                
                                with col3:
                                    volatility = hist['Close'].pct_change().std() * (252 ** 0.5) * 100
                                    st.metric("Annualized Volatility", f"{volatility:.2f}%")
                            
                            with analysis_tab:
                                # AI Analysis
                                st.markdown("### AI-Powered Analysis")
                                query = f"Provide a {analysis_type.lower()} for {stock_symbol}."
                                response = st.session_state.multi_ai_agent.print_response(query, stream=True)
                                
                                # Add to analysis history
                                st.session_state.analysis_history.append({
                                    'symbol': stock_symbol,
                                    'timestamp': datetime.now(),
                                    'analysis_type': analysis_type
                                })
                            
                            with news_tab:
                                st.markdown("### Latest News")
                                if 'news' in info:
                                    for news_item in info['news'][:5]:
                                        with st.container():
                                            st.markdown(f"""
                                            <div class="news-card">
                                                <h4>{news_item['title']}</h4>
                                                <p>{news_item['summary']}</p>
                                                <small>Source: {news_item.get('source', 'Unknown')} | 
                                                {datetime.fromtimestamp(news_item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}</small>
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.write("No recent news available")
                            
                            # Add refresh button - Experimental Rerun is not working 
                            # if st.button("🔄 Refresh Data"):
                            #     st.session_state.last_refresh = datetime.now()
                            #     st.experimental_rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### Recent Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(history_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This advanced stock market analysis tool combines:
        - Real-time market data analysis
        - AI-powered insights and predictions
        - Technical and fundamental analysis
        - News and sentiment analysis
        - Interactive charts and visualizations
        
        Features:
        - Support for both US and Indian markets (NSE/BSE)
        - Company name and symbol resolution
        - Watchlist management
        - Multiple timeframe analysis
        - Technical indicators
        
        Use the sidebar to configure your analysis preferences and manage your watchlist.
    """)
    
    # Display last refresh time if available
    if st.session_state.last_refresh:
        st.markdown(f"<div class='market-indicator'>Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
