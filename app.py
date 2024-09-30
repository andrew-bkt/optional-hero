import os
import re
import nltk
from nltk.corpus import stopwords
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import gradio as gr

# Ensure the OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Download NLTK data (stopwords)
nltk.download('stopwords')

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate the Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# Function to fetch historical volatility (using historical volatility as a proxy)
def get_historical_volatility(stock_data, window=30):
    """Calculate historical volatility over a rolling window."""
    log_returns = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    volatility = log_returns.rolling(window).std() * np.sqrt(252)  # Annualize the volatility
    return volatility

# Strategy Parser Function
def parse_strategy(strategy_description):
    """
    Parses the natural language strategy description and returns a structured dictionary.
    """
    # Convert to lowercase
    description = strategy_description.lower()

    # Remove punctuation except for the '%' symbol
    description = re.sub(r'[^\w\s%]', '', description)

    # Tokenize using simple whitespace split
    tokens = description.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]

    # Initialize strategy parameters
    strategy = {
        'option_type': 'call',       # default to 'call'
        'condition': None,           # e.g., 'price_drop', 'price_rise'
        'threshold': None,           # percentage threshold
        'hold_until': 'expiration',  # default holding period
        'expiration': 'next_week'    # default expiration
    }

    # Identify option type
    if 'put' in filtered_tokens:
        strategy['option_type'] = 'put'
    elif 'call' in filtered_tokens:
        strategy['option_type'] = 'call'

    # Identify conditions and thresholds
    if 'drop' in filtered_tokens or 'down' in filtered_tokens or 'decrease' in filtered_tokens:
        strategy['condition'] = 'price_drop'
    elif 'rise' in filtered_tokens or 'up' in filtered_tokens or 'increase' in filtered_tokens:
        strategy['condition'] = 'price_rise'

    # Extract percentage threshold
    threshold_match = re.search(r'(\d+)\s*(%|percent)?', description)
    if threshold_match:
        strategy['threshold'] = int(threshold_match.group(1))

    # Identify holding period
    if 'following week' in description or 'next week' in description:
        strategy['hold_until'] = 'next_week_expiration'
    elif 'month' in description:
        strategy['hold_until'] = 'next_month_expiration'

    # Validate parsed strategy
    if strategy['condition'] is None or strategy['threshold'] is None:
        raise ValueError("Unable to parse strategy condition or threshold from the description.")

    return strategy

# Enhanced backtest function
def backtest_option_strategy_dynamic(ticker: str, strategy_description: str, start_date: str, end_date: str) -> str:
    try:
        # Parse the strategy description
        strategy = parse_strategy(strategy_description)
    except ValueError as e:
        return f"Error parsing strategy description: {e}"

    # Download historical stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        return f"No historical data available for {ticker} between {start_date} and {end_date}."

    # Use adjusted close prices
    stock_data['Adjusted Close'] = stock_data['Adj Close']

    # Calculate daily percentage change using adjusted close prices
    stock_data['Percent_Change'] = stock_data['Adjusted Close'].pct_change() * 100

    # Calculate historical volatility (30-day rolling window)
    stock_data['Volatility'] = get_historical_volatility(stock_data)

    # Fill any missing volatility values with the mean
    stock_data['Volatility'] = stock_data['Volatility'].fillna(stock_data['Volatility'].mean())

    # Risk-free rate (using a constant rate for simplicity)
    r = 0.01  # 1% annual risk-free rate

    # Transaction costs
    commission = 1.0  # $1 per trade
    slippage = 0.02  # Assume $0.02 slippage per share

    # Identify signal dates based on the condition
    if strategy['condition'] == 'price_drop':
        signal_dates = stock_data[stock_data['Percent_Change'] <= -strategy['threshold']].index
    elif strategy['condition'] == 'price_rise':
        signal_dates = stock_data[stock_data['Percent_Change'] >= strategy['threshold']].index
    else:
        return "Unsupported condition in strategy."

    if signal_dates.empty:
        return f"No instances where {ticker} met the condition between {start_date} and {end_date}."

    trades = []
    for signal_date in signal_dates:
        # Purchase date is the next trading day
        purchase_date = signal_date + pd.tseries.offsets.BDay(1)
        if purchase_date not in stock_data.index:
            continue  # Skip if no data for next day

        # Determine expiration date
        if strategy['hold_until'] == 'next_week_expiration':
            expiration_date = purchase_date + pd.tseries.offsets.Week(weekday=4)
        elif strategy['hold_until'] == 'next_month_expiration':
            expiration_date = (purchase_date + pd.tseries.offsets.BMonthEnd(1))
        else:
            expiration_date = purchase_date + pd.Timedelta(days=7)  # Default to 1 week

        if expiration_date not in stock_data.index:
            continue  # Skip if no data for expiration date

        # Option parameters
        S = stock_data.loc[purchase_date]['Adjusted Close']  # Underlying stock price at purchase
        K = S  # ATM option
        T = (expiration_date - purchase_date).days / 365.0  # Time to expiration in years
        sigma = stock_data.loc[purchase_date]['Volatility']  # Historical volatility
        option_type = strategy['option_type']

        # Calculate option premium using Black-Scholes model
        try:
            purchase_price = black_scholes(S, K, T, r, sigma, option_type)
        except Exception as e:
            continue  # Skip if pricing fails

        # Adjust for slippage and commission
        purchase_price += slippage + commission

        # At expiration
        S_expiration = stock_data.loc[expiration_date]['Adjusted Close']
        if option_type == 'call':
            intrinsic_value = max(S_expiration - K, 0)
        else:  # put option
            intrinsic_value = max(K - S_expiration, 0)

        sell_price = intrinsic_value - slippage - commission  # Adjust for costs

        profit = sell_price - purchase_price
        trades.append({
            'Purchase Date': purchase_date.strftime('%Y-%m-%d'),
            'Expiration Date': expiration_date.strftime('%Y-%m-%d'),
            'Strike Price': K,
            'Purchase Price': purchase_price,
            'Sell Price': sell_price,
            'Profit': profit
        })

    if not trades:
        return f"No valid trades could be simulated for {ticker} between {start_date} and {end_date}."

    # Summarize results
    df_trades = pd.DataFrame(trades)
    total_profit = df_trades['Profit'].sum()
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['Profit'] > 0].shape[0]
    losing_trades = df_trades[df_trades['Profit'] <= 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    average_profit = df_trades['Profit'].mean()

    result = (
        f"Backtesting Results for Strategy on {ticker} from {start_date} to {end_date}:\n"
        f"- Strategy Description: {strategy_description}\n"
        f"- Total Trades: {total_trades}\n"
        f"- Winning Trades: {winning_trades}\n"
        f"- Losing Trades: {losing_trades}\n"
        f"- Win Rate: {win_rate:.2f}%\n"
        f"- Total Profit: ${total_profit:.2f}\n"
        f"- Average Profit per Trade: ${average_profit:.2f}\n"
    )

    # Include detailed trades if desired
    # result += "\nDetailed Trades:\n" + df_trades.to_string(index=False)

    return result

# Updated wrapper function
def backtest_option_strategy_dynamic_wrapper(input_str: str) -> str:
    try:
        # Remove any leading/trailing quotes from the input string
        input_str = input_str.strip('\'"')
        # Split the input string into parts
        parts = [s.strip() for s in input_str.split(',', 3)]
        if len(parts) < 4:
            return (
                "Invalid input format. Please provide input in the format: "
                "'ticker(s), strategy_description, start_date, end_date'."
            )
        ticker = parts[0]
        start_date = parts[-2]
        end_date = parts[-1]
        strategy_description = parts[1]
        # Handle multiple tickers separated by semicolons
        tickers = [t.strip('\'"') for t in ticker.split(';')]
        # Validate dates if necessary
    except Exception as e:
        return (
            f"Error parsing input: {e}\n"
            "Invalid input format. Please provide input in the format: "
            "'ticker(s), strategy_description, start_date, end_date'."
        )

    results = []
    for t in tickers:
        result = backtest_option_strategy_dynamic(t, strategy_description, start_date, end_date)
        results.append(result)

    return '\n'.join(results)

# Create tools
tools = [
    Tool(
        name="Backtest Custom Option Strategy",
        func=backtest_option_strategy_dynamic_wrapper,
        description=(
            "Use this tool to perform backtesting on any custom stock option strategy described in natural language. "
            "The input should be in the format: 'ticker(s), strategy_description, start_date, end_date'. "
            "For example: 'TSLA;AAPL, purchase put options after a 3% rise, 2020-01-01, 2021-12-31'. "
            "Dates should be in 'YYYY-MM-DD' format. Multiple tickers should be separated by semicolons."
        ),
    ),
]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Define the Gradio function
def chatbot(input_text):
    response = agent.run(input_text)
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=5, label="Enter your strategy query:"),
    outputs=gr.Textbox(label="Response:")
)

# Launch the app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))
)