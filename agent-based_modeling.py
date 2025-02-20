import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Parameters
n_agents = 1000  # Number of agents
n_steps = 1000    # Number of time steps
initial_price = 100  # Initial price of the financial instrument

# Function to generate truncated normal distribution
def truncated_normal(mean=0.5, std=0.1, low=0, upp=1, size=1):
    return truncnorm(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std).rvs(size)

# Initialize agents
wealth = np.random.lognormal(mean=np.log(5000), sigma=0.2, size=n_agents)
# Initial wealth for each agent
#for testing:
#wealth = np.full(n_agents, 5000)  # Initial wealth for each agent set to 5000
risk_aversion = truncated_normal(mean=0.5, std=0.1, low=0, upp=1, size=n_agents)  # Risk aversion for each agent (0 = risk-seeking, 1 = risk-averse)
#for testing:
#risk_aversion = np.full(n_agents, 0.5)  # Risk aversion for each agent set to 0.5
cash = wealth.copy()  # All wealth starts as cash
holdings = np.zeros(n_agents)  # No initial holdings of the financial instrument

# Create a DataFrame to store agent data
agent_data = pd.DataFrame({
    'Agent': np.arange(n_agents),
    'Initial Wealth': wealth,
    'Risk Aversion': risk_aversion,
    'Buys': np.zeros(n_agents, dtype=int),
    'Sales': np.zeros(n_agents, dtype=int)
})

# Initialize market
price = np.zeros(n_steps)  # Array to store price at each time step
price[0] = initial_price  # Set initial price
price_changes = np.zeros(n_steps-1)  # Track price changes for volatility calculations


# To include probabilities that depend on wealth and risk aversion 
# instead of using thresholds, you can use a logistic function 
# to determine the probability of buying or selling. 
# This way, the decision to buy or sell will be based on a continuous 
# probability distribution influenced by the agent's wealth and risk aversion.
def logistic(x):
    return 1 / (1 + np.exp(-x))

for t in range(1, n_steps):
    buy_orders = 0
    sell_orders = 0
    
    for i in range(n_agents):
        # Calculate expected return
        if t > 1 and price[t-2] != 0:
            expected_return = (price[t-1] - price[t-2]) / price[t-2]
        else:
            expected_return = 0  # No trend in the first step or if price[t-2] is zero
        
        # Add a small random factor to simulate market noise
        noise = np.random.normal(0, 0.01)
        
        # Calculate probabilities based on initial wealth and risk aversion
        buy_prob = logistic(0.02 + (wealth[i] - 1000) / 9000 * 0.02 - risk_aversion[i] * 0.6 + expected_return + noise)
        sell_prob = logistic(-0.01 + (wealth[i] - 1000) / 9000 * 0.01 + risk_aversion[i] * 0.2 - expected_return + noise)
        
        # Decision to buy or sell based on probabilities
        if np.random.rand() < buy_prob and cash[i] >= price[t-1]:
            buy_orders += 1
            holdings[i] += 1
            cash[i] -= price[t-1]
            agent_data.at[i, 'Buys'] += 1
        elif np.random.rand() < sell_prob and holdings[i] > 0:
            sell_orders += 1
            holdings[i] -= 1
            cash[i] += price[t-1]
            agent_data.at[i, 'Sales'] += 1
    
    # Update price based on supply and demand imbalance
    imbalance = buy_orders - sell_orders
    price[t] = price[t-1] * (1 + 0.01 * imbalance / n_agents)  # Normalize by number of agents
    price_changes[t-1] = price[t] - price[t-1]  # Track price change
    
    # Print the number of buy and sell orders
    print(f"Time step {t}: Buy orders = {buy_orders}, Sell orders = {sell_orders}, Imbalance = {imbalance}")
    
    # Update agent wealth
    wealth = cash + holdings * price[t]  # Wealth = cash + (holdings × current price)

# Update agent_data with final wealth
agent_data['Final Wealth'] = wealth

# Compute volatility (standard deviation of price changes)
volatility = np.std(price_changes)

def f(): 
    """Visualize the results of the agent-based model."""
    # Plot initial wealth distribution
    plt.figure(figsize=(12, 6))
    plt.hist(agent_data['Initial Wealth'], bins=30, color='g', alpha=0.7)
    plt.xlabel('Initial Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Initial Wealth Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot risk aversion distribution
    plt.figure(figsize=(12, 6))
    plt.hist(agent_data['Risk Aversion'], bins=30, color='b', alpha=0.7)
    plt.xlabel('Risk Aversion', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Risk Aversion Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot results
    plt.figure(figsize=(15, 8))

    # Plot price dynamics
    plt.subplot(2, 2, 1)
    plt.plot(range(n_steps), price, label='Price', color='b', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('Price Dynamics', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot wealth distribution at the end of the simulation
    plt.subplot(2, 2, 2)
    plt.hist(wealth, bins=30, color='g', alpha=0.7)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Wealth Distribution at t={n_steps}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot price volatility
    plt.subplot(2, 2, 3)
    plt.plot(range(n_steps-1), price_changes, label='Price Change', color='r', alpha=0.7)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price Change', fontsize=12)
    plt.title(f'Price Volatility (Std Dev: {volatility:.2f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Compare final wealth distributions
    plt.subplot(2, 2, 4)
    plt.hist(agent_data[agent_data['Risk Aversion'] > 0.5]['Final Wealth'], bins=30, color='r', alpha=0.6, label='Risk-Averse')
    plt.hist(agent_data[agent_data['Risk Aversion'] <= 0.5]['Final Wealth'], bins=30, color='b', alpha=0.6, label='Risk-Seeking')
    plt.xlabel('Final Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Final Wealth Distribution by Risk Group', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Further categorize risk-averse agents
    low_risk_averse_agents = wealth[(risk_aversion > 0) & (risk_aversion <= 0.3)]
    medium_risk_averse_agents = wealth[(risk_aversion > 0.3) & (risk_aversion <= 0.7)]
    high_risk_averse_agents = wealth[(risk_aversion > 0.7) & (risk_aversion <= 1)]
    


    # Plot wealth distribution for low risk-averse agents
    plt.subplot(2, 3, 1)
    plt.hist(low_risk_averse_agents, bins=30, color='orange', alpha=0.7)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Wealth Distribution of Low Risk-Averse Agents', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot wealth distribution for medium risk-averse agents
    plt.subplot(2, 3, 2)
    plt.hist(medium_risk_averse_agents, bins=30, color='purple', alpha=0.7)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Wealth Distribution of Medium Risk-Averse Agents', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot wealth distribution for high risk-averse agents
    plt.subplot(2, 3, 3)
    plt.hist(high_risk_averse_agents, bins=30, color='green', alpha=0.7)
    plt.xlabel('Wealth', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Wealth Distribution of High Risk-Averse Agents', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Scatter plot of initial wealth vs final wealth
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Initial Wealth'], agent_data['Final Wealth'], alpha=0.5)
    plt.xlabel('Initial Wealth', fontsize=12)
    plt.ylabel('Final Wealth', fontsize=12)
    plt.title('Initial Wealth vs Final Wealth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Scatter plot of risk aversion vs final wealth
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Risk Aversion'], agent_data['Final Wealth'], alpha=0.5, color='r')
    plt.xlabel('Risk Aversion', fontsize=12)
    plt.ylabel('Final Wealth', fontsize=12)
    plt.title('Risk Aversion vs Final Wealth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Scatter plot of initial wealth vs number of buys
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Initial Wealth'], agent_data['Buys'], alpha=0.5, color='g')
    plt.xlabel('Initial Wealth', fontsize=12)
    plt.ylabel('Number of Buys', fontsize=12)
    plt.title('Initial Wealth vs Number of Buys', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Scatter plot of initial wealth vs number of sales
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Initial Wealth'], agent_data['Sales'], alpha=0.5, color='b')
    plt.xlabel('Initial Wealth', fontsize=12)
    plt.ylabel('Number of Sales', fontsize=12)
    plt.title('Initial Wealth vs Number of Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Scatter plot of risk aversion vs number of buys
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Risk Aversion'], agent_data['Buys'], alpha=0.5, color='purple')
    plt.xlabel('Risk Aversion', fontsize=12)
    plt.ylabel('Number of Buys', fontsize=12)
    plt.title('Risk Aversion vs Number of Buys', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Scatter plot of risk aversion vs number of sales
    plt.figure(figsize=(12, 6))
    plt.scatter(agent_data['Risk Aversion'], agent_data['Sales'], alpha=0.5, color='orange')
    plt.xlabel('Risk Aversion', fontsize=12)
    plt.ylabel('Number of Sales', fontsize=12)
    plt.title('Risk Aversion vs Number of Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

f()

# The scatter plots show the relationship between initial wealth, risk aversion,
# and the number of buys and sales made by each agent.
# The plots provide insights into how different factors influence agent behavior
# and final wealth outcomes. For example, agents with higher initial wealth tend
# to make more buys and sales, while risk-averse agents make fewer trades overall.
# The relationship between risk aversion and trading activity is also evident,
# with risk-seeking agents making more trades compared to risk-averse agents.
# These visualizations help in understanding the dynamics of the agent-based model
# and how individual agent characteristics impact market behavior and outcomes.

# The agent-based model simulates the behavior of multiple agents in a financial market
# and tracks their trading decisions based on initial wealth, risk aversion, and market conditions. 
# The model captures the dynamics of supply and demand for a financial instrument,
# leading to price changes and wealth redistribution among agents. By analyzing the results
# of the simulation, we can gain insights into how different factors influence agent behavior,
# market outcomes, and wealth distribution. The model can be further extended to incorporate
# additional features such as different trading strategies, market conditions, and agent interactions
# to explore more complex scenarios and study the impact of various factors on market dynamics.
# Overall, agent-based modeling provides a powerful framework for studying complex systems
# and understanding emergent phenomena in financial markets and other domains.

# Prepare the data for linear regression
X = agent_data[['Risk Aversion', 'Initial Wealth']]
y = agent_data['Buys']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Predict the number of buys using the linear regression model
y_pred = model.predict(X)

# Calculate R-squared
r_squared = model.score(X, y)

print(f"R-squared: {r_squared:.2f}")


# Linear regression for y = Buys and x = Wealth
X_wealth = agent_data[['Initial Wealth']]
model_wealth = LinearRegression()
model_wealth.fit(X_wealth, y)
coefficients_wealth = model_wealth.coef_
intercept_wealth = model_wealth.intercept_
r_squared_wealth = model_wealth.score(X_wealth, y)

print(f"Wealth Model Coefficients: {coefficients_wealth}")
print(f"Wealth Model Intercept: {intercept_wealth}")
print(f"Wealth Model R-squared: {r_squared_wealth:.2f}")

# Linear regression for y = Buys and x = Risk Aversion
X_risk_aversion = agent_data[['Risk Aversion']]
model_risk_aversion = LinearRegression()
model_risk_aversion.fit(X_risk_aversion, y)
coefficients_risk_aversion = model_risk_aversion.coef_
intercept_risk_aversion = model_risk_aversion.intercept_
r_squared_risk_aversion = model_risk_aversion.score(X_risk_aversion, y)

print(f"Risk Aversion Model Coefficients: {coefficients_risk_aversion}")
print(f"Risk Aversion Model Intercept: {intercept_risk_aversion}")
print(f"Risk Aversion Model R-squared: {r_squared_risk_aversion:.2f}")

