import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
n_agents = 1000  # Number of agents
n_steps = 100    # Number of time steps
initial_price = 100  # Initial price of the financial instrument

# Initialize agents
#wealth = np.random.uniform(1000, 10000, n_agents)  # Initial wealth for each agent
#for testing:
wealth = np.full(n_agents, 5000)  # Initial wealth for each agent set to 5000
risk_aversion = np.random.uniform(0, 1, n_agents)  # Risk aversion for each agent (0 = risk-seeking, 1 = risk-averse)
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
        
        # Linear thresholds based on initial wealth and risk aversion
        buy_threshold = 0.02 - (wealth[i] - 1000) / 9000 * 0.02 - risk_aversion[i] * 0.01
        sell_threshold = -0.01 + (wealth[i] - 1000) / 9000 * 0.01 + risk_aversion[i] * 0.01
        
        # Decision to buy or sell
        if expected_return + noise > buy_threshold and cash[i] >= price[t-1]:
            buy_orders += 1
            holdings[i] += 1
            cash[i] -= price[t-1]
            agent_data.at[i, 'Buys'] += 1
        elif expected_return + noise < sell_threshold and holdings[i] > 0:
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
low_risk_averse_agents = wealth[(risk_aversion > 0) & (risk_aversion <= 0.33)]
medium_risk_averse_agents = wealth[(risk_aversion > 0.33) & (risk_aversion <= 0.66)]
high_risk_averse_agents = wealth[risk_aversion > 0.66]

plt.figure(figsize=(18, 12))

# Plot wealth distribution for low risk-averse agents
plt.subplot(2, 3, 3)
plt.hist(low_risk_averse_agents, bins=30, color='orange', alpha=0.7)
plt.xlabel('Wealth', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Wealth Distribution of Low Risk-Averse Agents', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot wealth distribution for medium risk-averse agents
plt.subplot(2, 3, 4)
plt.hist(medium_risk_averse_agents, bins=30, color='purple', alpha=0.7)
plt.xlabel('Wealth', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Wealth Distribution of Medium Risk-Averse Agents', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot wealth distribution for high risk-averse agents
plt.subplot(2, 3, 5)
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