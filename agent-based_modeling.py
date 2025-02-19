import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_agents = 1000  # Number of agents
n_steps = 100    # Number of time steps
initial_price = 100  # Initial price of the financial instrument

# Initialize agents
wealth = np.random.uniform(1000, 10000, n_agents)  # Initial wealth for each agent
risk_aversion = np.random.uniform(0, 1, n_agents)  # Risk aversion for each agent (0 = risk-seeking, 1 = risk-averse)
cash = wealth.copy()  # All wealth starts as cash
holdings = np.zeros(n_agents)  # No initial holdings of the financial instrument

# Initialize market
price = np.zeros(n_steps)  # Array to store price at each time step
price[0] = initial_price  # Set initial price

# Simulation loop
for t in range(1, n_steps):
    # Agents make decisions
    buy_orders = 0
    sell_orders = 0
    
    for i in range(n_agents):
        # Simple trend-following rule: expected return based on past price movement
        if t > 1 and price[t-2] != 0:
            expected_return = (price[t-1] - price[t-2]) / price[t-2]
        else:
            expected_return = 0  # No trend in the first step or if price[t-2] is zero
        
        # Add a small random factor to simulate market noise
        noise = np.random.normal(0, 0.01)
        
        # Risk-averse agents are less likely to buy or sell
        if risk_aversion[i] > 0.5:
            if expected_return + noise > 0 and cash[i] >= price[t-1]:  # Buy if expected return is positive and has enough cash
                buy_orders += 1
                holdings[i] += 1
                cash[i] -= price[t-1]
            elif expected_return + noise < 0 and holdings[i] > 0:  # Sell if expected return is negative and has holdings
                sell_orders += 1
                holdings[i] -= 1
                cash[i] += price[t-1]
        else:  # Risk-seeking agents are more likely to buy or sell
            if expected_return + noise > 0 and cash[i] >= price[t-1]:  # Buy if expected return is positive and has enough cash
                buy_orders += 1
                holdings[i] += 1
                cash[i] -= price[t-1]
            elif expected_return + noise < 0 and holdings[i] > 0:  # Sell if expected return is negative and has holdings
                sell_orders += 1
                holdings[i] -= 1
                cash[i] += price[t-1]
    
    # Update price based on supply and demand imbalance
    imbalance = buy_orders - sell_orders
    price[t] = price[t-1] * (1 + 0.01 * imbalance / n_agents)  # Normalize by number of agents
    
    # Print the number of buy and sell orders
    print(f"Time step {t}: Buy orders = {buy_orders}, Sell orders = {sell_orders}, Imbalance = {imbalance}")
    
    # Update agent wealth
    wealth = cash + holdings * price[t]  # Wealth = cash + (holdings × current price)

# Plot results
plt.figure(figsize=(12, 6))

# Plot price dynamics
plt.subplot(1, 2, 1)
plt.plot(range(n_steps), price, label='Price', color='b', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Price Dynamics', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot wealth distribution at the end of the simulation
plt.subplot(1, 2, 2)
plt.hist(wealth, bins=30, color='g', alpha=0.7)
plt.xlabel('Wealth', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Wealth Distribution at t={}'.format(n_steps), fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Analysis of agents with different parameters
risk_averse_agents = wealth[risk_aversion > 0.5]
risk_seeking_agents = wealth[risk_aversion <= 0.5]

plt.figure(figsize=(12, 6))

# Plot wealth distribution for risk-averse agents
plt.subplot(1, 2, 1)
plt.hist(risk_averse_agents, bins=30, color='r', alpha=0.7)
plt.xlabel('Wealth', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Wealth Distribution of Risk-Averse Agents', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot wealth distribution for risk-seeking agents
plt.subplot(1, 2, 2)
plt.hist(risk_seeking_agents, bins=30, color='b', alpha=0.7)
plt.xlabel('Wealth', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Wealth Distribution of Risk-Seeking Agents', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()