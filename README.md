# Agent-Based Modeling

This repository contains Python scripts for simulating agent-based models in a financial market. The models simulate the behavior of agents with different levels of risk aversion and their impact on the price dynamics of a financial instrument.

## Files

- `agent_based_modeling.py`: Simulates agent-based modeling with market noise.
- `agent_based_modeling_with_probabilities.py`: Simulates agent-based modeling with probabilistic decision-making.

## Simulation Parameters

- `n_agents`: Number of agents in the simulation.
- `n_steps`: Number of time steps in the simulation.
- `initial_price`: Initial price of the financial instrument.

## Agent Initialization

- `wealth`: Initial wealth for each agent, randomly assigned between 1000 and 10000.
- `risk_aversion`: Risk aversion for each agent, randomly assigned between 0 (risk-seeking) and 1 (risk-averse).
- `cash`: Initial cash for each agent, equal to their initial wealth.
- `holdings`: Initial holdings of the financial instrument, set to zero for all agents.

## Market Initialization

- `price`: Array to store the price of the financial instrument at each time step, initialized with the initial price.

## Simulation Loop

For each time step:
1. Agents make buy or sell decisions based on their risk aversion and expected return.
2. The price is updated based on the supply and demand imbalance.
3. The wealth of each agent is updated based on their cash and holdings.

## Plotting Results

The simulation results are plotted using `matplotlib`:
- Price dynamics over time.
- Wealth distribution at the end of the simulation.
- Wealth distribution for risk-averse and risk-seeking agents.

## Usage

To run the simulations, execute the Python scripts:

```sh
python agent_based_modeling.py
```