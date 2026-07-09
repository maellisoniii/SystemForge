# SystemForge
A modular energy systems modeling platform for evaluating solar-storage system designs under operational and economic uncertainty.

SystemForge combines deterministic optimization, economic analysis, and probabilistic simulation to evaluate how solar generation, battery storage, and electricity markets interact over time.

The platform is designed to support infrastructure planning, energy system analysis, and systems engineering research.

## Current Capabilities  
### Deterministic Energy System Simulation

SystemForge models hourly operation of an electrical system using:
- Building electrical demand
- Solar generation
- Battery storage
- Grid electricity
- Hourly electricity prices

For every simulated hour, the platform tracks:
- Grid imports
- Battery state of charge
- Battery charging/discharging
- Solar curtailment
- Annual grid cost
- Battery dispatch value

## Linear Programming Battery Dispatch

Battery operation is formulated as a linear optimization problem.
Decision variables include:

- Grid import
- Battery charging
- Battery discharging
- Solar curtailment
- Battery state of charge

The optimizer minimizes annual electricity cost while satisfying:
- Hourly energy balance
- Battery capacity limits
- Battery power limits
- Charge/discharge efficiency
- State-of-charge constraints

The deterministic optimizer assumes perfect foresight, making it appropriate as a planning benchmark.

## Design Space Exploration

Rather than evaluating only one system, SystemForge searches across combinations of:
- Solar penetration
- Storage duration

For every candidate design the platform computes:
- Annual grid cost
- Annualized battery cost
- Annualized solar cost
- Total annual cost
- Solar curtailment
- Grid dependence
- Battery dispatch savings

The lowest-cost configuration becomes the current optimal design.

## Economic Model

SystemForge annualizes infrastructure costs using the Capital Recovery Factor (CRF) rather than simple lifetime division.

Current economic assumptions include:
- Battery capital cost
- Solar capital cost
- Asset lifetime
- Discount rate

These assumptions are centralized in config.py, allowing rapid scenario analysis.

## Uncertainty Analysis

Future operating conditions are uncertain.
SystemForge evaluates uncertainty using Monte Carlo simulation.

Each scenario perturbs:
- Solar yield
- Electrical demand
- Electricity price
- Electricity price shift

Monte Carlo dispatch currently uses a threshold-based operating policy that approximates realistic battery behavior while remaining computationally efficient.

Outputs include:
- Expected annual cost
- Cost standard deviation
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Confidence intervals

## Software Architecture 
main.py 
├── config.py
├── data_profiles.py
├── simulation.py
├── optimization.py
├── design_search.py
├── monte_carlo.py
├── risk.py
├── reporting.py
└── plotting.py 


## Current Assumptions

Current implementation includes:
- Deterministic perfect-foresight dispatch
- Battery charge/discharge efficiency
- Battery power limits
- Capital Recovery Factor annualization
- Hourly energy balance
- Grid charging capability
- Solar curtailment

Current data source:
- Germany electrical load
- Germany solar generation
- Austrian day-ahead electricity prices

using the Open Power System Data project.

## Current Limitations

The current model intentionally simplifies several aspects of real power systems.

Examples include:
- Perfect information in deterministic dispatch
- Simplified battery degradation
- No transmission constraints
- Single-node electrical system
- Fixed design grid search
- Threshold-based Monte Carlo dispatch instead of repeated linear optimization




## Long-Term Vision 
SystemForge will become a portfolio of systems engineering tools focused on energy, sustainability, and optimized infrastructure. 

Potential future modules include:
- Microgrid planning
- Hydrogen systems
- Water infrastructure
- Transportation systems
- Industrial energy optimization
- Multi-energy network optimization



