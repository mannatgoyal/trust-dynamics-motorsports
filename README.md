# F1 Race Strategy Optimizer

**Game Theory & Machine Learning for Formula 1 Race Analysis**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Code Structure](#code-structure)
- [Results & Analysis](#results--analysis)
- [Driver & Strategy Recommendations](#driver--strategy-recommendations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

F1 Race Strategy Optimizer is an interactive Streamlit dashboard combining game theory, machine learning, and advanced visualization to analyze and optimize Formula 1 race strategies. It enables simulation of Nash and Stackelberg strategies, trust dynamics visualization, and actionable insights from real F1 data.

---

## Features

- Analyze 2019–2021 F1 race data via FastF1.
- Simulate Nash (conservative) and Stackelberg (aggressive) strategies.
- Compute trust metrics from lap times and positions.
- Use Random Forest to identify key performance drivers.
- Visualize trust dynamics, game theory payoffs, and feature importance heatmaps.
- Provide driver-specific strategic recommendations.
- Demo mode with fallback data for presentation.

---

## Installation

- git clone https://github.com/mannatgoyal/trust-dynamics-motorsports.git
- cd trust-dynamics-motorsports
- pip install -r requirements.txt
- streamlit run f1_ai_strategist.py
  
---

## Usage

- Select year, track, and driver in the sidebar.
- View trust dynamics and strategy comparisons.
- Explore feature importance heatmap and game theory payoff matrix.
- Toggle advanced analysis for detailed insights.
- If live data is unavailable, demo data is shown automatically.

---

## Data Sources

- [FastF1](https://theoehrly.github.io/Fast-F1/) for official timing and telemetry.
- Local CSV fallback for demo purposes.

---

## Code Structure

- f1_ai_strategist.py # Main dashboard app
- requirements.txt # Dependencies
- sample_race_data.csv # Demo fallback data
- assets/ # Images and assets
- README.md # This file
  
---

## Results & Analysis

- Trust dynamics reveal driver reliability lap-by-lap.
- Game theory payoff matrix quantifies strategy outcomes.
- Feature importance heatmap highlights critical race factors.
- Driver recommendations tailor strategy to strengths and track.
[Demo - Currently Deployed Results](https://trust-dynamics-motorsports-lp4lt5rzeuvryuqjedjd99.streamlit.app/)
---

## Driver & Strategy Recommendations

- **Hamilton**: Tire management and late-race pace focus.
- **Verstappen**: Early aggression and mid-race defense.
- **Leclerc**: Qualifying leverage and tire care.
- **Others**: Strategy tailored by feature importance insights.

---

## Future Work

- Support for 2022+ seasons.
- Weather and safety car integration.
- Advanced ML models (LSTM, XGBoost).
- Real-time telemetry and live race analysis.

---

## Acknowledgments

- FastF1 project for data access.
- Scikit-learn, Matplotlib, Streamlit for tooling.
- F1 community for inspiration.

---

## License

MIT License. See LICENSE file.

---

**Questions or contributions? Open an issue or pull request!**

---

*Ready to optimize your F1 race strategy? Clone, run, and explore!*
