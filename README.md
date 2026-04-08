## Football Wage Amplification Analysis

In football, where you are born matters almost as much as how good you are. This project measures how much — and evaluates whether that difference is justified.

This project goes beyond predicting player wages and instead focuses on understanding what those wages represent in a global economic context.


## Overview

Most football wage analysis answers the question:

“This player earns €X.”

This project extends that by asking:

“Should they earn €X, and what does €X mean relative to their country?”

The system combines:

* A machine learning model to predict player wages
* An economic framework to contextualize those wages globally


## Dataset

* Approximately 48,903 players
* 75 leagues
* 140 countries
* Combined with global minimum wage data


## Wage Prediction Model

The model is trained using the following features:

* Skill rating
* Potential
* Age
* Market value
* League
* League level

Performance:

* R² ≈ 0.79 on unseen data
* Explains approximately 79% of wage variation

This allows the model to estimate expected wages based on player attributes and competition level.


## Amplification Factor

The core contribution of this project is the Amplification Factor:

Wage Amplification = Player Wage ÷ Minimum Wage in Home Country

Example:

A player earning €5,000 per week from a country with a minimum wage of $44 per month results in an amplification of approximately 780×.

This metric provides a standardized way to compare economic impact across countries.


## Key Findings

* Skill is strongly correlated with wage amplification
* Higher-rated players experience disproportionately higher earnings relative to their home economies
* Nationality determines the baseline economic context
* Players from lower-income countries experience significantly higher amplification
* Players from higher-income countries experience relatively lower amplification


## Globalisation in Football

* Approximately 49% of male players compete outside their home country
* Migration patterns are driven by:

  * Infrastructure and league quality (e.g., UK, Germany)
  * Economic factors (e.g., Brazil, Haiti, Venezuela)

Football functions as a global pathway for economic mobility.


## Significance

Football is one of the few domains where:

* Individual skill can overcome geographic economic limitations
* Players can significantly exceed the economic constraints of their home country

At the same time, it highlights the scale of global inequality in baseline wages.


## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Data visualization tools


## Future Work

* Expand dataset coverage across additional leagues and seasons
* Incorporate transfer fee prediction
* Improve representation of women’s football data
* Develop a real-time prediction and visualization interface


## Conclusion

Place of origin defines the baseline.
Skill determines the extent of upward mobility.

This project explores the intersection of sports analytics, machine learning, and global economic inequality through the lens of football.
