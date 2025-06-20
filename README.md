# Portfolio
Hybrid Portfolio Optimization and Market Prediction using Machine Learning and Statistical Models

Project Overview

This project combines **quantitative finance** with **machine learning** to build a complete system for **portfolio optimization, stock trend forecasting, and gain/loss classification** using real stock market data.

We leverage 20 years of historical data from 42 companies to identify optimal asset allocations, predict future price trends, and classify future gains/losses — offering a hybrid framework that merges **traditional finance models** with **modern AI techniques**.

Objectives

* Optimize portfolio weights to **maximize return** and **minimize risk**.
* Predict future **stock price movements** using regression models.
* Classify stocks as **gain/loss** using classification algorithms.
* Evaluate all strategies with financial and ML metrics for performance comparison.

Technologies Used

* **Programming Language**: Python
* **Data Collection**: yfinance
* **Data Manipulation**: pandas, numpy
* **Visualization**: matplotlib, seaborn
* **Clustering**: KMeans (scikit-learn)
* **Optimization Models**:

  * Global Minimum Variance (GMV)
  * Maximum Sharpe Ratio (MSR)
  * Risk Parity (RP)
  * Equal Weight (EW)
  * Genetic Algorithm (custom implementation)
* **Regression Models**:

  * Random Forest Regressor
  * XGBoost Regressor
* **Classification Models**:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest Classifier
  * XGBoost Classifier
* **Evaluation Metrics**:

  * Sharpe Ratio, Cumulative Return, Volatility, Max Drawdown
  * MAE, RMSE (for regression)
  * Accuracy, Precision, Recall, F1-Score (for classification)

Workflow

1. **Data Preparation**

   * 42 stock tickers from Yahoo Finance over 20 years.
   * Cleaning, pivoting, and calculating daily returns and volatility.

2. **Portfolio Selection**

   * Using K-Means Clustering to group similar-performing stocks.

3. **Portfolio Optimization**

   * Compute GMV, MSR, RP, and EW portfolios.
   * Apply a Genetic Algorithm to discover optimal weights that maximize the Sharpe Ratio.

4. **Evaluation**

   * Compare portfolios using Sharpe Ratio, Volatility, Max Drawdown, and Cumulative Returns.

5. **Trend Prediction**

   * Use Random Forest and XGBoost regressors to forecast future prices.

6. **Gain/Loss Classification**

   * Predict if a stock will have a gain or loss using LR, SVM, RF, and XGBoost.

Results

* **MSR Portfolio** yielded the highest cumulative return (\~49.6%).
* **Genetic Algorithm** optimized Sharpe Ratio effectively.
* **XGBoost & Random Forest** gave strong predictions with low RMSE and MAE.
* **Tuned Logistic Regression and SVM** achieved classification accuracy > 93%.

Key Highlights

* End-to-end financial ML pipeline.
* Hybrid methodology combining finance and machine learning.
* Real data from Yahoo Finance ensures practical applicability.
* Scalable for dynamic rebalancing or real-time deployment in the future.

---

Let me know if you'd like a shorter version or if you want this split into `README.md` sections like Installation, Usage, or Results.
