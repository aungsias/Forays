# Dynamic Portfolio Optimization with Sector Rotation & Machine Learning
Aung Si<br>
September 9<sup>th</sup>, 2023

---

## Abstract
In this project, I've engineered an adaptive machine learning algorithm that undergoes biannual recalibration to select the most accurate model for sector-based investment strategies. To counteract the pitfalls of over-forecasting, the algorithm employs a custom loss function that penalizes overpredictions. It comprehensively integrates a diverse range of financial indicators, including equity, debt, commodities, and market volatility. To enhance computational efficiency and model precision, I employed Principal Component Analysis for feature reduction. The model's robustness was substantiated through a 15-year backtest, during which it outperformed the SPY index by an estimated 91.85%. The finalized, vetted model has been encapsulated in a real-time dashboard, effectively translating intricate analytics into actionable market insights.

## Introduction
Financial markets are inherently volatile and self-correcting, subject to externalities including but not limited to equities, debt instruments, commodities, and market sentiment. Traditional investment models often falter in this ever-changing landscape, and even adaptable strategies like sector rotation are usually limited by heuristic-based decision-making.

In this project, I've constructed a machine learning framework that operates as a "model of models." Rather than being a singular static model, it encompasses multiple machine learning algorithms. Every six months, each constituent model is retrained and assessed for performance. The best-performing model is then selected for the ensuing period, thereby ensuring that the framework remains attuned to the current market conditions.

To fortify the framework against the common pitfall of over-forecasting, I've implemented a custom loss function. This function penalizes overpredictions, thereby enhancing the framework's reliability. I've also integrated a wide array of financial indicators to achieve a comprehensive market analysis. For computational efficacy and precision, Principal Component Analysis (PCA) has been employed for feature selection and dimensionality reduction.

To substantiate the framework's efficacy, a 15-year backtest was conducted, during which it consistently outperformed the SPY index. The insights derived from this framework have been operationalized through a real-time dashboard, making the analytics readily actionable.

