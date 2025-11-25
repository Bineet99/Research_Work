# Research_Work
This paper explores how ARIMA and LSTM perform in short-term agricultural price forecasting using a synthetic dataset that behaves like real market data. The dataset includes trend, seasonality, noise, and sudden price jumps, so both models face the types of changes that usually appear in actual markets. After applying MinMax scaling and creating sliding windows, the models are trained and tested on the same series.

Their accuracy is measured using RMSE and MAE, which show how closely each model follows the true price movements. ARIMA produces stable predictions but struggles with fast or irregular changes. LSTM handles those shifts much better, giving lower RMSE and MAE values and staying closer to the real price curve.

Overall, the results suggest that LSTM offers a stronger approach for practical forecasting. The study also points toward future work with real market data and additional factors like weather or logistics to build a more useful real-time tool.
