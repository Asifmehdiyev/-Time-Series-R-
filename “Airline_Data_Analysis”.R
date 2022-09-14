# **** Import necessary libraries ******
library(tidymodels) # (Modelling and analysis purposes)
library(modeltime) # (Time Series modelling and analysis including ARIMA, Exponential Smoothing and Forecast)
library(tidyverse) # (Transform and understanding of dataset)
library(lubridate) # (Used for working with dates and times)
library(timetk)  # (Visualization, wrangling and feature engineering of time series data)
library(dygraphs) # (Interactive time series charting library)
library(zoo)  # (Infrastructure for Regular and Irregular Time Series)

interactive <- FALSE

# ******************* Import dataset ********
df <- read.csv('AirPassengers (3).csv')
View(df)


colnames(df)<-c('Date', 'Count')

#data is given monthly 
df$Date<-as.Date(as.yearmon(df$Date))
df$Count<-as.numeric(df$Count)

df %>% plot_time_series(Date, Count);

# *************** Train-Test split with 80/20 ratio ******
splits <- initial_time_split(df, prop = 0.8)

# ******** Model 1: auto_arima **********
model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Count ~ Date, data = training(splits))

# ******* Model 2: arima_boost *********
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Count ~ Date + as.numeric(Date) + factor(lubridate::month(Date, label = TRUE), ordered = F),
      data = training(splits))

# ********** Model: exp_smoothing() *****
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Count ~ Date, data = training(splits))

# ******* Model: Prophet_reg() ********
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Count ~ Date, data = training(splits))

# ******** Model: Linear_reg() ********
model_fit_lm <- linear_reg() %>%
  set_engine("lm") %>%
  fit(Count ~ as.numeric(Date) + factor(lubridate::month(Date, label = TRUE), ordered = FALSE),
      data = training(splits))


# ***************** Model table ********
models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet,
  model_fit_lm
)

models_tbl

# ****** Calibrate the model to a testing set ********
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

# ****** Testing Set Forecast & Accuracy Evaluation ********
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = df
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  )

# ************** Accuracy metrics ********
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = interactive
  )

# ******* Forecast forward *********
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = df)

refit_tbl %>%
  modeltime_forecast(h = "3 years", actual_data = df) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  )
# As we can see from Accuracy table, Prophet is performing
# better than other models
