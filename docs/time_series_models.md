# Time Series Models

The MFE Toolbox provides comprehensive tools for time series analysis with a focus on ARMA/ARMAX modeling, forecasting, diagnostics, and model selection. This document details the available functions, their parameters, and usage examples for time series analysis in financial econometrics.

## Table of Contents

## ARMA/ARMAX Modeling

Autoregressive Moving Average (ARMA) models and their extensions with exogenous variables (ARMAX) are fundamental tools for time series analysis. The MFE Toolbox provides robust implementations with support for various error distributions and comprehensive diagnostics.

### armaxfilter

Estimates parameters for ARMAX(p,q,r) models with robust optimization and comprehensive diagnostics.

```matlab
results = armaxfilter(data, p, q, [exogenous], [options])
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `data` | T×1 vector | Time series data to be modeled |
| `p` | non-negative integer | Autoregressive (AR) order |
| `q` | non-negative integer | Moving average (MA) order |
| `exogenous` | T×k matrix | Optional exogenous variables |
| `options` | structure | Optional estimation settings |

#### Returns

| Name | Type | Description |
|---|---|---|
| `results` | structure | Model estimation results including parameters, standard errors, diagnostics, and information criteria |

#### Options

| Option | Description |
|---|---|
| `distribution` | Error distribution: 'normal' (default), 'studentst', 'ged', or 'skewt' |
| `startingvals` | Initial parameter values for optimization |
| `constant` | Include constant term (default: true) |
| `robust` | Use robust standard errors (default: false) |

#### Example

```matlab
% Estimate ARMA(1,1) model with normal errors
results = armaxfilter(returns, 1, 1);

% Estimate ARMAX(2,1,1) with exogenous variable and t-distributed errors
exog = interest_rates;
options = struct('distribution', 'studentst');
results = armaxfilter(returns, 2, 1, exog, options);

% Display results
disp(['Log-likelihood: ' num2str(results.logL)]);
disp(['AIC: ' num2str(results.aic)]);
```

#### Notes

The function performs maximum likelihood estimation with support for various error distributions. Comprehensive diagnostics include Ljung-Box Q-statistics, LM tests, and information criteria.

## Forecasting

The MFE Toolbox provides robust forecasting capabilities for ARMA/ARMAX models, supporting both exact analytical forecasts and simulation-based methods with various error distributions.

### armafor

Generates multi-step ahead forecasts for ARMA/ARMAX models with confidence intervals.

```matlab
[forecasts, variances, paths] = armafor(parameters, data, [exogenous], horizon, [options])
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `parameters` | structure or vector | Model parameters, either as structure from armaxfilter or parameter vector |
| `data` | T×1 vector | Historical time series data |
| `exogenous` | (T+horizon)×k matrix | Optional exogenous variables including future values |
| `horizon` | positive integer | Forecast horizon (number of periods ahead) |
| `options` | structure | Optional forecasting settings |

#### Returns

| Name | Type | Description |
|---|---|---|
| `forecasts` | horizon×1 vector | Point forecasts for each period in the horizon |
| `variances` | horizon×1 vector | Forecast error variances |
| `paths` | horizon×nsim matrix | Simulated forecast paths (if simulation method used) |

#### Options

| Option | Description |
|---|---|
| `method` | Forecasting method: 'exact' (default) or 'simulation' |
| `nsim` | Number of simulation paths (default: 1000) |
| `distribution` | Error distribution for simulation: 'normal' (default), 'studentst', 'ged', or 'skewt' |
| `dist_params` | Distribution parameters for simulation |

#### Example

```matlab
% Estimate ARMA(1,1) model
results = armaxfilter(returns, 1, 1);

% Generate 10-step ahead forecasts with 95% confidence intervals
[forecasts, variances] = armafor(results.parameters, returns, 10);
upperCI = forecasts + 1.96*sqrt(variances);
lowerCI = forecasts - 1.96*sqrt(variances);

% Plot forecasts with confidence intervals
plot(forecasts, 'b-', 'LineWidth', 2);
hold on;
plot(upperCI, 'r--');
plot(lowerCI, 'r--');
legend('Forecast', '95% Confidence Interval');
```

#### Notes

The function supports both exact analytical forecasts and simulation-based forecasts with various error distributions. Confidence intervals can be computed from forecast variances.

## Diagnostic Tools

Diagnostic tools are essential for time series model identification, validation, and evaluation. The MFE Toolbox provides functions for computing autocorrelation, partial autocorrelation, and other diagnostic statistics.

### sacf

Computes sample autocorrelation function with optional confidence intervals.

```matlab
[acf, se, ci] = sacf(data, [lags], [options])
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `data` | T×1 vector | Time series data |
| `lags` | positive integer | Maximum number of lags (default: min(20, T/4)) |
| `options` | structure | Optional settings |

#### Returns

| Name | Type | Description |
|---|---|---|
| `acf` | lags×1 vector | Sample autocorrelation values |
| `se` | lags×1 vector | Standard errors for each autocorrelation |
| `ci` | lags×2 matrix | Confidence intervals for each autocorrelation |

#### Options

| Option | Description |
|---|---|
| `alpha` | Significance level for confidence intervals (default: 0.05) |
| `demean` | Remove mean from data before computation (default: true) |

#### Example

```matlab
% Compute autocorrelation up to lag 20 with 95% confidence intervals
[acf, se, ci] = sacf(returns, 20);

% Plot autocorrelation with confidence bounds
bar(acf);
hold on;
plot([0 21], [0 0], 'k-');
plot([0 21], [ci(1,1) ci(1,1)], 'r--');
plot([0 21], [ci(1,2) ci(1,2)], 'r--');
xlim([1 20]);
title('Sample Autocorrelation Function');
```

#### Notes

The sacf function is essential for model identification, particularly for determining the order of MA processes. The confidence intervals help identify statistically significant autocorrelations.

### spacf

Computes sample partial autocorrelation function with optional confidence intervals.

```matlab
[pacf, se, ci] = spacf(data, [lags], [options])
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `data` | T×1 vector | Time series data |
| `lags` | positive integer | Maximum number of lags (default: min(20, T/4)) |
| `options` | structure | Optional settings |

#### Returns

| Name | Type | Description |
|---|---|---|
| `pacf` | lags×1 vector | Sample partial autocorrelation values |
| `se` | lags×1 vector | Standard errors for each partial autocorrelation |
| `ci` | lags×2 matrix | Confidence intervals for each partial autocorrelation |

#### Options

| Option | Description |
|---|---|
| `alpha` | Significance level for confidence intervals (default: 0.05) |
| `demean` | Remove mean from data before computation (default: true) |

#### Example

```matlab
% Compute partial autocorrelation up to lag 20 with 95% confidence intervals
[pacf, se, ci] = spacf(returns, 20);

% Plot partial autocorrelation with confidence bounds
bar(pacf);
hold on;
plot([0 21], [0 0], 'k-');
plot([0 21], [ci(1,1) ci(1,1)], 'r--');
plot([0 21], [ci(1,2) ci(1,2)], 'r--');
xlim([1 20]);
title('Sample Partial Autocorrelation Function');
```

#### Notes

The spacf function is essential for model identification, particularly for determining the order of AR processes. The confidence intervals help identify statistically significant partial autocorrelations.

## Model Selection

Model selection is a critical step in time series analysis, balancing model fit with complexity. The MFE Toolbox provides information criteria for comparing alternative model specifications.

### aicsbic

Calculates Akaike Information Criterion (AIC) and Schwarz Bayesian Information Criterion (SBIC/BIC) for model selection.

```matlab
criteria = aicsbic(logL, k, T)
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `logL` | scalar or vector | Log-likelihood value(s) |
| `k` | integer or vector | Number of parameters in model(s) |
| `T` | integer | Sample size used in estimation |

#### Returns

| Name | Type | Description |
|---|---|---|
| `criteria` | structure | Structure with fields .aic and .sbic containing the computed criteria values |

#### Example

```matlab
% Estimate models with different orders
results1 = armaxfilter(returns, 1, 0);
results2 = armaxfilter(returns, 1, 1);
results3 = armaxfilter(returns, 2, 1);

% Calculate information criteria
c1 = aicsbic(results1.logL, length(results1.parameters), length(returns));
c2 = aicsbic(results2.logL, length(results2.parameters), length(returns));
c3 = aicsbic(results3.logL, length(results3.parameters), length(returns));

% Compare models
disp('Model Comparison:');
disp('            AIC      SBIC');
disp(['AR(1):     ' num2str(c1.aic) '  ' num2str(c1.sbic)]);
disp(['ARMA(1,1): ' num2str(c2.aic) '  ' num2str(c2.sbic)]);
disp(['ARMA(2,1): ' num2str(c3.aic) '  ' num2str(c3.sbic)]);
```

#### Notes

Lower values of information criteria indicate better models. AIC tends to select more complex models, while SBIC penalizes additional parameters more heavily, favoring more parsimonious models.

## Seasonal Models

Seasonal time series exhibit regular patterns at fixed intervals. The MFE Toolbox provides support for modeling seasonal patterns with Seasonal ARIMA (SARIMA) models.

### sarima

Estimates parameters for Seasonal ARIMA models with comprehensive diagnostics.

```matlab
results = sarima(data, p, d, q, P, D, Q, s, [options])
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `data` | T×1 vector | Time series data |
| `p` | non-negative integer | Non-seasonal AR order |
| `d` | non-negative integer | Non-seasonal differencing order |
| `q` | non-negative integer | Non-seasonal MA order |
| `P` | non-negative integer | Seasonal AR order |
| `D` | non-negative integer | Seasonal differencing order |
| `Q` | non-negative integer | Seasonal MA order |
| `s` | positive integer | Seasonality period (e.g., 12 for monthly, 4 for quarterly) |
| `options` | structure | Optional estimation settings |

#### Returns

| Name | Type | Description |
|---|---|---|
| `results` | structure | Model estimation results including parameters, standard errors, diagnostics, and information criteria |

#### Options

| Option | Description |
|---|---|
| `distribution` | Error distribution: 'normal' (default), 'studentst', 'ged', or 'skewt' |
| `constant` | Include constant term (default: true) |
| `robust` | Use robust standard errors (default: false) |

#### Example

```matlab
% Estimate SARIMA(1,1,1)(1,1,1)12 model for monthly data
results = sarima(monthly_data, 1, 1, 1, 1, 1, 1, 12);

% Display results
disp(['Log-likelihood: ' num2str(results.logL)]);
disp(['AIC: ' num2str(results.aic)]);

% Generate forecasts
[forecasts, variances] = armafor(results.parameters, monthly_data, 24);

% Plot forecasts
plot(forecasts);
title('24-Month SARIMA Forecast');
```

#### Notes

SARIMA models extend ARIMA to include seasonal patterns, making them suitable for data with regular seasonal fluctuations, such as monthly or quarterly financial time series.

## Integration with Volatility Models

Time series models in the MFE Toolbox integrate seamlessly with volatility models, providing a foundation for modeling both conditional means and conditional variances of financial time series.

Time series models are often combined with volatility models in a two-step procedure: first modeling the conditional mean with ARMA/ARMAX, then modeling the conditional variance of residuals with GARCH-type models. The MFE Toolbox facilitates this workflow through consistent interfaces and data structures.

```matlab
% Step 1: Model conditional mean with ARMA
mean_model = armaxfilter(returns, 1, 1);
residuals = mean_model.residuals;

% Step 2: Model conditional variance with GARCH
volatility_model = tarchfit(residuals);

% Combined forecast
mean_forecasts = armafor(mean_model.parameters, returns, 10);
volat_forecasts = garchfor(volatility_model.parameters, residuals, 10);

% Plot results
subplot(2,1,1);
plot(mean_forecasts);
title('Conditional Mean Forecast');
subplot(2,1,2);
plot(sqrt(volat_forecasts));
title('Conditional Volatility Forecast');
```

Cross-references:

- [volatility_models.md#garch-models](volatility_models.md) - Details on modeling conditional volatility

## Performance Considerations

The time series modeling functions in the MFE Toolbox are optimized for performance, with critical operations implemented in C via MEX interfaces. The `armaxerrors` MEX function provides significant performance improvements for ARMAX error computation, especially for long time series or when performing multiple estimations for model selection.

## See Also

- [Distribution Functions](distribution_functions.md) - Probability distributions used in time series modeling
- [Volatility Models](volatility_models.md) - Modeling conditional variances with GARCH and extensions
- [Statistical Tests](statistical_tests.md) - Tests for stationarity, serial correlation, and model adequacy
- [Bootstrap Methods](bootstrap_methods.md) - Resampling techniques for time series confidence intervals