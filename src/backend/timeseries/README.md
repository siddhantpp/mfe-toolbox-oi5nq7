# Time Series Analysis Module

The Time Series Analysis module provides comprehensive tools for modeling, estimating, and forecasting time series data with a focus on ARMA/ARMAX models in financial econometrics. This module forms a core component of the MFE Toolbox v4.0.

## ARMA/ARMAX Modeling

Implements ARMAX modeling with parameter estimation, residual diagnostics, and model selection criteria. Includes MEX-optimized core for high-performance computation.

### Key Functions:

- **armaxfilter.m**: Estimates parameters for ARMAX(p,q,r) models with robust optimization and comprehensive diagnostics
  ```matlab
  results = armaxfilter(data, p, q, exogenous, options)
  ```

- **armafor.m**: Generates multi-step forecasts for ARMA/ARMAX models with confidence intervals
  ```matlab
  forecasts = armafor(data, parameters, numPeriods, exogenous)
  ```

- **sarima.m**: Estimates Seasonal ARIMA models for seasonal time series analysis
  ```matlab
  results = sarima(data, p, d, q, P, D, Q, s, options)
  ```

## Diagnostic Tools

Functions for model identification, diagnostic checking, and model evaluation.

### Key Functions:

- **sacf.m**: Computes sample autocorrelation function with confidence intervals
  ```matlab
  acf = sacf(data, lags, options)
  ```

- **spacf.m**: Computes sample partial autocorrelation function with confidence intervals
  ```matlab
  pacf = spacf(data, lags, options)
  ```

- **aicsbic.m**: Calculates information criteria for model selection (AIC, SBIC)
  ```matlab
  criteria = aicsbic(logL, k, T, options)
  ```

## Performance Optimization

Performance-critical operations are implemented in C via MEX for significant computational speedup.

- **armaxerrors.c**: MEX implementation for efficient ARMAX error/residual computation
  * Achieves >50% performance improvement over equivalent MATLAB implementation

## Integration

The Time Series module integrates with other MFE Toolbox components:

- **GUI Integration**: ARMAX.m provides an interactive GUI for model configuration, estimation, and visualization
- **Volatility Models**: Time series functions provide preprocessing and modeling support for volatility analysis
- **Statistical Tests**: Integrates with diagnostic tests for model validation and residual analysis
- **Bootstrap Methods**: Supports bootstrap-based confidence intervals and variance estimation for time series models

## Examples

### Basic ARMA Modeling

```matlab
% Fit an ARMA(2,1) model
results = armaxfilter(returns, 2, 1);

% Generate forecasts
forecasts = armafor(returns, results.parameters, 10);

% Plot results with confidence intervals
plot(forecasts.forecasts, 'b');
hold on;
plot(forecasts.upper95, 'r--');
plot(forecasts.lower95, 'r--');
```

### Model Selection

```matlab
% Find optimal model order using information criteria
best_aic = Inf;
best_model = [];

for p = 0:3
    for q = 0:3
        results = armaxfilter(returns, p, q);
        if results.aic < best_aic
            best_aic = results.aic;
            best_model = [p q];
        end
    end
end

fprintf('Best model: ARMA(%d,%d)\n', best_model(1), best_model(2));
```