# Univariate Volatility Modeling

## Introduction

The Univariate Volatility Modeling module is a core component of the MFE Toolbox (MATLAB Financial Econometrics Toolbox) that provides sophisticated implementations of GARCH-family models for volatility modeling of financial time series. This module delivers high-performance computing capabilities optimized for analyzing and forecasting time-varying volatility in financial markets.

The module implements various GARCH-type models with robust estimation procedures, comprehensive diagnostics, and forecasting capabilities, making it suitable for both academic research and professional financial analysis.

## Module Components

The univariate module consists of the following key components:

### Core Components

- **garchcore.m** - Core implementation of GARCH variance recursion calculations
- **garchinit.m** - Parameter initialization functions for GARCH models
- **garchlikelihood.m** - Log-likelihood computation for GARCH model estimation
- **garchfor.m** - Volatility forecasting functions for GARCH models

### Model Implementations

- **agarchfit.m** - Asymmetric GARCH (AGARCH) model estimation
- **egarchfit.m** - Exponential GARCH (EGARCH) model estimation
- **igarchfit.m** - Integrated GARCH (IGARCH) model estimation
- **tarchfit.m** - Threshold ARCH (TARCH) model estimation
- **nagarchfit.m** - Nonlinear Asymmetric GARCH (NAGARCH) model estimation

## Model Features

All GARCH implementations in this module share the following features:

### Multiple Distributions

Support for various error distributions:
- Normal (Gaussian) distribution
- Student's t distribution
- Generalized Error Distribution (GED)
- Hansen's Skewed t distribution

These distribution options allow for modeling fat tails and asymmetry in financial returns.

### MEX Optimization

High-performance C implementations for computationally intensive operations:
- Variance recursion calculations
- Likelihood evaluation
- Parameter optimization

### Robust Estimation

- Constrained optimization using fmincon
- Parameter validation and boundary enforcement
- Multiple starting values for global optimization
- Analytical and numerical derivatives

### Comprehensive Diagnostics

- Log-likelihood values
- Information criteria (AIC, BIC)
- Parameter standard errors
- Model persistence measures
- Residual diagnostics

### Forecasting

- Multi-step ahead volatility forecasting
- Simulation-based forecast intervals
- Mean and variance path projections

## Performance Optimization

### MEX Integration

The module implements performance-critical operations in C via MEX for significant speed improvements:

- **agarch_core.c** - C implementation of AGARCH variance recursion
- **egarch_core.c** - C implementation of EGARCH variance recursion
- **igarch_core.c** - C implementation of IGARCH variance recursion
- **tarch_core.c** - C implementation of TARCH variance recursion

### Platform Support

Optimized MEX binaries are provided for:
- Windows platforms (*.mexw64)
- Unix platforms (*.mexa64)

### Memory Management

- Efficient memory allocation strategies
- Minimal data copying
- Pre-allocated arrays for recursive calculations

### Vectorized Operations

When MEX acceleration is unavailable, the module falls back to MATLAB-optimized vectorized algorithms for:
- Matrix-based recursion
- Efficient likelihood computation
- Optimized gradient calculations

## Usage Examples

### AGARCH Estimation

```matlab
% Estimate AGARCH(1,1) model with Student's t distribution
options = [];
options.p = 1;       % ARCH order
options.q = 1;       % GARCH order
options.error = 1;   % Student's t distribution
model = agarchfit(returns, options);
```

### Volatility Forecasting

```matlab
% Generate 10-step ahead forecasts
forecast_horizon = 10;
forecasts = garchfor(model, forecast_horizon);
```

## Dependencies

### Utility Functions
- backcast.m - Initial variance estimation
- columncheck.m - Input validation
- datacheck.m - Data validation
- parametercheck.m - Parameter validation

### Distribution Functions
- gedfit.m - GED parameter estimation
- gedloglik.m - GED log-likelihood
- stdtfit.m - Student's t parameter estimation
- stdtloglik.m - Student's t log-likelihood
- skewtfit.m - Skewed t parameter estimation
- skewtloglik.m - Skewed t log-likelihood

### MEX Components
- agarch_core.c - AGARCH implementation
- egarch_core.c - EGARCH implementation
- igarch_core.c - IGARCH implementation
- tarch_core.c - TARCH implementation

### Time Series Functions
- aicsbic.m - Information criteria calculation

### External
- MATLAB Optimization Toolbox (fmincon)

## References

- **GARCH**: Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics, 31, 307-327.

- **AGARCH**: Engle, R. F. (1990). Discussion: Stock Market Volatility and the Crash of '87. Review of Financial Studies, 3, 103-106.

- **EGARCH**: Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. Econometrica, 59, 347-370.

- **TARCH**: Zakoian, J. M. (1994). Threshold Heteroskedastic Models. Journal of Economic Dynamics and Control, 18, 931-955.