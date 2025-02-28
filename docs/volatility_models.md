# Volatility Models

The MFE Toolbox provides robust implementations of volatility models widely used in financial econometrics for modeling time-varying variance in financial time series. These models are essential tools for risk management, option pricing, portfolio optimization, and other financial applications. This document details both univariate and multivariate volatility models, their parameters, implementation details, and usage examples.

## Table of Contents
- [Overview of Volatility Models](#overview-of-volatility-models)
- [Univariate Volatility Models](#univariate-volatility-models)
    - [agarchfit - Asymmetric GARCH Model](#agarchfit---asymmetric-garch-model)
    - [egarchfit - Exponential GARCH Model](#egarchfit---exponential-garch-model)
    - [igarchfit - Integrated GARCH Model](#igarchfit---integrated-garch-model)
    - [tarchfit - Threshold ARCH/GARCH Model](#tarchfit---threshold-archgarch-model)
    - [nagarchfit - Nonlinear Asymmetric GARCH Model](#nagarchfit---nonlinear-asymmetric-garch-model)
- [Multivariate Volatility Models](#multivariate-volatility-models)
    - [ccc_mvgarch - Constant Conditional Correlation MVGARCH](#ccc_mvgarch---constant-conditional-correlation-mvgarch)
    - [dcc_mvgarch - Dynamic Conditional Correlation MVGARCH](#dcc_mvgarch---dynamic-conditional-correlation-mvgarch)
    - [bekk_mvgarch - BEKK MVGARCH Model](#bekk_mvgarch---bekk-mvgarch-model)
    - [gogarch - Generalized Orthogonal GARCH](#gogarch---generalized-orthogonal-garch)
- [Forecasting](#forecasting)
    - [garchfor - GARCH Model Forecasting](#garchfor---garch-model-forecasting)
- [MEX Optimization](#mex-optimization)
    - [MEX Implementation Details](#mex-implementation-details)
    - [Performance Comparison](#performance-comparison)
- [Usage Examples](#usage-examples)
    - [Basic Volatility Modeling](#basic-volatility-modeling)
    - [Model Comparison](#model-comparison)
    - [Value-at-Risk Estimation](#value-at-risk-estimation)
    - [Portfolio Optimization with DCC-MVGARCH](#portfolio-optimization-with-dcc-mvgarch)
- [Integration with Time Series Models](#integration-with-time-series-models)
    - [Two-Step Modeling Approach](#two-step-modeling-approach)
    - [Joint Estimation](#joint-estimation)
- [Technical Notes](#technical-notes)
    - [Implementation Details](#implementation-details-1)
    - [Precision Considerations](#precision-considerations)
    - [Performance Optimization](#performance-optimization-1)
- [See Also](#see-also)

## Overview of Volatility Models
Volatility models in the MFE Toolbox are divided into two main categories:

1. **Univariate Volatility Models** - For modeling conditional variance in single time series
   * AGARCH (Asymmetric GARCH)
   * EGARCH (Exponential GARCH)
   * IGARCH (Integrated GARCH)
   * TARCH (Threshold ARCH/GARCH)
   * NAGARCH (Nonlinear Asymmetric GARCH)

2. **Multivariate Volatility Models** - For modeling covariance structures across multiple time series
   * CCC (Constant Conditional Correlation)
   * DCC (Dynamic Conditional Correlation)
   * BEKK (Baba-Engle-Kraft-Kroner)
   * GO-GARCH (Generalized Orthogonal GARCH)

All models feature comprehensive error checking, robust numerical stability, support for various error distributions, and high-performance computation through MEX optimization.

## Univariate Volatility Models
Univariate GARCH-type models capture time-varying conditional variance in a single time series, with different specifications offering various advantages for modeling specific volatility characteristics like asymmetry, long memory, and leverage effects.

### agarchfit - Asymmetric GARCH Model
Estimates an Asymmetric GARCH (AGARCH) model that captures asymmetric effects of positive and negative shocks on volatility.

**Syntax**
```matlab
model = agarchfit(data, [p, q], [options])
```

**Parameters**
- `data`: numeric - Time series data to model
- `p`: integer - ARCH order (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `bounds`: Custom parameter bounds for estimation
- `optimOptions`: Options for fmincon optimizer

**Model Specification**
The AGARCH(p,q) model is specified as:

$$h_t = \omega + \sum_{i=1}^p \alpha_i (\epsilon_{t-i} - \gamma \sqrt{h_{t-i}})^2 + \sum_{j=1}^q \beta_j h_{t-j}$$

where:
- $h_t$ is the conditional variance at time t
- $\epsilon_t$ is the innovation at time t
- $\omega$, $\alpha_i$, $\gamma$, and $\beta_j$ are model parameters
- $\gamma$ represents the asymmetry parameter capturing the leverage effect

**Example**
```matlab
% Load data
returns = randn(1000, 1);

% Estimate AGARCH(1,1) model with Student's t errors
options = struct('distribution', 'STUDENTST');
model = agarchfit(returns, [1, 1], options);

% Display results
disp('AGARCH Parameter Estimates:');
disp(model.parameters);
```

**See Also**
- [egarchfit](#egarchfit) - Exponential GARCH model estimation
- [tarchfit](#tarchfit) - Threshold ARCH model estimation
- [garchfor](#garchfor) - GARCH model forecasting

### egarchfit - Exponential GARCH Model
Estimates an Exponential GARCH (EGARCH) model that captures asymmetric effects in volatility while ensuring positive variance through a logarithmic specification.

**Syntax**
```matlab
model = egarchfit(data, [p, o, q], [options])
```

**Parameters**
- `data`: numeric - Time series data to model
- `p`: integer - ARCH order (default: 1)
- `o`: integer - Order of asymmetry terms (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `bounds`: Custom parameter bounds for estimation
- `optimOptions`: Options for fmincon optimizer

**Model Specification**
The EGARCH(p,o,q) model is specified as:

$$\ln(h_t) = \omega + \sum_{i=1}^p \alpha_i \left|\frac{\epsilon_{t-i}}{\sqrt{h_{t-i}}}\right| + \sum_{k=1}^o \gamma_k \frac{\epsilon_{t-k}}{\sqrt{h_{t-k}}} + \sum_{j=1}^q \beta_j \ln(h_{t-j})$$

where:
- $h_t$ is the conditional variance at time t
- $\epsilon_t$ is the innovation at time t
- $\omega$, $\alpha_i$, $\gamma_k$, and $\beta_j$ are model parameters
- $\gamma_k$ captures the asymmetric impact of positive and negative shocks

**Example**
```matlab
% Load data
returns = randn(1000, 1);

% Estimate EGARCH(1,1,1) model with GED errors
options = struct('distribution', 'GED');
model = egarchfit(returns, [1, 1, 1], options);

% Display results
disp('EGARCH Parameter Estimates:');
disp(model.parameters);
```

**See Also**
- [agarchfit](#agarchfit) - Asymmetric GARCH model estimation
- [tarchfit](#tarchfit) - Threshold ARCH model estimation
- [garchfor](#garchfor) - GARCH model forecasting

### igarchfit - Integrated GARCH Model
Estimates an Integrated GARCH (IGARCH) model where the persistence of shocks is exactly unity, making it suitable for highly persistent volatility processes.

**Syntax**
```matlab
model = igarchfit(data, [p, q], [options])
```

**Parameters**
- `data`: numeric - Time series data to model
- `p`: integer - ARCH order (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `bounds`: Custom parameter bounds for estimation
- `optimOptions`: Options for fmincon optimizer

**Model Specification**
The IGARCH(p,q) model is specified as:

$$h_t = \omega + \sum_{i=1}^{p-1} \alpha_i \epsilon_{t-i}^2 + (1 - \sum_{i=1}^{p-1} \alpha_i - \sum_{j=1}^{q-1} \beta_j) \epsilon_{t-p}^2 + \sum_{j=1}^{q-1} \beta_j h_{t-j} + \beta_q h_{t-q}$$

with the constraint that:

$$\sum_{i=1}^{p} \alpha_i + \sum_{j=1}^{q} \beta_j = 1$$

**Example**
```matlab
% Load data
returns = randn(1000, 1);

% Estimate IGARCH(1,1) model with normal errors
model = igarchfit(returns, [1, 1]);

% Display results
disp('IGARCH Parameter Estimates:');
disp(model.parameters);
```

**See Also**
- [agarchfit](#agarchfit) - Asymmetric GARCH model estimation
- [egarchfit](#egarchfit) - Exponential GARCH model estimation
- [garchfor](#garchfor) - GARCH model forecasting

### tarchfit - Threshold ARCH/GARCH Model
Estimates a Threshold ARCH/GARCH (TARCH) model that incorporates asymmetric response to positive and negative shocks using a threshold specification.

**Syntax**
```matlab
model = tarchfit(data, [p, o, q], [options])
```

**Parameters**
- `data`: numeric - Time series data to model
- `p`: integer - ARCH order (default: 1)
- `o`: integer - Order of threshold terms (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `bounds`: Custom parameter bounds for estimation
- `optimOptions`: Options for fmincon optimizer

**Model Specification**
The TARCH(p,o,q) model is specified as:

$$h_t = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{k=1}^o \gamma_k I_{t-k} \epsilon_{t-k}^2 + \sum_{j=1}^q \beta_j h_{t-j}$$

where:
- $h_t$ is the conditional variance at time t
- $\epsilon_t$ is the innovation at time t
- $I_t$ is an indicator function where $I_t = 1$ if $\epsilon_t < 0$ and $I_t = 0$ otherwise
- $\omega$, $\alpha_i$, $\gamma_k$, and $\beta_j$ are model parameters

**Example**
```matlab
% Load data
returns = randn(1000, 1);

% Estimate TARCH(1,1,1) model with skewed t errors
options = struct('distribution', 'SKEWT');
model = tarchfit(returns, [1, 1, 1], options);

% Display results
disp('TARCH Parameter Estimates:');
disp(model.parameters);
```

**See Also**
- [agarchfit](#agarchfit) - Asymmetric GARCH model estimation
- [egarchfit](#egarchfit) - Exponential GARCH model estimation
- [garchfor](#garchfor) - GARCH model forecasting

### nagarchfit - Nonlinear Asymmetric GARCH Model
Estimates a Nonlinear Asymmetric GARCH (NAGARCH) model that accounts for the leverage effect through a nonlinear specification.

**Syntax**
```matlab
model = nagarchfit(data, [p, q], [options])
```

**Parameters**
- `data`: numeric - Time series data to model
- `p`: integer - ARCH order (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `bounds`: Custom parameter bounds for estimation
- `optimOptions`: Options for fmincon optimizer

**Model Specification**
The NAGARCH(p,q) model is specified as:

$$h_t = \omega + \sum_{i=1}^p \alpha_i (\epsilon_{t-i} + \gamma \sqrt{h_{t-i}})^2 + \sum_{j=1}^q \beta_j h_{t-j}$$

where:
- $h_t$ is the conditional variance at time t
- $\epsilon_t$ is the innovation at time t
- $\omega$, $\alpha_i$, $\gamma$, and $\beta_j$ are model parameters
- $\gamma$ captures the leverage effect

**Example**
```matlab
% Load data
returns = randn(1000, 1);

% Estimate NAGARCH(1,1) model with Student's t errors
options = struct('distribution', 'STUDENTST');
model = nagarchfit(returns, [1, 1], options);

% Display results
disp('NAGARCH Parameter Estimates:');
disp(model.parameters);
```

**See Also**
- [agarchfit](#agarchfit) - Asymmetric GARCH model estimation
- [egarchfit](#egarchfit) - Exponential GARCH model estimation
- [garchfor](#garchfor) - GARCH model forecasting

## Multivariate Volatility Models
Multivariate GARCH models extend the univariate framework to model the time-varying covariance structure between multiple time series, which is essential for portfolio optimization, risk management, and understanding cross-asset dependencies.

### ccc_mvgarch - Constant Conditional Correlation MVGARCH
Estimates a Constant Conditional Correlation (CCC) Multivariate GARCH model that combines univariate GARCH processes with constant correlation structure.

**Syntax**
```matlab
model = ccc_mvgarch(data, [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×k matrix for k series)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, conditional covariances, and diagnostics

**Options**
- `univariatePQ`: Cell array of [p,q] pairs for each series' GARCH specification
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `univariateOptions`: Cell array of option structures for univariate models
- `optimOptions`: Options for correlation parameter optimization

**Model Specification**
The CCC-MVGARCH model combines univariate GARCH processes with a constant correlation matrix:

$$h_{i,t} = \omega_i + \sum_{p=1}^{P_i} \alpha_{i,p} \epsilon_{i,t-p}^2 + \sum_{q=1}^{Q_i} \beta_{i,q} h_{i,t-q}$$
$$H_t = D_t R D_t$$

where:
- $h_{i,t}$ is the conditional variance of series i at time t
- $D_t$ is a diagonal matrix of conditional standard deviations $\sqrt{h_{i,t}}$
- $R$ is the constant correlation matrix
- $H_t$ is the conditional covariance matrix

**Example**
```matlab
% Generate sample multivariate data
returns = randn(1000, 3);

% Estimate CCC-MVGARCH with GARCH(1,1) for each series
options = struct();
options.univariatePQ = {[1,1], [1,1], [1,1]};
model = ccc_mvgarch(returns, options);

% Display correlation matrix
disp('Constant Correlation Matrix:');
disp(model.R);
```

**See Also**
- [dcc_mvgarch](#dcc_mvgarch) - Dynamic Conditional Correlation MVGARCH
- [bekk_mvgarch](#bekk_mvgarch) - BEKK MVGARCH model

### dcc_mvgarch - Dynamic Conditional Correlation MVGARCH
Estimates a Dynamic Conditional Correlation (DCC) Multivariate GARCH model that allows the correlation structure to evolve over time, capturing changing relationships between series.

**Syntax**
```matlab
model = dcc_mvgarch(data, [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×k matrix for k series)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, time-varying correlation matrices, conditional covariances, and diagnostics

**Options**
- `univariatePQ`: Cell array of [p,q] pairs for each series' GARCH specification
- `dccPQ`: [p,q] pair for DCC specification (default: [1,1])
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `univariateOptions`: Cell array of option structures for univariate models
- `optimOptions`: Options for DCC parameter optimization
- `forecast`: Forecast horizon for conditional correlations and covariances

**Model Specification**
The DCC-MVGARCH model extends the CCC framework with time-varying correlations:

$$h_{i,t} = \omega_i + \sum_{p=1}^{P_i} \alpha_{i,p} \epsilon_{i,t-p}^2 + \sum_{q=1}^{Q_i} \beta_{i,q} h_{i,t-q}$$
$$Q_t = (1-\sum_{i=1}^p a_i - \sum_{j=1}^q b_j)\bar{Q} + \sum_{i=1}^p a_i (\varepsilon_{t-i}\varepsilon_{t-i}') + \sum_{j=1}^q b_j Q_{t-j}$$
$$R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}$$
$$H_t = D_t R_t D_t$$

where:
- $h_{i,t}$ is the conditional variance of series i at time t
- $D_t$ is a diagonal matrix of conditional standard deviations $\sqrt{h_{i,t}}$
- $\varepsilon_t$ are standardized residuals
- $\bar{Q}$ is the unconditional covariance of standardized residuals
- $Q_t$ is a quasi-correlation matrix
- $R_t$ is the time-varying correlation matrix
- $H_t$ is the conditional covariance matrix
- $a_i$ and $b_j$ are DCC parameters

**Example**
```matlab
% Generate sample multivariate data
returns = randn(1000, 3);

% Estimate DCC-MVGARCH with GARCH(1,1) for each series
options = struct();
options.univariatePQ = {[1,1], [1,1], [1,1]};
options.dccPQ = [1,1];
options.forecast = 10;  % Generate 10-step ahead forecasts
model = dcc_mvgarch(returns, options);

% Plot time-varying correlations
t = 1:length(returns);
plot(t, squeeze(model.Rt(1,2,:)));
title('Dynamic Correlation Between Series 1 and 2');
xlabel('Time');
ylabel('Correlation');
```

**See Also**
- [ccc_mvgarch](#ccc_mvgarch) - Constant Conditional Correlation MVGARCH
- [bekk_mvgarch](#bekk_mvgarch) - BEKK MVGARCH model
- [gogarch](#gogarch) - Generalized Orthogonal GARCH

### bekk_mvgarch - BEKK MVGARCH Model
Estimates a BEKK Multivariate GARCH model that directly models the conditional covariance matrix without separating variances and correlations.

**Syntax**
```matlab
model = bekk_mvgarch(data, [p, q], [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×k matrix for k series)
- `p`: integer - ARCH order (default: 1)
- `q`: integer - GARCH order (default: 1)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, conditional covariance matrices, and diagnostics

**Options**
- `variant`: 'full', 'diagonal' (default), or 'scalar'
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `startingVals`: Starting parameter values for optimization
- `optimOptions`: Options for parameter optimization
- `forecast`: Forecast horizon for conditional covariances

**Model Specification**
The BEKK-MVGARCH model parameterizes the conditional covariance matrix directly:

$$H_t = C'C + \sum_{i=1}^p \sum_{k=1}^K A_{ik}' \epsilon_{t-i} \epsilon_{t-i}' A_{ik} + \sum_{j=1}^q \sum_{k=1}^K B_{jk}' H_{t-j} B_{jk}$$

where:
- $H_t$ is the conditional covariance matrix at time t
- $\epsilon_t$ is the vector of innovations at time t
- $C$, $A_{ik}$, and $B_{jk}$ are parameter matrices
- $K$ is the model order (K=1 for standard BEKK)

The 'diagonal' variant restricts A and B to be diagonal, while the 'scalar' variant further restricts them to be scalar multiples of identity matrices.

**Example**
```matlab
% Generate sample multivariate data
returns = randn(1000, 2);

% Estimate diagonal BEKK-MVGARCH(1,1) model
options = struct();
options.variant = 'diagonal';
model = bekk_mvgarch(returns, 1, 1, options);

% Extract conditional volatilities
vol1 = sqrt(squeeze(model.H(1,1,:)));
vol2 = sqrt(squeeze(model.H(2,2,:)));

% Plot conditional volatilities
plot(1:length(returns), [vol1, vol2]);
legend('Series 1 Volatility', 'Series 2 Volatility');
title('BEKK-MVGARCH Conditional Volatilities');
```

**See Also**
- [ccc_mvgarch](#ccc_mvgarch) - Constant Conditional Correlation MVGARCH
- [dcc_mvgarch](#dcc_mvgarch) - Dynamic Conditional Correlation MVGARCH
- [gogarch](#gogarch) - Generalized Orthogonal GARCH

### gogarch - Generalized Orthogonal GARCH
Estimates a Generalized Orthogonal GARCH (GO-GARCH) model that uses factor decomposition to reduce dimensionality and capture the variance-covariance structure.

**Syntax**
```matlab
model = gogarch(data, [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×k matrix for k series)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, conditional covariance matrices, factors, and diagnostics

**Options**
- `factorGarchPQ`: Cell array of [p,q] pairs for each factor's GARCH specification
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `k_factors`: Number of factors to use (default: number of series)
- `optimOptions`: Options for parameter optimization
- `forecast`: Forecast horizon for conditional covariances

**Model Specification**
The GO-GARCH model uses a factor approach to model multivariate volatility:

$$r_t = AZ_t$$
$$Z_t = \Lambda_t^{1/2} \eta_t$$
$$\lambda_{i,t} = \omega_i + \sum_{p=1}^{P_i} \alpha_{i,p} z_{i,t-p}^2 + \sum_{q=1}^{Q_i} \beta_{i,q} \lambda_{i,t-q}$$

where:
- $r_t$ is the vector of returns at time t
- $Z_t$ are latent factors
- $A$ is a mixing matrix
- $\Lambda_t$ is a diagonal matrix of factor conditional variances $\lambda_{i,t}$
- $\eta_t$ are i.i.d. standardized innovations

**Example**
```matlab
% Generate sample multivariate data
returns = randn(1000, 3);

% Estimate GO-GARCH model with GARCH(1,1) for each factor
options = struct();
options.k_factors = 2;  % Use 2 factors
options.factorGarchPQ = {[1,1], [1,1]};
model = gogarch(returns, options);

% Extract conditional covariances
cov12 = squeeze(model.H(1,2,:));
t = 1:length(returns);

% Plot conditional covariance between series 1 and 2
plot(t, cov12);
title('GO-GARCH Conditional Covariance');
xlabel('Time');
ylabel('Covariance');
```

**See Also**
- [ccc_mvgarch](#ccc_mvgarch) - Constant Conditional Correlation MVGARCH
- [dcc_mvgarch](#dcc_mvgarch) - Dynamic Conditional Correlation MVGARCH
- [bekk_mvgarch](#bekk_mvgarch) - BEKK MVGARCH model

## Forecasting
The MFE Toolbox provides robust forecasting capabilities for volatility models, supporting both point forecasts and simulation-based forecasts with various error distributions.

### garchfor - GARCH Model Forecasting
Generates multi-step ahead volatility forecasts for GARCH-type models.

**Syntax**
```matlab
[forecasts, errors] = garchfor(parameters, data, horizon, [model_type], [options])
```

**Parameters**
- `parameters`: struct or vector - Model parameters, either as structure from estimation functions or parameter vector
- `data`: vector - Historical time series data
- `horizon`: integer - Forecast horizon (number of periods ahead)
- `model_type`: string - (optional) Model type: 'GARCH' (default), 'EGARCH', 'AGARCH', 'TARCH', 'IGARCH', or 'NAGARCH'
- `options`: struct - (optional) Forecasting options

**Returns**
- `forecasts`: vector - Point forecasts of conditional variance for each period in the horizon
- `errors`: struct - Forecast error information including confidence intervals

**Options**
- `method`: Forecasting method: 'analytic' (default) or 'simulate'
- `nsim`: Number of simulation paths (default: 1000) when method='simulate'
- `probs`: Vector of probability levels for confidence intervals (default: [0.01, 0.05, 0.1, 0.9, 0.95, 0.99])
- `distribution`: Error distribution for simulation: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `dist_params`: Distribution parameters for simulation

**Example**
```matlab
% Estimate GARCH(1,1) model
model = tarchfit(returns, [1, 0, 1]);

% Generate 20-step ahead forecasts
[forecasts, errors] = garchfor(model.parameters, returns, 20, 'TARCH');

% Plot forecasts with 95% confidence intervals
plot(1:20, forecasts, 'b-', 'LineWidth', 2);
hold on;
plot(1:20, errors.upper(:,5), 'r--');  % 95% upper bound
plot(1:20, errors.lower(:,5), 'r--');  % 95% lower bound
legend('Forecast', '95% Confidence Interval');
title('GARCH Volatility Forecast');
xlabel('Horizon');
ylabel('Conditional Variance');
```

**See Also**
- [tarchfit](#tarchfit) - Threshold ARCH model estimation
- [egarchfit](#egarchfit) - Exponential GARCH model estimation
- [agarchfit](#agarchfit) - Asymmetric GARCH model estimation

## MEX Optimization
The MFE Toolbox leverages MEX optimization for performance-critical volatility model computations, offering significant speedups for model estimation and forecasting.

### MEX Implementation Details
The following volatility models benefit from MEX-accelerated implementations:

- **AGARCH**: Uses `agarch_core.mex*` for optimized computation of AGARCH recursion
- **EGARCH**: Uses `egarch_core.mex*` for optimized computation of EGARCH log-variance recursion
- **IGARCH**: Uses `igarch_core.mex*` for optimized computation of IGARCH recursion
- **TARCH**: Uses `tarch_core.mex*` for optimized computation of TARCH/GARCH variance recursion
- **Multivariate Models**: Use `composite_likelihood.mex*` for efficient likelihood computation

These MEX implementations provide several advantages:

1. **Performance**: Typically 5-10x faster than equivalent MATLAB code, especially for long time series
2. **Memory Efficiency**: Optimized memory management for large datasets
3. **Numerical Stability**: Careful implementation to maintain numerical precision
4. **Platform Independence**: Available for both Windows (*.mexw64) and Unix (*.mexa64) platforms

The MFE Toolbox automatically detects and uses MEX implementations when available, falling back to MATLAB implementations if necessary.

### Performance Comparison
The following table illustrates the performance advantage of MEX-accelerated implementations for volatility models:

| Model Type | Time Series Length | MATLAB Implementation | MEX Implementation | Speedup Factor |
|------------|-------------------|----------------------|-------------------|--------------|
| GARCH(1,1) | 1,000             | 0.65 sec             | 0.12 sec          | 5.4x           |
| GARCH(1,1) | 10,000            | 6.42 sec             | 0.58 sec          | 11.1x          |
| EGARCH(1,1,1) | 1,000           | 1.87 sec             | 0.22 sec          | 8.5x           |
| TARCH(1,1,1) | 1,000            | 0.93 sec             | 0.18 sec          | 5.2x           |
| DCC-MVGARCH | 1,000 (3 series)  | 8.21 sec             | 1.75 sec          | 4.7x           |

The performance advantage becomes more pronounced with larger datasets and more complex model specifications, making MEX optimization particularly valuable for production environments and large-scale analyses.

## Usage Examples
The following examples demonstrate how to use the volatility models for common financial applications.

### Basic Volatility Modeling
This example shows how to fit a TARCH model to financial returns and analyze the results.

```matlab
% Load sample financial returns data
load example_financial_data

% Fit TARCH(1,1,1) model with Student's t errors
options = struct('distribution', 'STUDENTST');
model = tarchfit(returns, [1, 1, 1], options);

% Display model parameters
disp('TARCH Parameter Estimates:');
disp(['omega: ' num2str(model.parameters.omega)]);
disp(['alpha: ' num2str(model.parameters.alpha')]);
disp(['gamma: ' num2str(model.parameters.gamma')]);
disp(['beta: ' num2str(model.parameters.beta')]);
disp(['DoF: ' num2str(model.parameters.nu)]);

% Plot conditional volatility
plot(sqrt(model.ht));
title('TARCH Conditional Volatility');
xlabel('Time');
ylabel('Conditional Standard Deviation');

% Basic model diagnostics
disp(['Log-likelihood: ' num2str(model.logL)]);
disp(['AIC: ' num2str(model.aic)]);
disp(['BIC: ' num2str(model.bic)]);
disp(['Persistence: ' num2str(model.persistence)]);
```

### Model Comparison
This example compares different volatility models to identify the best fit for a given dataset.