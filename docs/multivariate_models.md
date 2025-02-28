# Multivariate Models

The MFE Toolbox provides comprehensive implementations of multivariate time series and volatility models widely used in financial econometrics and quantitative finance. These models capture the dynamic interdependence between multiple financial time series, enabling sophisticated analysis of cross-asset relationships, cointegration, volatility spillovers, and portfolio risk. This document details the multivariate models implemented in the toolbox, their theoretical foundations, parameter specifications, implementation details, and usage examples.

## Table of Contents
- [Overview of Multivariate Models](#overview-of-multivariate-models)
- [Vector Autoregression (VAR)](#vector-autoregression-var)
    - [var_model - Vector Autoregression Estimation](#var_model---vector-autoregression-estimation)
    - [var_forecast - VAR Forecasting](#var_forecast---var-forecasting)
    - [var_irf - Impulse Response Functions](#var_irf---impulse-response-functions)
    - [var_fevd - Forecast Error Variance Decomposition](#var_fevd---forecast-error-variance-decomposition)
- [Vector Error Correction Model (VECM)](#vector-error-correction-model-vecm)
    - [vecm_model - Vector Error Correction Model Estimation](#vecm_model---vector-error-correction-model-estimation)
    - [Johansen Cointegration Testing](#johansen-cointegration-testing)
    - [vecm_forecast - VECM Forecasting](#vecm_forecast---vecm-forecasting)
- [Factor Models](#factor-models)
    - [factor_model - Factor Model Estimation](#factor_model---factor-model-estimation)
    - [Factor-Based Forecasting](#factor-based-forecasting)
    - [Factor Rotation Methods](#factor-rotation-methods)
- [Multivariate GARCH Models](#multivariate-garch-models)
    - [ccc_mvgarch - Constant Conditional Correlation MVGARCH](#ccc_mvgarch---constant-conditional-correlation-mvgarch)
    - [dcc_mvgarch - Dynamic Conditional Correlation MVGARCH](#dcc_mvgarch---dynamic-conditional-correlation-mvgarch)
    - [bekk_mvgarch - BEKK MVGARCH Model](#bekk_mvgarch---bekk-mvgarch-model)
    - [gogarch - Generalized Orthogonal GARCH](#gogarch---generalized-orthogonal-garch)
    - [Multivariate GARCH Forecasting](#multivariate-garch-forecasting)
- [High-Performance Computing](#high-performance-computing)
    - [MEX Implementation](#mex-implementation)
    - [Large-Scale Computation](#large-scale-computation)
    - [Performance Comparison](#performance-comparison)
- [Usage Examples](#usage-examples)
    - [VAR Analysis of Macroeconomic Variables](#var-analysis-of-macroeconomic-variables)
    - [Cointegration Analysis of Asset Prices](#cointegration-analysis-of-asset-prices)
    - [Dynamic Portfolio Optimization with DCC-MVGARCH](#dynamic-portfolio-optimization-with-dcc-mvgarch)
    - [Factor Model for Risk Decomposition](#factor-model-for-risk-decomposition)
- [Technical Notes](#technical-notes)
    - [Implementation Details](#implementation-details)
    - [Precision Considerations](#precision-considerations)
    - [Performance Optimization](#performance-optimization)
- [See Also](#see-also)

## Overview of Multivariate Models
Multivariate models in the MFE Toolbox are divided into two main categories:

1. **Multivariate Time Series Models** - For modeling conditional mean dynamics across multiple series
   * Vector Autoregression (VAR)
   * Vector Error Correction Model (VECM)
   * Factor Models

2. **Multivariate Volatility Models** - For modeling covariance structures across multiple time series
   * Constant Conditional Correlation (CCC-MVGARCH)
   * Dynamic Conditional Correlation (DCC-MVGARCH)
   * BEKK-MVGARCH
   * Generalized Orthogonal GARCH (GO-GARCH)

All models feature comprehensive error checking, robust numerical stability, support for various error distributions, and high-performance computation through MEX optimization where appropriate.

## Vector Autoregression (VAR)
Vector Autoregression (VAR) extends univariate autoregressive models to multiple time series, capturing linear interdependencies among multiple variables. VAR models are fundamental tools for analyzing multivariate time series relationships, forecasting, and performing impulse response analysis.

### var_model - Vector Autoregression Estimation
Estimates a Vector Autoregression (VAR) model for multivariate time series data.

**Syntax**
```matlab
model = var_model(y, p, [options])
```

**Parameters**
- `y`: matrix - Multivariate time series data (T×k matrix for k series)
- `p`: integer - Autoregressive order
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `constant`: logical - Include constant term (default: true)
- `trend`: logical - Include linear trend term (default: false)
- `seasonal`: integer - Include seasonal dummies (default: 0, no seasonality)
- `exogenous`: matrix - Exogenous variables (default: [])

**Model Specification**
The VAR(p) model is specified as:

$$Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \ldots + A_p Y_{t-p} + \varepsilon_t$$

where:
- $Y_t$ is a k×1 vector of variables at time t
- $c$ is a k×1 vector of constants
- $A_i$ are k×k coefficient matrices
- $\varepsilon_t$ is a k×1 vector of white noise innovations

**Example**
```matlab
% Load multivariate time series data
data = randn(1000, 3);  % 3 series, 1000 observations

% Estimate VAR(2) model with constant term
options = struct('constant', true);
model = var_model(data, 2, options);

% Display coefficient matrices
disp('VAR Coefficient Matrices:');
for i = 1:model.p
    disp(['A' num2str(i) ':']);
    disp(model.A(:,:,i));
end

% Display constant term
disp('Constant vector:');
disp(model.c);
```

### var_forecast - VAR Forecasting
Generates multi-step ahead forecasts from an estimated VAR model.

**Syntax**
```matlab
forecasts = var_forecast(model, h, [exo_future])
```

**Parameters**
- `model`: struct - Estimated VAR model structure from var_model
- `h`: integer - Forecast horizon (number of periods ahead)
- `exo_future`: matrix - (optional) Future values of exogenous variables

**Returns**
- `forecasts`: matrix - Point forecasts for each variable and horizon (h×k matrix)

**Example**
```matlab
% Generate 10-step ahead forecasts from VAR model
forecasts = var_forecast(model, 10);

% Plot forecasts for each variable
figure;
for i = 1:size(forecasts, 2)
    subplot(size(forecasts, 2), 1, i);
    plot(1:10, forecasts(:,i));
    title(['Forecast for Variable ' num2str(i)]);
    xlabel('Horizon');
    ylabel('Value');
end
```

### var_irf - Impulse Response Functions
Computes impulse response functions for an estimated VAR model, showing the dynamic effects of shocks to the system.

**Syntax**
```matlab
irf = var_irf(model, h, [method])
```

**Parameters**
- `model`: struct - Estimated VAR model structure from var_model
- `h`: integer - Impulse response horizon
- `method`: string - (optional) IRF identification method: 'cholesky' (default), 'generalized', or 'structural'

**Returns**
- `irf`: array - Impulse response coefficients (h×k×k array)

**Example**
```matlab
% Compute impulse responses for 20 periods using Cholesky decomposition
irfs = var_irf(model, 20, 'cholesky');

% Plot impulse response of variable 1 to a shock in variable 2
figure;
plot(0:20, squeeze(irfs(:,1,2)));
title('Response of Variable 1 to Shock in Variable 2');
xlabel('Periods');
ylabel('Response');
```

### var_fevd - Forecast Error Variance Decomposition
Computes forecast error variance decomposition for an estimated VAR model, showing the contribution of each variable's innovations to the forecast error variance.

**Syntax**
```matlab
fevd = var_fevd(model, h)
```

**Parameters**
- `model`: struct - Estimated VAR model structure from var_model
- `h`: integer - Decomposition horizon

**Returns**
- `fevd`: array - Variance decomposition proportions (h×k×k array)

**Example**
```matlab
% Compute variance decomposition for 20 periods
decomp = var_fevd(model, 20);

% Plot variance decomposition for variable 1
figure;
area(1:20, squeeze(decomp(:,1,:)));
title('Variance Decomposition for Variable 1');
xlabel('Horizon');
ylabel('Proportion');
legend('Shock 1', 'Shock 2', 'Shock 3');
```

## Vector Error Correction Model (VECM)
Vector Error Correction Models (VECM) extend VARs to incorporate cointegration relationships between non-stationary time series. VECMs capture both long-run equilibrium relationships and short-run dynamics, making them particularly valuable for analyzing financial and economic systems with common stochastic trends.

### vecm_model - Vector Error Correction Model Estimation
Estimates a Vector Error Correction Model for cointegrated multivariate time series data.

**Syntax**
```matlab
model = vecm_model(y, p, r, [options])
```

**Parameters**
- `y`: matrix - Multivariate time series data (T×k matrix for k series)
- `p`: integer - Lag order in VAR form (before differencing)
- `r`: integer - Cointegration rank (0 ≤ r < k)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, diagnostics, and fitted values

**Options**
- `constant`: integer - Constant specification: 0 (none), 1 (restricted), or 2 (unrestricted, default)
- `trend`: integer - Trend specification: 0 (none, default), 1 (restricted), or 2 (unrestricted)
- `ecdet`: string - Error correction deterministic term: 'none', 'const' (default), or 'trend'
- `method`: string - Estimation method: 'johansen' (default) or 'ml'

**Model Specification**
The VECM(p) model is specified as:

$$\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t$$

where:
- $Y_t$ is a k×1 vector of variables at time t
- $\Pi = \alpha \beta'$ where $\alpha$ is the adjustment matrix and $\beta$ contains the cointegrating vectors
- $\Gamma_i$ are k×k short-run coefficient matrices
- $\mu$ is a deterministic term (constant/trend)
- $\varepsilon_t$ is a k×1 vector of white noise innovations

**Example**
```matlab
% Load multivariate time series data
data = randn(1000, 3).cumsum();  % Non-stationary series

% Estimate VECM with cointegration rank 1
model = vecm_model(data, 2, 1);

% Display cointegrating vectors
disp('Cointegrating vectors (beta):');
disp(model.beta);

% Display adjustment coefficients
disp('Adjustment coefficients (alpha):');
disp(model.alpha);
```

### Johansen Cointegration Testing
The VECM implementation includes Johansen's procedure for testing cointegration rank and estimating cointegrating relationships.

**Syntax**
```matlab
[trace_stats, max_stats, crit_vals, r] = johansen_test(y, p, [options])
```

**Parameters**
- `y`: matrix - Multivariate time series data (T×k matrix for k series)
- `p`: integer - Lag order in VAR form
- `options`: struct - (optional) Test options

**Returns**
- `trace_stats`: vector - Trace test statistics
- `max_stats`: vector - Maximum eigenvalue test statistics
- `crit_vals`: matrix - Critical values for both tests
- `r`: integer - Estimated cointegration rank

**Example**
```matlab
% Test for cointegration rank
[trace, maxeig, cv, r] = johansen_test(data, 2);

% Display test results
disp('Johansen Cointegration Test Results:');
disp('Rank    Trace Statistic    Critical Value (5%)');
for i = 1:length(trace)
    disp([i-1, trace(i), cv(i,1)]);
end

disp(['Estimated cointegration rank: ' num2str(r)]);
```

### vecm_forecast - VECM Forecasting
Generates multi-step ahead forecasts from an estimated VECM model.

**Syntax**
```matlab
forecasts = vecm_forecast(model, h)
```

**Parameters**
- `model`: struct - Estimated VECM model structure from vecm_model
- `h`: integer - Forecast horizon (number of periods ahead)

**Returns**
- `forecasts`: matrix - Point forecasts for each variable and horizon (h×k matrix)

**Example**
```matlab
% Generate 10-step ahead forecasts from VECM model
forecasts = vecm_forecast(model, 10);

% Plot forecasts for each variable
figure;
for i = 1:size(forecasts, 2)
    subplot(size(forecasts, 2), 1, i);
    plot(1:10, forecasts(:,i));
    title(['Forecast for Variable ' num2str(i)]);
    xlabel('Horizon');
    ylabel('Value');
end
```

## Factor Models
Factor models in the MFE Toolbox provide dimensionality reduction techniques for multivariate financial time series, extracting common factors that drive asset returns. These models are essential for risk decomposition, structural analysis, and efficient parameter estimation in high-dimensional settings.

### factor_model - Factor Model Estimation
Estimates a factor model for multivariate financial time series using principal component analysis or maximum likelihood methods.

**Syntax**
```matlab
model = factor_model(data, k, [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×n matrix for n series)
- `k`: integer - Number of factors to extract
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with factors, loadings, and diagnostics

**Options**
- `method`: string - Estimation method: 'pca' (default) or 'ml' (maximum likelihood)
- `rotation`: string - Factor rotation method: 'none' (default), 'varimax', 'quartimax', or 'equamax'
- `scale`: logical - Standardize data before estimation (default: true)
- `threshold`: scalar - Variance explained threshold for automatic factor selection

**Model Specification**
The factor model is specified as:

$$X_t = \Lambda F_t + \varepsilon_t$$

where:
- $X_t$ is an n×1 vector of observed variables at time t
- $\Lambda$ is an n×k matrix of factor loadings
- $F_t$ is a k×1 vector of common factors at time t
- $\varepsilon_t$ is an n×1 vector of idiosyncratic errors

**Example**
```matlab
% Load multivariate financial returns
returns = randn(1000, 10);  % 10 assets, 1000 observations

% Estimate factor model with 3 factors using PCA
options = struct('method', 'pca', 'rotation', 'varimax');
model = factor_model(returns, 3, options);

% Display factor loadings
disp('Factor Loadings:');
disp(model.loadings);

% Display variance explained by each factor
disp('Variance Explained (%):');
disp(model.variance_explained);

% Extract factor time series
factors = model.factors;

% Plot factor time series
figure;
plot(factors);
title('Estimated Factors');
xlabel('Time');
legend('Factor 1', 'Factor 2', 'Factor 3');
```

### Factor-Based Forecasting
The factor model framework can be extended to forecasting applications by modeling factor dynamics.

**Example: Factor-Augmented VAR (FAVAR)**
```matlab
% Extract factors from large dataset
options = struct('method', 'pca');
factor_model = factor_model(data_large, 5, options);
factors = factor_model.factors;

% Combine factors with key observed variables
favar_data = [factors, key_variables];

% Estimate VAR model on the combined dataset
favar = var_model(favar_data, 2);

% Generate forecasts
forecasts = var_forecast(favar, 10);

% Map factor forecasts back to original variables using factor loadings
forecasted_factors = forecasts(:,1:5);
original_var_forecasts = forecasted_factors * factor_model.loadings';
```

### Factor Rotation Methods
The toolbox supports various factor rotation methods to enhance interpretability of factor loadings:

- **Varimax Rotation**: Maximizes the sum of variances of squared loadings, tending to produce factors with high loadings on few variables.

- **Quartimax Rotation**: Simplifies the rows of the loading matrix, making each variable load primarily on a single factor.

- **Equamax Rotation**: Compromise between varimax and quartimax, balancing the simplification of rows and columns.

**Example: Comparing Rotation Methods**
```matlab
% Estimate factor model with different rotation methods
rotation_methods = {'none', 'varimax', 'quartimax', 'equamax'};
loadings = cell(length(rotation_methods), 1);

for i = 1:length(rotation_methods)
    options = struct('method', 'pca', 'rotation', rotation_methods{i});
    model = factor_model(returns, 3, options);
    loadings{i} = model.loadings;
    
    % Display loadings with this rotation
    disp(['Loadings with ' rotation_methods{i} ' rotation:']);
    disp(loadings{i});
end
```

## Multivariate GARCH Models
Multivariate GARCH models extend univariate volatility models to capture time-varying covariance structures between multiple assets. These models are essential for portfolio optimization, risk management, and understanding volatility spillovers across financial markets.

### ccc_mvgarch - Constant Conditional Correlation MVGARCH
Estimates a Constant Conditional Correlation (CCC) Multivariate GARCH model that combines univariate GARCH processes with a constant correlation structure.

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
options.distribution = 'STUDENTST';
options.forecast = 10;  % Generate 10-step ahead forecasts
model = dcc_mvgarch(returns, options);

% Plot time-varying correlations
t = 1:length(returns);
plot(t, squeeze(model.Rt(1,2,:)));
title('Dynamic Correlation Between Series 1 and 2');
xlabel('Time');
ylabel('Correlation');
```

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

### gogarch - Generalized Orthogonal GARCH
Estimates a Generalized Orthogonal GARCH (GO-GARCH) model that uses orthogonal transformations to separate univariate GARCH processes.

**Syntax**
```matlab
model = gogarch(data, [options])
```

**Parameters**
- `data`: matrix - Multivariate time series data (T×k matrix for k series)
- `options`: struct - (optional) Estimation options

**Returns**
- `model`: struct - Estimated model structure with parameters, factors, factor loadings, conditional covariances, and diagnostics

**Options**
- `factorGarchPQ`: Cell array of [p,q] pairs for each factor's GARCH specification
- `distribution`: Error distribution: 'NORMAL' (default), 'STUDENTST', 'GED', or 'SKEWT'
- `k_factors`: Number of factors to use (default: number of series)
- `method`: Factor extraction method: 'pca' (default) or 'ica'
- `forecast`: Forecast horizon for conditional covariances

**Model Specification**
The GO-GARCH model uses orthogonal transformations and factor structure:

$$r_t = A z_t$$
$$z_t = \Lambda_t^{1/2} \eta_t$$

where:
- $r_t$ is the vector of returns at time t
- $A$ is the mixing matrix of factor loadings
- $z_t$ are independent factors with GARCH dynamics
- $\Lambda_t$ is a diagonal matrix with conditional variances of factors
- $\eta_t$ are i.i.d. standardized innovations

**Example**
```matlab
% Generate sample multivariate data
returns = randn(1000, 4);

% Estimate GO-GARCH model
options = struct();
options.factorGarchPQ = {[1,1], [1,1], [1,1], [1,1]};
options.method = 'pca';
model = gogarch(returns, options);

% Extract loadings matrix
disp('Factor Loadings:');
disp(model.A);

% Plot conditional correlations
t = 1:length(returns);
corr12 = squeeze(model.Rt(1,2,:));
plot(t, corr12);
title('GO-GARCH Conditional Correlation');
xlabel('Time');
ylabel('Correlation');
```

### Multivariate GARCH Forecasting
The multivariate GARCH models provide forecasting methods for conditional covariance matrices.

**For DCC models:**
```matlab
% Generate forecasts from estimated DCC model
forecasts = dcc_forecast(model, 10);

% Extract forecasted correlation between assets 1 and 2
corr_forecasts = squeeze(forecasts.R(1,2,:));

% Plot correlation forecasts
plot(1:10, corr_forecasts);
title('DCC Correlation Forecast');
xlabel('Horizon');
ylabel('Correlation');
```

**For BEKK models:**
```matlab
% Generate forecasts from estimated BEKK model
forecasts = bekk_forecast(model, 10);

% Extract forecasted variances
var1_forecasts = squeeze(forecasts.H(1,1,:));
var2_forecasts = squeeze(forecasts.H(2,2,:));

% Plot volatility forecasts
plot(1:10, [sqrt(var1_forecasts), sqrt(var2_forecasts)]);
legend('Asset 1 Volatility', 'Asset 2 Volatility');
title('BEKK Volatility Forecasts');
xlabel('Horizon');
ylabel('Conditional Standard Deviation');
```

## High-Performance Computing
The MFE Toolbox leverages MEX optimization for computationally intensive multivariate model operations, particularly focusing on likelihood evaluation and correlation updates in multivariate GARCH models.

### MEX Implementation
The following components benefit from MEX acceleration:

- **composite_likelihood.c**: Provides optimized computation of multivariate likelihood functions, especially useful for DCC models with large cross-sections
- **armaxerrors.c**: Optimized computation of residuals for multivariate time series models

These MEX implementations deliver several advantages:

1. **Performance**: Typically 5-10x faster than equivalent MATLAB code for large datasets
2. **Memory Efficiency**: Optimized memory management for large covariance matrices
3. **Numerical Stability**: Careful implementation for correlation matrix positive definiteness

The toolbox automatically detects and uses MEX implementations when available, falling back to MATLAB implementations when necessary.

### Large-Scale Computation
For large cross-sections of assets, the MFE Toolbox implements special techniques:

- **Two-Stage Estimation**: Sequential estimation of univariate models followed by correlation parameters
- **Composite Likelihood**: For very large systems (dozens or hundreds of assets), the composite likelihood approach offers tractable estimation by focusing on pairs or small groups of assets
- **Efficient Matrix Operations**: Leverages blocked operations and careful memory management for large covariance matrices

**Example: Large-Scale DCC Estimation**
```matlab
% For a large system (e.g., 50+ assets)
returns = randn(1000, 50);  % 50 assets, 1000 observations

% Configure options for large-scale estimation
options = struct();
options.univariatePQ = repmat({[1,1]}, 50, 1);  % GARCH(1,1) for all series
options.dccPQ = [1,1];
options.composite = true;  % Use composite likelihood approach
options.composite_size = 5;  % Consider groups of 5 assets

% Estimate model
model = dcc_mvgarch(returns, options);
```

### Performance Comparison
The following table illustrates performance advantages of optimized implementations:

| Model | Assets | Observations | MATLAB Implementation | MEX Implementation | Speedup |
|-------|--------|--------------|----------------------|-------------------|---------||
| VAR(2) | 5 | 1,000 | 0.21 sec | - | - |
| VECM(2) | 4 | 1,000 | 0.35 sec | - | - |
| Factor Model | 20 | 1,000 | 0.18 sec | - | - |
| CCC-MVGARCH | 5 | 1,000 | 3.27 sec | 0.92 sec | 3.6x |
| DCC-MVGARCH | 5 | 1,000 | 8.19 sec | 1.73 sec | 4.7x |
| DCC-MVGARCH | 10 | 1,000 | 31.5 sec | 4.2 sec | 7.5x |
| BEKK-MVGARCH | 3 | 1,000 | 7.82 sec | 2.15 sec | 3.6x |

The performance advantage becomes more pronounced with larger datasets and more complex model specifications, making MEX optimization particularly valuable for multivariate models with many assets.

## Usage Examples
The following examples demonstrate how to use multivariate models for common financial applications.

### VAR Analysis of Macroeconomic Variables
This example shows how to fit a VAR model to macroeconomic time series and perform impulse response analysis.