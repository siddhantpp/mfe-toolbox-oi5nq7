# High-Frequency Analysis

## Overview
High-frequency analysis is a specialized area of financial econometrics that focuses on analyzing market data sampled at very high frequencies, such as tick-by-tick or second-by-second. The MFE Toolbox provides a comprehensive suite of functions for high-frequency financial data analysis, focusing on estimating realized volatility, detecting price jumps, and addressing market microstructure noise.

High-frequency data provides rich information about market dynamics that cannot be captured with daily or lower-frequency data, including intraday volatility patterns, price jumps, market microstructure effects, and liquidity dynamics. The high-frequency analysis capabilities in the MFE Toolbox are primarily implemented through the `realized` module, which contains specialized functions for volatility estimation and jump detection.

## Theoretical Background
High-frequency financial econometrics is built on the theoretical foundation of continuous-time finance models, where asset prices are assumed to follow a continuous-time stochastic process, possibly with jumps:

$$dP_t = \mu_t dt + \sigma_t dW_t + J_t dN_t$$

Where $P_t$ is the logarithmic price process, $\mu_t$ is the drift term, $\sigma_t$ is the spot volatility, $W_t$ is a standard Brownian motion, $J_t$ is the jump size, and $N_t$ is a Poisson process governing the arrival of jumps.

In this framework, the quadratic variation of the price process represents the total variation, which can be decomposed into continuous variation (integrated variance) and discontinuous variation (jumps):

$$QV = \int_{0}^{T} \sigma_s^2 ds + \sum_{j=1}^{N_T} J_j^2$$

Realized volatility estimators aim to measure this quadratic variation using discretely sampled high-frequency returns. The MFE Toolbox implements various estimators to address challenges including market microstructure noise and jumps.

# Realized Volatility Measures

## Standard Realized Volatility
Standard realized volatility (RV) is the most basic estimator of integrated variance, computed as the sum of squared intraday returns:

$$RV_t = \sum_{i=1}^{n} r_{t,i}^2$$

Where $r_{t,i}$ represents the high-frequency returns within day $t$.

In the MFE Toolbox, standard realized volatility is implemented in the `rv_compute.m` function. This function provides several options for customization:

- Scaling factors for annualization
- Size adjustment
- Custom weighting schemes
- Subsampling methods to mitigate microstructure noise
- Jackknife bias correction

Example usage:

```matlab
% Basic usage to compute realized volatility from 5-minute returns
rv = rv_compute(fiveminreturns);

% With scaling for annualization (252 trading days)
options.scale = 252;
rv = rv_compute(fiveminreturns, options);

% With subsampling to reduce noise impact
options.method = 'subsample';
options.subSample = 3;
rv = rv_compute(fiveminreturns, options);
```

## Bipower Variation
Bipower variation (BV) is a jump-robust estimator of integrated variance that uses products of adjacent absolute returns:

$$BV_t = \mu_1^{-2} \sum_{i=2}^{n} |r_{t,i-1}| \cdot |r_{t,i}|$$

Where $\mu_1 = \sqrt{2/\pi}$ is a scaling factor.

The key advantage of bipower variation is its robustness to jumps. While standard RV captures both the continuous and jump components of price variation, BV converges to only the continuous component (integrated variance) even in the presence of jumps.

The MFE Toolbox implements bipower variation in the `bv_compute.m` function:

```matlab
% Basic usage
bv = bv_compute(returns);

% With custom scaling factor
options.scaleFactor = 0.8;
bv = bv_compute(returns, options);
```

Bipower variation is particularly useful when price jumps are suspected to impact volatility estimates or when disentangling continuous and jump components is important.

## Kernel-Based Estimation
Kernel-based realized volatility estimators are designed to address market microstructure noise, which can significantly bias standard realized volatility estimates at very high sampling frequencies.

The MFE Toolbox implements kernel-based estimation in the `rv_kernel.m` function, which supports various kernel types:
- Bartlett-Parzen kernel
- Quadratic kernel
- Cubic kernel
- Exponential kernel
- Tukey-Hanning kernel

The general form of a kernel-based realized volatility estimator is:

$$RV^K = \gamma_0 + \sum_{h=1}^{H} k\left(\frac{h}{H}\right) \gamma_h$$

Where $\gamma_h$ is the h-th order autocovariance of returns, $k(x)$ is the kernel function, and $H$ is the bandwidth parameter.

Usage example:

```matlab
% Basic usage with default settings (Bartlett-Parzen kernel)
rv = rv_kernel(returns);

% With custom kernel and bandwidth
options.kernelType = 'Tukey-Hanning';
options.bandwidth = 10;
rv = rv_kernel(returns, options);

% With automatic bandwidth selection
options.bandwidth = [];
rv = rv_kernel(returns, options);
```

## Spectral Analysis
Spectral analysis of realized volatility examines the frequency domain properties of volatility, which can reveal patterns and periodicity that are not apparent in the time domain.

The `realized_spectrum.m` function in the MFE Toolbox implements spectral analysis for high-frequency returns, providing insights into periodicity in volatility, long-range dependence, and volatility clustering in the frequency domain.

Example usage:
```matlab
% Basic spectral analysis
spectrum_results = realized_spectrum(returns);

% Plot the spectral density
plot(spectrum_results.frequencies, spectrum_results.spectral_density);
title('Volatility Spectrum');
xlabel('Frequency');
ylabel('Spectral Density');
```

# Jump Detection

## Jump Test Implementation
Detecting price jumps is crucial for understanding the dynamics of financial markets, risk management, and improving volatility forecasts. The MFE Toolbox implements jump detection through the `jump_test.m` function, which is based on the comparison of realized volatility (RV) and bipower variation (BV).

The test statistic is based on the ratio of RV to BV:

$$z = \frac{RV - BV}{\sqrt{\theta \cdot BV^2 / T}}$$

Where $\theta = (\pi^2/4 + \pi - 5)$ is a constant related to the asymptotic distribution, and $T$ is the sample size.

Under the null hypothesis of no jumps, this statistic follows a standard normal distribution asymptotically. A significantly positive value indicates the presence of jumps.

Usage:
```matlab
% Basic jump test
results = jump_test(returns);

% Display key results
disp(['Jump Test Statistic: ', num2str(results.zStatistic)]);
disp(['p-value: ', num2str(results.pValue)]);
disp(['Jump Component (%): ', num2str(results.jumpComponent * 100)]);

% Testing with specific significance level
options.alpha = 0.01;  % 1% significance level
results = jump_test(returns, options);
```

## Separating Continuous and Jump Components
Once jumps are detected, it's often useful to separate the total price variation into its continuous and jump components. The MFE Toolbox provides this decomposition as part of the `jump_test.m` function:

- Jump component: $JV = \max(0, RV - BV)$
- Continuous component: $CV = \min(RV, BV)$

The relative jump component can be calculated as:
$$JV_{\%} = \frac{JV}{RV} = \frac{RV - BV}{RV}$$

This decomposition is useful for analyzing the impact of jumps on total price variation, creating more robust volatility forecasts based only on the continuous component, and studying the relationship between news events and price jumps.

Example:
```matlab
% Compute jump test and extract components
results = jump_test(returns);

% Extract continuous and jump components
continuous_vol = results.contComponent;
jump_vol = results.jumpComponent * results.rv;

% Analyze the proportion of variance due to jumps
jump_proportion = results.jumpComponent;
disp(['Proportion of variance due to jumps: ', num2str(jump_proportion * 100), '%']);

% Plot the decomposition
figure;
bar([continuous_vol, jump_vol], 'stacked');
legend('Continuous Component', 'Jump Component');
title('Decomposition of Total Variance');
```

# Practical Implementation

## Data Preparation
Proper preparation of high-frequency data is critical for accurate realized volatility estimation. The MFE Toolbox functions expect well-prepared data, and several preprocessing steps are typically necessary:

1. **Regular Time Sampling**: 
   - Convert irregular tick data to regularly spaced intervals (e.g., 5-minute returns)
   - Use methods like previous-tick or linear interpolation for price aggregation

2. **Data Cleaning**:
   - Remove outliers and erroneous observations
   - Filter out non-trading periods (pre-market, after-hours)
   - Handle trading halts and market closures

3. **Return Calculation**:
   - Use log returns for additivity properties: $r_t = \log(P_t) - \log(P_{t-1})$
   - Ensure returns are properly aligned with timestamps

4. **Overnight Returns**:
   - Decide whether to include or exclude overnight returns
   - If excluded, ensure proper adjustment of daily measures

5. **Missing Data Handling**:
   - Implement consistent approaches for periods with missing observations
   - Consider interpolation methods or treating as zero returns

Example preprocessing workflow:
```matlab
% Load raw tick data
raw_data = load_tick_data('asset_ticks.csv');

% Convert to regular intervals (5-minute)
[regular_prices, timestamps] = convert_to_regular(raw_data.ticks, raw_data.timestamps, 5);

% Calculate log returns
log_returns = diff(log(regular_prices));

% Remove outliers (returns exceeding 5 standard deviations)
std_returns = std(log_returns);
valid_idx = abs(log_returns) <= 5 * std_returns;
clean_returns = log_returns .* valid_idx;

% Now ready for realized measures
rv = rv_compute(clean_returns);
```

## Sampling Frequency Considerations
Choosing the appropriate sampling frequency is a critical decision in high-frequency analysis that involves a fundamental trade-off:

- **Higher Frequency**:
  - More information about the price process
  - Theoretically more accurate volatility estimation
  - BUT: More susceptible to market microstructure noise

- **Lower Frequency**:
  - Reduces market microstructure effects
  - More stable estimates
  - BUT: Discards potentially valuable information

The optimal sampling frequency depends on the liquidity of the asset, the specific market microstructure, the purpose of the analysis, and the estimation method being used.

Common approaches to determining optimal frequency include volatility signature plots (plotting RV against sampling frequency) and minimizing mean squared error of volatility estimates.

As a general guideline, 5-minute to 15-minute sampling frequencies are often used for liquid assets, while less liquid assets may require 30-minute or hourly sampling.

## Handling Market Microstructure Noise
Market microstructure noise arises from various market frictions such as bid-ask bounce, discreteness of prices, and asynchronous trading. This noise can significantly bias realized volatility estimates, especially at high sampling frequencies.

The MFE Toolbox provides several approaches to mitigate microstructure noise:

1. **Subsampling** in `rv_compute`:
   ```matlab
   options.method = 'subsample';
   options.subSample = 5;
   rv = rv_compute(returns, options);
   ```
   This averages multiple RV estimates starting from different initial observations to reduce noise impact.

2. **Kernel-based Estimation** via `rv_kernel`:
   ```matlab
   options.kernelType = 'Bartlett-Parzen';
   options.bandwidth = 'auto';  % Automatic bandwidth selection
   rv = rv_kernel(returns, options);
   ```
   Kernel-based estimators weight autocovariances to reduce noise bias.

3. **Jackknife Bias Correction**:
   ```matlab
   options.jackknife = true;
   rv = rv_compute(returns, options);
   ```
   This approach corrects for first-order autocorrelation induced by noise.

The choice of method depends on the specific characteristics of the asset and market, the sampling frequency used, and the required level of accuracy.

## Overnight Returns
Handling overnight returns presents a unique challenge in high-frequency analysis. The market dynamics during non-trading hours differ significantly from those during trading hours, often with different volatility patterns.

Approaches to handling overnight returns in the MFE Toolbox:

1. **Exclude Overnight Returns**:
   The simplest approach is to compute realized measures using only intraday returns.

2. **Separate Treatment**:
   ```matlab
   % Separate overnight and intraday returns
   overnight_return = open_price / previous_close - 1;
   
   % Compute intraday RV
   intraday_rv = rv_compute(intraday_returns);
   
   % Combine for total daily variance
   total_variance = intraday_rv + overnight_return^2;
   ```

3. **Overnight Return Adjustment** in `rv_kernel`:
   ```matlab
   options.handleOvernight = true;
   rv = rv_kernel(returns, options);
   ```
   This option applies special weighting to returns that span overnight periods.

Considerations when dealing with overnight returns include overnight jumps due to news releases, different information processing rates outside trading hours, and the impact of global markets on overnight price movements.

# Performance Optimization

## Memory Management
High-frequency data analysis often involves processing large datasets, making efficient memory management crucial. The MFE Toolbox functions are designed with memory efficiency in mind, but additional considerations can further optimize performance:

1. **Batched Processing**:
   Process data in smaller time windows when dealing with very large datasets.
   ```matlab
   % Process one month at a time instead of the entire dataset
   monthly_rv = zeros(num_months, 1);
   for month = 1:num_months
       month_idx = month_start_idx(month):month_end_idx(month);
       month_returns = returns(month_idx);
       monthly_rv(month) = rv_compute(month_returns);
   end
   ```

2. **Pre-allocation**:
   Always pre-allocate arrays when performing iterative computations.
   ```matlab
   results = zeros(num_assets, 1);
   for i = 1:num_assets
       results(i) = rv_compute(returns(:,i));
   end
   ```

3. **Memory Mapping**:
   For extremely large datasets that don't fit in memory, consider using memory-mapped files.

## Computational Efficiency
The MFE Toolbox functions for high-frequency analysis are optimized for performance, but additional strategies can further improve computational efficiency:

1. **Vectorization**:
   Maximize the use of vectorized operations instead of loops when processing returns.
   ```matlab
   % Efficient:
   rv = sum(returns.^2);
   ```

2. **Parameter Selection**:
   Choose computation parameters wisely to balance accuracy and speed:
   - Lower bandwidth values in `rv_kernel` improve speed at the cost of some accuracy
   - Simpler kernel types generally compute faster

3. **Parallel Processing**:
   For multiple assets or multiple time periods, utilize parallel processing:
   ```matlab
   % Assuming returns is a matrix with each column representing an asset
   num_assets = size(returns, 2);
   rv_results = zeros(1, num_assets);
   
   parfor i = 1:num_assets
       rv_results(i) = rv_compute(returns(:,i));
   end
   ```

These optimization techniques are particularly important when working with very long time series, multiple assets simultaneously, or when computational resources are limited.

# Examples and Applications

## Basic Usage Example
Here's a step-by-step example of using the high-frequency analysis functions in the MFE Toolbox:

```matlab
% Load high-frequency return data
% Assuming your data is in a CSV file with timestamps and returns
data = readmatrix('high_frequency_data.csv');
timestamps = data(:,1);
returns = data(:,2);

% 1. Compute standard realized volatility
rv = rv_compute(returns);
disp(['Realized Volatility: ', num2str(rv)]);

% Convert to annualized percentage volatility (assuming daily data with 252 trading days)
annualized_vol = sqrt(252 * rv) * 100;
disp(['Annualized Volatility (%): ', num2str(annualized_vol)]);

% 2. Compute bipower variation (jump-robust measure)
bv = bv_compute(returns);
disp(['Bipower Variation: ', num2str(bv)]);
disp(['Jump Contribution (%): ', num2str((rv-bv)/rv*100)]);

% 3. Test for the presence of jumps
jump_results = jump_test(returns);
disp(['Jump Test Statistic: ', num2str(jump_results.zStatistic)]);
disp(['p-value: ', num2str(jump_results.pValue)]);

if jump_results.pValue < 0.05
    disp('Significant jumps detected at 5% level');
else
    disp('No significant jumps detected at 5% level');
end

% 4. Compute noise-robust kernel-based RV
options = struct();
options.kernelType = 'Bartlett-Parzen';
options.bandwidth = 10;
rv_kernel_est = rv_kernel(returns, options);
disp(['Kernel-based RV: ', num2str(rv_kernel_est)]);

% 5. Visualize results
figure;
bar([rv, bv, rv_kernel_est]);
set(gca, 'XTickLabel', {'RV', 'BV', 'Kernel RV'});
title('Comparison of Volatility Estimators');
ylabel('Variance');
```

## Advanced Applications
The high-frequency analysis functions in the MFE Toolbox can be combined with other components for more advanced applications:

### Volatility Forecasting with HAR-RV Model
```matlab
% Assuming rv_daily contains daily realized volatility estimates
% Implement the HAR-RV model of Corsi (2009)
T = length(rv_daily);
rv_weekly = zeros(T, 1);
rv_monthly = zeros(T, 1);

% Compute weekly and monthly RV components
for t = 5:T
    rv_weekly(t) = mean(rv_daily(t-4:t));
end
for t = 22:T
    rv_monthly(t) = mean(rv_daily(t-21:t));
end

% Prepare data for regression (lagged by 1)
Y = rv_daily(23:end);
X = [ones(T-22, 1), rv_daily(22:end-1), rv_weekly(22:end-1), rv_monthly(22:end-1)];

% Estimate HAR-RV model
beta = (X'*X)\(X'*Y);

% Generate forecasts
forecast = X * beta;

% Evaluate forecast performance
rmse = sqrt(mean((Y - forecast).^2));
disp(['Forecast RMSE: ', num2str(rmse)]);
```

### Jump-Diffusion Model Calibration
```matlab
% Assuming daily realized measures have been computed
rv_series = rv_daily;
bv_series = bv_daily;
jump_series = rv_daily - bv_daily;
jump_days = jump_series > 0;

% Analyze jump frequency
jump_frequency = sum(jump_days) / length(rv_series);
disp(['Jump Frequency: ', num2str(jump_frequency*100), '%']);

% Estimate jump size distribution
jump_sizes = sqrt(jump_series(jump_days));
jump_mean = mean(jump_sizes);
jump_std = std(jump_sizes);

disp(['Average Jump Size: ', num2str(jump_mean)]);
disp(['Jump Size Std Dev: ', num2str(jump_std)]);

% Calibrate continuous component parameters
continuous_vol = sqrt(bv_series);
vol_persistence = autocorr(continuous_vol, 1);
disp(['Volatility Persistence: ', num2str(vol_persistence(2))]);

% Visualize jump size distribution
figure;
histogram(jump_sizes, 20, 'Normalization', 'probability');
title('Jump Size Distribution');
xlabel('Jump Size');
ylabel('Probability');
```

## Worked Example
The example script `high_frequency_volatility.m` demonstrates a comprehensive high-frequency analysis workflow:

```matlab
% Run the example script
runHighFrequencyExample();
```

The script performs the following steps:

1. **Data Loading**: Loads example high-frequency return data sampled at a 5-minute frequency.
2. **Standard Realized Volatility (RV)**: Computes basic realized volatility.
3. **Bipower Variation (BV)**: Calculates jump-robust bipower variation.
4. **Kernel-Based Estimation**: Applies different kernel types to address microstructure noise.
5. **Jump Testing**: Performs statistical tests to identify significant jumps.
6. **Results Summary**: Displays comprehensive volatility estimation results.
7. **Visualization**: Creates various plots to visualize the results.
8. **Statistical Analysis**: Performs stationarity tests and autocorrelation analysis.
9. **Jump Component Analysis**: Analyzes jump frequency and magnitude.
10. **Practical Recommendations**: Provides guidelines for choosing appropriate estimators.

The example demonstrates how the high-frequency analysis tools in the MFE Toolbox can be combined to gain comprehensive insights into volatility dynamics, jump behavior, and microstructure effects.

# API Reference

## rv_compute
```matlab
function rv = rv_compute(returns, options)
```

Computes realized volatility (variance) from high-frequency financial return data.

### Inputs:
- **returns**: An m by n matrix of high-frequency returns where m is the number of observations and n is the number of assets.
- **options**: [Optional] A structure with the following fields:
  - **scale**: [Optional] Scaling factor for annualization or other adjustment. Default is 1.
  - **adjustSize**: [Optional] Logical flag to adjust for the total number of observations. Default is false.
  - **weights**: [Optional] Vector of weights for weighted realized volatility. Default is equal weighting.
  - **method**: [Optional] Method to compute RV: 'standard' (default) or 'subsample'.
  - **subSample**: [Optional] Subsampling period when method is 'subsample'. Default is 5.
  - **jackknife**: [Optional] Logical flag to use jackknife bias correction. Default is false.

### Outputs:
- **rv**: Realized volatility (variance) estimate based on the sum of squared returns. 1 by n vector where n is the number of assets or time series.

## bv_compute
```matlab
function bv = bv_compute(returns, options)
```

Computes bipower variation (BV) from high-frequency financial return data.

### Inputs:
- **returns**: T×N matrix of high-frequency returns where T is the number of observations and N is the number of assets.
- **options**: [Optional] Structure containing options for the BV computation:
  - **scaleFactor**: [Optional] Custom scaling factor to replace π/2. Default: pi/2 (theoretical scaling factor).

### Outputs:
- **bv**: N×1 vector of bipower variation estimates for each asset.

## rv_kernel
```matlab
function [rv] = rv_kernel(returns, options)
```

Computes kernel-based realized volatility estimation for high-frequency data.

### Inputs:
- **returns**: T by 1 vector of high-frequency returns (can be a matrix for multiple assets).
- **options**: Optional input structure with fields:
  - **kernelType**: String specifying kernel function ['Bartlett-Parzen']. Supported types: 'Bartlett-Parzen', 'Quadratic', 'Cubic', 'Exponential', 'Tukey-Hanning'.
  - **bandwidth**: Positive integer specifying kernel bandwidth (lag order). If not provided, automatically determined from data.
  - **autoCorrection**: Boolean indicating whether to apply asymptotic bias correction [false].
  - **handleOvernight**: Boolean indicating whether to apply overnight returns adjustment [false].
  - **removeOutliers**: Boolean indicating whether to detect and remove extreme outliers [false].

### Outputs:
- **rv**: Kernel-based realized volatility (variance) estimate, noise-robust.

## realized_spectrum
```matlab
function spectrum = realized_spectrum(returns, options)
```

Computes spectral analysis of realized volatility from high-frequency returns.

### Inputs:
- **returns**: T by 1 vector of high-frequency returns.
- **options**: [Optional] Structure containing options for spectral analysis.

### Outputs:
- **spectrum**: Structure containing spectral analysis results with fields:
  - **frequencies**: Vector of frequencies.
  - **spectral_density**: Vector of spectral density values.
  - **coherence**: [Optional] Coherence between multiple series if returns is a matrix.

## jump_test
```matlab
function results = jump_test(returns, options)
```

Implements statistical tests for identifying jumps in high-frequency financial time series.

### Inputs:
- **returns**: T×N matrix of high-frequency returns where T is the number of observations and N is the number of assets.
- **options**: [Optional] Structure containing options for the jump test:
  - **alpha**: [Optional] Significance level for the test (default: 0.05).
  - **bvOptions**: [Optional] Options to pass to bv_compute.
  - **rvOptions**: [Optional] Options to pass to rv_compute.

### Outputs:
- **results**: Structure containing jump test results with fields:
  - **zStatistic**: Z-statistic for the jump test.
  - **pValue**: p-value for the test.
  - **criticalValues**: Critical values at standard significance levels (0.01, 0.05, 0.10).
  - **jumpDetected**: Logical array indicating whether jumps are detected at various significance levels.
  - **jumpComponent**: Estimated jump component (when detected).
  - **contComponent**: Estimated continuous component.
  - **rv**: Realized volatility estimates.
  - **bv**: Bipower variation estimates.
  - **ratio**: RV/BV ratio.

# References

## Key Papers
The high-frequency analysis components in the MFE Toolbox are based on a rich academic literature in financial econometrics. The key papers that form the theoretical foundation for these methods include:

1. Andersen, T.G., Bollerslev, T., Diebold, F.X., & Labys, P. (2001). "The distribution of realized exchange rate volatility." *Journal of the American Statistical Association*, 96, 42-55.

2. Barndorff-Nielsen, O.E., & Shephard, N. (2004). "Power and bipower variation with stochastic volatility and jumps." *Journal of Financial Econometrics*, 2, 1-48.

3. Barndorff-Nielsen, O.E., & Shephard, N. (2006). "Econometrics of testing for jumps in financial economics using bipower variation." *Journal of Financial Econometrics*, 4, 1-30.

4. Zhang, L., Mykland, P.A., & Aït-Sahalia, Y. (2005). "A tale of two time scales: Determining integrated volatility with noisy high-frequency data." *Journal of the American Statistical Association*, 100, 1394-1411.

5. Hansen, P.R., & Lunde, A. (2006). "Realized variance and market microstructure noise." *Journal of Business & Economic Statistics*, 24, 127-161.

6. Corsi, F. (2009). "A simple approximate long-memory model of realized volatility." *Journal of Financial Econometrics*, 7, 174-196.

## Further Reading
For readers interested in exploring high-frequency financial econometrics in more depth, the following resources provide additional information:

1. Aït-Sahalia, Y., & Jacod, J. (2014). *High-Frequency Financial Econometrics*. Princeton University Press.

2. Andersen, T.G., Davis, R.A., Kreiss, J.P., & Mikosch, T. (Eds.) (2009). *Handbook of Financial Time Series*. Springer.

3. McAleer, M., & Medeiros, M.C. (2008). "Realized volatility: A review." *Econometric Reviews*, 27, 10-45.

4. Hautsch, N. (2012). *Econometrics of Financial High-Frequency Data*. Springer.

5. Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., & Shephard, N. (2008). "Designing realized kernels to measure the ex post variation of equity prices in the presence of noise." *Econometrica*, 76, 1481-1536.