% MFETOOLBOX MFE Toolbox Version 4.0 (28-Oct-2009)
% MFETOOLBOX MATLAB Financial Econometrics Toolbox
% Copyright © 2009 All Rights Reserved
%
% Bootstrap Functions
%   block_bootstrap     - Block bootstrap for dependent data
%   stationary_bootstrap - Stationary bootstrap with random block lengths
%   bootstrap_variance  - Variance estimation using bootstrap methods
%   bootstrap_confidence_intervals - Bootstrap-based confidence interval construction
%
% Cross-sectional Analysis
%   cross_section_filters    - Data filtering for cross-sectional analysis
%   cross_section_regression - Cross-sectional regression estimation
%   cross_section_analysis   - Comprehensive cross-sectional data analysis
%
% Distribution Functions
%   gedpdf     - Generalized Error Distribution PDF
%   gedcdf     - Generalized Error Distribution CDF
%   gedinv     - Generalized Error Distribution inverse CDF/quantile
%   gedrnd     - Generalized Error Distribution random number generator
%   gedloglik  - Generalized Error Distribution log-likelihood
%   gedfit     - Generalized Error Distribution parameter estimation
%   skewtpdf   - Hansen's Skewed T Distribution PDF
%   skewtcdf   - Hansen's Skewed T Distribution CDF
%   skewtinv   - Hansen's Skewed T Distribution inverse CDF/quantile
%   skewtrnd   - Hansen's Skewed T Distribution random number generator
%   skewtloglik - Hansen's Skewed T Distribution log-likelihood
%   skewtfit   - Hansen's Skewed T Distribution parameter estimation
%   stdtpdf    - Standardized Student's T Distribution PDF
%   stdtcdf    - Standardized Student's T Distribution CDF
%   stdtinv    - Standardized Student's T Distribution inverse CDF/quantile
%   stdtrnd    - Standardized Student's T Distribution random number generator
%   stdtloglik - Standardized Student's T Distribution log-likelihood
%   stdtfit    - Standardized Student's T Distribution parameter estimation
%
% GUI Components
%   ARMAX          - Interactive ARMAX modeling interface
%   ARMAX_viewer   - ARMAX results visualization
%   ARMAX_about    - About dialog for ARMAX GUI
%   ARMAX_close_dialog - Close confirmation dialog
%
% Multivariate Analysis
%   var_model   - Vector Autoregression model estimation
%   vecm_model  - Vector Error Correction model estimation
%   factor_model - Factor model implementation
%   ccc_mvgarch - Constant Conditional Correlation multivariate GARCH
%   dcc_mvgarch - Dynamic Conditional Correlation multivariate GARCH
%   bekk_mvgarch - BEKK multivariate GARCH model
%   gogarch     - Generalized Orthogonal GARCH model
%
% Realized Volatility
%   rv_compute  - Realized variance computation
%   bv_compute  - Bipower variation computation
%   rv_kernel   - Kernel-based realized variance estimation
%   realized_spectrum - Realized volatility spectral analysis
%   jump_test   - Jump detection in high-frequency data
%
% Statistical Tests
%   adf_test    - Augmented Dickey-Fuller unit root test
%   pp_test     - Phillips-Perron unit root test
%   kpss_test   - KPSS stationarity test
%   bds_test    - BDS test for nonlinear dependence
%   arch_test   - Test for ARCH effects
%   ljungbox    - Ljung-Box test for autocorrelation
%   jarque_bera - Jarque-Bera normality test
%   lmtest1     - Lagrange Multiplier test
%   white_test  - White's test for heteroskedasticity
%
% Time Series Analysis
%   aicsbic     - AIC and SBIC information criteria computation
%   sacf        - Sample autocorrelation function
%   spacf       - Sample partial autocorrelation function
%   armaxfilter - ARMAX model estimation
%   armafor     - ARMA/ARMAX forecasting
%   sarima      - Seasonal ARIMA model estimation
%
% Univariate Volatility
%   garchcore   - Core GARCH computation engine
%   garchinit   - GARCH parameter initialization
%   garchlikelihood - GARCH log-likelihood computation
%   agarchfit   - Asymmetric GARCH model estimation
%   egarchfit   - Exponential GARCH model estimation
%   igarchfit   - Integrated GARCH model estimation
%   tarchfit    - Threshold ARCH model estimation
%   nagarchfit  - Nonlinear Asymmetric GARCH model estimation
%   garchfor    - GARCH volatility forecasting
%
% Utility Functions
%   backcast    - Variance back-casting for GARCH initialization
%   columncheck - Data column/dimension validation
%   datacheck   - Input data validation
%   matrixdiagnostics - Matrix condition analysis
%   nwse        - Newey-West standard error computation
%   parametercheck - Parameter validation
%
% Installation
%   addToPath   - Configure MATLAB path for the MFE Toolbox
%
% Build System
%   buildZipFile - Create ZIP distribution package