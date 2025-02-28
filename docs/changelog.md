# MFE Toolbox Changelog

Version history and release notes for the MATLAB Financial Econometrics Toolbox

## Version 4.0 (October 28, 2009)
Major release with significant enhancements to volatility modeling, distribution functions, and performance optimization.

### New Features
- Added MEX optimization for AGARCH, EGARCH, IGARCH, and TARCH models
- Implemented Hansen's Skewed T-distribution (skewtpdf, skewtcdf, skewtinv, skewtrnd, skewtloglik, skewtfit)
- Added Realized Volatility module with jump detection and kernel-based estimation
- Implemented comprehensive bootstrap methods for dependent time series
- Added ARMAX GUI for interactive time series modeling
- Enhanced cross-sectional analysis tools

### Improvements
- Enhanced performance with C-based MEX implementations for critical components
- Improved memory efficiency for large dataset processing
- Enhanced numerical stability in parameter estimation
- Extended platform support with optimized binaries for Windows and Unix
- Added detailed function documentation and examples
- Improved error handling and validation across all components

### Fixed Issues
- Resolved numerical stability issues in GARCH estimation with extreme parameters
- Fixed memory leaks in certain MEX implementations
- Addressed convergence problems in multivariate models
- Corrected statistical distribution implementation for edge cases
- Fixed path configuration issues on certain platforms

### API Changes
- Standardized parameter ordering across all volatility model functions
- Modified function signatures for bootstrap methods to improve usability
- Added consistent error distribution options to all volatility models
- Added optional parameters for MEX optimization control

### Documentation
- Complete documentation overhaul with improved examples
- Added detailed installation guide with platform-specific instructions
- Enhanced API documentation for all functions
- Added cross-references between related components
- Included academic citations for implemented models and methods

## Version 3.5 (March 15, 2008)
Enhancement release focused on multivariate volatility models and distribution functions.

### New Features
- Added multivariate GARCH models (CCC, DCC, BEKK, GO-GARCH)
- Implemented standardized Student's T-distribution functions
- Added Vector Autoregression (VAR) and Vector Error Correction (VECM) models
- Introduced factor model implementation
- Added new statistical tests (BDS, White, KPSS)

### Improvements
- Enhanced optimization routines for faster convergence
- Improved numerical stability in multivariate models
- Extended error handling for edge cases
- Added additional diagnostic statistics
- Improved documentation and examples

### Fixed Issues
- Resolved estimation issues in certain GARCH variants
- Fixed incorrect standard errors in some statistical tests
- Addressed numerical overflow in certain distribution functions
- Corrected documentation inconsistencies

## Version 3.0 (July 22, 2007)
Major release adding univariate volatility models and distribution functions.

### New Features
- Implemented GARCH, EGARCH, TARCH, IGARCH models
- Added Generalized Error Distribution (GED) functions
- Introduced ARMA/ARMAX modeling and forecasting
- Added AIC/SBIC model selection criteria
- Implemented statistical tests for time series (ADF, PP, ARCH, Ljung-Box)

### Improvements
- Enhanced algorithm efficiency for time series computation
- Improved model initialization methods
- Added comprehensive validation for all functions
- Enhanced documentation with examples

## Version 2.0 (November 5, 2006)
Foundation release establishing core framework and utilities.

### Features
- Core statistical utilities and matrix operations
- Basic time series functions (ACF, PACF)
- Utility functions for data validation and processing
- Initial framework for financial econometrics
- Basic documentation structure

For installation instructions, see the installation guide in the docs directory.
For getting started with the latest version, see [Getting Started Guide](getting_started.md).