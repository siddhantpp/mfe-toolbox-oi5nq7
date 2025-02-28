# MFE Toolbox Examples

## Introduction

Welcome to the MFE (MATLAB Financial Econometrics) Toolbox examples directory. This collection of scripts demonstrates the usage and capabilities of the MFE Toolbox version 4.0, a comprehensive MATLAB-based suite for financial time series modeling, econometric analysis, and risk assessment.

These examples are designed to help users understand the full range of functionality available in the toolbox, from basic operations to advanced econometric techniques. Each example is extensively documented with comments explaining the underlying methodology and implementation details.

## Example Categories

### Basic Usage
**File:** `basic_usage.m`  
Introduction to fundamental operations of the MFE Toolbox, including path setup, data loading, and basic function calls. This is an excellent starting point for new users.

### Statistical Distribution Analysis
**File:** `distribution_analysis.m`  
Examples demonstrating distribution functions including GED, Hansen's skewed T, and standardized Student's T distributions. Showcases parameter estimation, random number generation, and distribution fitting.

### Time Series Modeling
**File:** `time_series_modeling.m`  
ARMA/ARMAX modeling with forecasting capabilities. Illustrates model specification, parameter estimation, and forecast generation for various time series models.

### Volatility Forecasting
**File:** `volatility_forecasting.m`  
Implementation of various volatility models including GARCH, EGARCH, TARCH and other variants. Demonstrates volatility estimation, parameter optimization, and forecast generation.

### Bootstrap Methods
**File:** `bootstrap_confidence_intervals.m`  
Implementation of block and stationary bootstrap methods for dependent time series. Shows how to generate confidence intervals and conduct hypothesis tests using resampling techniques.

### High-Frequency Analysis
**File:** `high_frequency_volatility.m`  
Examples of high-frequency data analysis techniques, including realized volatility estimation and intraday pattern recognition.

### Multivariate Volatility
**File:** `multivariate_volatility.m`  
Implementation of multivariate volatility models, demonstrating correlation structure analysis and portfolio risk assessment.

### Statistical Testing
**File:** `statistical_testing.m`  
Examples of statistical tests implementation, including Ljung-Box, ARCH effects tests, and other diagnostic procedures.

### Cross-Sectional Analysis
**File:** `cross_sectional_analysis.m`  
Demonstrations of cross-sectional analysis tools, including panel data methods and multi-asset analysis techniques.

### GUI Walkthrough
**File:** `gui_walkthrough.m`  
Guided examples for using the ARMAX GUI interface, including parameter configuration, model estimation, and results interpretation.

### Advanced Workflows
**File:** `advanced_workflows.m`  
Complex analysis examples combining multiple toolbox capabilities into complete analytical workflows relevant to financial analysis.

### Performance Optimization
**File:** `performance_optimization.m`  
Techniques for optimizing performance using MEX and other features, with examples of high-performance computing applications.

## Example Data

The examples directory includes several data files to use with the examples:

- `data/example_financial_data.mat` - Daily financial time series data for use with ARMA/GARCH modeling
- `data/example_high_frequency_data.mat` - Intraday data for high-frequency analysis
- `data/example_cross_sectional_data.mat` - Panel data for cross-sectional analysis

These files contain pre-processed, ready-to-use data structures that align with the example scripts.

## Usage Instructions

To run any example:

1. Ensure you have properly installed the MFE Toolbox by running `addToPath.m` from the toolbox root directory
2. Navigate to the examples directory
3. Run the desired example script in MATLAB, e.g.: `run time_series_modeling.m`

Most examples are designed to run without modification, but you can easily adapt them for your specific data by changing the input file paths or modifying model parameters.

For detailed explanations of the underlying methodology and implementation, refer to the extensive comments within each example script.

## Customizing Examples

The examples are designed to be modular and adaptable. To apply them to your own data:

1. Examine the data structures used in the example
2. Format your data to match the expected input structure
3. Replace the data loading section with your own data
4. Adjust parameters as needed for your specific analysis

Each example is carefully structured to make these modifications straightforward while maintaining the core analytical methodology.