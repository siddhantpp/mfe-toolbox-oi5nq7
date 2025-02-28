# MFE Toolbox - GUI Component

## Overview

The GUI component of the MFE Toolbox provides an interactive interface for ARMA/ARMAX (AutoRegressive Moving Average with eXogenous inputs) modeling, enabling users to perform sophisticated time series analysis without writing code. This document provides comprehensive information about the GUI components, their usage, and technical implementation details.

### Purpose

The ARMAX GUI serves as an accessible entry point to the powerful time series modeling capabilities of the MFE Toolbox. It allows users to:

- Configure and estimate ARMA and ARMAX models interactively
- Visualize time series data and model outputs
- Run comprehensive model diagnostics
- Export results for further analysis

This component bridges the gap between sophisticated econometric algorithms and user-friendly interfaces, making advanced time series modeling accessible to users with varying levels of programming expertise.

### Architecture

The GUI implementation follows a singleton pattern to ensure only one instance of each window is active at any time. The component relationships are structured as follows:

- **ARMAX.m/fig**: Main application window serving as the central controller
- **ARMAX_viewer.m/fig**: Results display window for detailed analysis
- **ARMAX_about.m/fig**: Information dialog with version details
- **ARMAX_close_dialog.m/fig**: Confirmation dialog for application exit

The components communicate through direct function calls and utilize MATLAB's guidata mechanism for persistent state management. The design separates presentation from computational logic, with the GUI layer interacting with the core time series functions through well-defined interfaces.

## Components

### ARMAX.m and ARMAX.fig

The main application window provides the primary interface for ARMAX model configuration, estimation, and visualization. Key features include:

- **Model Configuration Panel**: Controls for setting AR order, MA order, and distribution type
- **Time Series Visualization**: Interactive plot area for data display
- **Estimation Controls**: Buttons for model fitting and configuration
- **Diagnostic Selection**: Options for various statistical tests and plots
- **Results Navigation**: Access to detailed model output and statistics

This component handles user input validation, parameter configuration, and coordinates the modeling process by interfacing with core computational functions.

### ARMAX_viewer.m and ARMAX_viewer.fig

The results viewer provides a detailed presentation of model estimation results, including:

- **Parameter Estimates**: Table of coefficient values with standard errors
- **Model Equation**: LaTeX-rendered representation of the estimated model
- **Diagnostic Plots**: ACF, PACF, and residual analysis visualizations
- **Statistical Tests**: Ljung-Box Q-statistics, ARCH tests, and other diagnostics
- **Forecast Visualization**: Time series forecasts with confidence intervals

The viewer component is designed for efficient data presentation with pagination for complex models and selective plot rendering for performance optimization.

### ARMAX_about.m and ARMAX_about.fig

A simple dialog providing version information, copyright details, and acknowledgments. Features include:

- **Version Information**: Current MFE Toolbox version
- **Logo Display**: Oxford University logo (origin of the toolbox)
- **Copyright Notice**: License and usage information
- **Acknowledgments**: Credits to contributors and supporting institutions

### ARMAX_close_dialog.m and ARMAX_close_dialog.fig

A confirmation dialog that appears when closing the application, offering options to:

- Save the current model configuration
- Discard changes and exit
- Cancel the close operation and return to the application

This component ensures data preservation and prevents accidental work loss during the application lifecycle.

## Usage

### Launch Instructions

The ARMAX GUI can be launched in several ways:

1. **Direct launch without data**:
   ```matlab
   ARMAX
   ```

2. **Launch with time series data**:
   ```matlab
   % Prepare your time series data
   y = randn(100,1);  % Example: random data
   
   % Launch ARMAX with the data
   ARMAX(y)
   ```

3. **Launch with data and exogenous variables**:
   ```matlab
   % Prepare time series and exogenous data
   y = randn(100,1);
   X = randn(100,2);  % Two exogenous variables
   
   % Launch ARMAX with both datasets
   ARMAX(y, X)
   ```

### Data Input

The GUI accepts time series data in several formats:

- **Vector Input**: Single column of observations (most common)
- **Matrix Input**: Multiple columns for multivariate analysis
- **Dataset Import**: Options to import from MATLAB workspace or files
- **Transformations**: Data can be transformed (differencing, log, etc.) within the interface

Data validation is performed automatically, with error messages for incompatible inputs or structural issues.

### Model Configuration

To configure and estimate an ARMAX model:

1. Set the AR order (p) in the corresponding input field
2. Set the MA order (q) in the corresponding input field
3. Select the error distribution type (Normal, Student's t, GED)
4. Configure additional options if needed:
   - Mean inclusion/exclusion
   - Variance targeting
   - Estimation algorithm selection
5. Click the "Estimate" button to run the model

Progress is displayed during estimation, and results appear automatically upon completion.

### Results Interpretation

The results viewer provides comprehensive model information:

- **Parameter Table**: Estimated coefficients with standard errors and significance
- **Information Criteria**: AIC, BIC, and other model selection metrics
- **Diagnostic Tests**: Statistical tests for model adequacy
- **Residual Analysis**: Plots and tests for remaining patterns in residuals
- **Forecast Display**: Point forecasts with confidence intervals

Use the navigation controls to explore different aspects of the model results and diagnostic visualizations.

## Integration

### Time Series Functions

The GUI integrates with core computational components from the MFE Toolbox:

- **armaxfilter.m**: Primary function for ARMAX model estimation
- **armafor.m**: Forecasting function for estimated models
- **sacf.m/spacf.m**: Autocorrelation and partial autocorrelation functions
- **aicsbic.m**: Information criteria computation for model selection

The interface passes configured parameters to these functions and processes their outputs for display and analysis.

### Statistical Tests

The GUI incorporates various diagnostic tests:

- **ljungbox.m**: Ljung-Box Q-test for residual autocorrelation
- **lmtest1.m**: Lagrange Multiplier test for ARCH effects
- **berkowitz.m**: Test for forecast distribution accuracy
- **jarquebera.m**: Normality test for residuals

Test results are processed and presented in the results viewer with appropriate significance indicators.

## Implementation Notes

### GUIDE Framework

The GUI is implemented using MATLAB's GUIDE (GUI Development Environment) framework:

- **.fig files**: Generated by GUIDE, containing the layout information
- **.m files**: Containing callback functions and implementation logic
- **Property management**: Using MATLAB's property/value mechanism
- **Component hierarchy**: Structured for efficient event propagation

Developers should use GUIDE to modify the layout and then implement the corresponding callbacks in the .m files.

### Event Handling

The GUI implements a comprehensive event handling system:

- **Callbacks**: Functions triggered by user interactions
- **Property listeners**: For dynamic property updates
- **Timer objects**: For animations and progress indicators
- **Custom events**: For communication between components

Events follow a consistent naming convention: `ARMAX_eventName_Callback` for standard callbacks and `ARMAX_eventName_listener` for property change listeners.

### Performance Considerations

Several techniques are employed to ensure responsive performance:

- **Selective plot updates**: Only redrawing changed elements
- **Deferred computation**: Processing intensive calculations in background
- **Memory management**: Efficient data structures and cleanup routines
- **Progressive rendering**: Displaying results incrementally during computation
- **Cached results**: Storing intermediate calculations for reuse
- **Optimized callbacks**: Minimizing processing in high-frequency events

For large datasets, the GUI implements memory-mapped file access to reduce RAM usage during analysis.

## Examples

### Basic Usage

Simple example demonstrating basic GUI usage with synthetic data:

```matlab
% Generate some synthetic AR(2) data
n = 200;
phi = [0.8, -0.4];
e = randn(n, 1);
y = zeros(n, 1);

for t = 3:n
    y(t) = phi(1)*y(t-1) + phi(2)*y(t-2) + e(t);
end

% Launch the ARMAX GUI with this data
ARMAX(y);

% From the GUI:
% 1. Set AR order (p) to 2
% 2. Set MA order (q) to 0
% 3. Click "Estimate"
% 4. View the results
```

### Programmatic Control

Advanced example showing programmatic interface to the GUI components:

```matlab
% Load sample data
load('forex.mat');  % Example dataset with exchange rates

% Configure model options programmatically
options = struct('p', 2, 'q', 1, ...
                'distribution', 'student', ...
                'includeMean', true, ...
                'variantType', 'GARCH');

% Estimate model using core function
results = armaxfilter(usdeur, [], options);

% Display results in the viewer
ARMAX_viewer(results);

% Export results to workspace
assignin('base', 'armax_model', results);
```

## References

- **ARMAX.m**: Main GUI implementation file
- **ARMAX_viewer.m**: Results viewer component
- **armaxfilter.m**: Core ARMAX model implementation
- **Technical Specification**: MFE Toolbox technical documentation
- **MATLAB GUIDE**: [MATLAB GUI Development Environment](https://www.mathworks.com/discovery/matlab-gui.html)