# Getting Started with MFE Toolbox

Welcome to the MFE (MATLAB Financial Econometrics) Toolbox! This guide will help you get started with the toolbox, providing an introduction, initial setup guidance, basic usage examples, and navigation to more detailed documentation resources.

## Introduction

The MFE Toolbox is a comprehensive MATLAB-based software suite designed for financial time series modeling, econometric analysis, and risk assessment. Version 4.0, released on October 28, 2009, provides a robust set of tools for quantitative finance and econometrics. <!-- IE2 MATLAB 4.0 -->

Key capabilities include:

- Financial time series modeling
- Econometric analysis
- Risk assessment
- High-performance computing via MEX integration
- Pre-built, validated statistical routines
- Error checking and validation mechanisms

The toolbox is designed for financial analysts, econometricians, risk managers, academic researchers, and MATLAB platform users.

## Prerequisites

Before installing the MFE Toolbox, ensure your system meets the following requirements:

- **MATLAB**: R2009b or a compatible version. <!-- IE2 MATLAB 4.0 -->
- **MATLAB Statistics Toolbox**: Recommended for enhanced statistical functionality. <!-- IE2 MATLAB Statistics Toolbox compatible with MATLAB R2009b -->
- **MATLAB Optimization Toolbox**: Recommended for parameter estimation and optimization routines. <!-- IE2 MATLAB Optimization Toolbox compatible with MATLAB R2009b -->
- **Operating System**: Windows (PCWIN64) or Unix systems. <!-- IE3 Cross Platform Notes Platform-Specific Installation -->
- **Hardware**:
  - Processor: Intel Pentium 4 minimum, Core 2 Duo recommended
  - Memory: 2GB minimum, 4GB recommended
  - Storage: 50MB for installation

For advanced users requiring MEX compilation:

- **C Compiler**: A compatible C compiler (e.g., Microsoft Visual C++ for Windows, GCC for Unix).
- **MEX Setup**: Proper configuration of the MEX environment in MATLAB.

## Installation

Follow these steps to install the MFE Toolbox:

1. **Download**: Obtain the `MFEToolbox.zip` file from the official distribution channel.
2. **Extract**: Extract the contents of the ZIP file to a suitable location on your system (e.g., `C:\MFEToolbox` or `/opt/MFEToolbox`).
3. **MATLAB Path**: Open MATLAB and navigate to the extracted directory.
4. **Run `addToPath.m`**: Execute the `addToPath.m` script to configure the MATLAB path. This script automatically adds the required directories to your MATLAB path.
   ```matlab
   run('path/to/MFEToolbox/addToPath.m')
   ```
5. **Platform-Specific Considerations**:
   - **Windows**: The script adds the necessary DLL files for MEX functions.
   - **Unix**: Ensure that the MEX binaries are compatible with your system.
6. **Verification**: Confirm successful installation by running the following command in MATLAB:
   ```matlab
   ARMAX
   ```
   This should launch the ARMAX GUI. Additionally, verify that the core functions are available:
   ```matlab
   which gedpdf
   which tarchfit
   ```
   If these commands return the correct paths, the toolbox is installed correctly.

If you encounter issues, consult the [Troubleshooting Guide](troubleshooting.md). <!-- IE3 Troubleshooting Guide Installation Issues -->

## Toolbox Components

The MFE Toolbox is organized into three primary component groups:

1. **Core Statistical Modules**:
   - Distributions: Statistical distribution computations (GED, Hansen's skewed T, standardized Student's T).
   - Bootstrap: Robust resampling techniques.
   - Tests: Comprehensive statistical testing suite.
   - Cross-section: Analysis tools for sectional data.

2. **Time Series Components**:
   - ARMA/ARMAX: Modeling and forecasting.
   - Volatility: Univariate and multivariate volatility models.
   - High-frequency: Econometric analysis of high-frequency data.
   - Risk metrics: Computation of advanced risk metrics.

3. **Support Infrastructure**:
   - GUI: Interactive GUI for ARMAX modeling.
   - Utilities: Utility functions for data manipulation.
   - MEX: MEX-based performance optimization.
   - Platform support: Cross-platform compatibility layer.

The following workflow diagram illustrates the operational flow between components:

![MFE Toolbox Workflow Diagram](images/workflow_diagram.png) <!-- IE3 Workflow Diagram -->

## Basic Usage

Here are some basic usage examples to get you started:

1. **Loading and Preparing Data**:
   ```matlab
   load example_financial_data.mat;
   returns = example_financial_data;
   ```

2. **ARMA/ARMAX Modeling**:
   ```matlab
   % Estimate ARMA(1,1) model
   results = armaxfilter(returns, 1, 1);
   disp(results);
   ```

3. **Volatility Modeling**:
   ```matlab
   % Estimate GARCH(1,1) model
   results = tarchfit(returns);
   disp(results);
   ```

4. **Result Interpretation and Diagnostics**:
   ```matlab
   % Access estimated parameters
   omega = results.parameters(1);
   alpha = results.parameters(2);
   beta = results.parameters(3);

   % View diagnostics
   disp(['AIC: ', num2str(results.aic)]);
   disp(['BIC: ', num2str(results.bic)]);
   ```

5. **Visualization**:
   ```matlab
   % Plot the returns and conditional volatility
   plot(returns);
   hold on;
   plot(sqrt(results.ht), 'r', 'LineWidth', 1.5);
   legend('Returns', 'Volatility');
   hold off;
   ```

## GUI Usage

The MFE Toolbox includes an interactive GUI for ARMAX modeling. To launch the GUI, simply type:

```matlab
ARMAX
```

The GUI allows you to configure model parameters, estimate the model, and visualize the results.

![ARMAX GUI Interface](images/gui_screenshot.png) <!-- IE3 GUI Screenshot -->

For detailed instructions on using the GUI, refer to the [GUI Usage](gui_usage.md) documentation. <!-- IE3 GUI Usage ARMAX GUI -->

## Next Steps

To further explore the MFE Toolbox, consider the following:

- **Module-Specific Documentation**: Dive deeper into specific modules such as [Time Series Models](time_series_models.md) and [Volatility Models](volatility_models.md). <!-- IE3 Time Series Models ARMA/ARMAX Models, IE3 Volatility Models GARCH Models -->
- **Advanced Examples**: Explore the example files in the `examples` directory for more complex use cases.
- **API Reference**: Consult the [API Reference](api_reference.md) for comprehensive function documentation. <!-- IE3 API Reference Function References -->
- **Performance Optimization**: Learn about performance optimization techniques for large datasets.
- **Community Resources**: Engage with the MFE Toolbox community for support and additional learning materials.