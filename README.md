# MFE Toolbox

*MFE Toolbox v4.0 - 28-Oct-2009*

## Overview

The MFE (MATLAB Financial Econometrics) Toolbox is a comprehensive MATLAB-based software suite designed to provide tools for financial time series modeling, econometric analysis, and risk assessment. This toolbox enables accurate risk assessment, statistical inference, time series analysis, and high-performance computing for financial applications.

[![Version](https://img.shields.io/badge/Version-4.0-blue.svg)](src/backend/Contents.m)
[![Release Date](https://img.shields.io/badge/Release%20Date-28--Oct--2009-green.svg)](#)
[![Platform](https://img.shields.io/badge/Platform-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)

## Features

- **Statistical Distribution Computation Engine**: Comprehensive implementation of statistical distribution functions including GED, Hansen's skewed T, and standardized Student's T distributions.
- **Advanced Bootstrap Methods**: Implementation of block and stationary bootstrap methods for dependent time series.
- **Time Series Modeling Framework**: Comprehensive ARMA/ARMAX modeling with forecasting capabilities.
- **Advanced Volatility Models**: Implementation of univariate and multivariate volatility models.
- **MEX Optimization Framework**: Implementation of critical components in C via MEX for performance.

## Installation

To install the MFE Toolbox, follow these steps:

1.  Add the toolbox to your MATLAB path by navigating to the repository root directory and running the path configuration script:

    ```matlab
    run addToPath
    ```

2.  (Optional) To save the path permanently, you can use the `savepath` function.

    ```matlab
    savepath
    ```

    **Note**: You may need administrator privileges to save the path permanently.

3.  For detailed installation instructions, refer to the [Installation Guide](docs/installation.md).

## Documentation

The MFE Toolbox provides comprehensive documentation to help you effectively use its features.

-   [Installation Guide](docs/installation.md): Detailed instructions for installing and configuring the toolbox.
-   [Getting Started](docs/getting_started.md): A guide to help you get started with the toolbox.
-   [MEX Compilation Guide](docs/mex_compilation.md): Instructions for compiling MEX components.
-   [Cross-Platform Notes](docs/cross_platform_notes.md): Notes on cross-platform compatibility.
-   [Performance Optimization Guide](docs/performance_optimization.md): Guide to performance optimization techniques.
-   [API Reference](docs/api_reference.md): Comprehensive API reference for all functions.

## Examples

Explore the `examples` directory for example scripts demonstrating various functionalities of the MFE Toolbox. Key examples include:

-   `examples/distribution_analysis.m`: Demonstrates the usage of statistical distribution functions.
-   `examples/time_series_modeling.m`: Demonstrates a comprehensive time series modeling workflow.
-   `examples/volatility_forecasting.m`: Demonstrates volatility forecasting using GARCH models.

## System Requirements

The MFE Toolbox requires the following system components:

-   MATLAB (R2007a or later recommended)
-   MATLAB Statistics Toolbox
-   MATLAB Optimization Toolbox
-   C Compiler for MEX (optional, for MEX optimization)

For more details, refer to the [System Requirements](docs/mex_compilation.md) documentation.

## Contributing

We welcome contributions to the MFE Toolbox! Please see our [Contributing Guidelines](CONTRIBUTING.md) for information on how to contribute.

## License

The MFE Toolbox is licensed under the terms of the [LICENSE](LICENSE) file.

Copyright (c) 2009 All Rights Reserved