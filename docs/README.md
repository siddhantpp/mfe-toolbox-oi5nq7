# MFE Toolbox Documentation

## Overview

The MFE (MATLAB Financial Econometrics) Toolbox is a sophisticated MATLAB-based software suite designed to provide comprehensive tools for financial time series modeling, econometric analysis, and risk assessment. Version 4.0, released on October 28, 2009, represents a significant advancement in the field of quantitative finance and econometrics.

## Documentation Structure

This documentation is organized into several sections to help you effectively use the MFE Toolbox:

- **Getting Started**: Installation, configuration, and basic usage
- **User Guides**: Detailed explanations of key functionalities
- **API Reference**: Comprehensive documentation of functions and classes
- **Examples**: Practical examples demonstrating common use cases
- **Technical Details**: In-depth information about implementation
- **Release Notes**: Changes and improvements in each version

## Component Documentation

### Core Statistical Components

- [Distribution Analysis](../distributions/README.md) - Statistical distribution functions
- [Bootstrap Methods](../bootstrap/README.md) - Block and stationary bootstrap implementations
- [Statistical Tests](../tests/README.md) - Hypothesis testing suite
- [Cross-sectional Tools](../crosssection/README.md) - Analysis tools for sectional data

### Time Series Components

- [ARMA/ARMAX Models](../timeseries/README.md) - Time series modeling and forecasting
- [Univariate Volatility](../univariate/README.md) - GARCH family models
- [Multivariate Volatility](../multivariate/README.md) - Multivariate GARCH implementations
- [High-frequency Analysis](../realized/README.md) - Realized volatility estimation

### Support Infrastructure

- [GUI Interface](../GUI/README.md) - Interactive ARMAX modeling
- [Utility Functions](../utility/README.md) - Helper functions and tools
- [MEX Components](../mex_source/README.md) - High-performance C implementations

## Installation

The MFE Toolbox is distributed as a ZIP archive containing all required components. After extraction, use the `addToPath.m` utility to configure your MATLAB environment:

```matlab
run('path/to/MFEToolbox/addToPath.m')
```

This script will automatically:
- Add required directories to your MATLAB path
- Configure platform-specific MEX binaries
- Set up optional components if available
- Optionally save the path configuration permanently

## Requirements

- MATLAB (compatible with version available at release date)
- MATLAB Statistics Toolbox
- MATLAB Optimization Toolbox
- Platform-compatible C compiler (for MEX compilation)

## Quick Start

After installation, you can begin using the toolbox immediately. For example, to access the ARMAX modeling GUI:

```matlab
ARMAX
```

For programmatic time series analysis, try:

```matlab
% Example GARCH model estimation
data = randn(1000,1);
[parameters, logL, ht, VCVrobust, VCV, scores, diagnostics] = garch(data);
```

## Support & Resources

For additional information, please refer to:
- Contents.m file for version information
- Individual component documentation
- Technical specification for detailed implementation notes

## License

© 2009 All Rights Reserved

For licensing details, please see the license file included with the MFE Toolbox or contact the authors.