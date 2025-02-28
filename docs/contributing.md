# Contributing to the MFE Toolbox

## Introduction

The MFE (MATLAB Financial Econometrics) Toolbox is a sophisticated MATLAB-based software suite designed to provide comprehensive tools for financial time series modeling, econometric analysis, and risk assessment. This toolbox enables accurate risk assessment, statistical inference, time series analysis, and high-performance computing for financial applications.

Contributing to the MFE Toolbox helps maintain and enhance a high-quality financial econometrics toolbox that serves researchers, financial analysts, econometricians, and risk managers across both academic and industry sectors. Your contributions help strengthen the financial econometric community and promote advanced quantitative methods.

## Code of Conduct

All contributors to the MFE Toolbox project are expected to adhere to the following guidelines:

- **Respectful Communication**: Engage with others in a professional and respectful manner.
- **Inclusive Environment**: Welcome contributions from individuals of all backgrounds and experience levels.
- **Constructive Feedback**: Provide and accept constructive criticism that helps improve the project.
- **Focus on Technical Merit**: Evaluate contributions based on technical merit rather than personal attributes.
- **Acknowledge Contributions**: Recognize the work and efforts of others in the project.
- **Responsible Disclosure**: Report security vulnerabilities or critical bugs privately before disclosing publicly.

Contributors who violate these standards may be temporarily or permanently excluded from the project.

## Development Environment Setup

### MATLAB Requirements

- **MATLAB Version**: A recent version of MATLAB is required (2018b or newer recommended)
- **Required Toolboxes**:
  - MATLAB Statistics Toolbox
  - MATLAB Optimization Toolbox
- **Optional Toolboxes**:
  - MATLAB Parallel Computing Toolbox (for improved performance)

### C Compiler Setup

The MFE Toolbox uses MEX files to achieve high performance for critical numerical operations. Setting up the appropriate C compiler is essential for development and testing.

#### Windows Setup
1. Install a supported C compiler:
   - Microsoft Visual C++ (recommended)
   - MinGW-w64 C/C++ Compiler
   
2. Configure MATLAB to use the compiler:
   ```matlab
   mex -setup
   ```

3. Verify the compiler is correctly configured:
   ```matlab
   mexext % Should return 'mexw64' for 64-bit Windows
   ```

#### Unix/Linux Setup
1. Install a supported C compiler:
   - GCC (recommended)
   
2. Configure MATLAB to use the compiler:
   ```matlab
   mex -setup
   ```

3. Verify the compiler is correctly configured:
   ```matlab
   mexext % Should return 'mexa64' for 64-bit Unix/Linux
   ```

### Repository Structure

The MFE Toolbox repository is structured as follows:

```
MFEToolbox/
├── bootstrap/         # Bootstrap implementation
├── crosssection/      # Cross-sectional analysis tools
├── distributions/     # Statistical distribution functions
├── GUI/               # ARMAX modeling interface
├── multivariate/      # Multivariate analysis tools
├── tests/             # Statistical testing suite
├── timeseries/        # Time series analysis
├── univariate/        # Univariate analysis tools
├── utility/           # Helper functions
├── realized/          # High-frequency analysis
├── mex_source/        # C source files for MEX
├── dlls/              # Platform-specific MEX binaries
├── docs/              # Documentation
├── infrastructure/    # Build and support tools
|   └── templates/     # Code and documentation templates
├── .github/           # GitHub configuration files
├── addToPath.m        # Path configuration utility
└── Contents.m         # Version information
```

### Path Configuration

To set up your development environment:

1. Clone the repository to your local machine
2. Start MATLAB
3. Navigate to the repository root directory
4. Run the path configuration script:
   ```matlab
   run addToPath
   ```
5. Confirm that the toolbox components are accessible:
   ```matlab
   help MFE
   ```

## Coding Standards

### MATLAB Coding Standards

#### Naming Conventions
- **Functions**: Use camelCase (e.g., `armaxFilter`, `garchFit`)
- **Variables**: Use descriptive camelCase names (e.g., `parameterResults`, `timeSeriesData`)
- **Constants**: Use UPPER_CASE with underscores (e.g., `MAX_ITERATIONS`, `DEFAULT_TOLERANCE`)
- **Files**: Function files should match the function name (e.g., `armaxFilter.m`)

#### Code Structure
- Each function should perform a single, well-defined task
- Use helper functions for complex tasks
- Limit function length to improve readability (aim for under 200 lines)
- Use comments to explain complex algorithms or non-obvious code

#### Documentation
- All functions must have comprehensive help text using the standard template
- Include clear descriptions of inputs, outputs, and examples
- Document any reference to academic papers or algorithms
- Update Contents.m when adding new functions

#### Error Handling
- Validate all inputs at the start of each function
- Use the `parametercheck`, `datacheck`, and other validation utilities
- Provide meaningful error messages with specific information
- Use try-catch blocks for operations that might fail
- Always clean up resources in catch blocks

#### Performance Optimization
- Pre-allocate arrays where possible
- Use vectorized operations instead of loops
- Minimize memory usage for large datasets
- Consider MEX implementations for performance-critical sections

### C Coding Standards

#### Naming Conventions
- **Functions**: Use snake_case (e.g., `compute_garch_likelihood`, `process_matrix`)
- **Variables**: Use descriptive snake_case names (e.g., `input_matrix`, `result_array`)
- **Constants**: Use UPPER_CASE with underscores (e.g., `MAX_ARRAY_SIZE`, `DEFAULT_TOLERANCE`)
- **Files**: Use descriptive names ending with "_core.c" (e.g., `garch_core.c`)

#### Code Structure
- Include clear function headers with descriptions
- Limit function complexity
- Use consistent indentation (4 spaces recommended)
- Break complex calculations into meaningful steps

#### Memory Management
- Always check allocation success with error handling
- Free all allocated memory before function exit
- Avoid memory leaks in error conditions
- Use appropriate data types to optimize memory usage

#### MEX Interface
- Validate all inputs at the start of the function
- Check array dimensions before access
- Handle edge cases (empty arrays, NaN values)
- Return useful error messages to MATLAB

#### Error Handling
- Check return values from all library functions
- Provide detailed error messages
- Clean up resources properly in error conditions
- Use mexErrMsgIdAndTxt for error reporting

### Function Template

All new functions should follow the standard function template. The template includes:

1. Function signature
2. Comprehensive help text
3. Input validation
4. Primary computation
5. Error handling
6. Results formatting

See `infrastructure/templates/function_template.m` for the detailed template.

### Error Handling Patterns

Robust error handling is critical for production-quality financial software. Follow these patterns:

1. **Validation First**: Check all inputs before any computation
2. **Specific Error Messages**: Include parameter names and expected values
3. **Contextual Information**: Provide context for debugging
4. **Resource Cleanup**: Ensure all resources are freed in error conditions
5. **Graceful Degradation**: When possible, complete partial operations

Example:
```matlab
try
    % Main computation
catch ME
    % Enhanced error handling with context
    errorID = 'MFEToolbox:functionName';
    
    % Add additional context to the error message
    if strcmp(ME.identifier, 'MATLAB:nomem')
        errorMessage = sprintf(['Error in functionName: Out of memory. ' ...
            'Try using a smaller dataset or increasing available memory.']);
    else
        errorMessage = sprintf(['Error in functionName: %s\nFunction failed ' ...
            'at line %d of file %s.'], ME.message, ME.line, ME.stack(1).file);
    end
    
    % Throw error with context
    error(errorID, errorMessage);
end
```

### Performance Optimization

Performance is critical for financial computation. Follow these guidelines:

1. **Vectorization**: Use MATLAB's built-in vectorized operations
2. **Memory Preallocation**: Always preallocate arrays
3. **Data Types**: Use appropriate data types (e.g., sparse matrices when relevant)
4. **Algorithm Selection**: Choose algorithms with appropriate complexity
5. **MEX Implementation**: Consider C implementation for critical paths

## Contribution Workflow

### Fork and Clone

1. Fork the MFE Toolbox repository to your GitHub account
2. Clone your fork to your local machine:
   ```
   git clone https://github.com/your-username/MFEToolbox.git
   ```
3. Add the main repository as an upstream remote:
   ```
   git remote add upstream https://github.com/main-repo/MFEToolbox.git
   ```

### Branching Strategy

1. Create a new branch for each feature or bugfix:
   ```
   git checkout -b feature/your-feature-name
   ```
   or
   ```
   git checkout -b fix/your-bugfix-name
   ```

2. Keep branches focused on a single issue or feature
3. Regularly sync your fork with the upstream repository:
   ```
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

### Commit Guidelines

1. Write clear, descriptive commit messages
2. Use present tense ("Add feature" not "Added feature")
3. Reference issue numbers when applicable
4. Keep commits focused and logical
5. Example commit message:
   ```
   Add Hansen's skewed T distribution estimation

   - Implement maximum likelihood parameter estimation
   - Add random number generation function
   - Include numerical accuracy tests
   - Update documentation with examples

   Closes #123
   ```

### Pull Requests

1. Push your branch to your fork:
   ```
   git push origin feature/your-feature-name
   ```

2. Create a pull request (PR) to the main repository
3. Fill out the PR template completely
4. Include tests for your changes
5. Ensure all tests pass
6. Link to relevant issues

### Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Be responsive to questions during review
5. Keep PRs small and focused to speed up review

## Testing Guidelines

### Test Coverage Requirements

1. All new functions must have corresponding tests
2. Tests should cover normal operation, edge cases, and error conditions
3. Minimum coverage requirements:
   - 90% line coverage for core statistical functions
   - 80% line coverage for utility functions
   - 95% coverage for MEX wrapper functions

### Unit Testing

1. Create test classes inheriting from `BaseTest`
2. Name test files with the pattern `FunctionNameTest.m`
3. Verify numerical accuracy with appropriate tolerances
4. Test with representative financial data
5. Use the `infrastructure/templates/test_template.m` as a starting point

Example test method:
```matlab
function testBasicFunctionality(obj)
    % Prepare test data
    data = [1.2, 2.3, 3.4; 4.5, 5.6, 6.7];
    options.param1 = 0.95;
    
    % Call function under test
    result = functionUnderTest(data, options);
    
    % Verify results
    expectedValue = 1.234;
    obj.assertAlmostEqual(expectedValue, result.value, ...
        'Function produced incorrect value');
end
```

### Integration Testing

1. Create tests that verify interaction between components
2. Test complete workflows (e.g., ARMA model + volatility model)
3. Verify that results are consistent across integrated components
4. Test with realistic financial datasets

Example:
```matlab
function testArmaGarchIntegration(obj)
    % Test that ARMA and GARCH models work together correctly
    returns = obj.loadTestData('financial_returns.mat');
    
    % First estimate ARMA model
    armaResults = armax(returns, [1,1]);
    
    % Then use residuals for GARCH estimation
    garchResults = garch(armaResults.residuals);
    
    % Verify combined model behaves as expected
    % ...
end
```

### MEX Testing

1. Test MEX functions across all supported platforms
2. Verify memory management with large datasets
3. Check numerical precision against MATLAB implementations
4. Test error handling and validation in C code
5. Verify binary compatibility across environments

### Performance Testing

1. Measure execution time with various dataset sizes
2. Compare against baseline performance
3. Verify memory utilization is reasonable
4. Test scaling behavior with large datasets
5. Document performance characteristics

Example:
```matlab
function testPerformance(obj)
    sizes = [100, 1000, 10000];
    times = zeros(size(sizes));
    
    for i = 1:length(sizes)
        n = sizes(i);
        data = randn(n, 3);
        
        tic;
        functionUnderTest(data);
        times(i) = toc;
        
        fprintf('Size %d: %.6f seconds\n', n, times(i));
    end
    
    % Verify scaling behavior
    obj.assertTrue(times(3)/times(1) < 200, ...
        'Function does not scale well with dataset size');
end
```

### Running Tests

1. Navigate to the test directory
2. Run specific tests:
   ```matlab
   testObj = FunctionNameTest();
   results = testObj.runAllTests();
   ```
3. Run all tests:
   ```matlab
   runAllTests
   ```
4. Check test coverage:
   ```matlab
   generateCoverageReport
   ```

## Documentation Guidelines

### Function Documentation

All functions must be documented using the standardized header format:

```matlab
%% FUNCTIONNAME Brief description of the function's purpose
%
% Detailed description of the function, methodology, algorithm,
% and important concepts.
%
% USAGE:
%   results = functionName(data)
%   results = functionName(data, options)
%
% INPUTS:
%   data        - T by K matrix of input data
%                 Description of what each column represents
%                 First column: [Description of first column]
%
%   options     - [OPTIONAL] Structure containing configuration parameters
%                 Default: [] (Uses default values for all options)
%                 Fields:
%                   options.field1 - Description [default = value1]
%
% OUTPUTS:
%   results     - Structure containing function outputs with fields:
%                   results.field1 - Description of field1
%
% COMMENTS:
%   Implementation notes, assumptions, and important details.
%
% EXAMPLES:
%   % Basic usage example
%   data = [1 2; 3 4; 5 6];
%   results = functionName(data);
%
% REFERENCES:
%   [1] Author, A. (Year). "Title of Paper." Journal, Volume(Issue), Pages.
%
% SEE ALSO:
%   relatedFunction1, relatedFunction2
%
% MFE Toolbox v4.0
% Copyright (c) 2009
```

### Module Documentation

For each module (directory), provide an overview document that includes:

1. Module purpose and scope
2. List of main functions
3. Common usage patterns
4. Integration with other modules
5. Examples demonstrating typical workflows

### Example Creation

Examples should:

1. Use realistic financial data when possible
2. Demonstrate practical applications
3. Include full workflows (setup, execution, interpretation)
4. Show different parameter configurations
5. Include comments explaining key steps

Example:
```matlab
% GARCH Model Example
% Import daily stock returns
returns = csvread('daily_returns.csv');

% Configure GARCH options
options.model = 'EGARCH';
options.distribution = 'T';
options.p = 1;  % GARCH order
options.q = 1;  % ARCH order

% Estimate the model
results = garchfit(returns, options);

% Display results
disp('EGARCH(1,1) with t-distribution:');
disp(results.parameters);

% Create volatility forecast
forecast = garchforecast(results, 10);
plot(forecast.conditionalVolatility);
```

### Mathematical Documentation

When documenting mathematical or statistical methodology:

1. Include LaTeX formulas for key equations
2. Define all symbols and variables
3. Reference academic papers or textbooks
4. Explain parameter constraints and interpretation
5. Provide statistical properties and assumptions

Example:
```matlab
% EGARCH Model
%
% The EGARCH(p,q) model is specified as:
%
% log(σ_t²) = ω + Σ(i=1 to p) β_i log(σ_{t-i}²) + 
%             Σ(j=1 to q) α_j [θz_{t-j} + γ(|z_{t-j}| - E|z_{t-j}|)]
%
% where:
%   σ_t²: conditional variance at time t
%   ω, β, α, θ, γ: model parameters
%   z_t: standardized residual at time t
```

### Documentation Template

Use the `infrastructure/templates/documentation_template.md` as a starting point for any new documentation file.

## MEX Development Guidelines

### When to Use MEX

Consider MEX implementation when:

1. Function is computationally intensive
2. Function is called frequently
3. Operation involves large loops or recursion
4. Performance profiling shows a bottleneck
5. Memory efficiency is critical

Examples of good MEX candidates:
- Volatility model likelihood computation
- Bootstrap resampling implementation
- Large matrix operations
- Iterative numerical optimization

### C Implementation Standards

1. Follow best practices for numerical computing:
   - Use stable algorithms
   - Minimize cancellation errors
   - Avoid unnecessary branching in loops

2. Structure your C code logically:
   - Clear function organization
   - Separate computation from MEX interface
   - Document complex algorithms

3. Error handling and validation:
   - Check all memory allocations
   - Validate array dimensions
   - Handle NaN and Inf values
   - Provide meaningful error messages

### Memory Management

1. Allocate memory using `mxCalloc`/`mxMalloc` to leverage MATLAB's memory manager
2. Always check allocation success:
   ```c
   double *data = mxCalloc(n, sizeof(double));
   if (data == NULL) {
       mexErrMsgIdAndTxt("MFEToolbox:OutOfMemory", 
                         "Failed to allocate memory");
   }
   ```

3. Free memory explicitly if not returning it to MATLAB:
   ```c
   mxFree(data);
   ```

4. Be cautious with large allocations:
   - Consider chunking operations
   - Implement progress tracking for long operations
   - Use algorithms with lower memory footprint

### Error Handling in MEX

1. Input validation should occur before any computation:
   ```c
   /* Check input dimensions */
   if (mxGetM(prhs[0]) < 2) {
       mexErrMsgIdAndTxt("MFEToolbox:InvalidDimension",
                         "Input data must have at least 2 rows");
   }
   ```

2. Use consistent error identifiers:
   - Format: "MFEToolbox:ComponentName:ErrorType"
   - Example: "MFEToolbox:GARCH:InvalidParameter"

3. Memory cleanup in error conditions:
   ```c
   if (error_condition) {
       mxFree(buffer1);
       mxFree(buffer2);
       mexErrMsgIdAndTxt("MFEToolbox:Error", "Error message");
   }
   ```

4. Check for numerical issues:
   - NaN/Inf detection
   - Boundary cases
   - Numerical underflow/overflow

### Cross-Platform Compilation

1. Use platform-independent code:
   - Avoid compiler-specific features
   - Use standard C functions
   - Don't assume byte ordering

2. Compile with the `-largeArrayDims` flag:
   ```matlab
   mex -largeArrayDims garch_core.c
   ```

3. Test on all supported platforms:
   - Windows (PCWIN64)
   - Unix/Linux (UNIX)
   - macOS

4. Use conditional compilation for platform-specific code:
   ```c
   #ifdef _WIN32
       /* Windows-specific code */
   #else
       /* Unix/Linux/macOS code */
   #endif
   ```

### MEX Optimization Techniques

1. Minimize data copying:
   - Use in-place operations when possible
   - Access matrix data directly with mxGetPr/mxGetPi

2. Optimize memory access patterns:
   - Access arrays in row-major order (C style)
   - Use stride variables for clarity
   - Cache frequently accessed values

3. Take advantage of SIMD instructions:
   - Use compiler optimization flags
   - Consider aligned memory allocation
   - Use vector math libraries when available

4. Parallelization:
   - Use OpenMP for parallel loops
   - Balance parallelization overhead with gains
   - Consider thread safety

Example of optimized matrix access:
```c
/* Get matrix dimensions */
mwSize m = mxGetM(prhs[0]);
mwSize n = mxGetN(prhs[0]);

/* Get pointer to matrix data */
double *data = mxGetPr(prhs[0]);

/* Access using row-major indexing */
for (mwSize j = 0; j < n; j++) {
    for (mwSize i = 0; i < m; i++) {
        /* C style indexing (row-major) */
        double value = data[i + j*m];
        /* Process value */
    }
}
```

## Cross-Platform Considerations

### Windows-Specific Considerations

1. Path handling:
   - Use `filesep` in MATLAB code
   - Avoid hardcoding backslashes in paths

2. MEX compilation:
   - Use a compatible version of Visual Studio
   - Windows binaries have `.mexw64` extension
   - DLL dependencies go in the 'dlls' directory

3. File operations:
   - Be aware of file locking differences
   - File permissions behave differently than Unix

### Unix-Specific Considerations

1. Path handling:
   - Use `filesep` in MATLAB code
   - Avoid hardcoding forward slashes

2. MEX compilation:
   - Use GCC for Unix systems
   - Unix binaries have `.mexa64` extension
   - Ensure shared libraries are accessible

3. File operations:
   - Check file permissions
   - Use proper path separators

### Path Handling

1. Always use platform-independent path operations:
   ```matlab
   fullFilePath = fullfile(baseDir, 'data', 'financial_data.csv');
   ```

2. Check for file existence before operations:
   ```matlab
   if ~exist(filePath, 'file')
       error('MFEToolbox:FileNotFound', 'Required file not found: %s', filePath);
   end
   ```

3. Use relative paths from MATLAB's current directory when possible

### MEX Binary Compatibility

1. Compile MEX files on all supported platforms
2. Include platform-specific binaries in the distribution
3. Use the platform check in addToPath.m to load appropriate binaries
4. Test with various MATLAB versions and OS versions
5. Document minimum platform requirements

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Descriptive Title**: Brief summary of the issue
2. **Version Information**: MFE Toolbox version and MATLAB version
3. **Operating System**: Windows/Unix and version
4. **Steps to Reproduce**: Minimal code example that reproduces the issue
5. **Expected Behavior**: What you expected to happen
6. **Actual Behavior**: What actually happened (include error messages)
7. **Additional Context**: Any other relevant information

Example bug report:
```
Title: GARCH parameter estimation fails with t-distribution

Version: MFE Toolbox 4.0, MATLAB R2019b
OS: Windows 10 Pro 64-bit

Steps to Reproduce:
1. Load the example data:
   returns = [0.01, -0.02, 0.015, -0.01, 0.02];
2. Run GARCH with t-distribution:
   options.distribution = 'T';
   result = garchfit(returns, options);

Expected Behavior:
GARCH model should be estimated with t-distribution error

Actual Behavior:
Error: "Matrix is singular, close to singular or badly scaled."

Additional Context:
Works correctly with normal distribution
```

### Feature Requests

When requesting new features, please include:

1. **Descriptive Title**: Brief summary of the requested feature
2. **Use Case**: Why the feature is needed
3. **Proposed Solution**: If you have ideas for implementation
4. **Alternatives Considered**: Other approaches you've thought about
5. **Additional Context**: Any other relevant information

Example feature request:
```
Title: Add Copula-GARCH models for multivariate volatility

Use Case:
Financial risk management often requires modeling the dependence 
structure between multiple assets. Copula-GARCH models would allow 
for flexible multivariate volatility modeling beyond DCC-GARCH.

Proposed Solution:
Implement common copula functions (Gaussian, t, Clayton, Gumbel) 
with parameter estimation, random number generation, and 
model selection criteria.

Alternatives Considered:
DCC-GARCH models are available but cannot capture tail dependence 
effectively. Factor-GARCH is another option but less flexible for 
asymmetric dependencies.

Additional Context:
References:
[1] Author, A. (2015). "Copula-Based Volatility Models." Journal...
```

### Issue Prioritization

Issues are prioritized based on:

1. **Impact**: How severely the issue affects core functionality
2. **Frequency**: How often users encounter the issue
3. **Complexity**: How complex the issue is to resolve
4. **Dependencies**: Whether other components depend on the fix/feature
5. **Resources**: Available developer resources

Maintainers will assign appropriate priority labels to issues.

## Release Process

The MFE Toolbox follows a versioning scheme based on the following format:

```
MAJOR.MINOR (DD-MMM-YYYY)
```

For example: `4.0 (28-Oct-2009)`

### Version Components

- **MAJOR**: Incremented for significant, potentially breaking changes
- **MINOR**: Incremented for backward-compatible additions
- **Date**: Release date in DD-MMM-YYYY format

### Release Preparation

1. Maintainers consolidate changes for a release
2. All tests are verified on supported platforms
3. Documentation is updated
4. Contents.m is updated with version information
5. A release branch is created
6. Final verification and testing occurs
7. Release is tagged and distributed

### Change Documentation

All changes should be documented in:

1. Contents.m version history
2. Release notes
3. Function documentation
4. Test updates

## License Compliance

The MFE Toolbox is copyrighted software. Contributors must:

1. Ensure new code is original or properly licensed
2. Include copyright notices in all new files
3. Document any third-party code or algorithms
4. Reference academic papers for implemented methods
5. Comply with the MFE Toolbox license terms

## Getting Help

### Resources

- **Documentation**: Comprehensive documentation in the docs folder
- **Examples**: Example scripts in each component directory
- **Function Help**: Help text accessible via MATLAB's help system
- **Templates**: Code templates in the infrastructure/templates directory

### Contact

For questions or support, you can:

1. Create an issue on GitHub
2. Contact the maintainers
3. Join the discussion forum

### Additional Resources

- References to textbooks and papers on financial econometrics
- MATLAB documentation for related functions
- Academic papers implementing the relevant methods
```

# docs/contributing.md
```markdown
# Contributing to the MFE Toolbox

## Introduction

The MFE (MATLAB Financial Econometrics) Toolbox is a sophisticated MATLAB-based software suite designed to provide comprehensive tools for financial time series modeling, econometric analysis, and risk assessment. This toolbox enables accurate risk assessment, statistical inference, time series analysis, and high-performance computing for financial applications.

Contributing to the MFE Toolbox helps maintain and enhance a high-quality financial econometrics toolbox that serves researchers, financial analysts, econometricians, and risk managers across both academic and industry sectors. Your contributions help strengthen the financial econometric community and promote advanced quantitative methods.

## Code of Conduct

All contributors to the MFE Toolbox project are expected to adhere to the following guidelines:

- **Respectful Communication**: Engage with others in a professional and respectful manner.
- **Inclusive Environment**: Welcome contributions from individuals of all backgrounds and experience levels.
- **Constructive Feedback**: Provide and accept constructive criticism that helps improve the project.
- **Focus on Technical Merit**: Evaluate contributions based on technical merit rather than personal attributes.
- **Acknowledge Contributions**: Recognize the work and efforts of others in the project.
- **Responsible Disclosure**: Report security vulnerabilities or critical bugs privately before disclosing publicly.

Contributors who violate these standards may be temporarily or permanently excluded from the project.

## Development Environment Setup

### MATLAB Requirements

- **MATLAB Version**: A recent version of MATLAB is required (2018b or newer recommended)
- **Required Toolboxes**:
  - MATLAB Statistics Toolbox
  - MATLAB Optimization Toolbox
- **Optional Toolboxes**:
  - MATLAB Parallel Computing Toolbox (for improved performance)

### C Compiler Setup

The MFE Toolbox uses MEX files to achieve high performance for critical numerical operations. Setting up the appropriate C compiler is essential for development and testing.

#### Windows Setup
1. Install a supported C compiler:
   - Microsoft Visual C++ (recommended)
   - MinGW-w64 C/C++ Compiler
   
2. Configure MATLAB to use the compiler:
   ```matlab
   mex -setup
   ```

3. Verify the compiler is correctly configured:
   ```matlab
   mexext % Should return 'mexw64' for 64-bit Windows
   ```

#### Unix/Linux Setup
1. Install a supported C compiler:
   - GCC (recommended)
   
2. Configure MATLAB to use the compiler:
   ```matlab
   mex -setup
   ```

3. Verify the compiler is correctly configured:
   ```matlab
   mexext % Should return 'mexa64' for 64-bit Unix/Linux
   ```

### Repository Structure

The MFE Toolbox repository is structured as follows:

```
MFEToolbox/
├── bootstrap/         # Bootstrap implementation
├── crosssection/      # Cross-sectional analysis tools
├── distributions/     # Statistical distribution functions
├── GUI/               # ARMAX modeling interface
├── multivariate/      # Multivariate analysis tools
├── tests/             # Statistical testing suite
├── timeseries/        # Time series analysis
├── univariate/        # Univariate analysis tools
├── utility/           # Helper functions
├── realized/          # High-frequency analysis
├── mex_source/        # C source files for MEX
├── dlls/              # Platform-specific MEX binaries
├── docs/              # Documentation
├── infrastructure/    # Build and support tools
|   └── templates/     # Code and documentation templates
├── .github/           # GitHub configuration files
├── addToPath.m        # Path configuration utility
└── Contents.m         # Version information
```

### Path Configuration

To set up your development environment:

1. Clone the repository to your local machine
2. Start MATLAB
3. Navigate to the repository root directory
4. Run the path configuration script:
   ```matlab
   run addToPath
   ```
5. Confirm that the toolbox components are accessible:
   ```matlab
   help MFE
   ```

## Coding Standards

### MATLAB Coding Standards

#### Naming Conventions
- **Functions**: Use camelCase (e.g., `armaxFilter`, `garchFit`)
- **Variables**: Use descriptive camelCase names (e.g., `parameterResults`, `timeSeriesData`)
- **Constants**: Use UPPER_CASE with underscores (e.g., `MAX_ITERATIONS`, `DEFAULT_TOLERANCE`)
- **Files**: Function files should match the function name (e.g., `armaxFilter.m`)

#### Code Structure
- Each function should perform a single, well-defined task
- Use helper functions for complex tasks
- Limit function length to improve readability (aim for under 200 lines)
- Use comments to explain complex algorithms or non-obvious code

#### Documentation
- All functions must have comprehensive help text using the standard template
- Include clear descriptions of inputs, outputs, and examples
- Document any reference to academic papers or algorithms
- Update Contents.m when adding new functions

#### Error Handling
- Validate all inputs at the start of each function
- Use the `parametercheck`, `datacheck`, and other validation utilities
- Provide meaningful error messages with specific information
- Use try-catch blocks for operations that might fail
- Always clean up resources in catch blocks

#### Performance Optimization
- Pre-allocate arrays where possible
- Use vectorized operations instead of loops
- Minimize memory usage for large datasets
- Consider MEX implementations for performance-critical sections

### C Coding Standards

#### Naming Conventions
- **Functions**: Use snake_case (e.g., `compute_garch_likelihood`, `process_matrix`)
- **Variables**: Use descriptive snake_case names (e.g., `input_matrix`, `result_array`)
- **Constants**: Use UPPER_CASE with underscores (e.g., `MAX_ARRAY_SIZE`, `DEFAULT_TOLERANCE`)
- **Files**: Use descriptive names ending with "_core.c" (e.g., `garch_core.c`)

#### Code Structure
- Include clear function headers with descriptions
- Limit function complexity
- Use consistent indentation (4 spaces recommended)
- Break complex calculations into meaningful steps

#### Memory Management
- Always check allocation success with error handling
- Free all allocated memory before function exit
- Avoid memory leaks in error conditions
- Use appropriate data types to optimize memory usage

#### MEX Interface
- Validate all inputs at the start of the function
- Check array dimensions before access
- Handle edge cases (empty arrays, NaN values)
- Return useful error messages to MATLAB

#### Error Handling
- Check return values from all library functions
- Provide detailed error messages
- Clean up resources properly in error conditions
- Use mexErrMsgIdAndTxt for error reporting

### Function Template

All new functions should follow the standard function template. The template includes:

1. Function signature
2. Comprehensive help text
3. Input validation
4. Primary computation
5. Error handling
6. Results formatting

See `infrastructure/templates/function_template.m` for the detailed template.

### Error Handling Patterns

Robust error handling is critical for production-quality financial software. Follow these patterns:

1. **Validation First**: Check all inputs before any computation
2. **Specific Error Messages**: Include parameter names and expected values
3. **Contextual Information**: Provide context for debugging
4. **Resource Cleanup**: Ensure all resources are freed in error conditions
5. **Graceful Degradation**: When possible, complete partial operations

Example:
```matlab
try
    % Main computation
catch ME
    % Enhanced error handling with context
    errorID = 'MFEToolbox:functionName';
    
    % Add additional context to the error message
    if strcmp(ME.identifier, 'MATLAB:nomem')
        errorMessage = sprintf(['Error in functionName: Out of memory. ' ...
            'Try using a smaller dataset or increasing available memory.']);
    else
        errorMessage = sprintf(['Error in functionName: %s\nFunction failed ' ...
            'at line %d of file %s.'], ME.message, ME.line, ME.stack(1).file);
    end
    
    % Throw error with context
    error(errorID, errorMessage);
end
```

### Performance Optimization

Performance is critical for financial computation. Follow these guidelines:

1. **Vectorization**: Use MATLAB's built-in vectorized operations
2. **Memory Preallocation**: Always preallocate arrays
3. **Data Types**: Use appropriate data types (e.g., sparse matrices when relevant)
4. **Algorithm Selection**: Choose algorithms with appropriate complexity
5. **MEX Implementation**: Consider C implementation for critical paths

## Contribution Workflow

### Fork and Clone

1. Fork the MFE Toolbox repository to your GitHub account
2. Clone your fork to your local machine:
   ```
   git clone https://github.com/your-username/MFEToolbox.git
   ```
3. Add the main repository as an upstream remote:
   ```
   git remote add upstream https://github.com/main-repo/MFEToolbox.git
   ```

### Branching Strategy

1. Create a new branch for each feature or bugfix:
   ```
   git checkout -b feature/your-feature-name
   ```
   or
   ```
   git checkout -b fix/your-bugfix-name
   ```

2. Keep branches focused on a single issue or feature
3. Regularly sync your fork with the upstream repository:
   ```
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

### Commit Guidelines

1. Write clear, descriptive commit messages
2. Use present tense ("Add feature" not "Added feature")
3. Reference issue numbers when applicable
4. Keep commits focused and logical
5. Example commit message:
   ```
   Add Hansen's skewed T distribution estimation

   - Implement maximum likelihood parameter estimation
   - Add random number generation function
   - Include numerical accuracy tests
   - Update documentation with examples

   Closes #123
   ```

### Pull Requests

1. Push your branch to your fork:
   ```
   git push origin feature/your-feature-name
   ```

2. Create a pull request (PR) to the main repository
3. Fill out the PR template completely
4. Include tests for your changes
5. Ensure all tests pass
6. Link to relevant issues

### Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Be responsive to questions during review
5. Keep PRs small and focused to speed up review

## Testing Guidelines

### Test Coverage Requirements

1. All new functions must have corresponding tests
2. Tests should cover normal operation, edge cases, and error conditions
3. Minimum coverage requirements:
   - 90% line coverage for core statistical functions
   - 80% line coverage for utility functions
   - 95% coverage for MEX wrapper functions

### Unit Testing

1. Create test classes inheriting from `BaseTest`
2. Name test files with the pattern `FunctionNameTest.m`
3. Verify numerical accuracy with appropriate tolerances
4. Test with representative financial data
5. Use the `infrastructure/templates/test_template.m` as a starting point

Example test method:
```matlab
function testBasicFunctionality(obj)
    % Prepare test data
    data = [1.2, 2.3, 3.4; 4.5, 5.6, 6.7];
    options.param1 = 0.95;
    
    % Call function under test
    result = functionUnderTest(data, options);
    
    % Verify results
    expectedValue = 1.234;
    obj.assertAlmostEqual(expectedValue, result.value, ...
        'Function produced incorrect value');
end
```

### Integration Testing

1. Create tests that verify interaction between components
2. Test complete workflows (e.g., ARMA model + volatility model)
3. Verify that results are consistent across integrated components
4. Test with realistic financial datasets

Example:
```matlab
function testArmaGarchIntegration(obj)
    % Test that ARMA and GARCH models work together correctly
    returns = obj.loadTestData('financial_returns.mat');
    
    % First estimate ARMA model
    armaResults = armax(returns, [1,1]);
    
    % Then use residuals for GARCH estimation
    garchResults = garch(armaResults.residuals);
    
    % Verify combined model behaves as expected
    % ...
end
```

### MEX Testing

1. Test MEX functions across all supported platforms
2. Verify memory management with large datasets
3. Check numerical precision against MATLAB implementations
4. Test error handling and validation in C code
5. Verify binary compatibility across environments

### Performance Testing

1. Measure execution time with various dataset sizes
2. Compare against baseline performance
3. Verify memory utilization is reasonable
4. Test scaling behavior with large datasets
5. Document performance characteristics

Example:
```matlab
function testPerformance(obj)
    sizes = [100, 1000, 10000];
    times = zeros(size(sizes));
    
    for i = 1:length(sizes)
        n = sizes(i);
        data = randn(n, 3);
        
        tic;
        functionUnderTest(data);
        times(i) = toc;
        
        fprintf('Size %d: %.6f seconds\n', n, times(i));
    end
    
    % Verify scaling behavior
    obj.assertTrue(times(3)/times(1) < 200, ...
        'Function does not scale well with dataset size');
end
```

### Running Tests

1. Navigate to the test directory
2. Run specific tests:
   ```matlab
   testObj = FunctionNameTest();
   results = testObj.runAllTests();
   ```
3. Run all tests:
   ```matlab
   runAllTests
   ```
4. Check test coverage:
   ```matlab
   generateCoverageReport
   ```

## Documentation Guidelines

### Function Documentation

All functions must be documented using the standardized header format:

```matlab
%% FUNCTIONNAME Brief description of the function's purpose
%
% Detailed description of the function, methodology, algorithm,
% and important concepts.
%
% USAGE:
%   results = functionName(data)
%   results = functionName(data, options)
%
% INPUTS:
%   data        - T by K matrix of input data
%                 Description of what each column represents
%                 First column: [Description of first column]
%
%   options     - [OPTIONAL] Structure containing configuration parameters
%                 Default: [] (Uses default values for all options)
%                 Fields:
%                   options.field1 - Description [default = value1]
%
% OUTPUTS:
%   results     - Structure containing function outputs with fields:
%                   results.field1 - Description of field1
%
% COMMENTS:
%   Implementation notes, assumptions, and important details.
%
% EXAMPLES:
%   % Basic usage example
%   data = [1 2; 3 4; 5 6];
%   results = functionName(data);
%
% REFERENCES:
%   [1] Author, A. (Year). "Title of Paper." Journal, Volume(Issue), Pages.
%
% SEE ALSO:
%   relatedFunction1, relatedFunction2
%
% MFE Toolbox v4.0
% Copyright (c) 2009
```

### Module Documentation

For each module (directory), provide an overview document that includes:

1. Module purpose and scope
2. List of main functions
3. Common usage patterns
4. Integration with other modules
5. Examples demonstrating typical workflows

### Example Creation

Examples should:

1. Use realistic financial data when possible
2. Demonstrate practical applications
3. Include full workflows (setup, execution, interpretation)
4. Show different parameter configurations
5. Include comments explaining key steps

Example:
```matlab
% GARCH Model Example
% Import daily stock returns
returns = csvread('daily_returns.csv');

% Configure GARCH options
options.model = 'EGARCH';
options.distribution = 'T';
options.p = 1;  % GARCH order
options.q = 1;  % ARCH order

% Estimate the model
results = garchfit(returns, options);

% Display results
disp('EGARCH(1,1) with t-distribution:');
disp(results.parameters);

% Create volatility forecast
forecast = garchforecast(results, 10);
plot(forecast.conditionalVolatility);
```

### Mathematical Documentation

When documenting mathematical or statistical methodology:

1. Include LaTeX formulas for key equations
2. Define all symbols and variables
3. Reference academic papers or textbooks
4. Explain parameter constraints and interpretation
5. Provide statistical properties and assumptions

Example:
```matlab
% EGARCH Model
%
% The EGARCH(p,q) model is specified as:
%
% log(σ_t²) = ω + Σ(i=1 to p) β_i log(σ_{t-i}²) + 
%             Σ(j=1 to q) α_j [θz_{t-j} + γ(|z_{t-j}| - E|z_{t-j}|)]
%
% where:
%   σ_t²: conditional variance at time t
%   ω, β, α, θ, γ: model parameters
%   z_t: standardized residual at time t
```

### Documentation Template

Use the `infrastructure/templates/documentation_template.md` as a starting point for any new documentation file.

## MEX Development Guidelines

### When to Use MEX

Consider MEX implementation when:

1. Function is computationally intensive
2. Function is called frequently
3. Operation involves large loops or recursion
4. Performance profiling shows a bottleneck
5. Memory efficiency is critical

Examples of good MEX candidates:
- Volatility model likelihood computation
- Bootstrap resampling implementation
- Large matrix operations
- Iterative numerical optimization

### C Implementation Standards

1. Follow best practices for numerical computing:
   - Use stable algorithms
   - Minimize cancellation errors
   - Avoid unnecessary branching in loops

2. Structure your C code logically:
   - Clear function organization
   - Separate computation from MEX interface
   - Document complex algorithms

3. Error handling and validation:
   - Check all memory allocations
   - Validate array dimensions
   - Handle NaN and Inf values
   - Provide meaningful error messages

### Memory Management

1. Allocate memory using `mxCalloc`/`mxMalloc` to leverage MATLAB's memory manager
2. Always check allocation success:
   ```c
   double *data = mxCalloc(n, sizeof(double));
   if (data == NULL) {
       mexErrMsgIdAndTxt("MFEToolbox:OutOfMemory", 
                         "Failed to allocate memory");
   }
   ```

3. Free memory explicitly if not returning it to MATLAB:
   ```c
   mxFree(data);
   ```

4. Be cautious with large allocations:
   - Consider chunking operations
   - Implement progress tracking for long operations
   - Use algorithms with lower memory footprint

### Error Handling in MEX

1. Input validation should occur before any computation:
   ```c
   /* Check input dimensions */
   if (mxGetM(prhs[0]) < 2) {
       mexErrMsgIdAndTxt("MFEToolbox:InvalidDimension",
                         "Input data must have at least 2 rows");
   }
   ```

2. Use consistent error identifiers:
   - Format: "MFEToolbox:ComponentName:ErrorType"
   - Example: "MFEToolbox:GARCH:InvalidParameter"

3. Memory cleanup in error conditions:
   ```c
   if (error_condition) {
       mxFree(buffer1);
       mxFree(buffer2);
       mexErrMsgIdAndTxt("MFEToolbox:Error", "Error message");
   }
   ```

4. Check for numerical issues:
   - NaN/Inf detection
   - Boundary cases
   - Numerical underflow/overflow

### Cross-Platform Compilation

1. Use platform-independent code:
   - Avoid compiler-specific features
   - Use standard C functions
   - Don't assume byte ordering

2. Compile with the `-largeArrayDims` flag:
   ```matlab
   mex -largeArrayDims garch_core.c
   ```

3. Test on all supported platforms:
   - Windows (PCWIN64)
   - Unix/Linux (UNIX)
   - macOS

4. Use conditional compilation for platform-specific code:
   ```c
   #ifdef _WIN32
       /* Windows-specific code */
   #else
       /* Unix/Linux/macOS code */
   #endif
   ```

### MEX Optimization Techniques

1. Minimize data copying:
   - Use in-place operations when possible
   - Access matrix data directly with mxGetPr/mxGetPi

2. Optimize memory access patterns:
   - Access arrays in row-major order (C style)
   - Use stride variables for clarity
   - Cache frequently accessed values

3. Take advantage of SIMD instructions:
   - Use compiler optimization flags
   - Consider aligned memory allocation
   - Use vector math libraries when available

4. Parallelization:
   - Use OpenMP for parallel loops
   - Balance parallelization overhead with gains
   - Consider thread safety

Example of optimized matrix access:
```c
/* Get matrix dimensions */
mwSize m = mxGetM(prhs[0]);
mwSize n = mxGetN(prhs[0]);

/* Get pointer to matrix data */
double *data = mxGetPr(prhs[0]);

/* Access using row-major indexing */
for (mwSize j = 0; j < n; j++) {
    for (mwSize i = 0; i < m; i++) {
        /* C style indexing (row-major) */
        double value = data[i + j*m];
        /* Process value */
    }
}
```

## Cross-Platform Considerations

### Windows-Specific Considerations

1. Path handling:
   - Use `filesep` in MATLAB code
   - Avoid hardcoding backslashes in paths

2. MEX compilation:
   - Use a compatible version of Visual Studio
   - Windows binaries have `.mexw64` extension
   - DLL dependencies go in the 'dlls' directory

3. File operations:
   - Be aware of file locking differences
   - File permissions behave differently than Unix

### Unix-Specific Considerations

1. Path handling:
   - Use `filesep` in MATLAB code
   - Avoid hardcoding forward slashes

2. MEX compilation:
   - Use GCC for Unix systems
   - Unix binaries have `.mexa64` extension
   - Ensure shared libraries are accessible

3. File operations:
   - Check file permissions
   - Use proper path separators

### Path Handling

1. Always use platform-independent path operations:
   ```matlab
   fullFilePath = fullfile(baseDir, 'data', 'financial_data.csv');
   ```

2. Check for file existence before operations:
   ```matlab
   if ~exist(filePath, 'file')
       error('MFEToolbox:FileNotFound', 'Required file not found: %s', filePath);
   end
   ```

3. Use relative paths from MATLAB's current directory when possible

### MEX Binary Compatibility

1. Compile MEX files on all supported platforms
2. Include platform-specific binaries in the distribution
3. Use the platform check in addToPath.m to load appropriate binaries
4. Test with various MATLAB versions and OS versions
5. Document minimum platform requirements

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Descriptive Title**: Brief summary of the issue
2. **Version Information**: MFE Toolbox version and MATLAB version
3. **Operating System**: Windows/Unix and version
4. **Steps to Reproduce**: Minimal code example that reproduces the issue
5. **Expected Behavior**: What you expected to happen
6. **Actual Behavior**: What actually happened (include error messages)
7. **Additional Context**: Any other relevant information

Example bug report:
```
Title: GARCH parameter estimation fails with t-distribution

Version: MFE Toolbox 4.0, MATLAB R2019b
OS: Windows 10 Pro 64-bit

Steps to Reproduce:
1. Load the example data:
   returns = [0.01, -0.02, 0.015, -0.01, 0.02];
2. Run GARCH with t-distribution:
   options.distribution = 'T';
   result = garchfit(returns, options);

Expected Behavior:
GARCH model should be estimated with t-distribution error

Actual Behavior:
Error: "Matrix is singular, close to singular or badly scaled."

Additional Context:
Works correctly with normal distribution
```

### Feature Requests

When requesting new features, please include:

1. **Descriptive Title**: Brief summary of the requested feature
2. **Use Case**: Why the feature is needed
3. **Proposed Solution**: If you have ideas for implementation
4. **Alternatives Considered**: Other approaches you've thought about
5. **Additional Context**: Any other relevant information

Example feature request:
```
Title: Add Copula-GARCH models for multivariate volatility

Use Case:
Financial risk management often requires modeling the dependence 
structure between multiple assets. Copula-GARCH models would allow 
for flexible multivariate volatility modeling beyond DCC-GARCH.

Proposed Solution:
Implement common copula functions (Gaussian, t, Clayton, Gumbel) 
with parameter estimation, random number generation, and 
model selection criteria.

Alternatives Considered:
DCC-GARCH models are available but cannot capture tail dependence 
effectively. Factor-GARCH is another option but less flexible for 
asymmetric dependencies.

Additional Context:
References:
[1] Author, A. (2015). "Copula-Based Volatility Models." Journal...
```

### Issue Prioritization

Issues are prioritized based on:

1. **Impact**: How severely the issue affects core functionality
2. **Frequency**: How often users encounter the issue
3. **Complexity**: How complex the issue is to resolve
4. **Dependencies**: Whether other components depend on the fix/feature
5. **Resources**: Available developer resources

Maintainers will assign appropriate priority labels to issues.

## Release Process

The MFE Toolbox follows a versioning scheme based on the following format:

```
MAJOR.MINOR (DD-MMM-YYYY)
```

For example: `4.0 (28-Oct-2009)`

### Version Components

- **MAJOR**: Incremented for significant, potentially breaking changes
- **MINOR**: Incremented for backward-compatible additions
- **Date**: Release date in DD-MMM-YYYY format

### Release Preparation

1. Maintainers consolidate changes for a release
2. All tests are verified on supported platforms
3. Documentation is updated
4. Contents.m is updated with version information
5. A release branch is created
6. Final verification and testing occurs
7. Release is tagged and distributed

### Change Documentation

All changes should be documented in:

1. Contents.m version history
2. Release notes
3. Function documentation
4. Test updates

## License Compliance

The MFE Toolbox is copyrighted software. Contributors must:

1. Ensure new code is original or properly licensed
2. Include copyright notices in all new files
3. Document any third-party code or algorithms
4. Reference academic papers for implemented methods
5. Comply with the MFE Toolbox license terms

## Getting Help

### Resources

- **Documentation**: Comprehensive documentation in the docs folder
- **Examples**: Example scripts in each component directory
- **Function Help**: Help text accessible via MATLAB's help system
- **Templates**: Code templates in the infrastructure/templates directory

### Contact

For questions or support, you can:

1. Create an issue on GitHub
2. Contact the maintainers
3. Join the discussion forum

### Additional Resources

- References to textbooks and papers on financial econometrics
- MATLAB documentation for related functions
- Academic papers implementing the relevant methods