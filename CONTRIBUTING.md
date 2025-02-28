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
        errorMessage = sprintf(['Error in functionName: %s\\nFunction failed ' ...
            'at line %d of file %s.'], ME.message, ME.line, ME.stack(1).file);\n    end
    
    % Throw error with context
    error(errorID, errorMessage);\nend
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
    
    for i = 1:length(sizes))
        n = sizes(i);
        data = randn(n, 3);
        
        tic;
        functionUnderTest(data);
        times(i) = toc;
        
        fprintf('Size %d: %.6f seconds\\n', n, times(i));
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
%\n% Detailed description of the function, methodology, algorithm,\n% and important concepts.\n%\n% USAGE:\n%   results = functionName(data)\n%   results = functionName(data, options)\n%\n% INPUTS:\n%   data        - T by K matrix of input data\n%                 Description of what each column represents\n%                 First column: [Description of first column]\n%\n%   options     - [OPTIONAL] Structure containing configuration parameters\n%                 Default: [] (Uses default values for all options)\n%                 Fields:\n%                   options.field1 - Description [default = value1]\n%\n% OUTPUTS:\n%   results     - Structure containing function outputs with fields:\n%                   results.field1 - Description of field1\n%\n% COMMENTS:\n%   Implementation notes, assumptions, and important details.\n%\n% EXAMPLES:\n%   % Basic usage example\n%   data = [1 2; 3 4; 5 6];\n%   results = functionName(data);\n%\n% REFERENCES:\n%   [1] Author, A. (Year). \"Title of Paper.\" Journal, Volume(Issue), Pages.\n%\n% SEE ALSO:\n%   relatedFunction1, relatedFunction2\n%\n% MFE Toolbox v4.0\n% Copyright (c) 2009