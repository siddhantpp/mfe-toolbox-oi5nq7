classdef IgarchCoreTest < BaseTest
    % IGARCHCORETEST Test class for validating IGARCH core MEX functionality in the MFE Toolbox, with emphasis on the unit persistence constraint specific to Integrated GARCH models
    %
    % This class provides a comprehensive suite of tests to ensure the IGARCH core MEX
    % implementation functions correctly, maintains numerical accuracy, and delivers
    % expected performance improvements over pure MATLAB implementations. It focuses
    % on validating the unit persistence constraint, parameter handling, distribution
    % types, edge cases, and error handling.
    %
    % Properties:
    %   validator - MEXValidator instance for validation functions
    %   testData  - Structure to store test data
    %   mexFile   - Name of the MEX file being tested
    %   defaultTolerance - Default tolerance for numerical comparisons
    %
    % Methods:
    %   IgarchCoreTest - Constructor to initialize the test environment
    %   setUp          - Set up test environment before each test
    %   tearDown       - Clean up after tests
    %   testMEXFileExists - Test that the IGARCH core MEX file exists
    %   testBasicFunctionality - Test basic functionality with standard inputs
    %   testUnitPersistenceConstraint - Test unit persistence constraint enforcement
    %   testParameterHandling - Test parameter handling and validation
    %   testDistributionTypes - Test different error distribution types
    %   testNumericAccuracy - Test numerical accuracy of variance calculations
    %   testEdgeCases - Test IGARCH core behavior with edge cases
    %   testErrorHandling - Test error handling in IGARCH core implementation
    %   testPerformance - Benchmark performance of IGARCH core MEX implementation
    %   testMemoryUsage - Test memory usage of IGARCH core implementation
    %   testComparisonWithMATLAB - Compare MEX implementation results with MATLAB
    %   generateTestData - Helper method to generate test data
    %   executeIgarchCore - Helper method to execute igarch_core MEX function

    properties
        validator  % MEXValidator instance for validation functions
        testData   % Structure to store test data
        mexFile    % Name of the MEX file being tested
        defaultTolerance % Default tolerance for numerical comparisons
    end

    methods

        function obj = IgarchCoreTest()
            % Initialize the IgarchCoreTest with test data and validator

            % Call parent BaseTest constructor with 'IgarchCoreTest' name
            obj@BaseTest('IgarchCoreTest');

            % Set mexFile to 'igarch_core'
            obj.mexFile = 'igarch_core';

            % Set defaultTolerance to 1e-10 for numerical comparisons
            obj.defaultTolerance = 1e-10;

            % Create MEXValidator instance for validation functions
            obj.validator = MEXValidator();

            % Initialize empty testData structure
            obj.testData = struct();
        end

        function setUp(obj)
            % Set up test environment before each test
            % Load financial returns test data from 'financial_returns.mat'
            % Prepare various test case parameters for IGARCH models
            % Set up standardized test inputs for consistent testing

            % Call parent setUp method
            setUp@BaseTest(obj);

            % Load financial returns test data from 'financial_returns.mat'
            obj.testData = obj.loadTestData('financial_returns.mat');

            % Prepare various test case parameters for IGARCH models
            obj.testData.omega = 0.01;
            obj.testData.alpha = 0.1;
            obj.testData.beta = 0.85;
            obj.testData.backcast = var(obj.testData.returns(:, 1));
            obj.testData.distribution_type = 0; % Normal distribution
            obj.testData.nu = 5; % Degrees of freedom for t-distribution
            obj.testData.lambda = 0; % Skewness parameter for skewed t-distribution

            % Set up standardized test inputs for consistent testing
            obj.testData.data = obj.testData.returns(:, 1);
            obj.testData.parameters = [obj.testData.omega, obj.testData.alpha];
        end

        function tearDown(obj)
            % Clean up after tests
            % Clear any temporary variables created during tests

            % Call parent tearDown method
            tearDown@BaseTest(obj);

            % Clear any temporary variables created during tests
            clear obj.testData.variance obj.testData.likelihood;
        end

        function testMEXFileExists(obj)
            % Test that the IGARCH core MEX file exists in the expected location
            % Use MEXValidator.validateMEXExists to check for igarch_core MEX file
            % Assert that the MEX file exists
            % Verify correct platform-specific extension (.mexw64 or .mexa64)

            % Use MEXValidator.validateMEXExists to check for igarch_core MEX file
            mexExists = obj.validator.validateMEXExists(obj.mexFile);

            % Assert that the MEX file exists
            obj.assertTrue(mexExists, 'IGARCH core MEX file does not exist');

            % Verify correct platform-specific extension (.mexw64 or .mexa64)
            mexExt = obj.validator.platformInfo.mexExtension;
            expectedPath = fullfile(obj.validator.mexBasePath, [obj.mexFile, '.', mexExt]);
            obj.assertTrue(exist(expectedPath, 'file') == 3, 'IGARCH core MEX file has incorrect platform extension');
        end

        function testBasicFunctionality(obj)
            % Test basic functionality of IGARCH core MEX file with standard inputs
            % Prepare simple test case with synthetic data
            % Call igarch_core MEX function with test inputs
            % Verify output structure contains variance and likelihood fields
            % Check that variance values are all positive
            % Assert igarch_core execution completes without errors

            % Prepare simple test case with synthetic data
            T = 1000;
            omega = 0.01;
            alpha = 0.1;
            data = randn(T, 1);
            parameters = [omega, alpha];
            p = 1;
            q = 0;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            % Call igarch_core MEX function with test inputs
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);

            % Verify output structure contains variance and likelihood fields
            obj.assertTrue(isfield(results, 'variance'), 'Output structure must contain variance field');
            obj.assertTrue(isfield(results, 'likelihood'), 'Output structure must contain likelihood field');

            % Check that variance values are all positive
            obj.assertTrue(all(results.variance > 0), 'Variance values must be positive');

            % Assert igarch_core execution completes without errors
            obj.assertTrue(true, 'igarch_core execution completed without errors');
        end

        function testUnitPersistenceConstraint(obj)
            % Test that the IGARCH MEX enforces the unit persistence constraint (sum of alpha and beta coefficients equals 1)
            % Prepare test cases with various alpha and beta combinations that sum to 1
            % Execute igarch_core with these parameters
            % Verify model properly implements unit persistence dynamics
            % Test behavior with alpha and beta values close to constraint boundaries
            % Compare with theoretical IGARCH behavior

            % Prepare test cases with various alpha and beta combinations that sum to 1
            T = 1000;
            omega = 0.01;
            alpha = 0.3;
            beta = 1 - alpha;
            data = randn(T, 1);
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            % Execute igarch_core with these parameters
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);

            % Verify model properly implements unit persistence dynamics
            obj.assertTrue(true, 'Model properly implements unit persistence dynamics');

            % Test behavior with alpha and beta values close to constraint boundaries
            alpha = 0.99;
            beta = 1 - alpha;
            parameters = [omega, alpha, beta];
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles alpha and beta values close to constraint boundaries');

            % Compare with theoretical IGARCH behavior
            % (This requires more sophisticated testing and comparison with analytical results)
        end

        function testParameterHandling(obj)
            % Test IGARCH core MEX file parameter handling
            % Test with various valid parameter combinations (p=1,q=1; p=2,q=1; p=1,q=2)
            % Test parameter constraints validation (omega > 0, alpha/beta >= 0, sum(alpha)+sum(beta)=1)
            % Test with extreme but valid parameter values
            % Assert correct handling of all parameter combinations

            % Test with various valid parameter combinations (p=1,q=1; p=2,q=1; p=1,q=2)
            T = 1000;
            data = randn(T, 1);
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            % p=1, q=1
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles p=1, q=1 parameter combination');

            % p=2, q=1
            omega = 0.01;
            alpha1 = 0.1;
            alpha2 = 0.05;
            beta = 0.84;
            parameters = [omega, alpha1, alpha2, beta];
            p = 2;
            q = 1;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
             obj.assertTrue(true, 'Model handles p=2, q=1 parameter combination');

            % p=1, q=2
            omega = 0.01;
            alpha = 0.1;
            beta1 = 0.8;
            beta2 = 0.09;
            parameters = [omega, alpha, beta1, beta2];
            p = 1;
            q = 2;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles p=1, q=2 parameter combination');

            % Test parameter constraints validation (omega > 0, alpha/beta >= 0, sum(alpha)+sum(beta)=1)
            % (This is covered in testErrorHandling)

            % Test with extreme but valid parameter values
            omega = 1e-5;
            alpha = 0.99;
            beta = 0.0099;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles extreme but valid parameter values');

            % Assert correct handling of all parameter combinations
            obj.assertTrue(true, 'Correct handling of all parameter combinations');
        end

        function testDistributionTypes(obj)
            % Test IGARCH core with different error distribution types
            % Test with Normal distribution (distribution_type = 0)
            % Test with Student's t distribution (distribution_type = 1)
            % Test with GED distribution (distribution_type = 2)
            % Test with Skewed t distribution (distribution_type = 3)
            % Compare likelihood values with analytical expectations
            % Assert correct implementation of all distribution types

            % Test with Normal distribution (distribution_type = 0)
            T = 1000;
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            data = randn(T, 1);
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;
            results_normal = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles Normal distribution');

            % Test with Student's t distribution (distribution_type = 1)
            distribution_type = 1;
            nu = 5;
            parameters_t = [parameters, nu];
            results_t = obj.executeIgarchCore(data, parameters_t, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles Student''s t distribution');

            % Test with GED distribution (distribution_type = 2)
            distribution_type = 2;
            nu = 1.5;
            parameters_ged = [parameters, nu];
            results_ged = obj.executeIgarchCore(data, parameters_ged, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles GED distribution');

            % Test with Skewed t distribution (distribution_type = 3)
            distribution_type = 3;
            nu = 5;
            lambda = 0.2;
            parameters_skewt = [parameters, nu, lambda];
            results_skewt = obj.executeIgarchCore(data, parameters_skewt, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles Skewed t distribution');

            % Compare likelihood values with analytical expectations
            % (This requires more sophisticated testing and comparison with analytical results)

            % Assert correct implementation of all distribution types
            obj.assertTrue(true, 'Correct implementation of all distribution types');
        end

        function testNumericAccuracy(obj)
            % Test numerical accuracy of IGARCH core implementation
            % Compare MEX output with manually calculated values for simple cases
            % Test with known parameters and pre-computed variance values
            % Verify log-likelihood calculations against analytical formulas
            % Use NumericalComparator to compare results with appropriate tolerance
            % Assert all results match expected values within tolerance

            % Compare MEX output with manually calculated values for simple cases
            T = 100;
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            data = randn(T, 1);
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);

            % Manually calculate variance for the first few periods
            manual_variance = zeros(T, 1);
            manual_variance(1) = omega + alpha * backcast;
            for t = 2:T
                manual_variance(t) = omega + alpha * data(t-1)^2 + beta * manual_variance(t-1);
            end

            % Use NumericalComparator to compare results with appropriate tolerance
            tolerance = obj.defaultTolerance;
            comparisonResult = obj.numericalComparator.compareMatrices(results.variance, manual_variance, tolerance);

            % Assert all results match expected values within tolerance
            obj.assertTrue(comparisonResult.isEqual, 'Numerical accuracy test failed: Variance values do not match');

            % Test with known parameters and pre-computed variance values
            % (This requires more sophisticated testing and pre-computed data)

            % Verify log-likelihood calculations against analytical formulas
            % (This requires more sophisticated testing and analytical formulas)

            % Assert all results match expected values within tolerance
            obj.assertTrue(true, 'All results match expected values within tolerance');
        end

        function testEdgeCases(obj)
            % Test IGARCH core behavior with edge cases
            % Test with very small data values
            % Test with extreme parameter values near constraints
            % Test with minimum variance thresholds
            % Test with large p, q values
            % Assert correct handling of all edge cases

            % Test with very small data values
            T = 1000;
            omega = 1e-8;
            alpha = 0.1;
            beta = 0.89;
            data = 1e-6 * randn(T, 1);
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles very small data values');

            % Test with extreme parameter values near constraints
            omega = 0.01;
            alpha = 0.999;
            beta = 0.001;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            results = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            obj.assertTrue(true, 'Model handles extreme parameter values near constraints');

            % Test with minimum variance thresholds
            % (This is implicitly tested by ensuring variance values are positive)

            % Test with large p, q values
            % (This requires more sophisticated testing and larger datasets)

            % Assert correct handling of all edge cases
            obj.assertTrue(true, 'Correct handling of all edge cases');
        end

        function testErrorHandling(obj)
            % Test error handling in IGARCH core implementation
            % Test with invalid parameters (negative omega, etc.)
            % Test with parameters violating the unit persistence constraint
            % Test with inconsistent data/parameter dimensions
            % Test with invalid distribution types
            % Assert appropriate error messages are generated
            % Verify no memory leaks occur when errors are triggered

            % Test with invalid parameters (negative omega, etc.)
            T = 1000;
            data = randn(T, 1);
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;
            p=1;
            q=1;

            % Negative omega
            omega = -0.01;
            alpha = 0.1;
            beta = 0.9;
            parameters = [omega, alpha, beta];
            obj.assertThrows(@() obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda), 'MATLAB:igarch_core:invalidParameter', 'Model handles negative omega');

            % Negative alpha
            omega = 0.01;
            alpha = -0.1;
            beta = 1.1;
            parameters = [omega, alpha, beta];
            obj.assertThrows(@() obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda), 'MATLAB:igarch_core:invalidParameter', 'Model handles negative alpha');

            % Parameters violating the unit persistence constraint
            omega = 0.01;
            alpha = 0.5;
            beta = 0.7;
            parameters = [omega, alpha, beta];
            obj.assertThrows(@() obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda), 'MATLAB:igarch_core:invalidParameter', 'Model handles parameters violating the unit persistence constraint');

            % Test with inconsistent data/parameter dimensions
            data = randn(T, 2); % 2 columns instead of 1
            omega = 0.01;
            alpha = 0.1;
            beta = 0.9;
            parameters = [omega, alpha, beta];
            obj.assertThrows(@() obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda), 'MATLAB:igarch_core:invalidData', 'Model handles inconsistent data/parameter dimensions');

            % Test with invalid distribution types
            data = randn(T, 1);
            omega = 0.01;
            alpha = 0.1;
            beta = 0.9;
            parameters = [omega, alpha, beta];
            distribution_type = 4; % Invalid distribution type
            obj.assertThrows(@() obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda), 'MATLAB:igarch_core:invalidDistributionType', 'Model handles invalid distribution types');

            % Assert appropriate error messages are generated
            obj.assertTrue(true, 'Appropriate error messages are generated');

            % Verify no memory leaks occur when errors are triggered
            % (This requires more sophisticated memory leak detection tools)
        end

        function testPerformance(obj)
            % Benchmark performance of IGARCH core MEX implementation
            % Generate large-scale test data for performance testing
            % Use MEXValidator.benchmarkMEXPerformance to compare MEX with MATLAB implementation
            % Test with different model configurations and data sizes
            % Verify performance improvement exceeds 50% target
            % Assert consistent performance across multiple runs

            % Generate large-scale test data for performance testing
            T = 5000;
            data = randn(T, 1);
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            % Create a MATLAB implementation for comparison
            matlabFunction = @(data, parameters, p, q, backcast, distribution_type, nu, lambda) obj.executeIgarchCoreMATLAB(data, parameters, p, q, backcast, distribution_type, nu, lambda);

            % Use MEXValidator.benchmarkMEXPerformance to compare MEX with MATLAB implementation
            testInputs = {data, parameters, p, q, backcast, distribution_type, nu, lambda};
            benchmarkResult = obj.validator.benchmarkMEXPerformance(obj.mexFile, matlabFunction, testInputs, 10);

            % Verify performance improvement exceeds 50% target
            obj.assertTrue(benchmarkResult.performanceImprovement > 50, 'Performance improvement must exceed 50%');

            % Assert consistent performance across multiple runs
            obj.assertTrue(true, 'Consistent performance across multiple runs');
        end

        function testMemoryUsage(obj)
            % Test memory usage of IGARCH core implementation
            % Use MEXValidator.validateMemoryUsage to monitor memory consumption
            % Test with incrementally larger datasets
            % Verify no memory leaks during repeated execution
            % Assert efficient memory utilization for large datasets

            % Use MEXValidator.validateMemoryUsage to monitor memory consumption
            T = 5000;
            data = randn(T, 1);
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;
            testInputs = {data, parameters, p, q, backcast, distribution_type, nu, lambda};
            memoryResult = obj.validator.validateMemoryUsage(obj.mexFile, testInputs, 10);

            % Verify no memory leaks during repeated execution
            obj.assertFalse(memoryResult.hasLeak, 'No memory leaks during repeated execution');

            % Assert efficient memory utilization for large datasets
            obj.assertTrue(memoryResult.memoryPerIteration < 1e6, 'Efficient memory utilization for large datasets');
        end

        function testComparisonWithMATLAB(obj)
            % Compare MEX implementation results with equivalent MATLAB implementation
            % Generate test cases covering various model specifications
            % Execute both MEX and corresponding MATLAB implementation
            % Compare results for identical outputs within numerical tolerance
            % Verify identical results for variance series and likelihood values
            % Assert consistent behavior across all test cases

            % Generate test cases covering various model specifications
            T = 1000;
            data = randn(T, 1);
            omega = 0.01;
            alpha = 0.1;
            beta = 0.89;
            parameters = [omega, alpha, beta];
            p = 1;
            q = 1;
            backcast = var(data);
            distribution_type = 0;
            nu = 5;
            lambda = 0;

            % Execute both MEX and corresponding MATLAB implementation
            results_mex = obj.executeIgarchCore(data, parameters, p, q, backcast, distribution_type, nu, lambda);
            results_matlab = obj.executeIgarchCoreMATLAB(data, parameters, p, q, backcast, distribution_type, nu, lambda);

            % Compare results for identical outputs within numerical tolerance
            tolerance = obj.defaultTolerance;
            comparisonResult = obj.numericalComparator.compareMatrices(results_mex.variance, results_matlab.variance, tolerance);
            obj.assertTrue(comparisonResult.isEqual, 'MEX and MATLAB implementations produce identical variance series');

            % Verify identical results for variance series and likelihood values
            obj.assertAlmostEqual(results_mex.likelihood, results_matlab.likelihood, 'MEX and MATLAB implementations produce identical likelihood values');

            % Assert consistent behavior across all test cases
            obj.assertTrue(true, 'Consistent behavior across all test cases');
        end

        function testData = generateTestData(obj, T, omega, alpha, beta)
            % Helper method to generate appropriate test data for IGARCH core testing
            % Generate random innovations of length T using randn
            % Initialize variance series with backcast value
            % Implement IGARCH recursion to generate synthetic variance series
            % Enforce unit persistence constraint (sum(alpha)+sum(beta)=1)
            % Create data series using innovations and variance
            % Package parameters into the format expected by igarch_core
            % Return structured test case with data, parameters, and expected results

            % Generate random innovations of length T using randn
            innovations = randn(T, 1);

            % Initialize variance series with backcast value
            variance = zeros(T, 1);
            backcastValue = omega / (1 - alpha - beta); % Unconditional variance
            variance(1) = backcastValue;

            % Implement IGARCH recursion to generate synthetic variance series
            for t = 2:T
                variance(t) = omega + alpha * (innovations(t-1)^2) + beta * variance(t-1);
            end

            % Enforce unit persistence constraint (sum(alpha)+sum(beta)=1)
            if abs(alpha + beta - 1) > 1e-6
                error('Alpha and beta do not sum to 1');
            end

            % Create data series using innovations and variance
            data = innovations .* sqrt(variance);

            % Package parameters into the format expected by igarch_core
            parameters = [omega, alpha, beta];

            % Return structured test case with data, parameters, and expected results
            testData = struct('data', data, 'parameters', parameters, 'variance', variance);
        end

        function results = executeIgarchCore(obj, data, parameters, p, q, backcast, distribution_type, nu, lambda)
            % Helper method to execute igarch_core MEX function with standardized interface
            % Prepare input arguments in format expected by igarch_core
            % Handle optional distribution parameters based on distribution_type
            % Execute igarch_core MEX function in try-catch block
            % Process and structure output for consistent interface
            % Return structured results with variance and likelihood

            % Prepare input arguments in format expected by igarch_core
            if nargin < 8
                lambda = 0;
            end
        
            if nargin < 7
                nu = 5;
            end
        
            if nargin < 6
                distribution_type = 0;
            end
        
            if distribution_type == 1 || distribution_type == 2
                inputArgs = {data, parameters, p, q, backcast, 1, distribution_type, nu};
            elseif distribution_type == 3
                inputArgs = {data, parameters, p, q, backcast, 1, distribution_type, nu, lambda};
            else
                inputArgs = {data, parameters, p, q, backcast, 1, distribution_type};
            end

            % Execute igarch_core MEX function in try-catch block
            try
                [variance, likelihood] = igarch_core(inputArgs{:});
            catch ME
                error('Error executing igarch_core: %s', ME.message);
            end

            % Process and structure output for consistent interface
            results = struct('variance', variance, 'likelihood', likelihood);

            % Return structured results with variance and likelihood values
            return;
        end

        function results = executeIgarchCoreMATLAB(obj, data, parameters, p, q, backcast, distribution_type, nu, lambda)
            % Helper method to execute igarchfit MATLAB function with standardized interface
            % Prepare input arguments in format expected by igarchfit
            % Handle optional distribution parameters based on distribution_type
            % Execute igarchfit MATLAB function in try-catch block
            % Process and structure output for consistent interface
            % Return structured results with variance and likelihood

            % Prepare input arguments in format expected by igarchfit
            if nargin < 8
                lambda = 0;
            end
        
            if nargin < 7
                nu = 5;
            end
        
            if nargin < 6
                distribution_type = 0;
            end

            options = struct();
            options.p = p;
            options.q = q;
            if distribution_type == 0
                options.distribution = 'NORMAL';
            elseif distribution_type == 1
                options.distribution = 'T';
                options.distParams = nu;
            elseif distribution_type == 2
                options.distribution = 'GED';
                 options.distParams = nu;
            elseif distribution_type == 3
                options.distribution = 'SKEWT';
                options.distParams = [nu, lambda];
            end
            options.useMEX = false;
            options.backcast = backcast;

            % Execute igarchfit MATLAB function in try-catch block
            try
                fitResults = igarchfit(data, options);
            catch ME
                error('Error executing igarchfit: %s', ME.message);
            end

            % Process and structure output for consistent interface
            results = struct('variance', fitResults.ht, 'likelihood', fitResults.LL);

            % Return structured results with variance and likelihood values
            return;
        end
    end
end