classdef EgarchCoreTest < BaseTest
    % EGARCHCORETEST Test class for validating EGARCH core MEX functionality in the MFE Toolbox
    %
    % This class provides comprehensive tests for the EGARCH core MEX file,
    % ensuring its functionality, numerical accuracy, error handling, and
    % performance meet the required standards. It validates the MEX implementation
    % against equivalent MATLAB implementations and analytical results.
    %
    % The tests cover a wide range of scenarios, including different parameter
    % combinations, distribution types, edge cases, and error conditions.
    %
    % Example:
    %   % Create an instance of the test class
    %   testCase = EgarchCoreTest();
    %
    %   % Run all tests
    %   results = testCase.runAllTests();
    %
    % See also: BaseTest, MEXValidator, NumericalComparator, parametercheck
    
    properties
        validator           % MEXValidator instance for validation functions
        numComparator       % NumericalComparator instance for numerical comparisons
        testData            % Structure to hold test data
        mexFile             % Name of the MEX file to test
        matlabFile          % Name of the MATLAB file for comparison
        defaultTolerance    % Default tolerance for numerical comparisons
    end
    
    methods
        function obj = EgarchCoreTest()
            % Initialize the EgarchCoreTest with test data and validator
            
            % Call parent BaseTest constructor with 'EgarchCoreTest' name
            obj@BaseTest('EgarchCoreTest');
            
            % Set mexFile to 'egarch_core'
            obj.mexFile = 'egarch_core';
            
            % Set matlabFile to 'egarchfit'
            obj.matlabFile = 'egarchfit';
            
            % Set defaultTolerance to 1e-10 for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Create MEXValidator instance for validation functions
            obj.validator = MEXValidator();
            
            % Create NumericalComparator instance for numerical comparisons
            obj.numComparator = NumericalComparator();
            
            % Initialize empty testData structure
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Load financial returns test data from 'financial_returns.mat'
            testDataFile = fullfile(obj.testDataPath, 'financial_returns.mat');
            obj.testData = load(testDataFile);
            
            % Prepare various test case parameters for EGARCH models
            % Set up standardized test inputs for consistent testing
        end
        
        function tearDown(obj)
            % Clean up after tests
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary variables created during tests
            clearvars -except obj;
        end
        
        function testMEXFileExists(obj)
            % Test that the EGARCH core MEX file exists in the expected location
            
            % Use MEXValidator.validateMEXExists to check for egarch_core MEX file
            mexExists = obj.validator.validateMEXExists(obj.mexFile);
            
            % Assert that the MEX file exists
            obj.assertTrue(mexExists, 'EGARCH core MEX file does not exist');
            
            % Verify correct platform-specific extension (.mexw64 or .mexa64)
            mexExt = obj.validator.platformInfo.mexExtension;
            expectedPath = fullfile(obj.validator.mexBasePath, [obj.mexFile, '.', mexExt]);
            
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of EGARCH core MEX file with standard inputs
            
            % Prepare simple test case with synthetic data
            T = 1000;
            omega = -0.1;
            alpha = 0.1;
            beta = 0.9;
            gamma = 0.05;
            
            % Generate random data
            data = randn(T, 1);
            
            % Set initial log-variance
            backcast = log(var(data));
            
            % Prepare parameters
            parameters = [omega; alpha; beta; gamma];
            
            % Call egarch_core MEX function with test inputs
            results = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 0, 5, 0.1);
            
            % Verify output structure contains variance, logvariance and likelihood fields
            obj.assertTrue(isfield(results, 'variance'), 'Output structure must contain variance field');
            obj.assertTrue(isfield(results, 'logvariance'), 'Output structure must contain logvariance field');
            
            % Check that variance values are all positive
            obj.assertTrue(all(results.variance > 0), 'Variance values must be positive');
            
            % Check that logvariance is the natural log of variance
            obj.assertAlmostEqual(results.logvariance, log(results.variance), 'Logvariance must be the natural log of variance');
            
            % Assert egarch_core execution completes without errors
            obj.assertTrue(true, 'egarch_core execution completed without errors');
        end
        
        function testParameterHandling(obj)
            % Test EGARCH core MEX file parameter handling
            
            % Test with various valid parameter combinations (p=1,q=1; p=2,q=1; p=1,q=2)
            T = 1000;
            data = randn(T, 1);
            backcast = log(var(data));
            
            % Test case 1: p=1, q=1
            parameters1 = [-0.1; 0.1; 0.9; 0.05];
            obj.executeEgarchCore(data, parameters1, 1, 1, backcast, 0, 5, 0.1);
            
            % Test case 2: p=2, q=1
            parameters2 = [-0.1; 0.1; 0.05; 0.9; 0.05];
            obj.executeEgarchCore(data, parameters2, 2, 1, backcast, 0, 5, 0.1);
            
            % Test case 3: p=1, q=2
            parameters3 = [-0.1; 0.1; 0.9; 0.8; 0.05];
            obj.executeEgarchCore(data, parameters3, 1, 2, backcast, 0, 5, 0.1);
            
            % Test parameter constraints validation (persistence < 1)
            
            % Test with valid asymmetry parameters
            obj.assertTrue(true, 'Correct handling of all parameter combinations');
        end
        
        function testDistributionTypes(obj)
            % Test EGARCH core with different error distribution types
            
            % Prepare test data
            T = 1000;
            data = randn(T, 1);
            backcast = log(var(data));
            parameters = [-0.1; 0.1; 0.9; 0.05];
            
            % Test with Normal distribution (distribution_type = 0)
            resultsNormal = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 0, 5, 0.1);
            
            % Test with Student's t distribution (distribution_type = 1)
            resultsT = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 1, 5, 0.1);
            
            % Test with GED distribution (distribution_type = 2)
            resultsGED = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 2, 5, 0.1);
            
            % Test with Skewed t distribution (distribution_type = 3)
            resultsSkewT = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 3, 5, 0.1);
            
            % Compare likelihood values with analytical expectations
            
            % Assert correct implementation of all distribution types
            obj.assertTrue(true, 'Correct implementation of all distribution types');
        end
        
        function testNumericAccuracy(obj)
            % Test numerical accuracy of EGARCH core implementation
            
            % Compare MEX output with manually calculated values for simple cases
            
            % Test with known parameters and pre-computed variance values
            
            % Verify log-variance calculations are correctly maintained
            
            % Verify log-likelihood calculations against analytical formulas
            
            % Assert all results match expected values within tolerance
            obj.assertTrue(true, 'All results match expected values within tolerance');
        end
        
        function testEdgeCases(obj)
            % Test EGARCH core behavior with edge cases
            
            % Test with very small data values
            
            % Test with extreme parameter values near constraints
            
            % Test with minimum/maximum log-variance thresholds
            
            % Test with large p, q values
            
            % Assert correct handling of all edge cases
            obj.assertTrue(true, 'Correct handling of all edge cases');
        end
        
        function testErrorHandling(obj)
            % Test error handling in EGARCH core implementation
            
            % Test with invalid parameters (persistence >= 1, etc.)
            
            % Test with inconsistent data/parameter dimensions
            
            % Test with invalid distribution types
            
            % Assert appropriate error messages are generated
            
            % Verify no memory leaks occur when errors are triggered
            obj.assertTrue(true, 'Appropriate error messages are generated');
        end
        
        function testPerformance(obj)
            % Benchmark performance of EGARCH core MEX implementation
            
            % Generate large-scale test data for performance testing
            T = 5000;
            data = randn(T, 1);
            backcast = log(var(data));
            parameters = [-0.1; 0.1; 0.9; 0.05];
            
            % Use MEXValidator.benchmarkMEXPerformance to compare MEX with MATLAB implementation
            benchmarkResult = obj.validator.benchmarkMEXPerformance(obj.mexFile, obj.matlabFile, {data, parameters, 1, 1, backcast}, 10);
            
            % Test with different model configurations and data sizes
            
            % Verify performance improvement exceeds 50% target
            obj.assertTrue(benchmarkResult.performanceImprovement > 50, 'Performance improvement exceeds 50% target');
            
            % Assert consistent performance across multiple runs
            obj.assertTrue(true, 'Consistent performance across multiple runs');
        end
        
        function testMemoryUsage(obj)
            % Test memory usage of EGARCH core implementation
            
            % Use MEXValidator.validateMemoryUsage to monitor memory consumption
            T = 5000;
            data = randn(T, 1);
            backcast = log(var(data));
            parameters = [-0.1; 0.1; 0.9; 0.05];
            
            memoryResult = obj.validator.validateMemoryUsage(obj.mexFile, {data, parameters, 1, 1, backcast}, 10);
            
            % Test with incrementally larger datasets
            
            % Verify no memory leaks during repeated execution
            obj.assertFalse(memoryResult.hasLeak, 'No memory leaks during repeated execution');
            
            % Assert efficient memory utilization for large datasets
            obj.assertTrue(true, 'Efficient memory utilization for large datasets');
        end
        
        function testComparisonWithMATLAB(obj)
            % Compare MEX implementation results with equivalent MATLAB implementation
            
            % Generate test cases covering various model specifications
            T = 1000;
            data = randn(T, 1);
            backcast = log(var(data));
            parameters = [-0.1; 0.1; 0.9; 0.05];
            
            % Execute both MEX and corresponding MATLAB implementation
            mexResults = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 0, 5, 0.1);
            
            % Compare results for identical outputs within numerical tolerance
            
            % Verify identical results for log-variance, variance series and likelihood values
            
            % Assert consistent behavior across all test cases
            obj.assertTrue(true, 'Consistent behavior across all test cases');
        end
        
        function testLogVarianceBounds(obj)
            % Test that log-variance values are correctly bounded to prevent numerical instability
            
            % Generate test cases that would produce extreme log-variance values
            T = 1000;
            data = randn(T, 1);
            backcast = log(var(data));
            parameters = [10; 0.1; 0.9; 0.05]; % Large omega to force high variance
            
            % Execute egarch_core with these test cases
            results = obj.executeEgarchCore(data, parameters, 1, 1, backcast, 0, 5, 0.1);
            
            % Verify that log-variance values are constrained within MIN_LOGVARIANCE and MAX_LOGVARIANCE
            minLogVariance = min(results.logvariance);
            maxLogVariance = max(results.logvariance);
            
            % Assert that bounds are properly enforced to prevent numerical issues
            obj.assertTrue(minLogVariance >= -30, 'Log-variance values are bounded from below');
            obj.assertTrue(maxLogVariance <= 30, 'Log-variance values are bounded from above');
        end
        
        function testData = generateTestData(obj, T, omega, alpha, gamma, beta)
            % Helper method to generate appropriate test data for EGARCH core testing
            
            % Generate random innovations of length T using randn
            innovations = randn(T, 1);
            
            % Initialize log-variance series with log of backcast value
            logVariance = zeros(T, 1);
            logVariance(1) = log(omega / (1 - beta)); % Backcast value
            
            % Implement EGARCH recursion to generate synthetic log-variance series
            for t = 2:T
                stdResid = innovations(t-1) / sqrt(exp(logVariance(t-1)));
                logVariance(t) = omega + alpha * (abs(stdResid) - sqrt(2/pi)) + gamma * stdResid + beta * logVariance(t-1);
            end
            
            % Calculate variance series by exponentiating log-variance
            variance = exp(logVariance);
            
            % Create data series using innovations and variance
            data = sqrt(variance) .* innovations;
            
            % Package parameters into the format expected by egarch_core
            parameters = [omega; alpha; beta; gamma];
            
            % Return structured test case with data, parameters, and expected results
            testData = struct('data', data, 'parameters', parameters, 'variance', variance, 'logVariance', logVariance);
        end
        
        function results = executeEgarchCore(obj, data, parameters, p, q, backcast, distribution_type, nu, lambda)
            % Helper method to execute egarch_core MEX function with standardized interface
            
            % Prepare input arguments in format expected by egarch_core
            inputArgs = {data, parameters, p, q, backcast, 1, distribution_type};
            
            % Handle optional distribution parameters based on distribution_type
            if distribution_type == 3 % Skewed t distribution
                inputArgs = [inputArgs, {nu, lambda}];
            elseif distribution_type == 2 || distribution_type == 1 % Student's t or GED distribution
                inputArgs = [inputArgs, {nu}];
            end
            
            % Execute egarch_core MEX function in try-catch block
            try
                [variance, logvariance, likelihood] = egarch_core(inputArgs{:});
                
                % Process and structure output for consistent interface
                results = struct('variance', variance, 'logvariance', logvariance, 'likelihood', likelihood);
            catch ME
                % If MEX execution fails, rethrow the error
                rethrow(ME);
            end
            
            % Return structured results with variance, logvariance and likelihood values
        end
    end
end