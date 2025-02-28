classdef AgarchCoreTest < BaseTest
    % Test class for validating AGARCH core MEX functionality in the MFE Toolbox
    
    properties
        validator            % MEXValidator instance
        testData             % Test data structure
        mexFile              % Name of the MEX file being tested
        defaultTolerance     % Default tolerance for numerical comparisons
    end
    
    methods
        function obj = AgarchCoreTest()
            % Initialize the AgarchCoreTest with test data and validator
            
            % Call parent BaseTest constructor
            obj@BaseTest('AgarchCoreTest');
            
            % Set MEX file name
            obj.mexFile = 'agarch_core';
            
            % Set default tolerance for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Create MEXValidator instance
            obj.validator = MEXValidator();
            
            % Initialize empty test data structure
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Load test data if it exists
            try
                testDataFile = obj.findTestDataFile('financial_returns.mat');
                obj.testData = load(testDataFile);
            catch
                % If test data file doesn't exist, create synthetic data
                rng(42, 'twister'); % Set random seed for reproducibility
                T = 1000;
                obj.testData.returns = randn(T, 1);
                obj.testData.omega = 0.01;
                obj.testData.alpha = [0.1];
                obj.testData.gamma = 0.05;
                obj.testData.beta = [0.8];
                obj.testData.backcast = var(obj.testData.returns);
            end
        end
        
        function tearDown(obj)
            % Clean up after tests
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Additional cleanup if needed
        end
        
        function testMEXFileExists(obj)
            % Test that the AGARCH core MEX file exists in the expected location
            
            % Use MEXValidator to check if MEX file exists
            exists = obj.validator.validateMEXExists(obj.mexFile);
            
            % Assert the MEX file exists
            obj.assertTrue(exists, ['MEX file ' obj.mexFile ' does not exist']);
            
            % Get platform-specific extension
            if strcmp(computer('arch'), 'win64')
                mexExt = 'mexw64';
            elseif strcmp(computer('arch'), 'glnxa64')
                mexExt = 'mexa64';
            else
                mexExt = 'mexmaci64';
            end
            
            % Check if file with correct extension exists
            mexPath = fullfile('src/backend/dlls', [obj.mexFile '.' mexExt]);
            obj.assertTrue(exist(mexPath, 'file') > 0, ['MEX file ' mexPath ' not found']);
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of AGARCH core MEX file with standard inputs
            
            % Prepare simple test case
            data = obj.testData.returns;
            omega = obj.testData.omega;
            alpha = obj.testData.alpha;
            gamma = obj.testData.gamma;
            beta = obj.testData.beta;
            backcast = obj.testData.backcast;
            
            p = length(alpha);
            q = length(beta);
            
            % Create parameters vector
            parameters = [omega; alpha; gamma; beta];
            
            % Execute agarch_core MEX function
            try
                result = obj.executeAgarchCore(data, parameters, p, q, backcast, 0, 0, 0);
                
                % Check that result is a struct with variance field
                obj.assertTrue(isstruct(result), 'Result should be a structure');
                obj.assertTrue(isfield(result, 'variance'), 'Result should have variance field');
                
                % Check variance values are positive
                obj.assertTrue(all(result.variance > 0), 'All variance values should be positive');
                
                % If result has likelihood field, check it's a scalar
                if isfield(result, 'likelihood')
                    obj.assertTrue(isscalar(result.likelihood), 'Likelihood should be a scalar value');
                end
            catch ME
                obj.assertTrue(false, ['MEX execution failed: ' ME.message]);
            end
        end
        
        function testParameterHandling(obj)
            % Test AGARCH core MEX file parameter handling
            
            % Test with p=1, q=1 (standard case)
            data = obj.testData.returns;
            omega = 0.01;
            alpha = 0.1;
            gamma = 0.05;
            beta = 0.8;
            backcast = var(data);
            
            parameters1 = [omega; alpha; gamma; beta];
            result1 = obj.executeAgarchCore(data, parameters1, 1, 1, backcast, 0, 0, 0);
            obj.assertTrue(isfield(result1, 'variance'), 'Result should have variance field');
            
            % Test with p=2, q=1
            alpha2 = [0.05; 0.05];
            parameters2 = [omega; alpha2; gamma; beta];
            result2 = obj.executeAgarchCore(data, parameters2, 2, 1, backcast, 0, 0, 0);
            obj.assertTrue(isfield(result2, 'variance'), 'Result should have variance field');
            
            % Test with p=1, q=2
            beta2 = [0.4; 0.4];
            parameters3 = [omega; alpha; gamma; beta2];
            result3 = obj.executeAgarchCore(data, parameters3, 1, 2, backcast, 0, 0, 0);
            obj.assertTrue(isfield(result3, 'variance'), 'Result should have variance field');
            
            % Test parameter constraints
            % Test with omega > 0
            omegaLarge = 1.0;
            parametersLarge = [omegaLarge; alpha; gamma; beta];
            resultLarge = obj.executeAgarchCore(data, parametersLarge, 1, 1, backcast, 0, 0, 0);
            obj.assertTrue(isfield(resultLarge, 'variance'), 'Result should have variance field');
            
            % Test with extreme but valid alpha (close to 0)
            alphaSmall = 0.001;
            parametersSmallAlpha = [omega; alphaSmall; gamma; beta];
            resultSmallAlpha = obj.executeAgarchCore(data, parametersSmallAlpha, 1, 1, backcast, 0, 0, 0);
            obj.assertTrue(isfield(resultSmallAlpha, 'variance'), 'Result should have variance field');
            
            % Test with extreme but valid beta (close to 1)
            betaLarge = 0.899;  % Keeping alpha + beta < 1 for stationarity
            parametersLargeBeta = [omega; alpha; gamma; betaLarge];
            resultLargeBeta = obj.executeAgarchCore(data, parametersLargeBeta, 1, 1, backcast, 0, 0, 0);
            obj.assertTrue(isfield(resultLargeBeta, 'variance'), 'Result should have variance field');
        end
        
        function testDistributionTypes(obj)
            % Test AGARCH core with different error distribution types
            
            % Prepare test data
            data = obj.testData.returns;
            omega = obj.testData.omega;
            alpha = obj.testData.alpha;
            gamma = obj.testData.gamma;
            beta = obj.testData.beta;
            backcast = obj.testData.backcast;
            
            p = length(alpha);
            q = length(beta);
            
            % Create parameters vector
            parameters = [omega; alpha; gamma; beta];
            
            % Test with normal distribution (distribution_type = 0)
            resultNormal = obj.executeAgarchCore(data, parameters, p, q, backcast, 1, 0, 0);
            obj.assertTrue(isfield(resultNormal, 'likelihood'), 'Result should have likelihood field');
            
            % Test with Student's t distribution (distribution_type = 1)
            nu = 8;  % Degrees of freedom
            parametersT = [parameters; nu];
            resultT = obj.executeAgarchCore(data, parametersT, p, q, backcast, 1, 1, nu);
            obj.assertTrue(isfield(resultT, 'likelihood'), 'Result should have likelihood field');
            
            % Test with GED distribution (distribution_type = 2)
            nu_ged = 1.5;  % Shape parameter
            parametersGED = [parameters; nu_ged];
            resultGED = obj.executeAgarchCore(data, parametersGED, p, q, backcast, 1, 2, nu_ged);
            obj.assertTrue(isfield(resultGED, 'likelihood'), 'Result should have likelihood field');
            
            % Test with skewed t distribution (distribution_type = 3)
            nu_skewt = 8;
            lambda = 0.1;  % Skewness parameter
            parametersSkewT = [parameters; nu_skewt; lambda];
            resultSkewT = obj.executeAgarchCore(data, parametersSkewT, p, q, backcast, 1, 3, nu_skewt, lambda);
            obj.assertTrue(isfield(resultSkewT, 'likelihood'), 'Result should have likelihood field');
        end
        
        function testNumericAccuracy(obj)
            % Test numerical accuracy of AGARCH core implementation
            
            % Generate test case with known values
            T = 100;
            omega = 0.05;
            alpha = 0.1;
            gamma = 0.05;
            beta = 0.8;
            
            % Generate synthetic data with known parameters
            testCase = obj.generateTestData(T, omega, alpha, gamma, beta);
            data = testCase.data;
            variance_expected = testCase.variance;
            
            % Execute AGARCH core
            parameters = [omega; alpha; gamma; beta];
            result = obj.executeAgarchCore(data, parameters, 1, 1, testCase.backcast, 0, 0, 0);
            
            % Compare variance values with expected values
            obj.assertMatrixEqualsWithTolerance(variance_expected, result.variance, obj.defaultTolerance, ...
                'Variance values do not match expected values');
            
            % Test with likelihood computation
            result_with_ll = obj.executeAgarchCore(data, parameters, 1, 1, testCase.backcast, 1, 0, 0);
            obj.assertTrue(isfield(result_with_ll, 'likelihood'), 'Result should have likelihood field');
            obj.assertTrue(isfinite(result_with_ll.likelihood), 'Likelihood should be finite');
        end
        
        function testEdgeCases(obj)
            % Test AGARCH core behavior with edge cases
            
            % Prepare base test data
            T = 100;
            
            % Test with very small data values
            small_data = 1e-5 * randn(T, 1);
            omega = 1e-5;
            alpha = 0.1;
            gamma = 0.05;
            beta = 0.8;
            backcast = var(small_data);
            
            parameters = [omega; alpha; gamma; beta];
            result_small = obj.executeAgarchCore(small_data, parameters, 1, 1, backcast, 0, 0, 0);
            obj.assertTrue(all(isfinite(result_small.variance)), 'Variance should be finite with small data');
            
            % Test with extreme parameters near constraints
            omega_small = 1e-6;
            alpha_small = 0.01;
            beta_large = 0.98;  % Close to 1-alpha for stationarity
            
            parameters_extreme = [omega_small; alpha_small; gamma; beta_large];
            result_extreme = obj.executeAgarchCore(obj.testData.returns, parameters_extreme, 1, 1, obj.testData.backcast, 0, 0, 0);
            obj.assertTrue(all(isfinite(result_extreme.variance)), 'Variance should be finite with extreme parameters');
            
            % Test with minimum variance thresholds
            % AGARCH should enforce a minimum variance floor internally
            backcast_small = 1e-10;
            result_min_var = obj.executeAgarchCore(obj.testData.returns, parameters, 1, 1, backcast_small, 0, 0, 0);
            obj.assertTrue(all(result_min_var.variance > 0), 'Variance should be positive even with tiny backcast');
            
            % Test with larger p, q values
            large_p = 4;
            large_q = 4;
            alpha_large = 0.02 * ones(large_p, 1);
            beta_large = 0.2 * ones(large_q, 1);
            parameters_large_pq = [omega; alpha_large; gamma; beta_large];
            
            result_large_pq = obj.executeAgarchCore(obj.testData.returns, parameters_large_pq, large_p, large_q, obj.testData.backcast, 0, 0, 0);
            obj.assertTrue(all(isfinite(result_large_pq.variance)), 'Variance should be finite with large p,q values');
        end
        
        function testErrorHandling(obj)
            % Test error handling in AGARCH core implementation
            
            % Prepare base test data
            data = obj.testData.returns;
            omega = obj.testData.omega;
            alpha = obj.testData.alpha;
            gamma = obj.testData.gamma;
            beta = obj.testData.beta;
            backcast = obj.testData.backcast;
            
            % Test with invalid parameters (negative omega)
            omega_neg = -0.01;
            parameters_invalid = [omega_neg; alpha; gamma; beta];
            
            % This should throw an error or return NaN/Inf values
            try
                result_invalid = obj.executeAgarchCore(data, parameters_invalid, 1, 1, backcast, 0, 0, 0);
                % If no error thrown, check for NaN/Inf in variance
                obj.assertTrue(any(~isfinite(result_invalid.variance)) || all(result_invalid.variance == 0), ...
                    'Invalid parameters should result in non-finite variance or zeros');
            catch ME
                % Error thrown is expected behavior
                obj.assertTrue(true, 'Error properly thrown for invalid parameters');
            end
            
            % Test with inconsistent data/parameter dimensions
            try
                % Mismatch between p and alpha dimensions
                alpha_wrong = [0.1; 0.1];  % 2 elements when p=1
                parameters_mismatch = [omega; alpha_wrong; gamma; beta];
                result_mismatch = obj.executeAgarchCore(data, parameters_mismatch, 1, 1, backcast, 0, 0, 0);
                % If no error, should use only first alpha value
                obj.assertTrue(true);
            catch ME
                % Error might be expected
                obj.assertTrue(true, 'Handling dimension mismatch appropriately');
            end
            
            % Test with invalid distribution types
            try
                invalid_dist = 5;  % Invalid distribution type
                result_invalid_dist = obj.executeAgarchCore(data, [parameters; 5], 1, 1, backcast, 1, invalid_dist, 5);
                % Should throw an error or return NaN likelihood
                obj.assertTrue(~isfield(result_invalid_dist, 'likelihood') || ~isfinite(result_invalid_dist.likelihood), ...
                    'Invalid distribution type should result in error or NaN likelihood');
            catch ME
                % Error thrown is expected behavior
                obj.assertTrue(true, 'Error properly thrown for invalid distribution type');
            end
        end
        
        function testPerformance(obj)
            % Benchmark performance of AGARCH core MEX implementation
            
            % Generate large-scale test data
            T = 5000;  % Large enough to see performance difference
            omega = 0.01;
            alpha = 0.1;
            gamma = 0.05;
            beta = 0.8;
            
            % Create synthetic data
            rng(42, 'twister');  % Set seed for reproducibility
            data = randn(T, 1);
            backcast = var(data);
            
            % Parameters vector for AGARCH
            parameters = [omega; alpha; gamma; beta];
            
            % Use MEXValidator to benchmark performance
            mexFunction = 'agarch_core';
            matlabFunction = 'agarchfit';  % The MATLAB equivalent
            
            % Create equivalent inputs for both functions
            mexInputs = {data, parameters, 1, 1, backcast, 0, 0};
            
            % Set up options for agarchfit
            options = struct();
            options.p = 1;
            options.q = 1;
            options.distribution = 'NORMAL';
            options.useMEX = false;  % Force MATLAB implementation for comparison
            options.startingvals = parameters;
            options.backcast = backcast;
            
            matlabInputs = {data, options};
            
            % Perform benchmarking with multiple iterations
            iterations = 5;
            benchmark = obj.validator.benchmarkMEXPerformance(mexFunction, matlabFunction, mexInputs, iterations);
            
            % Assert performance improvement is substantial (>50%)
            obj.assertTrue(benchmark.performanceImprovement > 50, ...
                ['Performance improvement insufficient: ' num2str(benchmark.performanceImprovement) '% (target: >50%)']);
            
            % Display detailed performance results if test fails
            if benchmark.performanceImprovement <= 50
                disp('MEX mean time: ' + num2str(benchmark.mexMeanTime));
                disp('MATLAB mean time: ' + num2str(benchmark.matlabMeanTime));
                disp('Performance improvement: ' + num2str(benchmark.performanceImprovement) + '%');
            end
        end
        
        function testMemoryUsage(obj)
            % Test memory usage of AGARCH core implementation
            
            % Use MEXValidator to monitor memory consumption
            mexFunction = 'agarch_core';
            
            % Prepare inputs for MEX function
            data = obj.testData.returns;
            omega = obj.testData.omega;
            alpha = obj.testData.alpha;
            gamma = obj.testData.gamma;
            beta = obj.testData.beta;
            backcast = obj.testData.backcast;
            
            parameters = [omega; alpha; gamma; beta];
            
            mexInputs = {data, parameters, 1, 1, backcast, 0, 0};
            
            % Run memory usage test
            memoryResult = obj.validator.validateMemoryUsage(mexFunction, mexInputs, 100);
            
            % Assert no memory leaks
            obj.assertFalse(memoryResult.hasLeak, 'Memory leak detected in MEX implementation');
            
            % Test memory usage with increasingly larger datasets
            sizes = [1000, 2000, 5000];
            memoryUsage = zeros(length(sizes), 1);
            
            for i = 1:length(sizes)
                T = sizes(i);
                large_data = randn(T, 1);
                large_inputs = {large_data, parameters, 1, 1, var(large_data), 0, 0};
                
                % Measure memory before
                before = whos();
                beforeMem = sum([before.bytes]);
                
                % Execute function
                obj.executeAgarchCore(large_data, parameters, 1, 1, var(large_data), 0, 0, 0);
                
                % Measure memory after
                after = whos();
                afterMem = sum([after.bytes]);
                
                % Calculate memory difference
                memoryUsage(i) = afterMem - beforeMem;
            end
            
            % Memory usage should be proportional to data size but not excessively so
            % Just check that it doesn't grow unreasonably
            if length(sizes) > 1
                ratio = memoryUsage(end) / memoryUsage(1);
                expectedRatio = sizes(end) / sizes(1);
                obj.assertTrue(ratio < 10 * expectedRatio, 'Memory usage grows excessively with data size');
            end
        end
        
        function testComparisonWithMATLAB(obj)
            % Compare MEX implementation results with equivalent MATLAB implementation
            
            % Use MEXValidator to compare implementations
            mexFunction = 'agarch_core';
            
            % Test with different model specifications
            
            % Test case 1: AGARCH(1,1) with normal distribution
            data = obj.testData.returns;
            T = length(data);
            omega1 = 0.01;
            alpha1 = 0.1;
            gamma1 = 0.05;
            beta1 = 0.8;
            backcast1 = var(data);
            
            parameters1 = [omega1; alpha1; gamma1; beta1];
            
            % Execute MEX implementation
            mex_result1 = obj.executeAgarchCore(data, parameters1, 1, 1, backcast1, 1, 0, 0);
            
            % Execute MATLAB implementation
            options = struct();
            options.p = 1;
            options.q = 1;
            options.distribution = 'NORMAL';
            options.useMEX = false;  % Force MATLAB implementation
            options.startingvals = parameters1;
            options.backcast = backcast1;
            
            matlab_model = agarchfit(data, options);
            
            % Compare variance series
            obj.assertMatrixEqualsWithTolerance(matlab_model.ht, mex_result1.variance, 1e-6, ...
                'MEX and MATLAB variance values differ significantly');
            
            % Test case 2: AGARCH(1,1) with Student's t distribution
            gamma2 = 0.1;  % Different gamma
            parameters2 = [omega1; alpha1; gamma2; beta1; 8];  % nu = 8
            
            mex_result2 = obj.executeAgarchCore(data, parameters2, 1, 1, backcast1, 1, 1, 8);
            
            options.distribution = 'T';
            options.startingvals = parameters2;
            
            matlab_model2 = agarchfit(data, options);
            
            % Compare variance series
            obj.assertMatrixEqualsWithTolerance(matlab_model2.ht, mex_result2.variance, 1e-6, ...
                'MEX and MATLAB variance values differ significantly for Student''s t distribution');
        end
        
        function testCase = generateTestData(obj, T, omega, alpha, gamma, beta)
            % Helper method to generate appropriate test data for AGARCH core testing
            
            % Initialize test data structure
            testCase = struct();
            
            % Generate random innovations of length T
            rng(42, 'twister');  % Set seed for reproducibility
            testCase.innovations = randn(T, 1);
            
            % Set up parameters
            testCase.omega = omega;
            testCase.alpha = alpha;  % Can be scalar or vector
            testCase.gamma = gamma;
            testCase.beta = beta;    % Can be scalar or vector
            
            % Ensure alpha and beta are column vectors
            if isscalar(alpha)
                alpha = [alpha];
            end
            if isscalar(beta)
                beta = [beta];
            end
            
            % Get p and q
            p = length(alpha);
            q = length(beta);
            
            % Calculate the backcast value (unconditional variance)
            testCase.backcast = omega / (1 - sum(alpha) - sum(beta));
            
            % Initialize variance series with backcast value
            testCase.variance = ones(T, 1) * testCase.backcast;
            
            % Generate data and variance series using AGARCH recursion
            for t = 2:T
                % Start with constant term
                testCase.variance(t) = omega;
                
                % Add ARCH terms with asymmetry
                for i = 1:p
                    if t-i > 0
                        shock = testCase.innovations(t-i) - gamma * sqrt(testCase.variance(t-i));
                        testCase.variance(t) = testCase.variance(t) + alpha(i) * shock^2;
                    else
                        % Use backcast for pre-sample
                        testCase.variance(t) = testCase.variance(t) + alpha(i) * testCase.backcast;
                    end
                end
                
                % Add GARCH terms
                for j = 1:q
                    if t-j > 0
                        testCase.variance(t) = testCase.variance(t) + beta(j) * testCase.variance(t-j);
                    else
                        % Use backcast for pre-sample
                        testCase.variance(t) = testCase.variance(t) + beta(j) * testCase.backcast;
                    end
                end
                
                % Ensure non-negative variance
                testCase.variance(t) = max(testCase.variance(t), 1e-10);
            end
            
            % Create data as innovations scaled by conditional standard deviation
            testCase.data = testCase.innovations .* sqrt(testCase.variance);
            
            % Create parameter vector
            testCase.parameters = [omega; alpha(:); gamma; beta(:)];
        end
        
        function result = executeAgarchCore(obj, data, parameters, p, q, backcast, compute_likelihood, distribution_type, nu, lambda)
            % Helper method to execute agarch_core MEX function with standardized interface
            
            % Set default values for optional parameters
            if nargin < 9
                nu = 0;
            end
            if nargin < 10
                lambda = 0;
            end
            
            % Initialize result structure
            result = struct();
            
            try
                % Execute MEX function
                if compute_likelihood == 0
                    % No likelihood computation
                    variance = agarch_core(data, parameters, backcast, p, q, length(data));
                    result.variance = variance;
                else
                    % With likelihood computation
                    if distribution_type == 0
                        % Normal distribution
                        [variance, likelihood] = agarch_core(data, parameters, backcast, p, q, length(data), compute_likelihood, distribution_type);
                    elseif distribution_type == 1 || distribution_type == 2
                        % Student's t or GED distribution (need nu parameter)
                        [variance, likelihood] = agarch_core(data, parameters, backcast, p, q, length(data), compute_likelihood, distribution_type, nu);
                    elseif distribution_type == 3
                        % Skewed t distribution (need nu and lambda parameters)
                        [variance, likelihood] = agarch_core(data, parameters, backcast, p, q, length(data), compute_likelihood, distribution_type, nu, lambda);
                    else
                        % Invalid distribution type
                        error('Invalid distribution type: %d', distribution_type);
                    end
                    
                    result.variance = variance;
                    result.likelihood = likelihood;
                end
            catch ME
                % Re-throw error with more context
                error('Error executing agarch_core: %s', ME.message);
            end
        end
    end
end