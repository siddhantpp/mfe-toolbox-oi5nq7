classdef TarchCoreTest < BaseTest
    % Test class for validating TARCH core MEX functionality in the MFE Toolbox
    
    properties
        MEXValidator validator;
        struct testData;
        string mexFile;
        double defaultTolerance;
    end
    
    methods
        function obj = TarchCoreTest()
            % Initialize the TarchCoreTest with test data and validator
            obj@BaseTest('TarchCoreTest');
            
            % Set mexFile to 'tarch_core'
            obj.mexFile = 'tarch_core';
            
            % Set defaultTolerance to 1e-10 for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Create MEXValidator instance for validation functions
            obj.validator = MEXValidator();
            
            % Initialize empty testData structure
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Load financial returns test data from 'financial_returns.mat'
            try
                loadedData = obj.loadTestData('financial_returns.mat');
                obj.testData.returns = loadedData.returns;
            catch
                % If file not found, generate synthetic data
                obj.testData.returns = randn(1000, 1);
            end
            
            % Prepare various test case parameters for TARCH models
            obj.testData.params = struct();
            obj.testData.params.standard = [0.01; 0.05; 0.1; 0.8]; % omega, alpha, gamma, beta
            obj.testData.params.lowPersistence = [0.01; 0.02; 0.05; 0.6];
            obj.testData.params.highPersistence = [0.005; 0.1; 0.2; 0.85];
            obj.testData.params.noAsymmetry = [0.01; 0.1; 0.0; 0.8];
            
            % Set up standardized test inputs for consistent testing
            obj.testData.T = 1000;
            obj.testData.data = randn(obj.testData.T, 1);
            obj.testData.backcast = var(obj.testData.data);
        end
        
        function tearDown(obj)
            % Clean up after tests
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary variables created during tests
        end
        
        function testMEXFileExists(obj)
            % Test that the TARCH core MEX file exists in the expected location
            
            % Use MEXValidator.validateMEXExists to check for tarch_core MEX file
            mexExists = obj.validator.validateMEXExists(obj.mexFile);
            
            % Assert that the MEX file exists
            obj.assertTrue(mexExists, ['TARCH core MEX file "', obj.mexFile, '" not found']);
            
            % Verify correct platform-specific extension (.mexw64 or .mexa64)
            platform = computer();
            if strncmpi(platform, 'PCWIN', 5)
                expectedExt = 'mexw64';
            elseif strncmpi(platform, 'GLN', 3)
                expectedExt = 'mexa64';
            else
                expectedExt = obj.validator.getMEXExtension();
            end
            
            mexFilePath = fullfile(obj.validator.mexBasePath, [obj.mexFile, '.', expectedExt]);
            obj.assertTrue(exist(mexFilePath, 'file') == 3, ...
                ['TARCH core MEX file not found with expected extension: ', expectedExt]);
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of TARCH core MEX file with standard inputs
            
            % Prepare simple test case with synthetic data
            data = obj.testData.data;
            params = obj.testData.params.standard;
            backcast = obj.testData.backcast;
            
            % Call tarch_core MEX function with test inputs
            try
                result = tarch_core(data, params, backcast);
                
                % Verify output structure contains variance and likelihood fields
                obj.assertTrue(isstruct(result), 'TARCH core should return a structure');
                obj.assertTrue(isfield(result, 'ht'), 'Result should contain variance field "ht"');
                
                % Check that variance values are all positive
                ht = result.ht;
                obj.assertTrue(all(ht > 0), 'All variance values should be positive');
                obj.assertEqual(length(ht), length(data), 'Variance vector should match data length');
                
                % Assert tarch_core execution completes without errors
                obj.assertTrue(all(isfinite(ht)), 'All variance values should be finite');
            catch ME
                obj.assertTrue(false, ['TARCH core execution failed: ', ME.message]);
            end
        end
        
        function testParameterHandling(obj)
            % Test TARCH core MEX file parameter handling
            
            % Test with various valid parameter combinations
            data = obj.testData.data;
            backcast = obj.testData.backcast;
            
            % Test with standard parameters
            params1 = obj.testData.params.standard;
            try
                result1 = tarch_core(data, params1, backcast);
                obj.assertTrue(isfield(result1, 'ht'), 'Standard parameters should be accepted');
            catch ME
                obj.assertTrue(false, ['Standard parameters failed: ', ME.message]);
            end
            
            % Test parameter constraints validation (omega > 0, alpha/gamma/beta >= 0)
            % Test with boundary parameters (close to stationarity constraint)
            params2 = [0.001; 0.1; 0.2; 0.79]; % sum(alpha + gamma/2 + beta) = 0.1 + 0.1 + 0.79 = 0.99
            try
                result2 = tarch_core(data, params2, backcast);
                obj.assertTrue(isfield(result2, 'ht'), 'Boundary parameters should be accepted');
            catch ME
                obj.assertTrue(false, ['Boundary parameters failed: ', ME.message]);
            end
            
            % Test stability constraint (alpha + gamma/2 + beta < 1)
            params3 = [0.01; 0.2; 0.6; 0.8]; % sum(alpha + gamma/2 + beta) = 0.2 + 0.3 + 0.8 = 1.3 > 1
            try
                result3 = tarch_core(data, params3, backcast);
                % Implementation might handle this invalid case by enforcing constraints
                % or by throwing an error - adjust test accordingly
                if isfield(result3, 'error')
                    obj.assertTrue(true, 'Implementation correctly identified constraint violation');
                elseif isfield(result3, 'ht')
                    % If no error field, we should verify that results are still valid
                    obj.assertTrue(all(isfinite(result3.ht)), 'Variances should remain finite even with invalid parameters');
                end
            catch ME
                % If implementation throws error for invalid parameters, that's also valid
                obj.assertTrue(contains(lower(ME.message), 'stab') || ...
                               contains(lower(ME.message), 'constr'), ...
                               'Implementation should throw error for stationarity constraint violation');
            end
            
            % Test with extreme but valid parameter values
            params4 = [0.0001; 0.01; 0.01; 0.97]; % Very small omega, high persistence
            try
                result4 = tarch_core(data, params4, backcast);
                obj.assertTrue(isfield(result4, 'ht'), 'Extreme parameters should be accepted if valid');
            catch ME
                obj.assertTrue(false, ['Extreme parameters failed: ', ME.message]);
            end
            
            % Assert correct handling of all parameter combinations
            obj.assertTrue(true, 'All parameter handling tests passed');
        end
        
        function testAsymmetricEffect(obj)
            % Test TARCH core handling of asymmetric effects (leverage effect)
            
            % Generate test data with known positive and negative shocks
            T = 500;
            data = zeros(T, 1);
            
            % Create alternating positive and negative returns of equal magnitude
            for i = 1:T
                if mod(i, 2) == 0
                    data(i) = 1.0; % Positive shock
                else
                    data(i) = -1.0; % Negative shock
                end
            end
            
            % Parameters with different gamma values to test asymmetric effect
            omega = 0.01;
            alpha = 0.05;
            beta = 0.8;
            backcast = 1.0;
            
            % Test gamma parameter effects on variance response
            
            % Case 1: No asymmetry (gamma = 0)
            params1 = [omega; alpha; 0; beta];
            result1 = tarch_core(data, params1, backcast);
            ht1 = result1.ht;
            
            % Case 2: With asymmetry (gamma > 0)
            params2 = [omega; alpha; 0.2; beta];
            result2 = tarch_core(data, params2, backcast);
            ht2 = result2.ht;
            
            % Verify asymmetric variance response to negative vs positive returns
            neg_indices = find(data(1:end-1) < 0); % Indices of negative shocks
            pos_indices = find(data(1:end-1) > 0); % Indices of positive shocks
            
            % Get variances following negative and positive shocks
            neg_variances1 = ht1(neg_indices + 1);
            pos_variances1 = ht1(pos_indices + 1);
            neg_variances2 = ht2(neg_indices + 1);
            pos_variances2 = ht2(pos_indices + 1);
            
            % Compare response pattern with analytical expectations
            
            % With gamma = 0, variances after positive/negative shocks should be equal
            avg_neg1 = mean(neg_variances1);
            avg_pos1 = mean(pos_variances1);
            obj.assertAlmostEqual(avg_neg1, avg_pos1, 'Without asymmetry, volatility response should be symmetric');
            
            % With gamma > 0, variances after negative shocks should be higher
            avg_neg2 = mean(neg_variances2);
            avg_pos2 = mean(pos_variances2);
            obj.assertTrue(avg_neg2 > avg_pos2, 'With positive gamma, volatility should be higher after negative shocks');
            
            % Assert correct implementation of asymmetric effect
            ratio = avg_neg2 / avg_pos2;
            expected_ratio = (alpha + gamma) / alpha;
            obj.assertAlmostEqual(ratio, expected_ratio, 0.1, 'Asymmetric effect ratio should match theoretical value');
        end
        
        function testNumericAccuracy(obj)
            % Test numerical accuracy of TARCH core implementation
            
            % Compare MEX output with manually calculated values for simple cases
            
            % Create a simple test case with known solution
            T = 10;
            data = [1; -1; 2; -2; 1.5; -1.5; 0.5; -0.5; 1; -1];
            
            % Simple parameters for clear calculation
            omega = 0.1;
            alpha = 0.1;
            gamma = 0.2;
            beta = 0.6;
            params = [omega; alpha; gamma; beta];
            backcast = 1.0;
            
            % Test with known parameters and pre-computed variance values
            
            % Calculate expected variances manually
            expected_ht = zeros(T, 1);
            expected_ht(1) = backcast;
            
            for t = 2:T
                if data(t-1) < 0 % Negative previous return
                    asymmetric_term = gamma * data(t-1)^2;
                else
                    asymmetric_term = 0;
                end
                expected_ht(t) = omega + alpha * data(t-1)^2 + asymmetric_term + beta * expected_ht(t-1);
            end
            
            % Get variances from MEX implementation
            result = tarch_core(data, params, backcast);
            actual_ht = result.ht;
            
            % Verify log-likelihood calculations against analytical formulas
            
            % Compare manually calculated vs. MEX computed variances
            obj.assertMatrixEqualsWithTolerance(expected_ht, actual_ht, obj.defaultTolerance, ...
                'TARCH core variance calculations should match manual calculations');
            
            % Calculate and compare log-likelihood if available
            if isfield(result, 'loglik')
                % Calculate expected log-likelihood
                expected_loglik = 0;
                for t = 1:T
                    % Normal distribution log-likelihood
                    expected_loglik = expected_loglik - 0.5 * log(2 * pi) - 0.5 * log(expected_ht(t)) - 0.5 * data(t)^2 / expected_ht(t);
                end
                
                % Compare likelihoods with appropriate tolerance
                obj.assertAlmostEqual(expected_loglik, result.loglik, 'Log-likelihood calculations should match');
            end
            
            % Assert all results match expected values within tolerance
            obj.assertTrue(true, 'Numerical accuracy tests passed');
        end
        
        function testEdgeCases(obj)
            % Test TARCH core behavior with edge cases
            
            % Get standard parameters
            params = obj.testData.params.standard;
            backcast = obj.testData.backcast;
            
            % Test with very small data values
            small_data = 1e-8 * randn(100, 1);
            try
                small_result = tarch_core(small_data, params, backcast);
                obj.assertTrue(all(isfinite(small_result.ht)), 'Variances should be finite with very small data');
                obj.assertTrue(all(small_result.ht > 0), 'Variances should remain positive with very small data');
            catch ME
                obj.assertTrue(false, ['Failed with small data values: ', ME.message]);
            end
            
            % Test with extreme parameter values near constraints
            extreme_params = [0.001; 0.05; 0.05; 0.899]; % Close to constraint: 0.05 + 0.025 + 0.899 = 0.974
            try
                extreme_result = tarch_core(obj.testData.data, extreme_params, backcast);
                obj.assertTrue(all(isfinite(extreme_result.ht)), 'Variances should be finite with extreme parameters');
            catch ME
                obj.assertTrue(false, ['Failed with extreme parameters: ', ME.message]);
            end
            
            % Test with minimum variance thresholds
            zero_shock_data = zeros(100, 1);
            zero_result = tarch_core(zero_shock_data, params, backcast);
            min_variance = min(zero_result.ht);
            
            % The minimum variance should be positive
            obj.assertTrue(min_variance > 0, 'Minimum variance threshold should be positive');
            
            % Test with long time series data
            long_data = randn(10000, 1);
            try
                long_result = tarch_core(long_data, params, backcast);
                obj.assertTrue(length(long_result.ht) == length(long_data), 'Should handle long time series correctly');
            catch ME
                obj.assertTrue(false, ['Failed with long time series: ', ME.message]);
            end
            
            % Assert correct handling of all edge cases
            obj.assertTrue(true, 'All edge case tests passed');
        end
        
        function testErrorHandling(obj)
            % Test error handling in TARCH core implementation
            
            % Standard valid inputs
            data = obj.testData.data;
            params = obj.testData.params.standard;
            backcast = obj.testData.backcast;
            
            % Test with invalid parameters (negative omega, etc.)
            invalid_params1 = [-0.01; 0.05; 0.1; 0.8]; % Negative omega
            try
                result1 = tarch_core(data, invalid_params1, backcast);
                % If implementation doesn't throw error but returns a structure with error field
                if isfield(result1, 'error')
                    obj.assertTrue(true, 'Implementation detected negative omega parameter');
                else
                    % If implementation adjusts invalid parameters silently, verify results are still valid
                    obj.assertTrue(all(isfinite(result1.ht)), 'Results should remain valid with adjusted parameters');
                end
            catch ME
                % If implementation throws error for invalid parameters, that's also valid
                obj.assertTrue(contains(lower(ME.message), 'omega') || ...
                              contains(lower(ME.message), 'positive') || ...
                              contains(lower(ME.message), 'parameter'), ...
                              'Implementation should throw error for negative omega');
            end
            
            % Test with inconsistent data/parameter dimensions
            invalid_params2 = [0.01; 0.05; 0.1]; % Missing beta parameter
            try
                result2 = tarch_core(data, invalid_params2, backcast);
                obj.assertTrue(false, 'Implementation should detect incorrect parameter dimensions');
            catch ME
                obj.assertTrue(contains(lower(ME.message), 'parameter') || ...
                              contains(lower(ME.message), 'dimension') || ...
                              contains(lower(ME.message), 'size'), ...
                              'Implementation should throw error for incorrect parameter dimensions');
            end
            
            % Test with invalid parameter constraints
            invalid_params3 = [0.01; 0.3; 0.8; 0.6]; % Sum = 0.3 + 0.4 + 0.6 = 1.3 > 1
            try
                result3 = tarch_core(data, invalid_params3, backcast);
                if isfield(result3, 'error')
                    obj.assertTrue(true, 'Implementation detected stationarity constraint violation');
                elseif isfield(result3, 'ht')
                    % If implementation adjusts, results should still be valid
                    obj.assertTrue(all(isfinite(result3.ht)), 'Results should remain valid with adjusted parameters');
                end
            catch ME
                obj.assertTrue(contains(lower(ME.message), 'stab') || ...
                              contains(lower(ME.message), 'constr') || ...
                              contains(lower(ME.message), 'stationar'), ...
                              'Implementation should throw error for stationarity constraint violation');
            end
            
            % Assert appropriate error messages are generated
            
            % Verify no memory leaks occur when errors are triggered
            memory_result = obj.validator.validateMemoryUsage(obj.mexFile, {data, params, backcast}, 10);
            obj.assertFalse(memory_result.hasLeak, 'No memory leaks should occur during error handling');
        end
        
        function testPerformance(obj)
            % Benchmark performance of TARCH core MEX implementation
            
            % Generate large-scale test data for performance testing
            T = 10000;
            data = randn(T, 1);
            params = obj.testData.params.standard;
            backcast = var(data);
            
            % Use MEXValidator.benchmarkMEXPerformance to compare MEX with MATLAB implementation
            options = struct('useMEX', false, 'model', 'TARCH', 'p', 1, 'q', 1);
            
            % Test with different model configurations and data sizes
            mex_time = zeros(1, 5);
            matlab_time = zeros(1, 5);
            
            for i = 1:5
                % Benchmark single run
                tic;
                mex_result = tarch_core(data, params, backcast);
                mex_time(i) = toc;
                
                % Benchmark MATLAB implementation
                tic;
                matlab_result = garchcore(params, data, options);
                matlab_time(i) = toc;
            end
            
            % Calculate average times
            avg_mex_time = mean(mex_time);
            avg_matlab_time = mean(matlab_time);
            
            % Calculate performance improvement
            performance_gain = (avg_matlab_time - avg_mex_time) / avg_matlab_time * 100;
            
            % Verify performance improvement exceeds 50% target
            obj.assertTrue(performance_gain >= 50, ...
                ['MEX implementation should be at least 50% faster than MATLAB implementation. ', ...
                'Actual improvement: ', num2str(performance_gain), '%']);
            
            % Assert consistent performance across multiple runs
            mex_std = std(mex_time) / avg_mex_time;
            obj.assertTrue(mex_std < 0.3, 'MEX performance should be consistent across runs');
        end
        
        function testMemoryUsage(obj)
            % Test memory usage of TARCH core implementation
            
            % Use MEXValidator.validateMemoryUsage to monitor memory consumption
            params = obj.testData.params.standard;
            backcast = obj.testData.backcast;
            
            % Test with incrementally larger datasets
            test_sizes = [1000, 5000, 10000];
            
            for i = 1:length(test_sizes)
                T = test_sizes(i);
                data = randn(T, 1);
                
                % Monitor memory usage
                memory_result = obj.validator.validateMemoryUsage(obj.mexFile, {data, params, backcast}, 20);
                
                % Verify no memory leaks during repeated execution
                obj.assertFalse(memory_result.hasLeak, ...
                    ['Memory leak detected for data size T=', num2str(T)]);
                
                % Assert efficient memory utilization for large datasets
                expected_max_memory = 10 * T * 8; % 10x the data size (8 bytes per double)
                obj.assertTrue(memory_result.memoryDelta < expected_max_memory, ...
                    ['Memory usage too high for data size T=', num2str(T)]);
            end
        end
        
        function testComparisonWithMATLAB(obj)
            % Compare MEX implementation results with equivalent MATLAB implementation
            
            % Generate test cases covering various model specifications
            test_params = {
                obj.testData.params.standard,
                obj.testData.params.lowPersistence,
                obj.testData.params.highPersistence,
                obj.testData.params.noAsymmetry
            };
            
            data = obj.testData.data;
            backcast = obj.testData.backcast;
            
            for i = 1:length(test_params)
                params = test_params{i};
                
                % Execute both MEX and corresponding MATLAB implementation
                mex_result = tarch_core(data, params, backcast);
                
                % Use MATLAB implementation
                options = struct('model', 'TARCH', 'p', 1, 'q', 1, 'useMEX', false);
                matlab_result = garchcore(params, data, options);
                
                % Compare results for identical outputs within numerical tolerance
                comp_result = obj.numericalComparator.compareMatrices(matlab_result, mex_result.ht, obj.defaultTolerance);
                
                % Verify identical results for variance series and likelihood values
                obj.assertTrue(comp_result.isEqual, ...
                    ['MEX and MATLAB implementations produce different results in test case ', num2str(i), ...
                    '. Max difference: ', num2str(comp_result.maxAbsoluteDifference)]);
            end
            
            % Assert consistent behavior across all test cases
            obj.assertTrue(true, 'MEX implementation matches MATLAB implementation across all test cases');
        end
        
        function testData = generateTestData(obj, T, omega, alpha, gamma, beta)
            % Helper method to generate appropriate test data for TARCH core testing
            %
            % INPUTS:
            %   T - Length of time series
            %   omega, alpha, gamma, beta - TARCH model parameters
            %
            % OUTPUTS:
            %   testData - Test data structure with time series and parameters
            
            % Initialize parameters if not provided
            if nargin < 2 || isempty(T)
                T = 1000;
            end
            if nargin < 3 || isempty(omega)
                omega = 0.01;
            end
            if nargin < 4 || isempty(alpha)
                alpha = 0.05;
            end
            if nargin < 5 || isempty(gamma)
                gamma = 0.1;
            end
            if nargin < 6 || isempty(beta)
                beta = 0.8;
            end
            
            % Generate random innovations of length T using randn
            innovations = randn(T, 1);
            
            % Initialize variance series with backcast value
            ht = zeros(T, 1);
            data = zeros(T, 1);
            
            % Apply backcast for initial variance
            backcast = 1.0;
            ht(1) = backcast;
            
            % Implement TARCH recursion to generate synthetic variance series
            for t = 2:T
                % Scale previous innovation by volatility
                data(t-1) = sqrt(ht(t-1)) * innovations(t-1);
                
                % Apply asymmetric effect based on gamma parameter
                asymm_term = (data(t-1) < 0) * gamma * data(t-1)^2;
                
                % Update variance
                ht(t) = omega + alpha * data(t-1)^2 + asymm_term + beta * ht(t-1);
            end
            
            % Generate last data point
            data(T) = sqrt(ht(T)) * innovations(T);
            
            % Package parameters into the format expected by tarch_core
            params = [omega; alpha; gamma; beta];
            
            % Return structured test case with data, parameters, and expected results
            testData = struct();
            testData.data = data;
            testData.innovations = innovations;
            testData.ht = ht;
            testData.params = params;
            testData.backcast = backcast;
            testData.T = T;
        end
        
        function result = executeTarchCore(obj, data, parameters, backcast)
            % Helper method to execute tarch_core MEX function with standardized interface
            %
            % INPUTS:
            %   data - Time series data
            %   parameters - TARCH model parameters
            %   backcast - Initial variance value
            %
            % OUTPUTS:
            %   result - Results structure with variance and likelihood values
            
            % Prepare input arguments in format expected by tarch_core
            try
                % Execute tarch_core MEX function in try-catch block
                result = tarch_core(data, parameters, backcast);
            catch ME
                % Process and structure output for consistent interface
                error('TarchCoreTest:ExecutionError', 'Failed to execute tarch_core: %s', ME.message);
            end
            
            % Return structured results with variance and likelihood values
        end
    end
end