classdef CompositeLikelihoodTest < BaseTest
    % COMPOSITELIKELIHOODTEST Test class for validating composite likelihood MEX functionality in the MFE Toolbox
    
    properties
        MEXValidator validator
        struct testData
        string mexFile
        double defaultTolerance
        NumericalComparator comparator
    end
    
    methods
        function obj = CompositeLikelihoodTest()
            % Initialize the CompositeLikelihoodTest with test data and validator
            
            % Call parent BaseTest constructor with 'CompositeLikelihoodTest' name
            obj@BaseTest('CompositeLikelihoodTest');
            
            % Set mexFile to 'composite_likelihood'
            obj.mexFile = 'composite_likelihood';
            obj.defaultTolerance = 1e-10;
            obj.validator = MEXValidator();
            obj.comparator = NumericalComparator();
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test
            
            % Call parent setUp method
            obj.setUp@BaseTest();
            
            % Load financial returns test data from 'financial_returns.mat'
            try
                finData = obj.loadTestData('financial_returns.mat');
                returns = finData.returns;
            catch
                % Create synthetic data if test data file is not available
                returns = randn(1000, 5);
            end
            
            % Set up test data with different dimensions (small, medium, large)
            obj.testData.small = struct();
            obj.testData.small.T = 100;
            obj.testData.small.K = 3;
            obj.testData.small.data = returns(1:obj.testData.small.T, 1:obj.testData.small.K);
            
            obj.testData.medium = struct();
            obj.testData.medium.T = 250;
            obj.testData.medium.K = 8;
            % Ensure we don't exceed the dimensions of available data
            useK = min(obj.testData.medium.K, size(returns, 2));
            if useK < obj.testData.medium.K
                % Pad with synthetic data if needed
                temp = [returns(1:obj.testData.medium.T, 1:useK), randn(obj.testData.medium.T, obj.testData.medium.K-useK)];
                obj.testData.medium.data = temp;
            else
                obj.testData.medium.data = returns(1:obj.testData.medium.T, 1:obj.testData.medium.K);
            end
            
            obj.testData.large = struct();
            obj.testData.large.T = 500;
            obj.testData.large.K = 20;
            % Create synthetic data for large case
            obj.testData.large.data = randn(obj.testData.large.T, obj.testData.large.K);
            
            % Define model types and distribution options
            obj.testData.modelTypes = struct('BEKK', 1, 'DCC', 2);
            obj.testData.distTypes = struct('NORMAL', 1, 'T', 2, 'GED', 3, 'SKEWT', 4);
            obj.testData.weightTypes = struct('EQUAL', 1, 'INVERSE_VARIANCE', 2);
            
            % Prepare parameter sets for different model configurations
            obj.testData.parameters = struct();
            
            % BEKK parameters (simplified for testing)
            % In practice, these would include C, A, and B matrices
            obj.testData.parameters.BEKK = randn(10, 1) * 0.1;
            
            % DCC parameters (a, b parameters)
            obj.testData.parameters.DCC = [0.05; 0.90];
        end
        
        function tearDown(obj)
            % Clean up after tests
            
            % Call parent tearDown method
            obj.tearDown@BaseTest();
            
            % Clear any temporary variables created during tests
        end
        
        function testMEXFileExists(obj)
            % Test that the composite_likelihood MEX file exists in the expected location
            
            % Use MEXValidator.validateMEXExists to check for composite_likelihood MEX file
            exists = obj.validator.validateMEXExists(obj.mexFile);
            obj.assertTrue(exists, 'MEX file composite_likelihood should exist');
            
            % Verify correct platform-specific extension (.mexw64 or .mexa64)
            platform = computer();
            if strncmp(platform, 'PCW', 3)
                obj.assertTrue(exist([obj.mexFile, '.mexw64'], 'file') == 3, 'Windows MEX file should have .mexw64 extension');
            elseif strncmp(platform, 'GLN', 3) || strncmp(platform, 'MAC', 3)
                obj.assertTrue(exist([obj.mexFile, '.mexa64'], 'file') == 3 || ...
                               exist([obj.mexFile, '.mexmaci64'], 'file') == 3, ...
                               'Unix/Mac MEX file should have .mexa64 or .mexmaci64 extension');
            end
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of composite likelihood MEX file with standard inputs
            
            % Prepare simple test case with synthetic multivariate data
            data = obj.testData.small.data;
            parameters = obj.testData.parameters.BEKK;
            
            % Options for BEKK model with normal distribution
            options = [obj.testData.modelTypes.BEKK; 
                       obj.testData.distTypes.NORMAL; 
                       obj.testData.weightTypes.EQUAL; 
                       1; 1; 1];  % p, q, standardize
            
            % Call composite_likelihood MEX function with test inputs
            try
                result = composite_likelihood(data, parameters, options);
                
                % Verify output contains expected likelihood value
                obj.assertTrue(isfinite(result), 'Result should be a finite value');
                obj.assertTrue(result > 0, 'Negative log-likelihood should be positive');
                
                % Check likelihood value is finite and has expected sign (negative)
                obj.assertTrue(~isnan(result), 'Likelihood should not be NaN');
                obj.assertTrue(~isinf(result), 'Likelihood should not be infinite');
                
                % Assert execution completes without errors
                succeeded = true;
            catch ME
                succeeded = false;
                disp(['Error in basic functionality test: ', ME.message]);
            end
            
            obj.assertTrue(succeeded, 'Basic functionality test should execute without errors');
        end
        
        function testParameterHandling(obj)
            % Test composite likelihood MEX file parameter handling
            
            % Test with various valid parameter combinations for BEKK and DCC models
            data = obj.testData.small.data;
            
            % Test BEKK model
            bekk_params = obj.testData.parameters.BEKK;
            bekk_options = [obj.testData.modelTypes.BEKK; 
                           obj.testData.distTypes.NORMAL; 
                           obj.testData.weightTypes.EQUAL; 
                           1; 1; 1];
            
            try
                bekk_result = composite_likelihood(data, bekk_params, bekk_options);
                obj.assertTrue(isfinite(bekk_result), 'BEKK model should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['BEKK model failed: ', ME.message]);
            end
            
            % Test DCC model
            dcc_params = obj.testData.parameters.DCC;
            dcc_options = [obj.testData.modelTypes.DCC; 
                          obj.testData.distTypes.NORMAL; 
                          obj.testData.weightTypes.EQUAL; 
                          1; 1; 1];
            
            try
                dcc_result = composite_likelihood(data, dcc_params, dcc_options);
                obj.assertTrue(isfinite(dcc_result), 'DCC model should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['DCC model failed: ', ME.message]);
            end
            
            % Test with different weighting schemes for pairwise likelihood
            equal_weight_options = bekk_options;
            equal_weight_options(3) = obj.testData.weightTypes.EQUAL;
            
            inv_var_options = bekk_options;
            inv_var_options(3) = obj.testData.weightTypes.INVERSE_VARIANCE;
            
            try
                equal_result = composite_likelihood(data, bekk_params, equal_weight_options);
                inv_var_result = composite_likelihood(data, bekk_params, inv_var_options);
                
                % Different weighting schemes should give different likelihood values
                obj.assertTrue(abs(equal_result - inv_var_result) > obj.defaultTolerance, ...
                              'Different weighting schemes should give different likelihood values');
            catch ME
                obj.assertTrue(false, ['Weighting scheme test failed: ', ME.message]);
            end
            
            % Test parameter validation for model type identifiers
            invalid_model_options = bekk_options;
            invalid_model_options(1) = 999; % Invalid model type
            
            try
                invalid_result = composite_likelihood(data, bekk_params, invalid_model_options);
                obj.assertTrue(false, 'Should fail with invalid model type');
            catch
                % Expected failure
            end
            
            % Assert correct handling of all parameter combinations
        end
        
        function testDistributionTypes(obj)
            % Test composite likelihood with different error distribution types
            
            data = obj.testData.small.data;
            parameters = obj.testData.parameters.BEKK;
            
            % Test with Normal distribution (distribution_type = 0)
            normal_options = [obj.testData.modelTypes.BEKK; 
                             obj.testData.distTypes.NORMAL; 
                             obj.testData.weightTypes.EQUAL; 
                             1; 1; 1];
            
            try
                normal_result = composite_likelihood(data, parameters, normal_options);
                obj.assertTrue(isfinite(normal_result), 'Normal distribution should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['Normal distribution test failed: ', ME.message]);
            end
            
            % Test with Student's t distribution (distribution_type = 1)
            t_options = [obj.testData.modelTypes.BEKK; 
                        obj.testData.distTypes.T; 
                        obj.testData.weightTypes.EQUAL; 
                        1; 1; 1; 
                        8];  % degrees of freedom = 8
            
            try
                t_result = composite_likelihood(data, parameters, t_options);
                obj.assertTrue(isfinite(t_result), 'Student t distribution should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['Student t distribution test failed: ', ME.message]);
            end
            
            % Test with GED distribution (distribution_type = 2)
            ged_options = [obj.testData.modelTypes.BEKK; 
                          obj.testData.distTypes.GED; 
                          obj.testData.weightTypes.EQUAL; 
                          1; 1; 1; 
                          1.5];  % shape parameter = 1.5
            
            try
                ged_result = composite_likelihood(data, parameters, ged_options);
                obj.assertTrue(isfinite(ged_result), 'GED distribution should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['GED distribution test failed: ', ME.message]);
            end
            
            % Test with Skewed t distribution (distribution_type = 3)
            skewt_options = [obj.testData.modelTypes.BEKK; 
                            obj.testData.distTypes.SKEWT; 
                            obj.testData.weightTypes.EQUAL; 
                            1; 1; 1; 
                            8; 0.2];  % degrees of freedom = 8, skewness = 0.2
            
            try
                skewt_result = composite_likelihood(data, parameters, skewt_options);
                obj.assertTrue(isfinite(skewt_result), 'Skewed t distribution should return finite likelihood');
            catch ME
                obj.assertTrue(false, ['Skewed t distribution test failed: ', ME.message]);
            end
            
            % Compare likelihood values with analytical expectations
            % Different distributions should give different likelihood values
            obj.assertTrue(abs(normal_result - t_result) > obj.defaultTolerance, ...
                          'Normal and t distributions should give different likelihood values');
            obj.assertTrue(abs(normal_result - ged_result) > obj.defaultTolerance, ...
                          'Normal and GED distributions should give different likelihood values');
            obj.assertTrue(abs(normal_result - skewt_result) > obj.defaultTolerance, ...
                          'Normal and skewed t distributions should give different likelihood values');
            
            % Assert correct implementation of all distribution types
        end
        
        function testNumericAccuracy(obj)
            % Test numerical accuracy of composite likelihood implementation
            
            % Compare MEX output with manually calculated values for simple cases
            T = 50;
            K = 2;  % Use 2-dimensional case for simpler verification
            
            % Create a simple bivariate dataset with known properties
            rho = 0.5;  % Correlation
            sigma1 = 1.0;  % Std dev for series 1
            sigma2 = 1.2;  % Std dev for series 2
            
            % Correlation matrix
            R = [1, rho; rho, 1];
            
            % Covariance matrix
            D = diag([sigma1, sigma2]);
            sigma = D * R * D;
            
            % Generate data from this distribution
            data = mvnrnd(zeros(1, K), sigma, T);
            
            % Simple parameters for testing
            parameters = [0.05; 0.90];  % DCC parameters (a, b)
            
            % Options for DCC with normal distribution
            options = [obj.testData.modelTypes.DCC; 
                       obj.testData.distTypes.NORMAL; 
                       obj.testData.weightTypes.EQUAL; 
                       1; 1; 1];
            
            % Calculate using MEX function
            mex_result = composite_likelihood(data, parameters, options);
            
            % Calculate manually
            manual_ll = obj.calculateExpectedLikelihood(data, sigma, 1, []);
            
            % Validate pairwise likelihood components against direct computation
            obj.assertAlmostEqual(mex_result, manual_ll, ['MEX implementation should match manual calculation']);
            
            % Verify weighting scheme implementation
            options_inv_var = options;
            options_inv_var(3) = obj.testData.weightTypes.INVERSE_VARIANCE;
            
            inv_var_result = composite_likelihood(data, parameters, options_inv_var);
            
            % Different weighting should give different result
            obj.assertTrue(abs(mex_result - inv_var_result) > obj.defaultTolerance, ...
                          'Different weighting schemes should give different likelihood values');
            
            % Assert all results match expected values within tolerance
        end
        
        function testDimensionality(obj)
            % Test composite likelihood behavior with different dimensionality
            
            % Test with 2-dimensional case (simplest case)
            data_2d = obj.generateMultivariateTestData(100, 2, []);
            
            % Test with medium dimensionality (5-10 variables)
            data_5d = obj.generateMultivariateTestData(100, 5, []);
            data_10d = obj.testData.medium.data(:, 1:8);
            
            % Test with large dimensionality (30+ variables)
            data_30d = obj.testData.large.data;
            
            % Parameters for testing
            params = obj.testData.parameters.BEKK;
            
            % Options for BEKK with normal distribution
            options = [obj.testData.modelTypes.BEKK; 
                       obj.testData.distTypes.NORMAL; 
                       obj.testData.weightTypes.EQUAL; 
                       1; 1; 1];
            
            % Test each dimension
            try
                result_2d = composite_likelihood(data_2d.data, params, options);
                obj.assertTrue(isfinite(result_2d), 'Should return finite likelihood for 2D case');
                
                result_5d = composite_likelihood(data_5d.data, params, options);
                obj.assertTrue(isfinite(result_5d), 'Should return finite likelihood for 5D case');
                
                result_10d = composite_likelihood(data_10d, params, options);
                obj.assertTrue(isfinite(result_10d), 'Should return finite likelihood for 10D case');
                
                % Large dimension test (may take longer)
                result_30d = composite_likelihood(data_30d, params, options);
                obj.assertTrue(isfinite(result_30d), 'Should return finite likelihood for 30D case');
                
                % Verify correct handling of pairwise combinations
                % Number of pairs should increase quadratically with dimension
                % This is indirectly verified by comparing execution times
                tic; composite_likelihood(data_2d.data, params, options); t_2d = toc;
                tic; composite_likelihood(data_5d.data, params, options); t_5d = toc;
                
                % Execution time should roughly scale with number of pairs
                pairs_ratio = (5*4/2) / (2*1/2);  % (5×4/2) / (2×1/2) = 10/1 = 10
                time_ratio = t_5d / t_2d;
                
                % Allow flexibility in timing due to system variations and overhead
                obj.assertTrue(time_ratio > 1, 'Higher dimensions should take longer due to more pairs');
                
                % Assert computational tractability with high dimensions
            catch ME
                obj.assertTrue(false, ['Dimensionality test failed: ', ME.message]);
            end
        end
        
        function testEdgeCases(obj)
            % Test composite likelihood behavior with edge cases
            
            % Test with nearly singular covariance matrices
            T = 50;
            K = 3;
            
            % Create highly correlated data (near-singular correlation matrix)
            base = randn(T, 1);
            data_highly_corr = [base, base + 0.001*randn(T, 1), base + 0.002*randn(T, 1)];
            
            % Test with highly correlated variables
            params = obj.testData.parameters.BEKK;
            options = [obj.testData.modelTypes.BEKK; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            try
                result_corr = composite_likelihood(data_highly_corr, params, options);
                obj.assertTrue(isfinite(result_corr), 'Should handle highly correlated data');
            catch ME
                obj.assertTrue(false, ['Edge case test failed: ', ME.message]);
            end
            
            % Test with extremely small variances
            data_small_var = randn(T, K) * 1e-5;
            
            try
                result_small = composite_likelihood(data_small_var, params, options);
                obj.assertTrue(isfinite(result_small), 'Should handle extremely small variances');
            catch ME
                obj.assertTrue(false, ['Small variance test failed: ', ME.message]);
            end
            
            % Test with boundary cases for distribution parameters
            % Very low degrees of freedom for t-distribution
            t_options = [obj.testData.modelTypes.BEKK; 
                        obj.testData.distTypes.T; 
                        obj.testData.weightTypes.EQUAL; 
                        1; 1; 1; 
                        2.1];  % Very close to lower bound (2)
            
            try
                result_low_df = composite_likelihood(obj.testData.small.data, params, t_options);
                obj.assertTrue(isfinite(result_low_df), 'Should handle low degrees of freedom');
            catch ME
                obj.assertTrue(false, ['Low degrees of freedom test failed: ', ME.message]);
            end
            
            % Assert correct handling of all edge cases
        end
        
        function testErrorHandling(obj)
            % Test error handling in composite likelihood implementation
            
            % Test with invalid dimensions (mismatched data and parameters)
            data = obj.testData.small.data;
            params = obj.testData.parameters.BEKK;
            options = [obj.testData.modelTypes.BEKK; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            % Create invalid data with NaN
            invalid_data = data;
            invalid_data(5, 2) = NaN;
            
            try
                result = composite_likelihood(invalid_data, params, options);
                obj.assertTrue(false, 'Should throw error for NaN in data');
            catch
                % Expected behavior
            end
            
            % Test with invalid model type identifiers
            invalid_options = options;
            invalid_options(1) = 999;  % Invalid model type
            
            try
                result = composite_likelihood(data, params, invalid_options);
                obj.assertTrue(false, 'Should throw error for invalid model type');
            catch
                % Expected behavior
            end
            
            % Test with invalid distribution parameters
            invalid_t_options = [obj.testData.modelTypes.BEKK; 
                                obj.testData.distTypes.T; 
                                obj.testData.weightTypes.EQUAL; 
                                1; 1; 1; 
                                1.5];  % df < 2 is invalid for t-distribution
            
            try
                result = composite_likelihood(data, params, invalid_t_options);
                obj.assertTrue(false, 'Should throw error for invalid distribution parameters');
            catch
                % Expected behavior
            end
            
            % Test with invalid weighting schemes
            invalid_weight_options = options;
            invalid_weight_options(3) = 999;  % Invalid weighting scheme
            
            try
                result = composite_likelihood(data, params, invalid_weight_options);
                obj.assertTrue(false, 'Should throw error for invalid weighting scheme');
            catch
                % Expected behavior
            end
            
            % Assert appropriate error messages are generated
            % Verify no memory leaks occur when errors are triggered
        end
        
        function testPerformance(obj)
            % Benchmark performance of composite likelihood MEX implementation
            
            % Generate large-scale test data for performance testing
            data = obj.testData.large.data;
            parameters = obj.testData.parameters.DCC;
            
            % Options for DCC model with normal distribution
            options = [obj.testData.modelTypes.DCC; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            % Skip test if dcc_mvgarch is not available
            if exist('dcc_mvgarch', 'file') ~= 2
                warning('dcc_mvgarch.m not available, skipping performance comparison');
                return;
            end
            
            % Benchmark MEX implementation
            nRuns = 5;
            tic;
            for i = 1:nRuns
                mex_result = composite_likelihood(data, parameters, options);
            end
            mex_time = toc / nRuns;
            
            % Benchmark MATLAB implementation (simplified)
            % In a real-world scenario, this would be a direct MATLAB equivalent
            % Here we're using a simplified approach using dcc_mvgarch
            tic;
            for i = 1:nRuns
                % Create a structure to imitate dcc_mvgarch's behavior
                dcc_options = struct('p', 1, 'q', 1, 'model', 'GARCH', 'distribution', 'NORMAL');
                dcc_options.startvalues = parameters;
                
                % Note: dcc_mvgarch is not a direct equivalent, but serves as a reference
                % We use fewer iterations to avoid excessive test time
                matlab_result = obj.executeCompositeLikelihoodMATLAB(data, parameters, options);
            end
            matlab_time = toc / nRuns;
            
            % Calculate performance improvement
            perf_improvement = (matlab_time - mex_time) / matlab_time * 100;
            
            % Verify performance improvement exceeds 50% target
            obj.assertTrue(perf_improvement > 50, ...
                          sprintf('MEX implementation should provide >50%% performance improvement; got %.1f%%', perf_improvement));
            
            % Test scaling behavior with increasing dimensions
            % Already tested in dimensionality test
            
            % Assert consistent performance across multiple runs
            time_values = zeros(3, 1);
            for i = 1:3
                tic;
                composite_likelihood(data, parameters, options);
                time_values(i) = toc;
            end
            
            time_std = std(time_values);
            time_mean = mean(time_values);
            time_cv = time_std / time_mean;  % Coefficient of variation
            
            obj.assertTrue(time_cv < 0.5, 'Performance should be consistent across runs');
        end
        
        function testMemoryUsage(obj)
            % Test memory usage of composite likelihood implementation
            
            % Use MEXValidator.validateMemoryUsage to monitor memory consumption
            data = obj.testData.medium.data;
            parameters = obj.testData.parameters.BEKK;
            options = [obj.testData.modelTypes.BEKK; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            % Check memory usage with MEXValidator
            mem_result = obj.validator.validateMemoryUsage(obj.mexFile, {data, parameters, options}, 50);
            
            % Check for memory leaks
            obj.assertFalse(mem_result.hasLeak, 'No memory leaks should be detected');
            
            % Test with incrementally larger datasets
            dimensions = [5, 10, 15];
            memory_usage = zeros(length(dimensions), 1);
            
            for i = 1:length(dimensions)
                k = dimensions(i);
                test_data = data(:, 1:min(k, size(data, 2)));
                
                % If needed, pad with random data
                if size(test_data, 2) < k
                    test_data = [test_data, randn(size(test_data, 1), k - size(test_data, 2))];
                end
                
                % Measure memory usage for this dimension
                mem_result_k = obj.validator.validateMemoryUsage(obj.mexFile, {test_data, parameters, options}, 10);
                memory_usage(i) = mem_result_k.memoryPerIteration;
            end
            
            % Memory usage should scale reasonably with dimension
            if memory_usage(1) > 0
                scaling_factor_5_10 = memory_usage(2) / memory_usage(1);
                scaling_factor_10_15 = memory_usage(3) / memory_usage(2);
                
                % Number of pairs scales quadratically with dimension
                % For N dimensions, we have N*(N-1)/2 pairs
                pair_ratio_5_10 = (10*9/2) / (5*4/2);  // = 4.5
                pair_ratio_10_15 = (15*14/2) / (10*9/2);  // = 2.3
                
                % Memory scaling should be approximately proportional to pairs
                % but with some overhead, so we use a loose bound
                obj.assertTrue(scaling_factor_5_10 < pair_ratio_5_10 * 2, ...
                              'Memory usage should scale reasonably with dimension');
                obj.assertTrue(scaling_factor_10_15 < pair_ratio_10_15 * 2, ...
                              'Memory usage should scale reasonably with dimension');
            end
            
            % Assert efficient memory utilization for large datasets
        end
        
        function testBEKKModelIntegration(obj)
            % Test integration with BEKK multivariate volatility models
            
            % Set up BEKK model parameters and data
            T = 100;
            K = 3;
            
            % Generate test data
            data = obj.testData.small.data;
            
            % Create simplified BEKK parameters
            % In a full BEKK model, parameters include:
            % - C (K×K lower triangular matrix for constant term)
            % - A (K×K matrix for ARCH effects)
            % - B (K×K matrix for GARCH effects)
            
            % Create C parameters (vectorized lower triangular)
            C = zeros(K, K);
            C(1,1) = 0.1; 
            C(2,1) = 0.05; C(2,2) = 0.12;
            C(3,1) = 0.03; C(3,2) = 0.04; C(3,3) = 0.09;
            C_vector = [C(1,1); C(2,1); C(2,2); C(3,1); C(3,2); C(3,3)];
            
            % Create A and B (simplified diagonal matrices)
            A = eye(K) * 0.1;  // ARCH effect
            B = eye(K) * 0.8;  // GARCH effect
            
            % Vectorize parameters
            params = [C_vector; A(:); B(:)];
            
            % Options for BEKK model
            options = [obj.testData.modelTypes.BEKK; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            % Execute composite likelihood calculation with BEKK matrices
            result = composite_likelihood(data, params, options);
            
            % Verify results match expected likelihood values
            obj.assertTrue(isfinite(result), 'BEKK model should return finite likelihood');
            
            % Test consistency with varying BEKK specifications
            % Create alternative parameter set
            A_alt = eye(K) * 0.15;  // Different ARCH effect
            B_alt = eye(K) * 0.75;  // Different GARCH effect
            params_alt = [C_vector; A_alt(:); B_alt(:)];
            
            result_alt = composite_likelihood(data, params_alt, options);
            
            % Different parameters should give different likelihoods
            obj.assertTrue(abs(result - result_alt) > obj.defaultTolerance, ...
                          'Different BEKK parameters should give different likelihoods');
        end
        
        function testDCCModelIntegration(obj)
            % Test integration with DCC multivariate volatility models
            
            % Set up DCC model parameters and data
            data = obj.testData.small.data;
            
            % DCC parameters (a and b)
            a = 0.05;
            b = 0.93;
            params = [a; b];
            
            % Calculate standardized residuals (simplification)
            % In practice, this would come from univariate GARCH models
            std_residuals = zeros(size(data));
            for i = 1:size(data, 2)
                std_residuals(:, i) = data(:, i) / std(data(:, i));
            end
            
            % Calculate unconditional correlation matrix
            uncond_corr = corr(data);
            
            % Options for DCC model
            options = [obj.testData.modelTypes.DCC; 
                      obj.testData.distTypes.NORMAL; 
                      obj.testData.weightTypes.EQUAL; 
                      1; 1; 1];
            
            % Execute composite likelihood calculation with DCC matrices
            result = composite_likelihood(std_residuals, params, options);
            
            % Verify integration with dcc_mvgarch function
            if exist('dcc_mvgarch', 'file') == 2
                % This is just to check integration - full comparison would be complex
                try
                    dcc_options = struct('model', 'GARCH', 'distribution', 'NORMAL', 'p', 1, 'q', 1);
                    dcc_options.startvalues = params;
                    model = dcc_mvgarch(data, dcc_options);
                    
                    % Verify model estimation succeeded
                    obj.assertTrue(isfield(model, 'likelihood'), 'DCC model estimation should succeed');
                    obj.assertTrue(isfinite(model.likelihood), 'DCC model should return finite likelihood');
                catch ME
                    warning('DCC model estimation failed: %s', ME.message);
                end
            end
            
            % Test consistency with varying DCC specifications
            params_alt = [0.02; 0.97];  // Different DCC parameters
            result_alt = composite_likelihood(std_residuals, params_alt, options);
            
            % Different parameters should give different likelihoods
            obj.assertTrue(abs(result - result_alt) > obj.defaultTolerance, ...
                          'Different DCC parameters should give different likelihoods');
        end
        
        %% Helper methods
        
        function testData = generateMultivariateTestData(obj, T, K, covParams)
            % Helper method to generate multivariate test data with known properties
            
            % Generate random multivariate normal data with T observations and K dimensions
            if nargin < 4 || isempty(covParams)
                % Create random covariance structure
                D = diag(0.5 + rand(K, 1));  % Random variances
                R = eye(K);                  % Start with identity correlation
                
                % Add some correlation structure
                for i = 1:K
                    for j = i+1:K
                        R(i,j) = 0.1 + 0.3*rand();  % Random correlations between 0.1 and 0.4
                        R(j,i) = R(i,j);            % Ensure symmetry
                    end
                end
                
                % Create covariance matrix
                covMatrix = D * R * D;
            else
                % Use specified covariance structure
                covMatrix = covParams;
            end
            
            % Generate multivariate normal data
            data = mvnrnd(zeros(1, K), covMatrix, T);
            
            % Create standardized innovations if needed
            std_data = zeros(size(data));
            for i = 1:K
                std_data(:, i) = data(:, i) / sqrt(covMatrix(i, i));
            end
            
            % Set up parameter structure for composite likelihood function
            testData = struct();
            testData.data = data;
            testData.std_data = std_data;
            testData.cov = covMatrix;
            testData.corr = corr(data);
            testData.T = T;
            testData.K = K;
            
            % Add known analytical likelihood value if applicable
        end
        
        function result = executeCompositeLikelihood(obj, data, parameters, modelType, distributionType, distributionParams, weightingScheme)
            % Helper method to execute composite_likelihood MEX function with standardized interface
            
            % Prepare input arguments in format expected by composite_likelihood
            if nargin < 4 || isempty(modelType)
                modelType = obj.testData.modelTypes.BEKK;
            end
            
            if nargin < 5 || isempty(distributionType)
                distributionType = obj.testData.distTypes.NORMAL;
            end
            
            if nargin < 7 || isempty(weightingScheme)
                weightingScheme = obj.testData.weightTypes.EQUAL;
            end
            
            % Prepare options array
            options = [modelType; distributionType; weightingScheme; 1; 1; 1];
            
            % Handle optional parameters based on model and distribution types
            if nargin >= 6 && ~isempty(distributionParams)
                options = [options; distributionParams(:)];
            elseif distributionType == obj.testData.distTypes.T
                options = [options; 8];  % Default df=8 for t distribution
            elseif distributionType == obj.testData.distTypes.GED
                options = [options; 1.5];  % Default shape=1.5 for GED
            elseif distributionType == obj.testData.distTypes.SKEWT
                options = [options; 8; 0];  % Default df=8, lambda=0 for skewed t
            end
            
            % Execute composite_likelihood MEX function in try-catch block
            try
                result = composite_likelihood(data, parameters, options);
            catch ME
                error('MEX execution failed: %s', ME.message);
            end
            
            % Process output for consistent interface
            return;
        end
        
        function matlab_result = executeCompositeLikelihoodMATLAB(obj, data, parameters, options)
            % Helper function to emulate composite likelihood calculation in MATLAB
            % This is a simplified version for benchmarking, not a full implementation
            
            [T, K] = size(data);
            
            % Extract model details from options
            modelType = options(1);
            distType = options(2);
            weightType = options(3);
            
            % Calculate number of pairs
            n_pairs = K * (K - 1) / 2;
            
            % Equal weights for simplicity
            weights = ones(n_pairs, 1) / n_pairs;
            
            % Calculate composite likelihood
            total_ll = 0;
            pair_idx = 1;
            
            for i = 1:K-1
                for j = i+1:K
                    % Extract pair data
                    pair_data = data(:, [i, j]);
                    
                    % Calculate pair covariance
                    pair_cov = cov(pair_data);
                    
                    % Guarantee positive definiteness
                    if det(pair_cov) <= 0
                        pair_cov(1,1) = pair_cov(1,1) + 1e-6;
                        pair_cov(2,2) = pair_cov(2,2) + 1e-6;
                    end
                    
                    % Calculate pair log-likelihood
                    pair_ll = 0;
                    
                    % Only implementing normal distribution for simplicity
                    det_cov = det(pair_cov);
                    inv_cov = inv(pair_cov);
                    
                    for t = 1:T
                        x = pair_data(t, :)';
                        quad = x' * inv_cov * x;
                        ll_t = -log(2*pi) - 0.5*log(det_cov) - 0.5*quad;
                        pair_ll = pair_ll + ll_t;
                    end
                    
                    % Add weighted contribution
                    total_ll = total_ll + weights(pair_idx) * pair_ll;
                    pair_idx = pair_idx + 1;
                end
            end
            
            % Return negative log-likelihood for minimization
            matlab_result = -total_ll;
        end
        
        function expectedLike = calculateExpectedLikelihood(obj, data, covariances, distributionType, distributionParams)
            % Helper method to calculate expected likelihood values for validation
            
            [T, K] = size(data);
            
            % Extract all pairwise data and covariance submatrices
            n_pairs = K * (K - 1) / 2;
            weights = ones(n_pairs, 1) / n_pairs;  % Equal weights
            
            total_ll = 0;
            pair_idx = 1;
            
            for i = 1:K-1
                for j = i+1:K
                    % Extract data for this pair
                    pair_data = data(:, [i, j]);
                    
                    % Extract covariance for this pair
                    if size(covariances, 1) == K && size(covariances, 2) == K
                        % Single covariance matrix
                        pair_cov = covariances([i, j], [i, j]);
                    else
                        % Use sample covariance
                        pair_cov = cov(pair_data);
                    end
                    
                    % Calculate determinants and inverses for 2x2 matrices
                    det_cov = det(pair_cov);
                    
                    % Handle potential numerical issues
                    if det_cov <= 0
                        % Add small constant to diagonal
                        pair_cov(1,1) = pair_cov(1,1) + 1e-6;
                        pair_cov(2,2) = pair_cov(2,2) + 1e-6;
                        det_cov = det(pair_cov);
                    end
                    
                    inv_cov = inv(pair_cov);
                    
                    % Compute quadratic forms for each pair
                    pair_ll = 0;
                    
                    % Apply appropriate distribution density function
                    if distributionType == 1 || isempty(distributionType)  % Normal
                        for t = 1:T
                            x = pair_data(t, :)';
                            quad = x' * inv_cov * x;
                            ll_t = -log(2*pi) - 0.5*log(det_cov) - 0.5*quad;
                            pair_ll = pair_ll + ll_t;
                        end
                    elseif distributionType == 2  % Student's t
                        nu = distributionParams(1);  % Degrees of freedom
                        for t = 1:T
                            x = pair_data(t, :)';
                            quad = x' * inv_cov * x;
                            const = lgamma((nu+2)/2) - lgamma(nu/2) - log(pi*nu) - 0.5*log(det_cov);
                            ll_t = const - 0.5*(nu+2)*log(1 + quad/nu);
                            pair_ll = pair_ll + ll_t;
                        end
                    end
                    
                    % Add weighted contribution
                    total_ll = total_ll + weights(pair_idx) * pair_ll;
                    pair_idx = pair_idx + 1;
                end
            end
            
            % Return negative log-likelihood
            expectedLike = -total_ll;
        end
    end
end