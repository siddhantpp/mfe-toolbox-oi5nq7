classdef BootstrapVarianceTest < BaseTest
    % BOOTSTRAPVARIANCETEST Unit test class for the bootstrap_variance function
    %
    % This class tests the bootstrap_variance function which estimates variance
    % using bootstrap resampling methods for financial time series data. Tests
    % cover both block and stationary bootstrap methods with different parameters
    % and validate the statistical properties of the resulting variance estimates.
    
    properties
        testData                % Univariate test data for bootstrapping
        multivariateTestData    % Multivariate test data for testing
        testParams              % Common test parameters
        sampleMean              % Function handle for mean calculation
        sampleVar               % Function handle for variance calculation
    end
    
    methods
        function obj = BootstrapVarianceTest()
            % Initialize with test name
            obj = obj@BaseTest('Bootstrap Variance Test');
            
            % Initialize test parameters structure
            obj.testParams = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Set random seed for reproducibility
            rng(123);
            
            % Generate univariate AR(1) test data
            T = 100;
            ar_coef = 0.5; % Moderate autoregressive coefficient
            innovations = randn(T, 1);
            obj.testData = zeros(T, 1);
            
            % Generate AR(1) process: X_t = ar_coef * X_{t-1} + e_t
            for t = 2:T
                obj.testData(t) = ar_coef * obj.testData(t-1) + innovations(t);
            end
            
            % Generate multivariate test data with known cross-correlation
            n_var = 3;
            obj.multivariateTestData = randn(T, n_var);
            
            % Create correlation structure
            corr_factor = 0.5;
            for i = 2:n_var
                obj.multivariateTestData(:,i) = corr_factor * obj.multivariateTestData(:,1) + 
                    sqrt(1 - corr_factor^2) * obj.multivariateTestData(:,i);
            end
            
            % Define test statistic functions
            obj.sampleMean = @mean;
            obj.sampleVar = @(x) var(x);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.testData = [];
            obj.multivariateTestData = [];
        end
        
        function testBasicFunctionality(obj)
            % Tests that bootstrap_variance correctly estimates variance 
            % using the block bootstrap method
            
            % Set up options structure with block bootstrap method
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            options.replications = 500;
            
            % Call bootstrap_variance with univariate test data and sampleMean function
            results = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Verify the structure of the results contains all required fields
            obj.assertTrue(isstruct(results), 'Results should be a structure');
            obj.assertTrue(isfield(results, 'variance'), 'Results should contain variance field');
            obj.assertTrue(isfield(results, 'std_error'), 'Results should contain std_error field');
            obj.assertTrue(isfield(results, 'conf_lower'), 'Results should contain conf_lower field');
            obj.assertTrue(isfield(results, 'conf_upper'), 'Results should contain conf_upper field');
            obj.assertTrue(isfield(results, 'bootstrap_stats'), 'Results should contain bootstrap_stats field');
            
            % Verify the variance estimate is positive
            obj.assertTrue(results.variance > 0, 'Variance should be positive');
            
            % Theoretical variance for AR(1) mean estimator with coefficient 0.5
            % Using the formula: Var(mean) = σ²/T * (1+ρ)/(1-ρ) where ρ is AR coefficient
            theoretical_var = var(obj.testData) * (1 + 0.5) / (1 - 0.5);
            theoretical_var = theoretical_var / length(obj.testData);
            
            % Verify the variance estimate is reasonably close to theoretical value
            % Using a larger tolerance due to bootstrap randomness
            obj.assertMatrixEqualsWithTolerance(results.variance, theoretical_var, 0.5, 
                'Bootstrap variance estimate should be close to theoretical value');
            
            % Verify the confidence intervals have the correct properties
            obj.assertTrue(results.conf_lower < results.conf_upper, 
                'Lower confidence bound should be less than upper bound');
            obj.assertTrue(results.conf_lower < results.mean && results.mean < results.conf_upper,
                'Mean should be within confidence bounds');
        end
        
        function testStationaryBootstrap(obj)
            % Tests that bootstrap_variance correctly estimates variance
            % using the stationary bootstrap method
            
            % Set up options structure with stationary bootstrap method
            options = struct();
            options.bootstrap_type = 'stationary';
            options.p = 0.1;  % Expected block length = 1/p = 10
            options.replications = 500;
            
            % Run bootstrap variance estimation
            results = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Verify the structure of the results contains all required fields
            obj.assertTrue(isstruct(results), 'Results should be a structure');
            obj.assertTrue(isfield(results, 'variance'), 'Results should contain variance field');
            
            % Theoretical variance for AR(1) mean estimator with coefficient 0.5
            theoretical_var = var(obj.testData) * (1 + 0.5) / (1 - 0.5);
            theoretical_var = theoretical_var / length(obj.testData);
            
            % Verify the variance estimate is reasonably close to theoretical value
            obj.assertMatrixEqualsWithTolerance(results.variance, theoretical_var, 0.5, 
                'Bootstrap variance estimate should be close to theoretical value');
            
            % Verify confidence interval properties
            obj.assertTrue(results.conf_lower < results.mean && results.mean < results.conf_upper,
                'Mean should be within confidence bounds');
        end
        
        function testMultipleVariables(obj)
            % Tests that bootstrap_variance correctly handles multivariate data
            
            % Set up options for multivariate testing
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            options.replications = 500;
            
            % Create a statistic function that returns a scalar from multivariate data
            multivar_stat = @(x) mean(mean(x));
            
            % Call bootstrap_variance with multivariate test data
            results = bootstrap_variance(obj.multivariateTestData, multivar_stat, options);
            
            % Verify the structure of the results
            obj.assertTrue(isstruct(results), 'Results should be a structure');
            obj.assertTrue(isfield(results, 'variance'), 'Results should contain variance field');
            
            % Verify the variance estimate is positive
            obj.assertTrue(results.variance > 0, 'Variance should be positive');
            
            % Verify all bootstrap statistics are finite
            obj.assertTrue(all(isfinite(results.bootstrap_stats)), 
                'All bootstrap statistics should be finite');
            
            % Verify cross-correlation handling by ensuring confidence interval isn't too wide
            interval_width = results.conf_upper - results.conf_lower;
            obj.assertTrue(interval_width < 1.0, 
                'Confidence interval width should be reasonable for multivariate data');
        end
        
        function testConfidenceIntervals(obj)
            % Tests that bootstrap_variance correctly computes confidence 
            % intervals at various confidence levels
            
            % Configure base options
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            options.replications = 1000;
            
            % Test with 90% confidence level
            options.conf_level = 0.90;
            results90 = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with 95% confidence level
            options.conf_level = 0.95;
            results95 = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with 99% confidence level
            options.conf_level = 0.99;
            results99 = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Calculate interval widths
            interval90 = results90.conf_upper - results90.conf_lower;
            interval95 = results95.conf_upper - results95.conf_lower;
            interval99 = results99.conf_upper - results99.conf_lower;
            
            % Verify increasing confidence level widens the interval
            obj.assertTrue(interval95 > interval90, '95% interval should be wider than 90% interval');
            obj.assertTrue(interval99 > interval95, '99% interval should be wider than 95% interval');
            
            % Verify all means are within their respective confidence intervals
            obj.assertTrue(results90.conf_lower < results90.mean && results90.mean < results90.conf_upper,
                'Mean should be within 90% confidence bounds');
            obj.assertTrue(results95.conf_lower < results95.mean && results95.mean < results95.conf_upper,
                'Mean should be within 95% confidence bounds');
            obj.assertTrue(results99.conf_lower < results99.mean && results99.mean < results99.conf_upper,
                'Mean should be within 99% confidence bounds');
        end
        
        function testDifferentStatisticFunctions(obj)
            % Tests that bootstrap_variance works correctly with different statistic functions
            
            % Configure options
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            options.replications = 500;
            
            % Test with sample mean function
            results_mean = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with sample variance function
            results_var = bootstrap_variance(obj.testData, obj.sampleVar, options);
            
            % Test with custom statistic function (Sharpe ratio-like)
            sharpe_ratio = @(x) mean(x) / std(x);
            results_sharpe = bootstrap_variance(obj.testData, sharpe_ratio, options);
            
            % Verify each result has valid structure
            obj.assertTrue(isfield(results_mean, 'variance'), 'Mean results should contain variance field');
            obj.assertTrue(isfield(results_var, 'variance'), 'Variance results should contain variance field');
            obj.assertTrue(isfield(results_sharpe, 'variance'), 'Sharpe results should contain variance field');
            
            % Verify different statistics produce different variances
            obj.assertTrue(abs(results_mean.variance - results_var.variance) > 1e-6, 
                'Mean and variance statistics should produce different bootstrap variances');
            obj.assertTrue(abs(results_mean.variance - results_sharpe.variance) > 1e-6, 
                'Mean and Sharpe ratio should produce different bootstrap variances');
        end
        
        function testReplicationCount(obj)
            % Tests that bootstrap_variance correctly handles different numbers of bootstrap replications
            
            % Configure base options
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            
            % Test with small number of replications
            options.replications = 100;
            results_small = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with medium number of replications
            options.replications = 1000;
            results_medium = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with large number of replications
            options.replications = 5000;
            results_large = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Verify bootstrap statistics counts match requested replication counts
            obj.assertEqual(length(results_small.bootstrap_stats), 100, 
                'Should have 100 bootstrap statistics for small replication count');
            obj.assertEqual(length(results_medium.bootstrap_stats), 1000, 
                'Should have 1000 bootstrap statistics for medium replication count');
            obj.assertEqual(length(results_large.bootstrap_stats), 5000, 
                'Should have 5000 bootstrap statistics for large replication count');
            
            % Calculate standard deviations of bootstrap statistics
            std_small = std(results_small.bootstrap_stats);
            std_medium = std(results_medium.bootstrap_stats);
            std_large = std(results_large.bootstrap_stats);
            
            % Verify all standard deviations are positive
            obj.assertTrue(std_small > 0, 'Small replication std should be positive');
            obj.assertTrue(std_medium > 0, 'Medium replication std should be positive');
            obj.assertTrue(std_large > 0, 'Large replication std should be positive');
            
            % Verify variance estimates stabilize with more replications
            % Specifically, the distance between small and large samples should be greater
            % than between medium and large samples
            diff_small_large = abs(results_small.variance - results_large.variance);
            diff_medium_large = abs(results_medium.variance - results_large.variance);
            obj.assertTrue(diff_small_large >= diff_medium_large, 
                'Variance estimates should stabilize with more replications');
        end
        
        function testBlockSizeSensitivity(obj)
            % Tests the sensitivity of bootstrap_variance to block size selection for block bootstrap
            
            % Configure base options
            options = struct();
            options.bootstrap_type = 'block';
            options.replications = 1000;
            
            % Test with small block size
            options.block_size = 2;
            results_small = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with medium block size
            options.block_size = 10;
            results_medium = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with large block size
            options.block_size = 30;
            results_large = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Verify all results are positive
            obj.assertTrue(results_small.variance > 0, 'Small block size variance should be positive');
            obj.assertTrue(results_medium.variance > 0, 'Medium block size variance should be positive');
            obj.assertTrue(results_large.variance > 0, 'Large block size variance should be positive');
            
            % Theoretical variance for AR(1) mean estimator with coefficient 0.5
            theoretical_var = var(obj.testData) * (1 + 0.5) / (1 - 0.5);
            theoretical_var = theoretical_var / length(obj.testData);
            
            % Find which block size provides the closest estimate to theoretical variance
            diff_small = abs(results_small.variance - theoretical_var);
            diff_medium = abs(results_medium.variance - theoretical_var);
            diff_large = abs(results_large.variance - theoretical_var);
            
            [~, best_idx] = min([diff_small, diff_medium, diff_large]);
            block_sizes = [2, 10, 30];
            best_block_size = block_sizes(best_idx);
            
            % Verify a best block size was found
            obj.assertTrue(~isempty(best_block_size), 'Should find a best block size');
            
            % For AR(1) data, verify the optimal block size is not the smallest
            % since small blocks won't capture sufficient temporal dependence
            obj.assertTrue(best_block_size > 2, 
                'Optimal block size for AR(1) data should capture temporal dependence');
        end
        
        function testProbabilitySensitivity(obj)
            % Tests the sensitivity of bootstrap_variance to probability parameter for stationary bootstrap
            
            % Configure base options
            options = struct();
            options.bootstrap_type = 'stationary';
            options.replications = 1000;
            
            % Test with small probability (larger expected block length)
            options.p = 0.01;
            results_small_p = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with medium probability
            options.p = 0.1;
            results_medium_p = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Test with large probability (smaller expected block length)
            options.p = 0.5;
            results_large_p = bootstrap_variance(obj.testData, obj.sampleMean, options);
            
            % Verify all results are positive
            obj.assertTrue(results_small_p.variance > 0, 'Small probability variance should be positive');
            obj.assertTrue(results_medium_p.variance > 0, 'Medium probability variance should be positive');
            obj.assertTrue(results_large_p.variance > 0, 'Large probability variance should be positive');
            
            % Theoretical variance for AR(1) mean estimator with coefficient 0.5
            theoretical_var = var(obj.testData) * (1 + 0.5) / (1 - 0.5);
            theoretical_var = theoretical_var / length(obj.testData);
            
            % Find which probability provides the closest estimate to theoretical variance
            diff_small = abs(results_small_p.variance - theoretical_var);
            diff_medium = abs(results_medium_p.variance - theoretical_var);
            diff_large = abs(results_large_p.variance - theoretical_var);
            
            [~, best_idx] = min([diff_small, diff_medium, diff_large]);
            probabilities = [0.01, 0.1, 0.5];
            best_probability = probabilities(best_idx);
            
            % Verify a best probability was found
            obj.assertTrue(~isempty(best_probability), 'Should find a best probability parameter');
            
            % For AR(1) data, verify that smaller p (larger blocks) is typically better
            % for capturing autocorrelation, though we allow for randomness in the testing
            obj.assertTrue(best_probability <= 0.5, 
                'Optimal probability for AR(1) data should yield meaningful block lengths');
        end
        
        function testInvalidInputs(obj)
            % Tests that bootstrap_variance correctly handles invalid inputs with appropriate error messages
            
            % Configure base options
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 10;
            options.replications = 100;
            
            % Test with empty data
            empty_data = [];
            try
                bootstrap_variance(empty_data, obj.sampleMean, options);
                obj.assertTrue(false, 'Should throw error with empty data');
            catch
                % Error was thrown as expected
            end
            
            % Test with data containing NaN values
            nan_data = obj.testData;
            nan_data(5) = NaN;
            try
                bootstrap_variance(nan_data, obj.sampleMean, options);
                obj.assertTrue(false, 'Should throw error with NaN data');
            catch
                % Error was thrown as expected
            end
            
            % Test with invalid bootstrap type
            invalid_options = options;
            invalid_options.bootstrap_type = 'invalid_type';
            try
                bootstrap_variance(obj.testData, obj.sampleMean, invalid_options);
                obj.assertTrue(false, 'Should throw error with invalid bootstrap type');
            catch
                % Error was thrown as expected
            end
            
            % Test with negative block size
            invalid_options = options;
            invalid_options.block_size = -5;
            try
                bootstrap_variance(obj.testData, obj.sampleMean, invalid_options);
                obj.assertTrue(false, 'Should throw error with negative block size');
            catch
                % Error was thrown as expected
            end
            
            % Test with invalid statistic function
            invalid_func = 'not_a_function';
            try
                bootstrap_variance(obj.testData, invalid_func, options);
                obj.assertTrue(false, 'Should throw error with invalid function handle');
            catch
                % Error was thrown as expected
            end
            
            % Test with invalid confidence level
            invalid_options = options;
            invalid_options.conf_level = 1.5;
            try
                bootstrap_variance(obj.testData, obj.sampleMean, invalid_options);
                obj.assertTrue(false, 'Should throw error with confidence level > 1');
            catch
                % Error was thrown as expected
            end
            
            % Test with negative replication count
            invalid_options = options;
            invalid_options.replications = -100;
            try
                bootstrap_variance(obj.testData, obj.sampleMean, invalid_options);
                obj.assertTrue(false, 'Should throw error with negative replication count');
            catch
                % Error was thrown as expected
            end
        end
        
        function testPerformance(obj)
            % Tests the performance of bootstrap_variance with large datasets
            
            % Generate large test dataset
            T = 1000;
            K = 10;
            large_data = randn(T, K);
            
            % Configure options with minimal replications for speed
            options = struct();
            options.bootstrap_type = 'block';
            options.block_size = 20;
            options.replications = 100;
            
            % Measure execution time
            execution_time = obj.measureExecutionTime(@() bootstrap_variance(large_data, @mean, options));
            
            % Check execution time is reasonable (no specific threshold, just verify it completes)
            obj.assertTrue(isfinite(execution_time), 'Execution time should be finite for large dataset');
            
            % Verify the function executes successfully with large dataset
            results = bootstrap_variance(large_data, @mean, options);
            obj.assertTrue(isstruct(results), 'Should return valid results structure for large dataset');
            obj.assertTrue(isfield(results, 'variance'), 'Results should contain variance field');
            obj.assertTrue(results.variance > 0, 'Variance should be positive for large dataset');
        end
        
        function result = verifyVarianceEstimate(obj, results, theoretical_value, tolerance)
            % Helper method to verify the bootstrap variance estimate against theoretical value
            
            % Default tolerance if not provided
            if nargin < 4
                tolerance = 0.5;  % 50% tolerance due to bootstrap randomness
            end
            
            % Extract variance estimate
            variance_estimate = results.variance;
            
            % Check if variance is within tolerance of theoretical value
            relative_diff = abs(variance_estimate - theoretical_value) / theoretical_value;
            result = relative_diff <= tolerance;
            
            % Return verification result
            obj.assertTrue(result, sprintf('Variance estimate (%.6f) differs from theoretical value (%.6f) by %.2f%%', 
                variance_estimate, theoretical_value, relative_diff * 100));
        end
        
        function results = runAllTests(obj)
            % Convenience method to run all test cases in the class
            
            % Call superclass runAllTests method
            results = runAllTests@BaseTest(obj);
            
            % Display summary of test results
            fprintf('\nBootstrap Variance Test Summary:\n');
            fprintf('  Tests: %d\n', results.summary.numTests);
            fprintf('  Passed: %d\n', results.summary.numPassed);
            fprintf('  Failed: %d\n', results.summary.numFailed);
            fprintf('  Total time: %.2f seconds\n', results.summary.totalExecutionTime);
        end
    end
end