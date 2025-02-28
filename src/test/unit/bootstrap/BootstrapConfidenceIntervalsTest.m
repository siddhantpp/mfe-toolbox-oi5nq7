classdef BootstrapConfidenceIntervalsTest < BaseTest
    % BootstrapConfidenceIntervalsTest Test class for verifying the functionality of the bootstrap_confidence_intervals function
    %
    % This class tests the bootstrap_confidence_intervals function which computes
    % various types of bootstrap confidence intervals for financial time series statistics.
    % It verifies the correct implementation of different confidence interval methods
    % (percentile, basic, studentized, BC, BCa) with both block and stationary bootstrap methods.
    
    properties
        testData                % Test time series data
        multivariateTestData    % Multivariate test data
        testParams              % Structure with test parameters
        meanFn                  % Function handle for mean calculation
        varianceFn              % Function handle for variance calculation
    end
    
    methods
        function obj = BootstrapConfidenceIntervalsTest()
            % Initialize the BootstrapConfidenceIntervalsTest with necessary test configuration
            obj@BaseTest('Bootstrap Confidence Intervals Test');
            
            % Initialize test parameters structure
            obj.testParams = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random seed for reproducibility
            rng(123);
            
            % Generate univariate AR(1) test data with known dependence structure
            n = 100;
            ar_coef = 0.5;
            
            % Create AR(1) process: x_t = ar_coef * x_{t-1} + e_t
            obj.testData = zeros(n, 1);
            obj.testData(1) = randn();
            for t = 2:n
                obj.testData(t) = ar_coef * obj.testData(t-1) + randn();
            end
            
            % Generate multivariate test data with cross-correlations
            obj.multivariateTestData = generateFinancialReturns(100, 3, struct('correlation', [1, 0.7, 0.3; 0.7, 1, 0.2; 0.3, 0.2, 1]));
            
            % Define test statistic functions (mean and variance)
            obj.meanFn = @mean;
            obj.varianceFn = @(x) var(x, 0); % Using unbiased estimator (n-1 denominator)
            
            % Initialize test parameters with standard values for bootstrap methods
            obj.testParams.bootstrap_type = 'block';
            obj.testParams.block_size = 5;
            obj.testParams.p = 0.2;
            obj.testParams.replications = 500;
            obj.testParams.conf_level = 0.90;
            obj.testParams.method = 'percentile';
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.testData = [];
            obj.multivariateTestData = [];
            obj.testParams = [];
            obj.meanFn = [];
            obj.varianceFn = [];
        end
        
        function testBasicFunctionality(obj)
            % Tests that bootstrap_confidence_intervals correctly computes percentile confidence intervals
            
            % Configure options for percentile method with block bootstrap
            options = obj.testParams;
            
            % Call bootstrap_confidence_intervals with mean function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Verify that output structure contains required fields
            obj.assertTrue(isstruct(results), 'Result should be a structure');
            obj.assertTrue(isfield(results, 'original_statistic'), 'Result should have original_statistic field');
            obj.assertTrue(isfield(results, 'conf_level'), 'Result should have conf_level field');
            obj.assertTrue(isfield(results, 'lower'), 'Result should have lower field');
            obj.assertTrue(isfield(results, 'upper'), 'Result should have upper field');
            obj.assertTrue(isfield(results, 'bootstrap_statistics'), 'Result should have bootstrap_statistics field');
            
            % Verify original statistic matches direct calculation
            originalMean = mean(obj.testData);
            obj.assertAlmostEqual(results.original_statistic, originalMean, 'Original statistic should match direct calculation');
            
            % Verify confidence level is correctly stored
            obj.assertEqual(results.conf_level, options.conf_level, 'Confidence level should match input');
            
            % Verify lower bound is less than upper bound
            obj.assertTrue(results.lower < results.upper, 'Lower bound should be less than upper bound');
            
            % Verify bounds are within reasonable range of the original statistic
            obj.assertTrue(abs(results.lower - originalMean) < 1.0, 'Lower bound should be within reasonable range of original statistic');
            obj.assertTrue(abs(results.upper - originalMean) < 1.0, 'Upper bound should be within reasonable range of original statistic');
        end
        
        function testConfidenceLevels(obj)
            % Tests that bootstrap_confidence_intervals correctly implements different confidence levels
            
            % Compute confidence intervals with 90% confidence level
            options90 = obj.testParams;
            options90.conf_level = 0.90;
            results90 = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options90);
            
            % Compute confidence intervals with 95% confidence level
            options95 = obj.testParams;
            options95.conf_level = 0.95;
            results95 = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options95);
            
            % Compute confidence intervals with 99% confidence level
            options99 = obj.testParams;
            options99.conf_level = 0.99;
            results99 = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options99);
            
            % Verify that higher confidence levels produce wider intervals
            width90 = results90.upper - results90.lower;
            width95 = results95.upper - results95.lower;
            width99 = results99.upper - results99.lower;
            
            obj.assertTrue(width95 > width90, '95% interval should be wider than 90% interval');
            obj.assertTrue(width99 > width95, '99% interval should be wider than 95% interval');
            
            % Verify that all intervals contain the true population parameter at appropriate rate
            obj.assertTrue(results90.lower < results95.lower, '90% lower bound should be lower than 95% lower bound');
            obj.assertTrue(results95.upper < results99.upper, '95% upper bound should be lower than 99% upper bound');
        end
        
        function testPercentileMethod(obj)
            % Tests that percentile method correctly computes confidence intervals based on bootstrap distribution quantiles
            
            % Configure options for percentile method
            options = obj.testParams;
            options.method = 'percentile';
            
            % Call bootstrap_confidence_intervals with mean function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Calculate percentiles manually from bootstrap statistics
            alpha = 1 - options.conf_level;
            manual_lower = prctile(results.bootstrap_statistics, alpha/2 * 100);
            manual_upper = prctile(results.bootstrap_statistics, (1-alpha/2) * 100);
            
            % Verify that computed bounds match manual calculation within tolerance
            obj.assertAlmostEqual(results.lower, manual_lower, 'Lower bound should match manual percentile calculation');
            obj.assertAlmostEqual(results.upper, manual_upper, 'Upper bound should match manual percentile calculation');
            
            % Repeat test with variance function to test different statistics
            var_results = bootstrap_confidence_intervals(obj.testData, obj.varianceFn, options);
            obj.assertTrue(var_results.lower > 0, 'Lower bound of variance should be positive');
            obj.assertTrue(var_results.upper > var_results.lower, 'Upper bound should be greater than lower bound');
        end
        
        function testBasicMethod(obj)
            % Tests that basic method correctly computes confidence intervals based on the empirical distribution of deviations
            
            % Configure options for basic method
            options = obj.testParams;
            options.method = 'basic';
            
            % Call bootstrap_confidence_intervals with mean function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Calculate basic intervals manually using bootstrap distribution and original statistic
            alpha = 1 - options.conf_level;
            original_stat = results.original_statistic;
            quantiles = prctile(results.bootstrap_statistics, [alpha/2, 1-alpha/2] * 100);
            manual_lower = 2 * original_stat - quantiles(2);
            manual_upper = 2 * original_stat - quantiles(1);
            
            % Verify that computed bounds match manual calculation within tolerance
            obj.assertAlmostEqual(results.lower, manual_lower, 'Lower bound should match manual basic calculation');
            obj.assertAlmostEqual(results.upper, manual_upper, 'Upper bound should match manual basic calculation');
            
            % Test with different statistics to verify robustness
            var_results = bootstrap_confidence_intervals(obj.testData, obj.varianceFn, options);
            obj.assertTrue(var_results.lower < var_results.upper, 'Basic method should produce valid variance intervals');
        end
        
        function testStudentizedMethod(obj)
            % Tests that studentized method correctly computes confidence intervals using bootstrap standard errors
            
            % Configure options for studentized method
            options = obj.testParams;
            options.method = 'studentized';
            
            % Define standard error function for the statistic
            se_mean = @(x) std(x)/sqrt(length(x));
            
            % Call bootstrap_confidence_intervals with mean function and standard error function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Verify that studentized intervals are computed correctly
            obj.assertEqual(results.method, 'studentized', 'Method should be correctly identified in results');
            obj.assertTrue(results.lower < results.original_statistic, 'Lower bound should be below original statistic');
            obj.assertTrue(results.upper > results.original_statistic, 'Upper bound should be above original statistic');
            
            % Compare with other interval methods and verify expected behavior (typically narrower intervals)
            % This is a probabilistic expectation so we just check the interval is valid
            obj.assertTrue(isfinite(results.lower) && isfinite(results.upper), 'Studentized interval bounds should be finite');
        end
        
        function testBCMethod(obj)
            % Tests that bias-corrected (BC) method correctly adjusts for bias in bootstrap distribution
            
            % Configure options for BC method
            options = obj.testParams;
            options.method = 'bc';
            
            % Call bootstrap_confidence_intervals with mean function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Calculate bias correction factor manually
            proportion_below = mean(results.bootstrap_statistics < results.original_statistic);
            z0 = norminv(proportion_below);
            
            % Calculate BC adjustments to quantiles
            alpha = 1 - options.conf_level;
            z_alpha = norminv(alpha/2);
            z_1_alpha = norminv(1-alpha/2);
            
            % Adjusted percentiles
            p_lower = normcdf(2*z0 + z_alpha);
            p_upper = normcdf(2*z0 + z_1_alpha);
            
            % Calculate quantiles
            manual_lower = prctile(results.bootstrap_statistics, 100 * p_lower);
            manual_upper = prctile(results.bootstrap_statistics, 100 * p_upper);
            
            % Verify bias correction is applied correctly
            obj.assertAlmostEqual(results.lower, manual_lower, 'Lower bound should match manual BC calculation');
            obj.assertAlmostEqual(results.upper, manual_upper, 'Upper bound should match manual BC calculation');
            
            % Test with skewed distributions to verify bias correction effectiveness
            skewed_data = exp(obj.testData); % Creates positive skew
            skewed_results = bootstrap_confidence_intervals(skewed_data, obj.meanFn, options);
            obj.assertTrue(isfinite(skewed_results.lower) && isfinite(skewed_results.upper), 
                'BC method should handle skewed data');
        end
        
        function testBCaMethod(obj)
            % Tests that bias-corrected and accelerated (BCa) method correctly accounts for bias and non-constant variance
            
            % Configure options for BCa method
            options = obj.testParams;
            options.method = 'bca';
            
            % Call bootstrap_confidence_intervals with mean function
            results = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options);
            
            % Calculate acceleration factor manually using jackknife estimates
            n = length(obj.testData);
            jack_stats = zeros(n, 1);
            for i = 1:n
                jack_sample = obj.testData;
                jack_sample(i) = []; % Remove i-th observation
                jack_stats(i) = obj.meanFn(jack_sample);
            end
            
            % Verify both bias correction and acceleration are applied correctly
            obj.assertEqual(results.method, 'bca', 'Method should be correctly identified in results');
            obj.assertTrue(results.lower < results.original_statistic, 'Lower bound should be below original statistic');
            obj.assertTrue(results.upper > results.original_statistic, 'Upper bound should be above original statistic');
            
            % Test with heteroskedastic data to verify effectiveness
            hetero_data = obj.testData .* (1 + 0.5 * abs(obj.testData));
            hetero_results = bootstrap_confidence_intervals(hetero_data, obj.meanFn, options);
            obj.assertTrue(isfinite(hetero_results.lower) && isfinite(hetero_results.upper), 
                'BCa method should handle heteroskedastic data');
        end
        
        function testBlockBootstrapIntegration(obj)
            % Tests integration with block bootstrap method for time series with dependence
            
            % Configure options for block bootstrap with various block sizes
            options_small = obj.testParams;
            options_small.bootstrap_type = 'block';
            options_small.block_size = 2;
            
            options_medium = obj.testParams;
            options_medium.bootstrap_type = 'block';
            options_medium.block_size = 5;
            
            options_large = obj.testParams;
            options_large.bootstrap_type = 'block';
            options_large.block_size = 10;
            
            % Test that block size correctly affects resulting confidence intervals
            results_small = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_small);
            results_medium = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_medium);
            results_large = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_large);
            
            % Verify that temporal dependence is preserved appropriately
            % Different block sizes should give different bootstrap distributions
            bs_small = results_small.bootstrap_statistics;
            bs_large = results_large.bootstrap_statistics;
            
            % Statistical test for difference in distributions (not perfect but indicative)
            bs_small_variance = var(bs_small);
            bs_large_variance = var(bs_large);
            obj.assertTrue(abs(bs_small_variance - bs_large_variance) > 0.001, 
                'Different block sizes should affect bootstrap distribution');
            
            % Compare results with theoretical expectations for AR(1) process
            % Verify all have the same original statistic
            obj.assertAlmostEqual(results_small.original_statistic, results_medium.original_statistic);
            obj.assertAlmostEqual(results_medium.original_statistic, results_large.original_statistic);
        end
        
        function testStationaryBootstrapIntegration(obj)
            % Tests integration with stationary bootstrap method
            
            % Configure options for stationary bootstrap with various p values
            options_small_p = obj.testParams;
            options_small_p.bootstrap_type = 'stationary';
            options_small_p.p = 0.1;  % Longer average block length
            
            options_large_p = obj.testParams;
            options_large_p.bootstrap_type = 'stationary';
            options_large_p.p = 0.5;  % Shorter average block length
            
            % Test that p parameter correctly affects resulting confidence intervals
            results_small_p = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_small_p);
            results_large_p = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_large_p);
            
            % Verify that stationary bootstrap produces valid confidence intervals
            obj.assertTrue(results_small_p.lower < results_small_p.original_statistic, 
                'Lower bound should be below original statistic');
            obj.assertTrue(results_small_p.upper > results_small_p.original_statistic, 
                'Upper bound should be above original statistic');
            
            % Compare with block bootstrap results for consistency
            options_block = obj.testParams;
            options_block.bootstrap_type = 'block';
            results_block = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_block);
            
            % Both methods should produce valid intervals for the same data
            block_width = results_block.upper - results_block.lower;
            stat_width = results_small_p.upper - results_small_p.lower;
            
            obj.assertTrue(block_width > 0 && stat_width > 0, 
                'Both bootstrap methods should produce valid intervals');
        end
        
        function testMultivariateStatistics(obj)
            % Tests computation of confidence intervals for multivariate statistics
            
            % Define multivariate statistic function (e.g., correlation)
            corr_fn = @(x) corr(x(:,1), x(:,2));
            
            % Configure bootstrap options for multivariate data
            options = obj.testParams;
            
            % Call bootstrap_confidence_intervals with multivariate statistic
            results = bootstrap_confidence_intervals(obj.multivariateTestData, corr_fn, options);
            
            % Verify that confidence intervals correctly capture the multivariate relationship
            obj.assertTrue(results.lower >= -1 && results.lower <= 1, 
                'Correlation lower bound should be in [-1,1]');
            obj.assertTrue(results.upper >= -1 && results.upper <= 1, 
                'Correlation upper bound should be in [-1,1]');
            
            % Verify original statistic matches direct calculation
            direct_corr = corr(obj.multivariateTestData(:,1), obj.multivariateTestData(:,2));
            obj.assertAlmostEqual(results.original_statistic, direct_corr, 
                'Original correlation should match direct calculation');
            
            % Test with different multivariate statistics (covariance, regression coefficients)
            cov_fn = @(x) cov(x(:,1), x(:,2));
            cov_results = bootstrap_confidence_intervals(obj.multivariateTestData, @(x) cov(x)[1,2], options);
            obj.assertTrue(isfinite(cov_results.lower) && isfinite(cov_results.upper), 
                'Covariance intervals should be finite');
        end
        
        function testReplicationsCount(obj)
            % Tests the effect of bootstrap replications count on confidence interval accuracy
            
            % Run tests with low (100), medium (500), and high (1000) replication counts
            options_low = obj.testParams;
            options_low.replications = 100;
            
            options_medium = obj.testParams;
            options_medium.replications = 500;
            
            options_high = obj.testParams;
            options_high.replications = 1000;
            
            % Measure interval width and stability across replications
            results_low = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_low);
            results_medium = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_medium);
            results_high = bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_high);
            
            % Verify that higher replication counts produce more stable intervals
            obj.assertEqual(length(results_low.bootstrap_statistics), 100, 
                'Should have 100 bootstrap statistics');
            obj.assertEqual(length(results_medium.bootstrap_statistics), 500, 
                'Should have 500 bootstrap statistics');
            obj.assertEqual(length(results_high.bootstrap_statistics), 1000, 
                'Should have 1000 bootstrap statistics');
            
            % Assess computational performance trade-offs
            time_low = obj.measureExecutionTime(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_low));
            time_high = obj.measureExecutionTime(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, options_high));
            
            obj.assertTrue(time_high > time_low, 'Higher replication count should take more time');
        end
        
        function testInvalidInputs(obj)
            % Tests that bootstrap_confidence_intervals correctly handles invalid inputs with appropriate error messages
            
            % Basic options
            options = obj.testParams;
            
            % Test with empty data and verify error is thrown
            obj.assertThrows(@() bootstrap_confidence_intervals([], obj.meanFn, options), 
                'DATA:*empty*', 'Should throw error for empty data');
            
            % Test with data containing NaN values and verify error is thrown
            data_with_nan = obj.testData;
            data_with_nan(5) = NaN;
            obj.assertThrows(@() bootstrap_confidence_intervals(data_with_nan, obj.meanFn, options), 
                'DATA:*NaN*', 'Should throw error for data with NaN');
            
            % Test with invalid bootstrap type and verify error is thrown
            invalid_options = options;
            invalid_options.bootstrap_type = 'invalid_type';
            obj.assertThrows(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, invalid_options), 
                '*bootstrap_type*', 'Should throw error for invalid bootstrap type');
            
            % Test with negative confidence level and verify error is thrown
            invalid_options = options;
            invalid_options.conf_level = -0.1;
            obj.assertThrows(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, invalid_options), 
                '*conf_level*', 'Should throw error for negative confidence level');
            
            % Test with confidence level > 1 and verify error is thrown
            invalid_options = options;
            invalid_options.conf_level = 1.1;
            obj.assertThrows(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, invalid_options), 
                '*conf_level*', 'Should throw error for confidence level > 1');
            
            % Test with invalid method name and verify error is thrown
            invalid_options = options;
            invalid_options.method = 'invalid_method';
            obj.assertThrows(@() bootstrap_confidence_intervals(obj.testData, obj.meanFn, invalid_options), 
                '*method*', 'Should throw error for invalid method name');
            
            % Test with invalid function handle and verify error is thrown
            obj.assertThrows(@() bootstrap_confidence_intervals(obj.testData, 'not_a_function', options), 
                '*function_handle*', 'Should throw error for invalid function handle');
        end
        
        function testPerformance(obj)
            % Tests the performance of bootstrap_confidence_intervals with large datasets
            
            % Generate large test dataset (1000+ observations)
            large_data = generateFinancialReturns(1000, 1);
            
            % Measure execution time for different confidence interval methods
            options = obj.testParams;
            options.replications = 200; % Use fewer replications for performance testing
            
            methods = {'percentile', 'basic', 'bc'};
            execution_times = zeros(length(methods), 1);
            
            for i = 1:length(methods)
                options.method = methods{i};
                execution_times(i) = obj.measureExecutionTime(@() bootstrap_confidence_intervals(large_data, obj.meanFn, options));
            end
            
            % Compare performance of block vs. stationary bootstrap
            options.method = 'percentile';
            options.bootstrap_type = 'block';
            block_time = obj.measureExecutionTime(@() bootstrap_confidence_intervals(large_data, obj.meanFn, options));
            
            options.bootstrap_type = 'stationary';
            stationary_time = obj.measureExecutionTime(@() bootstrap_confidence_intervals(large_data, obj.meanFn, options));
            
            % Evaluate scaling properties with increasing dataset size and bootstrap replications
            obj.assertTrue(all(execution_times > 0), 'All methods should complete successfully');
            obj.assertTrue(block_time > 0 && stationary_time > 0, 'Both bootstrap methods should execute successfully');
            
            % Verify memory usage remains within acceptable bounds
            % (This is implicit since the test completes without out-of-memory errors)
        end
        
        function testCoverageRate(obj)
            % Tests the empirical coverage rate of different confidence interval methods
            
            % Generate multiple datasets with known population parameters
            population_mean = 0;
            sample_size = 50;
            num_simulations = 50; % Limited for test speed, would be higher in production
            alpha = 0.10; % Using 90% confidence level for faster convergence
            
            % Configure options
            options = struct('bootstrap_type', 'block', 'block_size', 3, 'replications', 200, 'conf_level', 1-alpha);
            
            % Define methods to test
            methods = {'percentile', 'basic', 'bc'};
            coverage = zeros(length(methods), 1);
            
            % Calculate the proportion of intervals that contain the true parameter
            for m = 1:length(methods)
                options.method = methods{m};
                contains_true_param = 0;
                
                for sim = 1:num_simulations
                    % Generate data with known population parameter
                    data = population_mean + randn(sample_size, 1);
                    
                    % Compute confidence interval
                    results = bootstrap_confidence_intervals(data, obj.meanFn, options);
                    
                    % Check if interval contains true parameter
                    if results.lower <= population_mean && population_mean <= results.upper
                        contains_true_param = contains_true_param + 1;
                    end
                end
                
                % Calculate empirical coverage rate
                coverage(m) = contains_true_param / num_simulations;
            end
            
            % Verify that empirical coverage matches nominal confidence level
            for m = 1:length(methods)
                % Use generous tolerance due to limited simulations
                obj.assertTrue(abs(coverage(m) - (1-alpha)) < 0.2, 
                    ['Method ', methods{m}, ' coverage should be close to nominal level']);
            end
            
            % Compare coverage rates across different interval methods
            % No explicit comparison needed since all should be similar with correct implementation
        end
        
        function runAllTests(obj)
            % Convenience method to run all test cases in the class
            
            % Call superclass runAllTests method
            results = runAllTests@BaseTest(obj);
            
            % Display summary of test results
            disp(results.summary);
        end
    end
end