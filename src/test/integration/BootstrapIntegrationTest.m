classdef BootstrapIntegrationTest < BaseTest
    % BootstrapIntegrationTest Integration test class for the Bootstrap component of the MFE Toolbox
    %
    % This test class verifies the correct integration between different bootstrap methods
    % and their application to financial time series analysis. It tests the Bootstrap 
    % Framework implementation, ensuring proper block formation, sampling methods, and
    % statistical consistency between different bootstrap approaches.
    %
    % The tests include:
    % - Integration of block bootstrap with variance estimation
    % - Integration of stationary bootstrap with variance estimation
    % - Confidence interval computation using different bootstrap methods
    % - Comparison between different bootstrap methods
    % - End-to-end testing of bootstrap analysis workflows with financial data
    % - Statistical validation of bootstrap distributions
    %
    % See also: BaseTest, block_bootstrap, stationary_bootstrap, bootstrap_variance,
    % bootstrap_confidence_intervals
    
    properties
        testData            % Simulated test data with known properties
        financialReturns    % Financial returns data for testing
        numBootstrapReplications % Number of bootstrap replications to use
        tolerance           % Tolerance for numerical comparisons
        statisticalProperties % Structure with original data statistical properties
        comparator          % NumericalComparator instance for test validations
    end
    
    methods
        function obj = BootstrapIntegrationTest()
            % Initialize the test class and set default values
            obj = obj@BaseTest();
            obj.tolerance = 1e-8;
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method
            setUp@BaseTest(obj);
            
            % Generate simulated time series data with known properties
            obj.testData = generateSimulatedData(struct('mean', 0.1, ...
                                                       'variance', 2, ...
                                                       'autocorr', [0.3, 0.2, 0.1]), 500);
            obj.testData = obj.testData.data;
            
            % Generate financial returns data for realistic testing
            obj.financialReturns = generateFinancialReturns(1000, 1);
            
            % Set number of bootstrap replications to use in tests
            obj.numBootstrapReplications = 1000;
            
            % Create NumericalComparator instance
            obj.comparator = NumericalComparator();
            
            % Store original statistical properties
            obj.statisticalProperties = struct();
            obj.statisticalProperties.mean = mean(obj.testData);
            obj.statisticalProperties.variance = var(obj.testData);
            obj.statisticalProperties.std = std(obj.testData);
        end
        
        function tearDown(obj)
            % Clean up after each test method execution
            obj.testData = [];
            obj.financialReturns = [];
            obj.statisticalProperties = [];
            tearDown@BaseTest(obj);
        end
        
        function testBlockBootstrapWithVariance(obj)
            % Tests the integration between block bootstrap and variance estimation
            
            % Set up block bootstrap options
            blockSize = 25; % Appropriate block size for test data
            options = struct('bootstrap_type', 'block', ...
                           'block_size', blockSize, ...
                           'replications', obj.numBootstrapReplications);
            
            % Define statistic function to calculate mean
            statFun = @(x) mean(x);
            
            % Generate bootstrap samples using block bootstrap
            bs_data = block_bootstrap(obj.testData, blockSize, obj.numBootstrapReplications);
            
            % Apply bootstrap_variance using block bootstrap samples
            var_results = bootstrap_variance(obj.testData, statFun, options);
            
            % Verify variance estimates are within expected range
            obj.assertTrue(var_results.variance > 0, 'Variance estimate should be positive');
            obj.assertTrue(var_results.std_error > 0, 'Standard error should be positive');
            
            % Verify standard errors are properly calculated
            bs_means = zeros(obj.numBootstrapReplications, 1);
            for i = 1:obj.numBootstrapReplications
                bs_means(i) = mean(bs_data(:,:,i));
            end
            expected_var = var(bs_means);
            obj.assertAlmostEqual(var_results.variance, expected_var, ...
                'Bootstrap variance should match variance of bootstrap means');
            
            % Verify confidence intervals have correct coverage
            obj.assertTrue(var_results.conf_lower < obj.statisticalProperties.mean, ...
                'Lower confidence bound should be below true mean');
            obj.assertTrue(var_results.conf_upper > obj.statisticalProperties.mean, ...
                'Upper confidence bound should be above true mean');
            
            % Assert that returned structure has all required fields
            expectedFields = {'variance', 'std_error', 'conf_lower', 'conf_upper', ...
                'bootstrap_stats', 'mean', 'median', 'min', 'max', 'q25', 'q75'};
            for i = 1:length(expectedFields)
                obj.assertTrue(isfield(var_results, expectedFields{i}), ...
                    ['Result should contain field ' expectedFields{i}]);
            end
        end
        
        function testStationaryBootstrapWithVariance(obj)
            % Tests the integration between stationary bootstrap and variance estimation
            
            % Set up stationary bootstrap options
            p = 0.04; % Expected block length = 1/p = 25
            options = struct('bootstrap_type', 'stationary', ...
                           'p', p, ...
                           'replications', obj.numBootstrapReplications);
            
            % Define statistic function to calculate mean
            statFun = @(x) mean(x);
            
            % Generate bootstrap samples using stationary bootstrap
            bs_data = stationary_bootstrap(obj.testData, p, obj.numBootstrapReplications);
            
            % Apply bootstrap_variance using stationary bootstrap samples
            var_results = bootstrap_variance(obj.testData, statFun, options);
            
            % Verify variance estimates are within expected range
            obj.assertTrue(var_results.variance > 0, 'Variance estimate should be positive');
            obj.assertTrue(var_results.std_error > 0, 'Standard error should be positive');
            
            % Verify standard errors are properly calculated
            bs_means = zeros(obj.numBootstrapReplications, 1);
            for i = 1:obj.numBootstrapReplications
                bs_means(i) = mean(bs_data(:,:,i));
            end
            expected_var = var(bs_means);
            obj.assertAlmostEqual(var_results.variance, expected_var, ...
                'Bootstrap variance should match variance of bootstrap means');
            
            % Verify confidence intervals have correct coverage
            obj.assertTrue(var_results.conf_lower < obj.statisticalProperties.mean, ...
                'Lower confidence bound should be below true mean');
            obj.assertTrue(var_results.conf_upper > obj.statisticalProperties.mean, ...
                'Upper confidence bound should be above true mean');
            
            % Assert that returned structure has all required fields
            expectedFields = {'variance', 'std_error', 'conf_lower', 'conf_upper', ...
                'bootstrap_stats', 'mean', 'median', 'min', 'max', 'q25', 'q75'};
            for i = 1:length(expectedFields)
                obj.assertTrue(isfield(var_results, expectedFields{i}), ...
                    ['Result should contain field ' expectedFields{i}]);
            end
        end
        
        function testBlockBootstrapWithConfidenceIntervals(obj)
            % Tests the integration between block bootstrap and confidence interval estimation
            
            % Set up block bootstrap options
            blockSize = 25; % Appropriate block size for test data
            options = struct('bootstrap_type', 'block', ...
                          'block_size', blockSize, ...
                          'replications', obj.numBootstrapReplications, ...
                          'conf_level', 0.95);
            
            % Define statistic function to calculate mean
            statFun = @(x) mean(x);
            
            % Generate bootstrap samples
            bs_data = block_bootstrap(obj.testData, blockSize, obj.numBootstrapReplications);
            
            % Apply bootstrap_confidence_intervals with different methods
            methods = {'percentile', 'basic', 'studentized'};
            
            for i = 1:length(methods)
                % Set method for this iteration
                options.method = methods{i};
                
                % Calculate confidence intervals
                ci_results = bootstrap_confidence_intervals(obj.testData, statFun, options);
                
                % Check that original statistic is returned correctly
                obj.assertAlmostEqual(ci_results.original_statistic, mean(obj.testData), ...
                    'Original statistic should match data mean');
                
                % Verify confidence intervals contain original statistic
                obj.assertTrue(ci_results.lower < obj.statisticalProperties.mean && ...
                              ci_results.upper > obj.statisticalProperties.mean, ...
                    ['Confidence interval for ' methods{i} ' method should contain true mean']);
                
                % Verify confidence intervals have expected width relative to data variability
                expected_width = 2 * 1.96 * sqrt(obj.statisticalProperties.variance / length(obj.testData));
                ci_width = ci_results.upper - ci_results.lower;
                
                % The ratio should be approximately 1, but bootstrap CI widths can vary
                ratio = ci_width / expected_width;
                obj.assertTrue(ratio > 0.5 && ratio < 2.0, ...
                    ['CI width for ' methods{i} ' should be reasonable compared to expected width']);
                
                % Verify different methods have expected structure
                obj.assertEqual(ci_results.method, methods{i}, ...
                    'Method in results should match requested method');
                
                % Assert that returned structure has all required fields
                expectedFields = {'original_statistic', 'conf_level', 'lower', 'upper', ...
                    'bootstrap_statistics', 'method', 'bootstrap_options'};
                for j = 1:length(expectedFields)
                    obj.assertTrue(isfield(ci_results, expectedFields{j}), ...
                        ['Result should contain field ' expectedFields{j}]);
                end
            end
        end
        
        function testStationaryBootstrapWithConfidenceIntervals(obj)
            % Tests the integration between stationary bootstrap and confidence interval estimation
            
            % Set up stationary bootstrap options
            p = 0.04; % Expected block length = 1/p = 25
            options = struct('bootstrap_type', 'stationary', ...
                          'p', p, ...
                          'replications', obj.numBootstrapReplications, ...
                          'conf_level', 0.95);
            
            % Define statistic function to calculate mean
            statFun = @(x) mean(x);
            
            % Generate bootstrap samples
            bs_data = stationary_bootstrap(obj.testData, p, obj.numBootstrapReplications);
            
            % Apply bootstrap_confidence_intervals with different methods
            methods = {'percentile', 'basic', 'studentized'};
            
            for i = 1:length(methods)
                % Set method for this iteration
                options.method = methods{i};
                
                % Calculate confidence intervals
                ci_results = bootstrap_confidence_intervals(obj.testData, statFun, options);
                
                % Check that original statistic is returned correctly
                obj.assertAlmostEqual(ci_results.original_statistic, mean(obj.testData), ...
                    'Original statistic should match data mean');
                
                % Verify confidence intervals contain original statistic
                obj.assertTrue(ci_results.lower < obj.statisticalProperties.mean && ...
                              ci_results.upper > obj.statisticalProperties.mean, ...
                    ['Confidence interval for ' methods{i} ' method should contain true mean']);
                
                % Verify confidence intervals have expected width relative to data variability
                expected_width = 2 * 1.96 * sqrt(obj.statisticalProperties.variance / length(obj.testData));
                ci_width = ci_results.upper - ci_results.lower;
                
                % The ratio should be approximately 1, but bootstrap CI widths can vary
                ratio = ci_width / expected_width;
                obj.assertTrue(ratio > 0.5 && ratio < 2.0, ...
                    ['CI width for ' methods{i} ' should be reasonable compared to expected width']);
                
                % Verify different methods have expected structure
                obj.assertEqual(ci_results.method, methods{i}, ...
                    'Method in results should match requested method');
                
                % Assert that returned structure has all required fields
                expectedFields = {'original_statistic', 'conf_level', 'lower', 'upper', ...
                    'bootstrap_statistics', 'method', 'bootstrap_options'};
                for j = 1:length(expectedFields)
                    obj.assertTrue(isfield(ci_results, expectedFields{j}), ...
                        ['Result should contain field ' expectedFields{j}]);
                end
            end
        end
        
        function testCompareDifferentBootstrapMethods(obj)
            % Compares results from block and stationary bootstrap methods to verify consistency
            
            % Define statistic function to calculate mean
            statFun = @(x) mean(x);
            
            % Set up block bootstrap options
            blockSize = 25;
            block_options = struct('bootstrap_type', 'block', ...
                                 'block_size', blockSize, ...
                                 'replications', obj.numBootstrapReplications);
            
            % Set up stationary bootstrap options
            p = 0.04; % Expected block length = 1/p = 25
            stat_options = struct('bootstrap_type', 'stationary', ...
                                'p', p, ...
                                'replications', obj.numBootstrapReplications);
            
            % Apply both bootstrap variance methods
            block_var_results = bootstrap_variance(obj.testData, statFun, block_options);
            stat_var_results = bootstrap_variance(obj.testData, statFun, stat_options);
            
            % Compare bootstrap means from both methods
            obj.assertAlmostEqual(block_var_results.mean, stat_var_results.mean, ...
                'Mean bootstrap statistics should be similar between methods');
            
            % Compare bootstrap variances from both methods
            % Allow for some difference as methods are not identical
            relativeDiff = abs(block_var_results.variance - stat_var_results.variance) / ...
                           max(block_var_results.variance, stat_var_results.variance);
            obj.assertTrue(relativeDiff < 0.2, ...
                'Variance estimates should be reasonably similar between methods');
            
            % Compare confidence intervals
            block_width = block_var_results.conf_upper - block_var_results.conf_lower;
            stat_width = stat_var_results.conf_upper - stat_var_results.conf_lower;
            width_ratio = block_width / stat_width;
            
            obj.assertTrue(width_ratio > 0.8 && width_ratio < 1.25, ...
                'Confidence interval widths should be similar between methods');
            
            % Now test confidence interval methods directly
            block_options.method = 'percentile';
            stat_options.method = 'percentile';
            
            block_ci = bootstrap_confidence_intervals(obj.testData, statFun, block_options);
            stat_ci = bootstrap_confidence_intervals(obj.testData, statFun, stat_options);
            
            % Compare confidence interval bounds
            obj.assertTrue(obj.comparator.compareScalars(block_ci.lower, stat_ci.lower, 0.1).isEqual, ...
                'Lower confidence bounds should be reasonably similar');
            obj.assertTrue(obj.comparator.compareScalars(block_ci.upper, stat_ci.upper, 0.1).isEqual, ...
                'Upper confidence bounds should be reasonably similar');
        end
        
        function testEndToEndBootstrapWorkflow(obj)
            % Tests a complete bootstrap analysis workflow from sampling to confidence intervals
            
            % Set up simulated financial time series with known properties
            % Use autocorrelated data to test bootstrap's ability to preserve dependence
            acf = [0.4, 0.3, 0.2, 0.1];
            simData = generateSimulatedData(struct('mean', 0.05, ...
                                                  'variance', 1.5, ...
                                                  'autocorr', acf), 400);
            testSeries = simData.data;
            
            % Define statistic function - use multiple statistics
            meanFun = @(x) mean(x);
            varFun = @(x) var(x);
            
            % Configure bootstrap options for both methods
            blockSize = 20;
            p = 0.05; % Expected block length = 1/p = 20
            
            % 1. Start workflow with bootstrap sampling
            block_samples = block_bootstrap(testSeries, blockSize, obj.numBootstrapReplications);
            stat_samples = stationary_bootstrap(testSeries, p, obj.numBootstrapReplications);
            
            % 2. Calculate bootstrap statistics manually from samples
            block_means = zeros(obj.numBootstrapReplications, 1);
            block_vars = zeros(obj.numBootstrapReplications, 1);
            stat_means = zeros(obj.numBootstrapReplications, 1);
            stat_vars = zeros(obj.numBootstrapReplications, 1);
            
            for i = 1:obj.numBootstrapReplications
                block_means(i) = meanFun(block_samples(:,:,i));
                block_vars(i) = varFun(block_samples(:,:,i));
                stat_means(i) = meanFun(stat_samples(:,:,i));
                stat_vars(i) = varFun(stat_samples(:,:,i));
            end
            
            % 3. Calculate variance of bootstrap statistics
            block_mean_var = var(block_means);
            stat_mean_var = var(stat_means);
            
            % 4. Apply bootstrap variance estimation
            block_options = struct('bootstrap_type', 'block', ...
                                 'block_size', blockSize, ...
                                 'replications', obj.numBootstrapReplications);
            stat_options = struct('bootstrap_type', 'stationary', ...
                                'p', p, ...
                                'replications', obj.numBootstrapReplications);
            
            block_var_results = bootstrap_variance(testSeries, meanFun, block_options);
            stat_var_results = bootstrap_variance(testSeries, meanFun, stat_options);
            
            % 5. Apply confidence interval estimation
            block_options.method = 'percentile';
            stat_options.method = 'percentile';
            
            block_ci = bootstrap_confidence_intervals(testSeries, meanFun, block_options);
            stat_ci = bootstrap_confidence_intervals(testSeries, meanFun, stat_options);
            
            % 6. Verify consistency of results through the workflow
            % Compare manual variance calculation with bootstrap_variance results
            obj.assertAlmostEqual(block_mean_var, block_var_results.variance, ...
                'Manual calculation of block bootstrap variance should match bootstrap_variance result');
            obj.assertAlmostEqual(stat_mean_var, stat_var_results.variance, ...
                'Manual calculation of stationary bootstrap variance should match bootstrap_variance result');
            
            % Verify bootstrap distribution properties preserved key data properties
            % Test that bootstrap means are centered near the original mean
            original_mean = mean(testSeries);
            bootstrap_mean_of_means = mean([block_means; stat_means]);
            obj.assertAlmostEqual(original_mean, bootstrap_mean_of_means, 0.1, ...
                'Bootstrap distribution should preserve original mean');
            
            % Verify that confidence intervals contain the original statistic
            obj.assertTrue(block_ci.lower < original_mean && block_ci.upper > original_mean, ...
                'Block bootstrap CI should contain the original mean');
            obj.assertTrue(stat_ci.lower < original_mean && stat_ci.upper > original_mean, ...
                'Stationary bootstrap CI should contain the original mean');
            
            % Verify workflow produces expected statistical accuracy
            % Compare CI width with theoretical width (assuming independence)
            theoretical_width = 2 * 1.96 * sqrt(var(testSeries)/length(testSeries));
            block_width = block_ci.upper - block_ci.lower;
            stat_width = stat_ci.upper - stat_ci.lower;
            
            % Bootstrap CIs for dependent data should be wider than theoretical width for independent data
            obj.assertTrue(block_width > theoretical_width, ...
                'Block bootstrap CI width should account for dependence');
            obj.assertTrue(stat_width > theoretical_width, ...
                'Stationary bootstrap CI width should account for dependence');
        end
        
        function testBootstrapWithFinancialData(obj)
            % Tests bootstrap methods with realistic financial return data
            
            % Use generated financial returns data
            returns = obj.financialReturns;
            
            % Define statistic functions relevant to financial data
            meanFun = @(x) mean(x);
            sharpeRatio = @(x) mean(x) / std(x);
            
            % Set up bootstrap options
            blockSize = 22; % Approximately one month of trading days
            p = 0.05;       % Expected block length = 1/p = 20 (similar to blockSize)
            
            % Block bootstrap configuration
            block_options = struct('bootstrap_type', 'block', ...
                                 'block_size', blockSize, ...
                                 'replications', obj.numBootstrapReplications);
            
            % Stationary bootstrap configuration
            stat_options = struct('bootstrap_type', 'stationary', ...
                                'p', p, ...
                                'replications', obj.numBootstrapReplications);
            
            % Test for mean estimation
            block_mean_results = bootstrap_variance(returns, meanFun, block_options);
            stat_mean_results = bootstrap_variance(returns, meanFun, stat_options);
            
            % Test for Sharpe ratio estimation
            block_sharpe_results = bootstrap_variance(returns, sharpeRatio, block_options);
            stat_sharpe_results = bootstrap_variance(returns, sharpeRatio, stat_options);
            
            % Verify results preserve key statistical properties
            % Both methods should produce similar bootstrap distributions
            mean_diff = abs(block_mean_results.mean - stat_mean_results.mean);
            sharpe_diff = abs(block_sharpe_results.mean - stat_sharpe_results.mean);
            
            obj.assertTrue(mean_diff < 0.05 * abs(block_mean_results.mean), ...
                'Mean estimates should be similar between bootstrap methods');
            obj.assertTrue(sharpe_diff < 0.05 * abs(block_sharpe_results.mean), ...
                'Sharpe ratio estimates should be similar between bootstrap methods');
            
            % Test robustness to heteroskedasticity and autocorrelation
            % Compute autocorrelation in squared returns (ARCH effect)
            sq_returns = returns.^2;
            lag1_autocorr = corr(sq_returns(1:end-1), sq_returns(2:end));
            
            % If there's significant ARCH effect, bootstrap should capture it
            if abs(lag1_autocorr) > 0.1
                % Check if bootstrap variance properly accounts for this
                theoretical_stderr = std(returns)/sqrt(length(returns));
                
                % Bootstrap standard errors should typically be larger than theoretical 
                % if ARCH effects are present
                obj.assertTrue(block_mean_results.std_error >= 0.9 * theoretical_stderr, ...
                    'Block bootstrap should account for heteroskedasticity');
                obj.assertTrue(stat_mean_results.std_error >= 0.9 * theoretical_stderr, ...
                    'Stationary bootstrap should account for heteroskedasticity');
            end
            
            % Test that confidence intervals are consistent across methods
            block_options.method = 'percentile';
            stat_options.method = 'percentile';
            
            block_ci = bootstrap_confidence_intervals(returns, meanFun, block_options);
            stat_ci = bootstrap_confidence_intervals(returns, meanFun, stat_options);
            
            % Compare confidence interval bounds
            lower_diff = abs(block_ci.lower - stat_ci.lower);
            upper_diff = abs(block_ci.upper - stat_ci.upper);
            ci_width = max(block_ci.upper - block_ci.lower, stat_ci.upper - stat_ci.lower);
            
            obj.assertTrue(lower_diff < 0.2 * ci_width, ...
                'Lower confidence bounds should be reasonably similar between methods');
            obj.assertTrue(upper_diff < 0.2 * ci_width, ...
                'Upper confidence bounds should be reasonably similar between methods');
        end
        
        function testBootstrapStatisticDistribution(obj)
            % Tests the distribution properties of bootstrap statistics
            
            % Generate data with specific distribution properties
            simData = generateSimulatedData(struct('mean', 0.2, ...
                                                  'variance', 1.2, ...
                                                  'skewness', -0.3, ...
                                                  'kurtosis', 4), 500);
            testData = simData.data;
            
            % Get original data properties
            orig_mean = mean(testData);
            orig_var = var(testData);
            
            % Define multiple test statistics
            meanFun = @(x) mean(x);
            varFun = @(x) var(x);
            
            % Generate bootstrap samples using both methods
            blockSize = 25;
            p = 0.04;
            block_samples = block_bootstrap(testData, blockSize, obj.numBootstrapReplications);
            stat_samples = stationary_bootstrap(testData, p, obj.numBootstrapReplications);
            
            % Calculate multiple test statistics from bootstrap samples
            block_means = zeros(obj.numBootstrapReplications, 1);
            block_vars = zeros(obj.numBootstrapReplications, 1);
            stat_means = zeros(obj.numBootstrapReplications, 1);
            stat_vars = zeros(obj.numBootstrapReplications, 1);
            
            for i = 1:obj.numBootstrapReplications
                block_means(i) = meanFun(block_samples(:,:,i));
                block_vars(i) = varFun(block_samples(:,:,i));
                stat_means(i) = meanFun(stat_samples(:,:,i));
                stat_vars(i) = varFun(stat_samples(:,:,i));
            end
            
            % Analyze distribution properties
            % Mean of bootstrap distributions should be close to original statistics
            obj.assertAlmostEqual(mean(block_means), orig_mean, 0.1, ...
                'Mean of block bootstrap means should be close to original mean');
            obj.assertAlmostEqual(mean(stat_means), orig_mean, 0.1, ...
                'Mean of stationary bootstrap means should be close to original mean');
            
            % Standard error of means should be related to original std dev
            expected_se = sqrt(orig_var / length(testData));  % Theoretical standard error
            
            % For dependent data, bootstrap SE might be different from theoretical SE
            % but they should be in a reasonable range
            block_se = std(block_means);
            stat_se = std(stat_means);
            
            obj.assertTrue(block_se > 0.5 * expected_se && block_se < 2.0 * expected_se, ...
                'Block bootstrap SE should be in reasonable range of theoretical SE');
            obj.assertTrue(stat_se > 0.5 * expected_se && stat_se < 2.0 * expected_se, ...
                'Stationary bootstrap SE should be in reasonable range of theoretical SE');
            
            % Verify bootstrap distribution for variances
            % Mean of bootstrap variances should be related to original variance
            obj.assertTrue(obj.verifyBootstrapDistributionProperties(testData, block_samples), ...
                'Block bootstrap should preserve key distribution properties');
            obj.assertTrue(obj.verifyBootstrapDistributionProperties(testData, stat_samples), ...
                'Stationary bootstrap should preserve key distribution properties');
            
            % Test consistency between different bootstrap methods
            % Using Kolmogorov-Smirnov test or similar could be ideal here,
            % but for simplicity, we'll just compare the moments
            mean_diff = abs(mean(block_means) - mean(stat_means)) / max(abs(mean(block_means)), abs(mean(stat_means)));
            var_diff = abs(var(block_means) - var(stat_means)) / max(var(block_means), var(stat_means));
            
            obj.assertTrue(mean_diff < 0.1, ...
                'Mean of bootstrap means should be similar between methods');
            obj.assertTrue(var_diff < 0.2, ...
                'Variance of bootstrap means should be similar between methods');
        end
        
        %% Helper methods
        
        function coverageRate = verifyCoverageRate(obj, trueValue, lowerBounds, upperBounds, targetCoverage)
            % Helper method to verify the coverage rate of bootstrap confidence intervals
            
            % Count how many intervals contain the true value
            containsTrue = (lowerBounds <= trueValue) & (upperBounds >= trueValue);
            
            % Calculate coverage rate
            coverageRate = mean(containsTrue);
            
            % The coverage rate should be close to the target coverage
            obj.assertTrue(abs(coverageRate - targetCoverage) < 0.1, ...
                sprintf('Coverage rate (%.3f) should be close to target (%.3f)', coverageRate, targetCoverage));
        end
        
        function isValid = verifyBootstrapDistributionProperties(obj, originalData, bootstrapSamples)
            % Helper method to verify the statistical properties of bootstrap distributions
            
            % Get original data properties
            orig_mean = mean(originalData);
            orig_var = var(originalData);
            
            % Compute bootstrap sample statistics
            [~, ~, numSamples] = size(bootstrapSamples);
            bs_means = zeros(numSamples, 1);
            bs_vars = zeros(numSamples, 1);
            
            for i = 1:numSamples
                bs_means(i) = mean(bootstrapSamples(:,:,i));
                bs_vars(i) = var(bootstrapSamples(:,:,i));
            end
            
            % Compare first moment (mean)
            mean_of_means = mean(bs_means);
            mean_diff = abs(mean_of_means - orig_mean) / max(abs(orig_mean), eps);
            mean_valid = mean_diff < 0.1;
            
            % Compare second moment (variance)
            mean_of_vars = mean(bs_vars);
            var_diff = abs(mean_of_vars - orig_var) / max(abs(orig_var), eps);
            var_valid = var_diff < 0.2;
            
            % Overall validation
            isValid = mean_valid && var_valid;
        end
        
        function result = calculateMeanStatistic(obj, data)
            % Helper method to calculate mean statistic for bootstrap tests
            result = mean(data);
        end
        
        function result = calculateVarianceStatistic(obj, data)
            % Helper method to calculate variance statistic for bootstrap tests
            result = var(data);
        end
    end
end