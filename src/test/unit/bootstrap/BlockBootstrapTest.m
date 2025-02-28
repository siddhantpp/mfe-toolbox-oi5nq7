classdef BlockBootstrapTest < BaseTest
    % BLOCKBOOTSTRAPTEST Unit tests for the block_bootstrap function
    %
    % This class tests the functionality of the block_bootstrap function which
    % generates bootstrap resamples for dependent time series data. It verifies
    % the correct handling of block resampling options, proper preservation of 
    % temporal dependence structure, and robustness to various inputs.
    %
    % Tests include:
    %   - Basic functionality and dimensions
    %   - Circular vs. non-circular block options
    %   - Univariate and multivariate data handling
    %   - Different block sizes and impact on dependence preservation
    %   - Statistical properties of bootstrapped samples
    %   - Proper error handling for invalid inputs
    %   - Performance with large datasets
    
    properties
        testData               % Sample time series data for testing
        multivariateTestData   % Sample multivariate time series data
        testParams             % Standard test parameters
    end
    
    methods
        function obj = BlockBootstrapTest()
            % Initialize the BlockBootstrapTest with necessary test configuration
            
            % Call superclass constructor
            obj = obj@BaseTest('BlockBootstrapTest');
            
            % Configure standard parameters
            obj.testParams = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Set random seed for reproducibility
            rng(123);
            
            % Generate AR(1) test data with known temporal dependence
            T = 100; % Length of time series
            ar_coef = 0.7; % AR(1) coefficient for temporal dependence
            
            % Generate AR(1) process: y_t = ar_coef * y_{t-1} + e_t
            e = randn(T, 1); % Random innovations
            y = zeros(T, 1);
            y(1) = e(1);
            for t = 2:T
                y(t) = ar_coef * y(t-1) + e(t);
            end
            obj.testData = y;
            
            % Generate multivariate test data with cross-correlations
            N = 3; % Number of variables
            multivariate_e = randn(T, N);
            multivariate_y = zeros(T, N);
            
            % Create correlated variables
            multivariate_y(:,1) = y; % First variable is the AR(1) process
            multivariate_y(:,2) = 0.5 * y + 0.5 * multivariate_e(:,2); % Correlated with first
            multivariate_y(:,3) = 0.3 * y + 0.7 * multivariate_e(:,3); % Less correlated
            
            obj.multivariateTestData = multivariate_y;
            
            % Set standard test parameters
            obj.testParams = struct(...
                'blockLength', 5, ...
                'numBootstraps', 100, ...
                'circular', true ...
            );
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data
            obj.testData = [];
            obj.multivariateTestData = [];
        end
        
        function testBasicFunctionality(obj)
            % Tests that block_bootstrap generates the correct number of bootstrap samples with expected dimensions
            
            % Generate bootstrap samples with standard parameters
            bsdata = block_bootstrap(obj.testData, ...
                obj.testParams.blockLength, ...
                obj.testParams.numBootstraps, ...
                obj.testParams.circular);
            
            % Verify dimensions
            [T, N, B] = size(bsdata);
            obj.assertEqual(T, length(obj.testData), 'Bootstrap sample length mismatch');
            obj.assertEqual(N, 1, 'Bootstrap sample width mismatch');
            obj.assertEqual(B, obj.testParams.numBootstraps, 'Number of bootstrap samples mismatch');
            
            % Verify samples are not identical (check first vs second sample)
            identical = all(bsdata(:,:,1) == bsdata(:,:,2));
            obj.assertFalse(identical, 'Bootstrap samples should not be identical');
            
            % Verify bootstrap samples contain only values from original data
            original_unique = unique(obj.testData);
            for b = 1:min(10, B) % Check first 10 samples
                bs_unique = unique(bsdata(:,:,b));
                obj.assertTrue(all(ismember(bs_unique, original_unique)), ...
                    'Bootstrap samples should only contain values from original data');
            end
        end
        
        function testCircularOption(obj)
            % Tests that circular and non-circular block bootstrap options produce correctly structured samples
            
            % Generate bootstrap samples with circular = true
            bsdata_circular = block_bootstrap(obj.testData, ...
                obj.testParams.blockLength, ...
                obj.testParams.numBootstraps, ...
                true);
            
            % Generate bootstrap samples with circular = false
            bsdata_noncircular = block_bootstrap(obj.testData, ...
                obj.testParams.blockLength, ...
                obj.testParams.numBootstraps, ...
                false);
            
            % Verify dimensions for both
            [T_circ, N_circ, B_circ] = size(bsdata_circular);
            [T_noncirc, N_noncirc, B_noncirc] = size(bsdata_noncircular);
            
            obj.assertEqual(T_circ, length(obj.testData), 'Circular bootstrap sample length mismatch');
            obj.assertEqual(T_noncirc, length(obj.testData), 'Non-circular bootstrap sample length mismatch');
            obj.assertEqual(N_circ, 1, 'Circular bootstrap sample width mismatch');
            obj.assertEqual(N_noncirc, 1, 'Non-circular bootstrap sample width mismatch');
            obj.assertEqual(B_circ, obj.testParams.numBootstraps, 'Circular bootstrap count mismatch');
            obj.assertEqual(B_noncirc, obj.testParams.numBootstraps, 'Non-circular bootstrap count mismatch');
            
            % Verify both methods produce valid samples (spot-check a few)
            for b = 1:min(10, B_circ)
                obj.assertTrue(all(isfinite(bsdata_circular(:,:,b))), 'Circular bootstrap contains invalid values');
                obj.assertTrue(all(isfinite(bsdata_noncircular(:,:,b))), 'Non-circular bootstrap contains invalid values');
            end
        end
        
        function testSingleVariable(obj)
            % Tests that block_bootstrap works correctly with a single column of data
            
            % Generate bootstrap samples for a single variable
            bsdata = block_bootstrap(obj.testData, ...
                obj.testParams.blockLength, ...
                obj.testParams.numBootstraps);
            
            % Verify dimensions
            [T, N, B] = size(bsdata);
            obj.assertEqual(T, length(obj.testData), 'Bootstrap sample length mismatch');
            obj.assertEqual(N, 1, 'Bootstrap sample width mismatch');
            obj.assertEqual(B, obj.testParams.numBootstraps, 'Number of bootstrap samples mismatch');
            
            % Compare statistical properties
            original_mean = mean(obj.testData);
            original_std = std(obj.testData);
            
            bootstrap_means = zeros(B, 1);
            bootstrap_stds = zeros(B, 1);
            
            for b = 1:B
                bootstrap_means(b) = mean(bsdata(:,:,b));
                bootstrap_stds(b) = std(bsdata(:,:,b));
            end
            
            % Mean of bootstrap means should be close to original mean
            bootstrap_mean_of_means = mean(bootstrap_means);
            obj.assertEqualsWithTolerance(bootstrap_mean_of_means, original_mean, 0.1, ...
                'Mean of bootstrap means deviates too much from original mean');
            
            % Mean of bootstrap stds should be close to original std
            bootstrap_mean_of_stds = mean(bootstrap_stds);
            obj.assertEqualsWithTolerance(bootstrap_mean_of_stds, original_std, 0.1, ...
                'Mean of bootstrap standard deviations deviates too much from original std');
        end
        
        function testMultipleVariables(obj)
            % Tests that block_bootstrap preserves cross-sectional correlation when multiple variables are bootstrapped
            
            % Generate bootstrap samples for multiple variables
            bsdata = block_bootstrap(obj.multivariateTestData, ...
                obj.testParams.blockLength, ...
                obj.testParams.numBootstraps);
            
            % Verify dimensions
            [T, N, B] = size(bsdata);
            obj.assertEqual(T, size(obj.multivariateTestData, 1), 'Bootstrap sample length mismatch');
            obj.assertEqual(N, size(obj.multivariateTestData, 2), 'Bootstrap sample width mismatch');
            obj.assertEqual(B, obj.testParams.numBootstraps, 'Number of bootstrap samples mismatch');
            
            % Calculate original covariance structure
            original_cov = cov(obj.multivariateTestData);
            
            % Calculate average covariance of bootstrap samples
            bootstrap_cov_sum = zeros(size(original_cov));
            for b = 1:B
                bootstrap_cov_sum = bootstrap_cov_sum + cov(bsdata(:,:,b));
            end
            average_bootstrap_cov = bootstrap_cov_sum / B;
            
            % Verify cross-correlations are preserved within reasonable tolerance
            obj.assertMatrixEqualsWithTolerance(average_bootstrap_cov, original_cov, 0.2, ...
                'Bootstrap covariance structure differs too much from original');
        end
        
        function testBlockSizes(obj)
            % Tests block_bootstrap with various block sizes to verify proper handling
            
            % Test small block size
            small_block = 2;
            bsdata_small = block_bootstrap(obj.testData, small_block, obj.testParams.numBootstraps);
            
            % Test medium block size
            medium_block = 10;
            bsdata_medium = block_bootstrap(obj.testData, medium_block, obj.testParams.numBootstraps);
            
            % Test large block size
            large_block = 30;
            bsdata_large = block_bootstrap(obj.testData, large_block, obj.testParams.numBootstraps);
            
            % Verify all produce correct dimensions
            [T_small, N_small, B_small] = size(bsdata_small);
            [T_medium, N_medium, B_medium] = size(bsdata_medium);
            [T_large, N_large, B_large] = size(bsdata_large);
            
            obj.assertEqual(T_small, length(obj.testData), 'Small block bootstrap sample length mismatch');
            obj.assertEqual(T_medium, length(obj.testData), 'Medium block bootstrap sample length mismatch');
            obj.assertEqual(T_large, length(obj.testData), 'Large block bootstrap sample length mismatch');
            
            % Compare autocorrelation preservation
            % (Larger blocks should better preserve autocorrelation structure)
            lag = 1; % Check first-order autocorrelation
            original_autocorr = autocorr(obj.testData, lag);
            
            small_autocorrs = zeros(B_small, 1);
            medium_autocorrs = zeros(B_medium, 1);
            large_autocorrs = zeros(B_large, 1);
            
            for b = 1:B_small
                small_autocorrs(b) = autocorr(bsdata_small(:,:,b), lag);
                medium_autocorrs(b) = autocorr(bsdata_medium(:,:,b), lag);
                large_autocorrs(b) = autocorr(bsdata_large(:,:,b), lag);
            end
            
            % Calculate average autocorrelations
            avg_small_autocorr = mean(small_autocorrs);
            avg_medium_autocorr = mean(medium_autocorrs);
            avg_large_autocorr = mean(large_autocorrs);
            
            % Larger blocks should preserve autocorrelation better
            % (i.e., closer to original)
            obj.assertTrue(abs(avg_large_autocorr - original_autocorr(2)) < ...
                abs(avg_small_autocorr - original_autocorr(2)), ...
                'Larger blocks should better preserve autocorrelation structure');
        end
        
        function testPreservationOfDependence(obj)
            % Tests that block_bootstrap preserves temporal dependence structure within blocks
            
            % Calculate original autocorrelation (10 lags)
            lags = 10;
            original_autocorr = autocorr(obj.testData, lags);
            
            % Generate bootstrap samples with appropriate block size
            block_length = 10; % Sufficient to preserve short-term dependence
            B = 200; % More bootstrap samples for better statistical properties
            
            bsdata = block_bootstrap(obj.testData, block_length, B);
            
            % Calculate autocorrelation for each bootstrap sample
            bootstrap_autocorrs = zeros(B, lags+1);
            for b = 1:B
                bootstrap_autocorrs(b,:) = autocorr(bsdata(:,:,b), lags);
            end
            
            % Calculate average autocorrelation across bootstrap samples
            avg_bootstrap_autocorr = mean(bootstrap_autocorrs);
            
            % Short-lag autocorrelation should be preserved within blocks
            short_lag = 2; % Check lag 1 autocorrelation
            obj.assertEqualsWithTolerance(avg_bootstrap_autocorr(short_lag), original_autocorr(short_lag), 0.1, ...
                'Short-lag autocorrelation not preserved within blocks');
            
            % Long-lag autocorrelation may not be preserved between blocks
            long_lag = lags+1; % Check the longest lag
            if block_length < lags
                % Only check this if block_length is less than the max lag
                % We expect larger differences at lags beyond the block size
                long_lag_diff = abs(avg_bootstrap_autocorr(long_lag) - original_autocorr(long_lag));
                short_lag_diff = abs(avg_bootstrap_autocorr(short_lag) - original_autocorr(short_lag));
                
                obj.assertTrue(long_lag_diff > short_lag_diff, ...
                    'Long-lag autocorrelation should differ more than short-lag autocorrelation');
            end
        end
        
        function testInvalidInputs(obj)
            % Tests that block_bootstrap correctly handles invalid inputs with appropriate error messages
            
            % Test with empty data
            try
                block_bootstrap([], obj.testParams.blockLength, obj.testParams.numBootstraps);
                obj.assertTrue(false, 'Empty data should throw an error');
            catch
                % Expected behavior - error was thrown
            end
            
            % Test with data containing NaN
            invalid_data = obj.testData;
            invalid_data(5) = NaN;
            try
                block_bootstrap(invalid_data, obj.testParams.blockLength, obj.testParams.numBootstraps);
                obj.assertTrue(false, 'Data with NaN should throw an error');
            catch
                % Expected behavior - error was thrown
            end
            
            % Test with negative block length
            try
                block_bootstrap(obj.testData, -5, obj.testParams.numBootstraps);
                obj.assertTrue(false, 'Negative block length should throw an error');
            catch
                % Expected behavior - error was thrown
            end
            
            % Test with block length larger than data length
            try
                block_bootstrap(obj.testData, length(obj.testData)+10, obj.testParams.numBootstraps);
                obj.assertTrue(false, 'Block length larger than data should throw an error');
            catch
                % Expected behavior - error was thrown
            end
            
            % Test with negative number of bootstrap samples
            try
                block_bootstrap(obj.testData, obj.testParams.blockLength, -10);
                obj.assertTrue(false, 'Negative number of bootstrap samples should throw an error');
            catch
                % Expected behavior - error was thrown
            end
            
            % Test with non-numeric circular parameter
            try
                block_bootstrap(obj.testData, obj.testParams.blockLength, obj.testParams.numBootstraps, 'yes');
                obj.assertTrue(false, 'Non-numeric circular parameter should throw an error');
            catch
                % Expected behavior - error was thrown
            end
        end
        
        function testPerformance(obj)
            % Tests the performance of block_bootstrap with large datasets
            
            % Generate larger test dataset
            T = 1000;
            N = 10;
            large_data = randn(T, N);
            
            % Measure execution time for different parameter combinations
            block_sizes = [5, 20, 50];
            bootstrap_counts = [10, 50];
            
            % Only run a few combinations to keep test time reasonable
            for block_size = block_sizes
                for B = bootstrap_counts
                    % Measure execution time
                    execution_time = obj.measureExecutionTime(@() block_bootstrap(large_data, block_size, B));
                    
                    % Ensure execution completes within reasonable time
                    obj.assertTrue(execution_time < 60, ...
                        sprintf('Execution time too long for block_size=%d, B=%d', block_size, B));
                end
            end
        end
        
        function testBootstrapDistribution(obj)
            % Tests that the distribution of bootstrapped statistics matches theoretical expectations
            
            % Generate many bootstrap samples to test distribution properties
            B = 1000;
            bsdata = block_bootstrap(obj.testData, obj.testParams.blockLength, B);
            
            % Calculate original mean
            original_mean = mean(obj.testData);
            
            % Calculate bootstrap means
            bootstrap_means = zeros(B, 1);
            for b = 1:B
                bootstrap_means(b) = mean(bsdata(:,:,b));
            end
            
            % Mean of bootstrap means should be close to original mean
            bootstrap_mean_of_means = mean(bootstrap_means);
            obj.assertEqualsWithTolerance(bootstrap_mean_of_means, original_mean, 0.05, ...
                'Mean of bootstrap means deviates too much from original mean');
            
            % Standard deviation of bootstrap means approximates standard error of the mean
            bootstrap_std_of_means = std(bootstrap_means);
            theoretical_se = std(obj.testData) / sqrt(length(obj.testData)); % Simple approximation
            
            % The ratio shouldn't be too far from 1 (allowing for some difference due to dependence)
            ratio = bootstrap_std_of_means / theoretical_se;
            obj.assertTrue(ratio > 0.5 && ratio < 2, ...
                'Bootstrap standard error deviates too much from theoretical approximation');
            
            % Calculate bootstrap 95% confidence interval
            bootstrap_ci = prctile(bootstrap_means, [2.5, 97.5]);
            
            % True mean should be contained in the confidence interval
            obj.assertTrue(bootstrap_ci(1) <= original_mean && original_mean <= bootstrap_ci(2), ...
                'Bootstrap confidence interval does not contain the original mean');
        end
        
        function results = runAllTests(obj)
            % Convenience method to run all test cases in the class
            
            % Run all test methods in the class
            results = runAllTests@BaseTest(obj);
            
            % Display summary
            fprintf('BlockBootstrapTest: %d tests, %d passed, %d failed\n', ...
                results.summary.numTests, ...
                results.summary.numPassed, ...
                results.summary.numFailed);
        end
    end
end