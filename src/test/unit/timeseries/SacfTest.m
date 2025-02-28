classdef SacfTest < BaseTest
    % SACFTEST Test class for the sacf function
    %
    % Tests the functionality, robustness, and performance of the sample
    % autocorrelation function (sacf) implementation in the MFE Toolbox.
    %
    % This test class validates the correctness of autocorrelation calculations,
    % checks numerical stability across different types of time series data,
    % and verifies proper error handling for invalid inputs.
    
    properties
        testData       % Matrix for test data
        testTolerance  % Tolerance for numerical comparisons
    end
    
    methods
        function obj = SacfTest()
            % Initialize the SacfTest class with default test data
            obj = obj@BaseTest();
            obj.testTolerance = 1e-10; % Default tolerance for numerical comparisons
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Generate test data - AR(1) process with phi = 0.7
            rng(123); % Set random seed for reproducibility
            T = 1000;
            phi = 0.7;
            noise = randn(T, 1);
            obj.testData = zeros(T, 1);
            
            % Generate AR(1) process: x_t = phi*x_{t-1} + noise
            for t = 2:T
                obj.testData(t) = phi * obj.testData(t-1) + noise(t);
            end
            
            % Validate test data
            datacheck(obj.testData, 'testData');
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of sacf with simple time series data
            
            % For an AR(1) process with parameter phi, the theoretical
            % autocorrelation at lag k is phi^k
            phi = 0.7;
            lags = 1:10;
            theoretical_acf = phi.^lags';
            
            % Compute sample ACF using sacf function
            computed_acf = sacf(obj.testData, lags);
            
            % Compare with theoretical values (with tolerance due to sample variation)
            for i = 1:length(lags)
                obj.assertAlmostEqual(theoretical_acf(i), computed_acf(i), ...
                    sprintf('ACF at lag %d differs from theoretical value', lags(i)));
            end
            
            % Verify that ACF at lag 0 would be 1
            lag0_acf = sacf(obj.testData, 0);
            obj.assertAlmostEqual(1.0, lag0_acf, 'ACF at lag 0 should be 1');
        end
        
        function testWithMultipleOutputs(obj)
            % Test sacf with multiple output arguments (ACF, standard errors, confidence intervals)
            
            % Test with second output: standard errors
            [acf, se] = sacf(obj.testData, 1:5);
            
            % Verify dimensions
            obj.assertEqual(size(acf), size(se), 'ACF and SE should have same dimensions');
            
            % Verify standard error calculation (approx 1/sqrt(T) for stationary process)
            expected_se = ones(5, 1) / sqrt(length(obj.testData));
            obj.assertMatrixEqualsWithTolerance(expected_se, se, obj.testTolerance, ...
                'Standard errors should be approximately 1/sqrt(T)');
            
            % Test with third output: confidence intervals
            [acf, se, ci] = sacf(obj.testData, 1:5);
            
            % Verify dimensions
            obj.assertEqual([length(acf), 2], size(ci), 'CI should be an N×2 matrix');
            
            % Verify confidence interval calculation: ACF ± z_(α/2) * SE
            alpha = 0.05; % Default alpha
            z = norminv(1 - alpha/2);
            expected_ci = [acf - z*se, acf + z*se];
            obj.assertMatrixEqualsWithTolerance(expected_ci, ci, obj.testTolerance, ...
                'Confidence intervals should be ACF ± z_(α/2) * SE');
        end
        
        function testInputValidation(obj)
            % Test error handling for invalid inputs
            
            % Helper function to assert that an error is thrown
            function assertErrorThrown(func, message)
                try
                    func();
                    obj.assertTrue(false, message);
                catch
                    % Error was thrown as expected
                end
            end
            
            % Test with empty data
            assertErrorThrown(@() sacf([]), 'sacf should error with empty data');
            
            % Test with non-numeric data
            assertErrorThrown(@() sacf('string'), 'sacf should error with non-numeric data');
            
            % Test with NaN values
            data_with_nan = obj.testData;
            data_with_nan(5) = NaN;
            assertErrorThrown(@() sacf(data_with_nan), 'sacf should error with NaN values');
            
            % Test with Inf values
            data_with_inf = obj.testData;
            data_with_inf(10) = Inf;
            assertErrorThrown(@() sacf(data_with_inf), 'sacf should error with Inf values');
            
            % Test with invalid lag parameter (negative)
            assertErrorThrown(@() sacf(obj.testData, -1), 'sacf should error with negative lags');
            
            % Test with invalid lag parameter (non-integer)
            assertErrorThrown(@() sacf(obj.testData, 1.5), 'sacf should error with non-integer lags');
            
            % Test with invalid alpha (outside 0-1 range)
            options = struct('alpha', 1.5);
            assertErrorThrown(@() sacf(obj.testData, [], options), 'sacf should error with alpha outside 0-1 range');
        end
        
        function testLagOptions(obj)
            % Test sacf with various lag parameter options
            
            % Test with explicit lag parameter as a scalar
            explicit_lag = 5;
            acf_explicit = sacf(obj.testData, explicit_lag);
            obj.assertEqual(explicit_lag, length(acf_explicit), ...
                'When lags is a scalar, output should have length equal to lags');
            
            % Test with explicit lag parameter as a vector
            lag_vector = [1, 3, 5, 7, 9];
            acf_vector = sacf(obj.testData, lag_vector);
            obj.assertEqual(length(lag_vector), length(acf_vector), ...
                'When lags is a vector, output should have same length');
            
            % Test with default lag parameter
            default_acf = sacf(obj.testData);
            expected_length = min(20, floor(length(obj.testData)/4));
            obj.assertEqual(expected_length, length(default_acf), ...
                'Default lags should be 1:min(20,floor(T/4))');
            
            % Test with lag larger than data size
            large_lag = length(obj.testData) + 10;
            acf_large = sacf(obj.testData, large_lag);
            obj.assertEqual(large_lag, length(acf_large), ...
                'Output should be same length as lags, even if lags exceed data size');
            
            % Test that ACF for lag values larger than data size should be 0
            obj.assertEqual(0, acf_large(end), ...
                'ACF for lags exceeding data size should be 0');
        end
        
        function testDemeanOption(obj)
            % Test sacf with demean option for non-centered data
            
            % Create non-centered data (add constant)
            mean_value = 5.0;
            non_centered_data = obj.testData + mean_value;
            
            % Calculate ACF with demean=true (default)
            acf_demeaned = sacf(non_centered_data, 1:5);
            
            % Calculate ACF with demean=false
            options = struct('demean', false);
            acf_non_demeaned = sacf(non_centered_data, 1:5, options);
            
            % For demeaned data, ACF should match the original data's ACF
            acf_original = sacf(obj.testData, 1:5);
            obj.assertMatrixEqualsWithTolerance(acf_original, acf_demeaned, obj.testTolerance, ...
                'ACF with demean=true should match ACF of centered data');
            
            % For non-demeaned data with a constant mean, all autocorrelations should be close to 1
            expected_acf = ones(5, 1);
            obj.assertMatrixEqualsWithTolerance(expected_acf, acf_non_demeaned, 0.1, ...
                'ACF with demean=false for non-centered data should be close to 1');
            
            % Verify that the non-demeaned ACF differs from the demeaned ACF
            obj.assertTrue(any(abs(acf_demeaned - acf_non_demeaned) > obj.testTolerance), ...
                'ACF with and without demeaning should differ for non-centered data');
        end
        
        function testAlphaOption(obj)
            % Test sacf with different alpha values for confidence intervals
            
            lags = 1:5;
            
            % Test with default alpha (0.05 - 95% confidence)
            [~, ~, ci_default] = sacf(obj.testData, lags);
            
            % Test with alpha = 0.01 (99% confidence)
            options1 = struct('alpha', 0.01);
            [~, ~, ci_01] = sacf(obj.testData, lags, options1);
            
            % Test with alpha = 0.10 (90% confidence)
            options2 = struct('alpha', 0.10);
            [~, ~, ci_10] = sacf(obj.testData, lags, options2);
            
            % 99% confidence intervals should be wider than 95% confidence intervals
            for i = 1:length(lags)
                width_default = ci_default(i, 2) - ci_default(i, 1);
                width_01 = ci_01(i, 2) - ci_01(i, 1);
                obj.assertTrue(width_01 > width_default, ...
                    sprintf('99%% CI should be wider than 95%% CI at lag %d', lags(i)));
            end
            
            % 90% confidence intervals should be narrower than 95% confidence intervals
            for i = 1:length(lags)
                width_default = ci_default(i, 2) - ci_default(i, 1);
                width_10 = ci_10(i, 2) - ci_10(i, 1);
                obj.assertTrue(width_10 < width_default, ...
                    sprintf('90%% CI should be narrower than 95%% CI at lag %d', lags(i)));
            end
        end
        
        function testWithRealData(obj)
            % Test sacf with real financial return data
            
            % Load example financial returns data or simulate typical returns
            try
                financial_returns = obj.loadTestData('financial_returns.mat');
                returns = financial_returns.returns;
            catch
                % Simulate financial returns with stylized facts:
                % small autocorrelation, volatility clustering
                rng(456);
                T = 1000;
                returns = 0.0005 + 0.05*randn(T, 1);
                returns = returns .* (1 + 0.7*abs(returns([T; 1:T-1])));
            end
            
            % Compute ACF for returns
            acf_returns = sacf(returns, 1:10);
            
            % Financial returns typically show little autocorrelation
            obj.assertTrue(all(abs(acf_returns) < 0.3), ...
                'Financial returns should have relatively small autocorrelations');
            
            % Compute ACF for squared returns (proxy for volatility)
            acf_squared = sacf(returns.^2, 1:10);
            
            % Squared returns often show significant autocorrelation (volatility clustering)
            obj.assertTrue(any(abs(acf_squared) > 0.2), ...
                'Squared returns should exhibit some significant autocorrelations');
            
            % Compute ACF for absolute returns (another proxy for volatility)
            acf_absolute = sacf(abs(returns), 1:10);
            
            % Absolute returns often show significant autocorrelation
            obj.assertTrue(any(abs(acf_absolute) > 0.2), ...
                'Absolute returns should exhibit some significant autocorrelations');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of sacf with challenging datasets
            
            % Test with very small values (near machine epsilon)
            small_data = eps * randn(100, 1);
            acf_small = sacf(small_data, 1:5);
            obj.assertTrue(all(isfinite(acf_small)), ...
                'ACF should remain finite with very small input values');
            
            % Test with very large values
            large_data = 1e10 * randn(100, 1);
            acf_large = sacf(large_data, 1:5);
            obj.assertTrue(all(isfinite(acf_large)), ...
                'ACF should remain finite with very large input values');
            
            % Test with high persistence series (near unit root)
            rng(789);
            T = 500;
            phi = 0.99;  % High persistence
            persistent_data = zeros(T, 1);
            for t = 2:T
                persistent_data(t) = phi * persistent_data(t-1) + randn();
            end
            acf_persistent = sacf(persistent_data, 1:10);
            
            % High persistence should result in slowly decaying ACF
            for i = 2:10
                obj.assertTrue(acf_persistent(i-1) > acf_persistent(i), ...
                    'ACF of high persistence series should decay monotonically');
            end
            
            % Verify ACF is approximately phi^k
            theoretical_acf = phi.^(1:10)';
            max_diff = max(abs(acf_persistent - theoretical_acf));
            obj.assertTrue(max_diff < 0.2, ...
                'ACF of high persistence series should approximate phi^k');
        end
        
        function testPerformance(obj)
            % Test performance of sacf with large datasets
            
            % Generate large dataset
            rng(101112);
            large_T = 10000;
            large_data = randn(large_T, 1);
            
            % Measure execution time
            tic;
            sacf(large_data, 1:20);
            execution_time = toc;
            
            % Check that execution time is reasonable
            obj.assertTrue(execution_time < 1.0, ...
                sprintf('sacf execution time for large dataset (%.2f sec) exceeds threshold', execution_time));
            
            % Verify that execution time scales approximately linearly with data size
            small_T = 1000;
            small_data = large_data(1:small_T);
            
            tic;
            sacf(small_data, 1:20);
            small_execution_time = toc;
            
            % Allow some flexibility in the ratio (between 5 and 15 instead of exactly 10)
            time_ratio = execution_time / small_execution_time;
            expected_ratio = large_T / small_T;
            
            obj.assertTrue(time_ratio < expected_ratio * 1.5, ...
                sprintf('Execution time does not scale appropriately with data size (ratio: %.2f, expected: %.2f)', ...
                time_ratio, expected_ratio));
        end
    end
end