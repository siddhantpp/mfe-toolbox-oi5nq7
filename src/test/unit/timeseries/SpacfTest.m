classdef SpacfTest < BaseTest
    % SpacfTest Test class for the spacf function which computes sample partial 
    % autocorrelation functions for time series data, essential for ARMA model 
    % order identification
    
    properties
        testData
        testTolerance
    end
    
    methods
        function obj = SpacfTest()
            % Initialize the SpacfTest class with default test data
            obj = obj@BaseTest();
            obj.testTolerance = 1e-10;
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Set random seed for reproducibility
            rng(123);
            
            % Create standard test data
            T = 500;
            obj.testData = struct();
            
            % Create random data
            obj.testData.random = randn(T, 1);
            
            % Create AR(1) process with phi = 0.7
            phi = 0.7;
            y_ar1 = zeros(T, 1);
            epsilon = randn(T, 1);
            for t = 2:T
                y_ar1(t) = phi * y_ar1(t-1) + epsilon(t);
            end
            obj.testData.ar1 = y_ar1;
            obj.testData.ar1_phi = phi;
            
            % Create AR(2) process with phi = [0.7, -0.2]
            phi = [0.7, -0.2];
            y_ar2 = zeros(T, 1);
            for t = 3:T
                y_ar2(t) = phi(1) * y_ar2(t-1) + phi(2) * y_ar2(t-2) + epsilon(t);
            end
            obj.testData.ar2 = y_ar2;
            obj.testData.ar2_phi = phi;
            
            % Create data with non-zero mean
            obj.testData.nonzero_mean = obj.testData.random + 5;
            
            % Validate test data
            obj.testData.random = datacheck(obj.testData.random, 'random');
            obj.testData.ar1 = datacheck(obj.testData.ar1, 'ar1');
            obj.testData.ar2 = datacheck(obj.testData.ar2, 'ar2');
            obj.testData.nonzero_mean = datacheck(obj.testData.nonzero_mean, 'nonzero_mean');
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            tearDown@BaseTest(obj);
            
            % No specific cleanup needed for this test class
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of spacf with simple time series data
            
            % For an AR(1) process with parameter phi, the PACF is:
            % PACF(1) = phi
            % PACF(k) ≈ 0 for k > 1
            
            % Compute PACF
            lags = 5;
            pacf = spacf(obj.testData.ar1, lags);
            
            % Check that PACF(1) is approximately phi
            obj.assertAlmostEqual(obj.testData.ar1_phi, pacf(1), ['AR(1) PACF at lag 1 should be close to ', num2str(obj.testData.ar1_phi)]);
            
            % Check that PACF at higher lags is approximately 0
            for i = 2:lags
                obj.assertAlmostEqual(0, pacf(i), ['AR(1) PACF at lag ', num2str(i), ' should be close to 0'], 0.1);
            end
            
            % Test with AR(2) process
            pacf = spacf(obj.testData.ar2, lags);
            
            % Check first two PACF values match AR coefficients
            obj.assertAlmostEqual(obj.testData.ar2_phi(1), pacf(1), 'AR(2) PACF at lag 1 should match first AR coefficient', 0.1);
            obj.assertAlmostEqual(obj.testData.ar2_phi(2), pacf(2), 'AR(2) PACF at lag 2 should match second AR coefficient', 0.1);
            
            % Higher lags should be close to zero
            for i = 3:lags
                obj.assertAlmostEqual(0, pacf(i), ['AR(2) PACF at lag ', num2str(i), ' should be close to 0'], 0.15);
            end
        end
        
        function testWithMultipleOutputs(obj)
            % Test spacf with multiple output arguments (PACF, standard errors, confidence intervals)
            
            % Test with standard errors
            [pacf, se] = spacf(obj.testData.random, 10);
            
            % Verify SE dimensions
            obj.assertEqual(length(pacf), length(se), 'PACF and SE should have same length');
            
            % Verify SE values (for a random process, SE should be approximately 1/sqrt(T))
            expectedSE = 1/sqrt(length(obj.testData.random));
            for i = 1:length(se)
                obj.assertAlmostEqual(expectedSE, se(i), 'SE should be approximately 1/sqrt(T)');
            end
            
            % Test with confidence intervals (default alpha = 0.05)
            [pacf, se, ci] = spacf(obj.testData.random, 10);
            
            % Verify CI dimensions
            obj.assertEqual([length(pacf), 2], size(ci), 'CI should be a matrix with rows = lags and columns = 2');
            
            % Verify CI values (CI = PACF ± z_(alpha/2) * SE where z_(0.025) ≈ 1.96)
            z = norminv(0.975); % 97.5% quantile for 95% CI
            for i = 1:length(pacf)
                expectedLower = pacf(i) - z * se(i);
                expectedUpper = pacf(i) + z * se(i);
                obj.assertAlmostEqual(expectedLower, ci(i,1), 'Lower CI bound is incorrect');
                obj.assertAlmostEqual(expectedUpper, ci(i,2), 'Upper CI bound is incorrect');
            end
        end
        
        function testInputValidation(obj)
            % Test error handling for invalid inputs
            
            % Helper function to test that a function throws an error
            function testErrorThrown(func, errorMsg)
                try
                    func();
                    obj.assertTrue(false, ['Expected error with message containing: ', errorMsg]);
                catch ME
                    % Check if error message contains expected text
                    obj.assertTrue(contains(ME.message, errorMsg), ['Error message should contain: ', errorMsg]);
                end
            end
            
            % Set up valid data for testing
            validData = randn(10, 1);
            
            % Test with empty data
            testErrorThrown(@() spacf([]), 'cannot be empty');
            
            % Test with non-numeric data
            testErrorThrown(@() spacf('string'), 'must be numeric');
            
            % Test with NaN values
            nanData = validData;
            nanData(5) = NaN;
            testErrorThrown(@() spacf(nanData), 'cannot contain NaN');
            
            % Test with Inf values
            infData = validData;
            infData(5) = Inf;
            testErrorThrown(@() spacf(infData), 'cannot contain Inf');
            
            % Test with invalid lags (negative)
            testErrorThrown(@() spacf(validData, -1), 'greater than or equal to 0');
            
            % Test with invalid lags (non-integer)
            testErrorThrown(@() spacf(validData, 1.5), 'integer values');
            
            % Test with invalid alpha (outside [0,1])
            testErrorThrown(@() spacf(validData, 1, struct('alpha', -0.1)), 'greater than or equal to 0');
            testErrorThrown(@() spacf(validData, 1, struct('alpha', 1.1)), 'less than or equal to 1');
            
            % Test with invalid demean option
            testErrorThrown(@() spacf(validData, 1, struct('demean', 'yes')), 'logical value');
        end
        
        function testLagOptions(obj)
            % Test spacf with various lag parameter options
            
            % Create test data
            y = obj.testData.random;
            T = length(y);
            
            % Test with scalar lag
            lag = 5;
            pacf = spacf(y, lag);
            obj.assertEqual(lag, length(pacf), 'PACF length should match lag parameter');
            
            % Test with vector of lags
            lags = [1, 3, 5, 10];
            pacf = spacf(y, lags);
            obj.assertEqual(length(lags), length(pacf), 'PACF length should match number of lags');
            
            % Test with default lag (1:min(20,floor(T/4)))
            pacf = spacf(y);
            expectedLength = min(20, floor(T/4));
            obj.assertEqual(expectedLength, length(pacf), 'Default PACF length incorrect');
            
            % Test with large lag
            largeLag = min(50, T-5);  % Choose a large but valid lag
            pacf = spacf(y, largeLag);
            obj.assertEqual(largeLag, length(pacf), 'PACF should handle large lags');
            
            % Test with empty lag parameter (uses default)
            pacf_default = spacf(y);
            pacf_empty = spacf(y, []);
            obj.assertMatrixEqualsWithTolerance(pacf_default, pacf_empty, obj.testTolerance, 'Empty lag should use default');
        end
        
        function testDemeanOption(obj)
            % Test spacf with demean option for non-centered data
            
            % Use test data with non-zero mean
            y = obj.testData.nonzero_mean;
            y_zeromean = y - mean(y);
            
            % Compute PACF with demean=true (default)
            pacf_demean = spacf(y, 5);
            
            % Compute PACF with demean=false
            pacf_no_demean = spacf(y, 5, struct('demean', false));
            
            % The PACFs should be different
            differentPACFs = any(abs(pacf_demean - pacf_no_demean) > obj.testTolerance * 100);
            obj.assertTrue(differentPACFs, 'PACFs should differ with/without demeaning for non-centered data');
            
            % PACF of zero-mean data should match PACF of original data with demeaning
            pacf_zeromean = spacf(y_zeromean, 5, struct('demean', false));
            obj.assertMatrixEqualsWithTolerance(pacf_demean, pacf_zeromean, obj.testTolerance * 10, 'Zero-mean PACF should match demeaned PACF');
        end
        
        function testAlphaOption(obj)
            % Test spacf with different alpha values for confidence intervals
            
            % Use random test data
            y = obj.testData.random;
            lags = 5;
            
            % Compute confidence intervals with different alpha values
            [~, ~, ci_01] = spacf(y, lags, struct('alpha', 0.01)); % 99% CI
            [~, ~, ci_05] = spacf(y, lags, struct('alpha', 0.05)); % 95% CI
            [~, ~, ci_10] = spacf(y, lags, struct('alpha', 0.10)); % 90% CI
            
            % Verify that wider confidence level (smaller alpha) gives wider intervals
            for i = 1:size(ci_01, 1)
                width_01 = ci_01(i,2) - ci_01(i,1);
                width_05 = ci_05(i,2) - ci_05(i,1);
                width_10 = ci_10(i,2) - ci_10(i,1);
                
                obj.assertTrue(width_01 > width_05, 'CI with alpha=0.01 should be wider than with alpha=0.05');
                obj.assertTrue(width_05 > width_10, 'CI with alpha=0.05 should be wider than with alpha=0.10');
            end
            
            % Check that the ratio of widths approximately matches the ratio of z-values
            z_01 = norminv(0.995); % 99.5% quantile for 99% CI
            z_05 = norminv(0.975); % 97.5% quantile for 95% CI
            z_10 = norminv(0.95);  % 95% quantile for 90% CI
            
            width_ratio_01_05 = (ci_01(1,2) - ci_01(1,1)) / (ci_05(1,2) - ci_05(1,1));
            expected_ratio_01_05 = z_01 / z_05;
            
            obj.assertAlmostEqual(expected_ratio_01_05, width_ratio_01_05, 'CI width ratio should match z-value ratio', 0.1);
        end
        
        function testYuleWalkerImplementation(obj)
            % Test the Yule-Walker equations implementation used in PACF calculation
            
            % Use AR(2) test data with known parameters
            y = obj.testData.ar2;
            phi = obj.testData.ar2_phi;
            
            % Get ACF values for lags 0 to 2
            acf = sacf(y, 0:2, struct('demean', true));
            
            % Manually solve Yule-Walker equations for lag 2
            % For lag 1: phi_1 = acf(1)
            % For lag 2: [1 acf(1); acf(1) 1] * [phi_1; phi_2] = [acf(1); acf(2)]
            R = toeplitz(acf(1:2));  % 2x2 Toeplitz matrix using acf for lags 0-1
            b = acf(2:3);            % acf for lags 1-2
            phi_manual = R \ b;
            
            % PACF at lag 1 should be phi_manual(1), PACF at lag 2 should be phi_manual(2)
            pacf = spacf(y, 2);
            
            obj.assertAlmostEqual(phi_manual(1), pacf(1), 'PACF at lag 1 should match manual calculation', 0.1);
            obj.assertAlmostEqual(phi_manual(2), pacf(2), 'PACF at lag 2 should match manual calculation', 0.1);
            
            % For an AR(2) process, PACF at lags > 2 should be approximately 0
            pacf_extended = spacf(y, 5);
            for i = 3:5
                obj.assertAlmostEqual(0, pacf_extended(i), ['PACF at lag ', num2str(i), ' should be close to 0'], 0.15);
            end
        end
        
        function testWithRealData(obj)
            % Test spacf with real financial return data
            
            % Try to load financial returns data
            try
                data = obj.loadTestData('financial_returns.mat');
                returns = data.returns;
            catch
                % If file doesn't exist, create synthetic financial returns
                warning('Could not load financial_returns.mat. Using synthetic data instead.');
                T = 1000;
                returns = 0.001 + 0.01 * randn(T, 1);
            end
            
            % Financial returns typically have very little autocorrelation
            returns = datacheck(returns, 'returns');
            
            % Compute PACF for financial returns
            [pacf, se, ci] = spacf(returns, 10);
            
            % Financial returns typically have very little autocorrelation
            % Most PACF values should be inside the confidence bounds based on SE
            significantCount = sum(abs(pacf) > 1.96 * se);
            
            % A few significant values can occur by chance (roughly 5% with alpha=0.05)
            obj.assertTrue(significantCount <= 2, 'Too many significant PACF values for financial returns');
            
            % Verify that confidence intervals contain 0 for most lags
            zeroInsideCI = sum((ci(:,1) <= 0) & (0 <= ci(:,2)));
            obj.assertTrue(zeroInsideCI >= 8, 'CI should contain 0 for most lags in financial returns');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of spacf with challenging datasets
            
            % Test with very small values
            T = 200;
            y_small = 1e-10 * randn(T, 1);
            pacf_small = spacf(y_small, 5);
            
            % Results should still be finite
            obj.assertTrue(all(isfinite(pacf_small)), 'PACF with small values should be finite');
            
            % Test with very large values
            y_large = 1e10 * randn(T, 1);
            pacf_large = spacf(y_large, 5);
            
            % Results should still be finite
            obj.assertTrue(all(isfinite(pacf_large)), 'PACF with large values should be finite');
            
            % Test with high-persistence AR(1) process
            phi = 0.999;
            y = zeros(T, 1);
            epsilon = randn(T, 1);
            for t = 2:T
                y(t) = phi * y(t-1) + epsilon(t);
            end
            
            pacf = spacf(y, 5);
            
            % PACF at lag 1 should be close to phi
            obj.assertAlmostEqual(phi, pacf(1), 'PACF at lag 1 should be close to phi even for high persistence', 0.05);
            
            % PACF at higher lags should be finite (not NaN or Inf)
            obj.assertTrue(all(isfinite(pacf)), 'PACF values should be finite even for high persistence');
        end
        
        function testPerformance(obj)
            % Test performance of spacf with large datasets
            
            % Skip detailed performance testing in regular test runs
            if ~obj.verbose
                return;
            end
            
            % Create a large dataset
            T = 10000;
            y = randn(T, 1);
            
            % Measure time for computing PACF
            tic;
            pacf = spacf(y, 20);
            executionTime = toc;
            
            % Print execution time
            fprintf('PACF computation for %d observations took %.4f seconds\n', T, executionTime);
            
            % Execution time should scale approximately linearly with data size
            % Compare with a smaller dataset
            T_small = 1000;
            y_small = randn(T_small, 1);
            
            tic;
            pacf_small = spacf(y_small, 20);
            executionTime_small = toc;
            
            % Ratio of execution times should be roughly proportional to ratio of data sizes
            % but with some overhead, so we use a loose bound
            timeRatio = executionTime / executionTime_small;
            sizeRatio = T / T_small;
            
            fprintf('Time ratio: %.2f, Size ratio: %.2f\n', timeRatio, sizeRatio);
            obj.assertTrue(timeRatio < sizeRatio * 2, 'Execution time should scale reasonably with data size');
        end
    end
end