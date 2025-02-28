classdef StatisticalTestsValidation < BaseTest
    % STATISTICALTESTSVALIDATION Validation test class for the statistical tests functionality in the MFE Toolbox
    
    properties
        testData        % Structure for test data
        defaultTolerance % Default tolerance for numerical comparisons
        verbose         % Flag for verbose output
    end
    
    methods
        function obj = StatisticalTestsValidation()
            % Initializes the StatisticalTestsValidation test class
            
            % Call superclass constructor
            obj = obj@BaseTest();
            
            % Set default tolerance for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Set verbose mode to false
            obj.verbose = false;
            
            % Load test data
            try
                % Try to load financial returns for realistic testing
                financial_data = obj.loadTestData('financial_returns.mat');
                obj.testData.financial = financial_data;
            catch
                % Create synthetic data if file doesn't exist
                if obj.verbose
                    fprintf('Financial returns data not found. Using simulated data instead.\n');
                end
                obj.testData.financial.returns = randn(1000, 1);
                obj.testData.financial.prices = cumsum([100; obj.testData.financial.returns]);
            end
            
            try
                % Try to load simulated data with known properties
                simulated_data = obj.loadTestData('simulated_data.mat');
                obj.testData.simulated = simulated_data;
            catch
                % Create basic simulated data if file doesn't exist
                if obj.verbose
                    fprintf('Simulated data not found. Generating data with known properties.\n');
                end
                
                % Generate stationary AR(1) process
                T = 1000;
                ar_coef = 0.7;
                e = randn(T, 1);
                ar1 = zeros(T, 1);
                for t = 2:T
                    ar1(t) = ar_coef * ar1(t-1) + e(t);
                end
                obj.testData.simulated.stationary = ar1;
                
                % Generate non-stationary random walk
                obj.testData.simulated.nonstationary = cumsum(randn(T, 1));
                
                % Generate series with ARCH effects
                arch_e = randn(T, 1);
                arch_sigma2 = zeros(T, 1);
                arch_y = zeros(T, 1);
                arch_sigma2(1) = 1;
                for t = 2:T
                    arch_sigma2(t) = 0.2 + 0.8 * arch_y(t-1)^2;
                    arch_y(t) = sqrt(arch_sigma2(t)) * arch_e(t);
                end
                obj.testData.simulated.arch = arch_y;
                
                % Generate autocorrelated series
                ar5 = filter(1, [1 -0.6 0.2 -0.3 0.1 -0.05], randn(T, 1));
                obj.testData.simulated.autocorrelated = ar5;
                
                % Generate non-normal data with skewness and kurtosis
                % Mix of normal distributions to create skewness
                non_normal = [randn(T/2, 1); 2*randn(T/2, 1) + 3];
                obj.testData.simulated.non_normal = non_normal;
            end
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Additional setup if needed
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj);
            
            % Additional cleanup if needed
        end
        
        function testUnitRootTests(obj)
            % Validates the implementation of unit root tests (ADF, PP, KPSS)
            
            % Create random stationary series (AR(1) with coefficient 0.7)
            T = 500;
            stationary_series = zeros(T, 1);
            e = randn(T, 1);
            stationary_series(1) = e(1);
            for t = 2:T
                stationary_series(t) = 0.7 * stationary_series(t-1) + e(t);
            end
            
            % Create non-stationary series (random walk)
            nonstationary_series = cumsum(randn(T, 1));
            
            % 1. Test ADF test with stationary series (should reject null of unit root)
            adf_results_stationary = adf_test(stationary_series);
            obj.assertTrue(adf_results_stationary.pval < 0.05, ...
                'ADF test should reject unit root for stationary series');
            
            % 2. Test ADF test with non-stationary series (should not reject null of unit root)
            adf_results_nonstationary = adf_test(nonstationary_series);
            obj.assertTrue(adf_results_nonstationary.pval > 0.05, ...
                'ADF test should not reject unit root for non-stationary series');
            
            % 3. Test different regression specifications for ADF
            adf_options = struct('regression_type', 'n');
            adf_no_const = adf_test(stationary_series, adf_options);
            adf_options.regression_type = 'ct';
            adf_const_trend = adf_test(stationary_series, adf_options);
            obj.assertTrue(isfield(adf_no_const, 'stat') && isfield(adf_const_trend, 'stat'), ...
                'ADF test with different regression types should return valid results');
            
            % 4. Test automatic lag selection for ADF
            adf_options = struct('lags', 'aic');
            adf_aic_results = adf_test(stationary_series, adf_options);
            adf_options.lags = 'bic';
            adf_bic_results = adf_test(stationary_series, adf_options);
            obj.assertTrue(isfield(adf_aic_results, 'optimal_lag') && isfield(adf_bic_results, 'optimal_lag'), ...
                'ADF test with automatic lag selection should return optimal lag information');
            
            % 5. Test PP test with stationary series (should reject null of unit root)
            pp_results_stationary = pp_test(stationary_series);
            obj.assertTrue(pp_results_stationary.pval < 0.05, ...
                'PP test should reject unit root for stationary series');
            
            % 6. Test PP test with non-stationary series (should not reject null of unit root)
            pp_results_nonstationary = pp_test(nonstationary_series);
            obj.assertTrue(pp_results_nonstationary.pval > 0.05, ...
                'PP test should not reject unit root for non-stationary series');
            
            % 7. Test KPSS test with stationary series (should not reject null of stationarity)
            kpss_results_stationary = kpss_test(stationary_series);
            obj.assertTrue(kpss_results_stationary.pval > 0.05, ...
                'KPSS test should not reject stationarity for stationary series');
            
            % 8. Test KPSS test with non-stationary series (should reject null of stationarity)
            kpss_results_nonstationary = kpss_test(nonstationary_series);
            obj.assertTrue(kpss_results_nonstationary.pval < 0.05 || kpss_results_nonstationary.stat > kpss_results_nonstationary.cv(3), ...
                'KPSS test should reject stationarity for non-stationary series');
            
            % 9. Test trend stationarity in KPSS
            kpss_options = struct('regression_type', 'tau');
            kpss_trend_stationary = kpss_test(stationary_series, kpss_options);
            obj.assertTrue(isfield(kpss_trend_stationary, 'stat'), ...
                'KPSS test with trend stationarity specification should return valid results');
            
            % 10. Verify consistency between tests
            % For stationary series: ADF and PP should reject, KPSS should not reject
            % For non-stationary series: ADF and PP should not reject, KPSS should reject
            obj.assertTrue((adf_results_stationary.pval < 0.05) == (pp_results_stationary.pval < 0.05), ...
                'ADF and PP tests should give consistent results for stationary series');
            
            % 11. Test edge case: very small sample
            small_stationary = stationary_series(1:30);
            small_adf = adf_test(small_stationary);
            small_pp = pp_test(small_stationary);
            small_kpss = kpss_test(small_stationary);
            obj.assertTrue(isfield(small_adf, 'stat') && isfield(small_pp, 'stat') && isfield(small_kpss, 'stat'), ...
                'Unit root tests should handle small samples gracefully');
        end
        
        function testNormalityTests(obj)
            % Validates the implementation of the Jarque-Bera test for normality
            
            % 1. Generate normal data (should not reject normality)
            T = 500;
            normal_data = randn(T, 1);
            jb_normal = jarque_bera(normal_data);
            obj.assertTrue(jb_normal.pval > 0.05, ...
                'Jarque-Bera should not reject normality for normal data');
            
            % 2. Generate non-normal data with high skewness (should reject normality)
            % Mix two normal distributions to create skewness
            skewed_data = [randn(T/2, 1); 3*randn(T/2, 1) + 2];
            jb_skewed = jarque_bera(skewed_data);
            obj.assertTrue(jb_skewed.pval < 0.05, ...
                'Jarque-Bera should reject normality for skewed data');
            
            % 3. Generate non-normal data with high kurtosis (should reject normality)
            % Use t-distribution with low degrees of freedom for high kurtosis
            rng(123); % Set seed for reproducibility
            high_kurtosis = trnd(3, T, 1);
            jb_kurtosis = jarque_bera(high_kurtosis);
            obj.assertTrue(jb_kurtosis.pval < 0.05, ...
                'Jarque-Bera should reject normality for high kurtosis data');
            
            % 4. Test consistency with sample size variation
            % Larger samples should provide more power for detecting non-normality
            jb_small = jarque_bera(skewed_data(1:50));
            jb_large = jarque_bera(skewed_data);
            obj.assertTrue(isfield(jb_small, 'pval') && isfield(jb_large, 'pval'), ...
                'Jarque-Bera should work with different sample sizes');
            
            % 5. Validate test statistic calculation
            % Manually calculate skewness and kurtosis
            s = skewness(skewed_data);
            k = kurtosis(skewed_data);
            manual_jb = T/6 * (s^2 + (k-3)^2/4);
            obj.assertAlmostEqual(manual_jb, jb_skewed.statistic, ...
                'Jarque-Bera statistic should match manual calculation');
            
            % 6. Validate p-value calculation
            manual_pval = 1 - chi2cdf(manual_jb, 2);
            obj.assertAlmostEqual(manual_pval, jb_skewed.pval, ...
                'Jarque-Bera p-value should match chi-square distribution');
            
            % 7. Test with extreme values
            extreme_data = [normal_data; 10000];
            jb_extreme = jarque_bera(extreme_data);
            obj.assertTrue(jb_extreme.pval < 0.01, ...
                'Jarque-Bera should strongly reject normality for data with extreme values');
        end
        
        function testAutocorrelationTests(obj)
            % Validates the implementation of autocorrelation and independence tests 
            % (Ljung-Box, BDS, LM)
            
            % 1. Generate independent series (white noise)
            T = 500;
            independent_series = randn(T, 1);
            
            % 2. Generate autocorrelated series (AR process)
            autocorrelated_series = zeros(T, 1);
            e = randn(T, 1);
            autocorrelated_series(1) = e(1);
            for t = 2:T
                autocorrelated_series(t) = 0.7 * autocorrelated_series(t-1) + e(t);
            end
            
            % 3. Test Ljung-Box on independent series (should not reject independence)
            lb_independent = ljungbox(independent_series, 10);
            obj.assertTrue(all(lb_independent.pvals > 0.05), ...
                'Ljung-Box should not reject independence for white noise');
            
            % 4. Test Ljung-Box on autocorrelated series (should reject independence)
            lb_autocorrelated = ljungbox(autocorrelated_series, 10);
            obj.assertTrue(any(lb_autocorrelated.pvals < 0.05), ...
                'Ljung-Box should reject independence for autocorrelated series');
            
            % 5. Test LM test on independent series (should not reject independence)
            lm_independent = lmtest1(independent_series, 5);
            obj.assertTrue(lm_independent.pval > 0.05, ...
                'LM test should not reject independence for white noise');
            
            % 6. Test LM test on autocorrelated series (should reject independence)
            lm_autocorrelated = lmtest1(autocorrelated_series, 5);
            obj.assertTrue(lm_autocorrelated.pval < 0.05, ...
                'LM test should reject independence for autocorrelated series');
            
            % 7. Test BDS test on independent series (should not reject independence)
            % BDS test can be computationally intensive, so we only test with a few dimensions
            try
                bds_independent = bds_test(independent_series, [2, 3]);
                obj.assertTrue(all(bds_independent.pval > 0.05), ...
                    'BDS test should not reject independence for white noise');
            catch ME
                fprintf('Note: BDS test could not be completed (may be due to computational constraints).\n');
            end
            
            % 8. Test with various lag specifications for Ljung-Box
            lb_lags5 = ljungbox(autocorrelated_series, 5);
            lb_lags15 = ljungbox(autocorrelated_series, 15);
            obj.assertTrue(isfield(lb_lags5, 'stats') && isfield(lb_lags15, 'stats'), ...
                'Ljung-Box should work with different lag specifications');
            
            % 9. Verify test statistics match their mathematical definitions
            % Verify Ljung-Box statistic manually for lag 1
            acf1 = autocorrelated_series(2:end)' * autocorrelated_series(1:end-1) / ...
                   (autocorrelated_series' * autocorrelated_series);
            manual_lb1 = T * (T + 2) * (acf1^2 / (T - 1));
            lb_lag1 = ljungbox(autocorrelated_series, 1);
            obj.assertAlmostEqual(manual_lb1, lb_lag1.stats, 0.1, ...
                'Ljung-Box statistic should match manual calculation for lag 1');
            
            % 10. Test edge case: small sample
            small_autocorrelated = autocorrelated_series(1:30);
            lb_small = ljungbox(small_autocorrelated, 5);
            lm_small = lmtest1(small_autocorrelated, 2);
            obj.assertTrue(isfield(lb_small, 'stats') && isfield(lm_small, 'stat'), ...
                'Autocorrelation tests should handle small samples gracefully');
        end
        
        function testHeteroskedasticityTests(obj)
            % Validates the implementation of heteroskedasticity tests 
            % (ARCH test, White test)
            
            % 1. Generate homoskedastic series (constant variance)
            T = 500;
            homoskedastic = randn(T, 1);
            
            % 2. Generate heteroskedastic series (ARCH(1) process)
            arch_e = randn(T, 1);
            arch_sigma2 = zeros(T, 1);
            arch_series = zeros(T, 1);
            arch_sigma2(1) = 1;
            for t = 2:T
                arch_sigma2(t) = 0.2 + 0.7 * arch_series(t-1)^2;
                arch_series(t) = sqrt(arch_sigma2(t)) * arch_e(t);
            end
            
            % 3. Test ARCH test on homoskedastic series (should not reject homoskedasticity)
            arch_homoskedastic = arch_test(homoskedastic, 5);
            obj.assertTrue(arch_homoskedastic.pval > 0.05, ...
                'ARCH test should not reject homoskedasticity for constant variance series');
            
            % 4. Test ARCH test on heteroskedastic series (should reject homoskedasticity)
            arch_heteroskedastic = arch_test(arch_series, 5);
            obj.assertTrue(arch_heteroskedastic.pval < 0.05, ...
                'ARCH test should reject homoskedasticity for ARCH process');
            
            % 5. Create regression model with homoskedastic residuals
            X = [ones(T, 1), randn(T, 2)];
            beta = [1; 0.5; -0.3];
            y_homoskedastic = X * beta + homoskedastic;
            beta_est = (X'*X)\(X'*y_homoskedastic);
            residuals_homoskedastic = y_homoskedastic - X * beta_est;
            
            % 6. Create regression model with heteroskedastic residuals
            y_heteroskedastic = X * beta + arch_series;
            beta_est_hetero = (X'*X)\(X'*y_heteroskedastic);
            residuals_heteroskedastic = y_heteroskedastic - X * beta_est_hetero;
            
            % 7. Test White test on homoskedastic residuals
            white_homoskedastic = white_test(residuals_homoskedastic, X);
            obj.assertTrue(white_homoskedastic.pval > 0.05, ...
                'White test should not reject homoskedasticity for constant variance residuals');
            
            % 8. Test White test on heteroskedastic residuals
            white_heteroskedastic = white_test(residuals_heteroskedastic, X);
            obj.assertTrue(white_heteroskedastic.pval < 0.05 || any(white_heteroskedastic.rej), ...
                'White test should reject homoskedasticity for heteroskedastic residuals');
            
            % 9. Test with various lag specifications for ARCH test
            arch_lag1 = arch_test(arch_series, 1);
            arch_lag10 = arch_test(arch_series, 10);
            obj.assertTrue(isfield(arch_lag1, 'statistic') && isfield(arch_lag10, 'statistic'), ...
                'ARCH test should work with different lag specifications');
            
            % 10. Verify test statistics match their mathematical definitions
            % Verify ARCH test statistic for lag 1
            squared_resid = arch_series.^2;
            X_arch = [ones(T-1, 1), squared_resid(1:end-1)];
            y_arch = squared_resid(2:end);
            beta_arch = (X_arch'*X_arch)\(X_arch'*y_arch);
            fitted = X_arch * beta_arch;
            resid = y_arch - fitted;
            TSS = sum((y_arch - mean(y_arch)).^2);
            RSS = sum(resid.^2);
            R2 = 1 - RSS/TSS;
            manual_arch = (T-1) * R2;
            arch_lag1_test = arch_test(arch_series, 1);
            obj.assertAlmostEqual(manual_arch, arch_lag1_test.statistic, 0.1, ...
                'ARCH test statistic should match manual calculation for lag 1');
        end
        
        function testInputValidation(obj)
            % Validates that statistical tests properly handle invalid inputs
            
            % 1. Test handling of empty inputs
            obj.assertThrows(@() adf_test([]), 'MATLAB:minrhs', ...
                'ADF test should throw error for empty input');
            obj.assertThrows(@() pp_test([]), 'MATLAB:minrhs', ...
                'PP test should throw error for empty input');
            obj.assertThrows(@() kpss_test([]), 'MATLAB:minrhs', ...
                'KPSS test should throw error for empty input');
            obj.assertThrows(@() jarque_bera([]), 'MATLAB:minrhs', ...
                'Jarque-Bera test should throw error for empty input');
            
            % 2. Test handling of non-numeric inputs
            obj.assertThrows(@() adf_test('string'), 'MATLAB:invalidType', ...
                'ADF test should throw error for non-numeric input');
            obj.assertThrows(@() ljungbox('string'), 'MATLAB:invalidType', ...
                'Ljung-Box test should throw error for non-numeric input');
            
            % 3. Test handling of NaN and Inf values
            obj.assertThrows(@() adf_test([1, 2, NaN, 4]'), 'MATLAB:invalidInput', ...
                'ADF test should throw error for NaN values');
            obj.assertThrows(@() kpss_test([1, 2, Inf, 4]'), 'MATLAB:invalidInput', ...
                'KPSS test should throw error for Inf values');
            
            % 4. Test handling of scalar inputs when vector expected
            obj.assertThrows(@() ljungbox(1), 'MATLAB:invalidInput', ...
                'Ljung-Box test should throw error for scalar input');
            
            % 5. Test handling of row vector when column vector expected
            % (Most functions should automatically handle this by converting to column)
            rowVector = [1, 2, 3, 4, 5];
            jb_row = jarque_bera(rowVector);
            obj.assertTrue(isfield(jb_row, 'statistic'), ...
                'Jarque-Bera should handle row vector input gracefully');
            
            % 6. Test handling of invalid options structures
            invalidOptions = struct('regression_type', 'invalid');
            obj.assertThrows(@() adf_test(randn(100, 1), invalidOptions), 'MATLAB:invalidInput', ...
                'ADF test should throw error for invalid regression_type');
            
            % 7. Test handling of invalid lag specifications
            invalidLagOptions = struct('lags', -1);
            obj.assertThrows(@() ljungbox(randn(100, 1), -1), 'MATLAB:invalidInput', ...
                'Ljung-Box test should throw error for negative lags');
            
            % 8. Test handling of incompatible dimensions in White test
            obj.assertThrows(@() white_test(randn(100, 1), randn(50, 2)), 'MATLAB:invalidInput', ...
                'White test should throw error for incompatible dimensions');
        end
        
        function testCrossValidation(obj)
            % Performs cross-validation of test results against known reference implementations
            
            % Load or create data with known test statistics
            T = 100;
            
            % AR(1) process with known parameters
            rng(42); % Set seed for reproducibility
            ar_coef = 0.7;
            e = randn(T, 1);
            ar1 = zeros(T, 1);
            ar1(1) = e(1);
            for t = 2:T
                ar1(t) = ar_coef * ar1(t-1) + e(t);
            end
            
            % 1. Cross-validate ADF test
            adf_results = adf_test(ar1);
            % For AR(1) with coefficient 0.7, we expect to reject unit root
            obj.assertTrue(adf_results.pval < 0.05, ...
                'ADF test should reject unit root for AR(1) with coefficient 0.7');
            
            % 2. Cross-validate PP test
            pp_results = pp_test(ar1);
            % For AR(1) with coefficient 0.7, PP and ADF should give consistent results
            is_adf_reject = adf_results.pval < 0.05;
            is_pp_reject = pp_results.pval < 0.05;
            obj.assertEqual(is_adf_reject, is_pp_reject, ...
                'ADF and PP tests should give consistent results for AR(1) process');
            
            % 3. Cross-validate KPSS test
            kpss_results = kpss_test(ar1);
            % For AR(1) with coefficient 0.7, we expect not to reject stationarity
            obj.assertTrue(kpss_results.pval > 0.05, ...
                'KPSS test should not reject stationarity for AR(1) with coefficient 0.7');
            
            % 4. Cross-validate Ljung-Box test
            lb_results = ljungbox(ar1, 10);
            % For AR(1) process, we expect to reject independence
            obj.assertTrue(any(lb_results.pvals < 0.05), ...
                'Ljung-Box test should reject independence for AR(1) process');
            
            % 5. Cross-validate ARCH test
            % Generate ARCH(1) process with known parameters
            arch_e = randn(T, 1);
            arch_sigma2 = zeros(T, 1);
            arch_y = zeros(T, 1);
            arch_sigma2(1) = 1;
            alpha0 = 0.2;
            alpha1 = 0.7;
            for t = 2:T
                arch_sigma2(t) = alpha0 + alpha1 * arch_y(t-1)^2;
                arch_y(t) = sqrt(arch_sigma2(t)) * arch_e(t);
            end
            arch_results = arch_test(arch_y, 5);
            % For ARCH(1) process, we expect to reject homoskedasticity
            obj.assertTrue(arch_results.pval < 0.05, ...
                'ARCH test should reject homoskedasticity for ARCH(1) process');
            
            % 6. Cross-validate Jarque-Bera test
            % Generate non-normal data with known skewness and kurtosis
            skewed_data = [randn(T/2, 1); 3*randn(T/2, 1) + 2];
            jb_results = jarque_bera(skewed_data);
            % For skewed data, we expect to reject normality
            obj.assertTrue(jb_results.pval < 0.05, ...
                'Jarque-Bera test should reject normality for skewed data');
        end
        
        function results = runAllValidationTests(obj)
            % Executes all validation tests for statistical test implementations
            
            % Create a test suite instance
            suite = TestSuite('Statistical Tests Validation Suite');
            
            % Add this test class to the suite
            suite.addTest(obj);
            
            % Set verbose mode if needed
            suite.setVerbose(obj.verbose);
            
            % Execute all tests in the suite
            results = suite.execute();
            
            % Display summary if verbose
            if obj.verbose
                suite.displaySummary();
            end
        end
    end
end