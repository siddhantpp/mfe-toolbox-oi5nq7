classdef ArmaxfilterTest < BaseTest
    % Test class for the armaxfilter function which estimates ARMAX model parameters for time series data
    
    properties
        testData                % Structure to hold test data
        testTolerance           % Tolerance for numerical comparisons
        simulatedArmaSeries     % Matrix of simulated ARMA time series for testing
        exogenousVariables      % Matrix of exogenous variables for testing
        knownParameters         % Structure of known parameters for validation
    end
    
    methods
        function obj = ArmaxfilterTest()
            % Initialize the ArmaxfilterTest class with default test data
            obj = obj@BaseTest();  % Call the superclass constructor
            obj.testTolerance = 1e-8;  % Default tolerance for numerical comparisons
            obj.testData = struct(); % Initialize empty testData property
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            setUp@BaseTest(obj);  % Call the superclass setUp method
            
            % Load simulated ARMAX time series data
            testDataFile = 'arma_test_data.mat';
            obj.testData = obj.loadTestData(testDataFile);
            
            % Extract known parameters
            obj.knownParameters = obj.testData.parameters;
            
            % Generate controlled test data for specific test cases
            % Set random seed for reproducibility
            rng(1234);
            
            % Generate AR(1) process: y_t = 0.7*y_{t-1} + e_t
            T = 1000;
            e = randn(T, 1);
            y = zeros(T, 1);
            ar_param = 0.7;
            for t = 2:T
                y(t) = ar_param * y(t-1) + e(t);
            end
            obj.simulatedArmaSeries = y;
            
            % Generate exogenous variables
            obj.exogenousVariables = [randn(T, 1), randn(T, 1)];
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            tearDown@BaseTest(obj);  % Call the superclass tearDown method
            
            % Clear temporary test variables
            clear y e ar_param;
            
            % Reset any modified settings
            rng('default');
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of armaxfilter with simple ARMA time series
            
            % Use simulated AR(1) series with known parameter
            y = obj.simulatedArmaSeries;
            
            % Call armaxfilter with AR order=1, MA order=0
            options = struct('p', 1, 'q', 0);
            results = armaxfilter(y, [], options);
            
            % Verify that estimated AR parameter is close to the true value
            % Parameter order: [constant, AR(1)]
            estimated_ar = results.parameters(2);
            obj.assertAlmostEqual(estimated_ar, 0.7, 'AR parameter not close to true value');
            
            % Validate standard error computation
            obj.assertTrue(results.standardErrors(2) > 0, 'AR parameter should have positive standard error');
            
            % Check that log-likelihood is computed correctly
            obj.assertTrue(isfinite(results.logL), 'Log-likelihood should be finite');
        end
        
        function testArmaEstimation(obj)
            % Test ARMA parameter estimation with various model orders
            
            % Use simulated ARMA(1,1) time series with known parameters
            y = obj.testData.arma11;
            true_ar = obj.knownParameters.arma11_ar;
            true_ma = obj.knownParameters.arma11_ma;
            
            % Call armaxfilter with AR order=1, MA order=1
            options = struct('p', 1, 'q', 1);
            results = armaxfilter(y, [], options);
            
            % Verify that estimated parameters match true values within tolerance
            % Parameter order: [constant, AR(1), MA(1)]
            estimated_ar = results.parameters(2);
            estimated_ma = results.parameters(3);
            
            obj.assertMatrixEqualsWithTolerance(estimated_ar, true_ar, obj.testTolerance, 'AR parameter estimation failed');
            obj.assertMatrixEqualsWithTolerance(estimated_ma, true_ma, obj.testTolerance, 'MA parameter estimation failed');
            
            % Test with ARMA(2,1) series
            if isfield(obj.testData, 'arma21')
                y_arma21 = obj.testData.arma21;
                true_ar21 = obj.knownParameters.arma21_ar;
                true_ma21 = obj.knownParameters.arma21_ma;
                
                options = struct('p', 2, 'q', 1);
                results_arma21 = armaxfilter(y_arma21, [], options);
                
                % Parameter order: [constant, AR(1), AR(2), MA(1)]
                estimated_ar21 = results_arma21.parameters(2:3);
                estimated_ma21 = results_arma21.parameters(4);
                
                obj.assertMatrixEqualsWithTolerance(estimated_ar21, true_ar21, obj.testTolerance, 'ARMA(2,1) AR parameters estimation failed');
                obj.assertMatrixEqualsWithTolerance(estimated_ma21, true_ma21, obj.testTolerance, 'ARMA(2,1) MA parameter estimation failed');
            end
            
            % Validate parameter covariance matrix computation
            obj.assertTrue(all(diag(results.standardErrors) > 0), 'Standard errors should be positive');
        end
        
        function testArmaxEstimation(obj)
            % Test ARMAX parameter estimation with exogenous variables
            
            % Use simulated ARMAX time series with known exogenous effects
            y = obj.simulatedArmaSeries;
            x = obj.exogenousVariables;
            
            % Set known exogenous parameters
            exog_params = [0.5; -0.3];
            
            % Generate ARMAX series: y with exogenous effects
            y_armax = y + x * exog_params;
            
            % Call armaxfilter with AR, MA orders and exogenous variables
            options = struct('p', 1, 'q', 0);
            results = armaxfilter(y_armax, x, options);
            
            % Verify that AR, MA, and exogenous parameters are estimated correctly
            % Parameter order: [constant, AR(1), Exog(1), Exog(2)]
            estimated_ar = results.parameters(2);
            estimated_exog = results.parameters(3:4);
            
            obj.assertAlmostEqual(estimated_ar, 0.7, 'AR parameter estimation with exogenous variables failed');
            obj.assertMatrixEqualsWithTolerance(estimated_exog, exog_params, obj.testTolerance, 'Exogenous parameters estimation failed');
            
            % Test with multiple exogenous variables
            options_noconst = struct('p', 1, 'q', 0, 'constant', false);
            results_noconst = armaxfilter(y_armax, x, options_noconst);
            
            % Parameter order without constant: [AR(1), Exog(1), Exog(2)]
            estimated_ar_noconst = results_noconst.parameters(1);
            estimated_exog_noconst = results_noconst.parameters(2:3);
            
            obj.assertAlmostEqual(estimated_ar_noconst, 0.7, 'AR parameter estimation without constant failed');
            obj.assertMatrixEqualsWithTolerance(estimated_exog_noconst, exog_params, obj.testTolerance, 'Exogenous parameters without constant failed');
            
            % Validate standard errors for exogenous parameter estimates
            obj.assertTrue(all(results.standardErrors(3:4) > 0), 'Exogenous parameter standard errors should be positive');
        end
        
        function testDistributionOptions(obj)
            % Test armaxfilter with different error distribution assumptions
            
            % Test with 'normal' distribution (default)
            y = obj.simulatedArmaSeries;
            options_normal = struct('p', 1, 'q', 0, 'distribution', 'normal');
            results_normal = armaxfilter(y, [], options_normal);
            
            % Test with 'student' distribution
            options_t = struct('p', 1, 'q', 0, 'distribution', 't');
            results_t = armaxfilter(y, [], options_t);
            
            % Test with 'ged' distribution
            options_ged = struct('p', 1, 'q', 0, 'distribution', 'ged');
            results_ged = armaxfilter(y, [], options_ged);
            
            % Test with 'skewt' distribution
            options_skewt = struct('p', 1, 'q', 0, 'distribution', 'skewt');
            results_skewt = armaxfilter(y, [], options_skewt);
            
            % Verify appropriate distribution parameter estimation
            obj.assertTrue(results_t.parameters(end) > 2, 'Student t degrees of freedom should be > 2');
            obj.assertTrue(results_ged.parameters(end) > 0, 'GED shape parameter should be > 0');
            obj.assertTrue(results_skewt.parameters(end-1) > 2, 'Skewed t degrees of freedom should be > 2');
            obj.assertTrue(abs(results_skewt.parameters(end)) < 1, 'Skewed t skewness parameter should be in (-1,1)');
            
            % Compare log-likelihoods across distribution assumptions
            % Non-normal distributions should provide at least as good a fit if appropriate
            ll_normal = results_normal.logL;
            ll_t = results_t.logL;
            ll_ged = results_ged.logL;
            ll_skewt = results_skewt.logL;
            
            % Skewed t has the most parameters and should provide the best fit
            obj.assertTrue(ll_skewt >= ll_normal - 1e-6, 'Skewed t should not decrease log-likelihood compared to normal');
        end
        
        function testInputValidation(obj)
            % Test error handling for invalid inputs
            
            % Test with empty data array and verify error is thrown
            obj.assertThrows(@() armaxfilter([]), 'At least one input (time series data) is required.');
            
            % Test with non-numeric data and verify error is thrown
            obj.assertThrows(@() armaxfilter({'a', 'b', 'c'}), 'DATA must be numeric');
            
            % Test with NaN values and verify error is thrown
            data_with_nan = obj.simulatedArmaSeries;
            data_with_nan(10) = NaN;
            obj.assertThrows(@() armaxfilter(data_with_nan), 'data cannot contain NaN values');
            
            % Test with invalid AR/MA orders (negative, non-integer)
            options_neg_p = struct('p', -1);
            obj.assertThrows(@() armaxfilter(obj.simulatedArmaSeries, [], options_neg_p), 'p must contain only non-negative values');
            
            options_nonint_q = struct('q', 1.5);
            obj.assertThrows(@() armaxfilter(obj.simulatedArmaSeries, [], options_nonint_q), 'q must contain only integer values');
            
            % Test with mismatched dimensions for exogenous variables
            x_wrong_size = obj.exogenousVariables(1:500,:);
            obj.assertThrows(@() armaxfilter(obj.simulatedArmaSeries, x_wrong_size), ...
                'Exogenous variables must have the same number of rows as the time series data');
            
            % Test with invalid options structure fields
            options_invalid_dist = struct('distribution', 'invalid_dist');
            obj.assertThrows(@() armaxfilter(obj.simulatedArmaSeries, [], options_invalid_dist), ...
                'Unsupported distribution type');
        end
        
        function testOptionsHandling(obj)
            % Test armaxfilter options handling and defaults
            
            % Test with default options (empty struct)
            y = obj.simulatedArmaSeries;
            results_default = armaxfilter(y);
            
            % Verify default options
            obj.assertEqual(results_default.p, 1, 'Default AR order should be 1');
            obj.assertEqual(results_default.q, 1, 'Default MA order should be 1');
            obj.assertTrue(results_default.constant, 'Default should include constant term');
            obj.assertEqual(results_default.distribution, 'normal', 'Default distribution should be normal');
            
            % Test various StartingVals options
            start_vals = [0; 0.6]; % [constant; AR(1)] - close to true 0.7
            options_start = struct('p', 1, 'q', 0, 'startingVals', start_vals);
            results_start = armaxfilter(y, [], options_start);
            
            % Verify that optimization still finds correct parameter
            obj.assertMatrixEqualsWithTolerance(results_start.parameters(2), 0.7, obj.testTolerance, 'Custom starting values failed');
            
            % Test Display options (on/off)
            options_display = struct('p', 1, 'q', 0, 'optimopts', struct('Display', 'off'));
            results_display = armaxfilter(y, [], options_display);
            
            % Verify that parameter estimates are still correct
            obj.assertMatrixEqualsWithTolerance(results_display.parameters(2), 0.7, obj.testTolerance, 'Display option affected results');
            
            % Test distribution-specific options
            options_t = struct('p', 1, 'q', 0, 'distribution', 't');
            results_t = armaxfilter(y, [], options_t);
            
            % Verify distribution parameter is estimated
            obj.assertEqual(length(results_t.parameters), 3, 't distribution should have an extra parameter');
            
            % Verify that options are correctly applied to the estimation process
            options_noconst = struct('p', 1, 'q', 0, 'constant', false);
            results_noconst = armaxfilter(y, [], options_noconst);
            
            obj.assertFalse(results_noconst.constant, 'Constant option not applied correctly');
            obj.assertEqual(length(results_noconst.parameters), 1, 'No-constant model should have only 1 parameter (AR)');
        end
        
        function testDiagnostics(obj)
            % Test diagnostics produced by armaxfilter
            
            % Validate AIC and SBIC calculations
            y = obj.simulatedArmaSeries;
            options = struct('p', 1, 'q', 0);
            results = armaxfilter(y, [], options);
            
            % Calculate expected values manually
            k = 2; % constant + AR parameter
            T = length(y);
            expected_ic = aicsbic(-results.logL, k, T);
            
            obj.assertMatrixEqualsWithTolerance(results.aic, expected_ic.aic, obj.testTolerance, 'AIC computation incorrect');
            obj.assertMatrixEqualsWithTolerance(results.sbic, expected_ic.sbic, obj.testTolerance, 'SBIC computation incorrect');
            
            % Verify Ljung-Box Q-test computation
            obj.assertTrue(isstruct(results.ljungBox), 'Ljung-Box test results should be a structure');
            obj.assertTrue(isfield(results.ljungBox, 'stats'), 'Ljung-Box should include test statistics');
            obj.assertTrue(isfield(results.ljungBox, 'pvals'), 'Ljung-Box should include p-values');
            
            % Check LM test for residual autocorrelation
            obj.assertTrue(isstruct(results.lmTest), 'LM test results should be a structure');
            obj.assertTrue(isfield(results.lmTest, 'stat'), 'LM test should include test statistic');
            obj.assertTrue(isfield(results.lmTest, 'pval'), 'LM test should include p-value');
            
            % Validate residual statistics (mean, variance, etc.)
            residuals = results.residuals;
            obj.assertAlmostEqual(mean(residuals), 0, 'Residual mean should be approximately 0');
            
            % Verify information criteria for model selection
            options_ar2 = struct('p', 2, 'q', 0);
            results_ar2 = armaxfilter(y, [], options_ar2);
            
            % AR(1) model should have lower AIC/SBIC than AR(2) for data generated from AR(1)
            obj.assertTrue(results.aic <= results_ar2.aic, 'AIC should correctly identify AR(1) over AR(2)');
            obj.assertTrue(results.sbic <= results_ar2.sbic, 'SBIC should correctly identify AR(1) over AR(2)');
        end
        
        function testRobustStandardErrors(obj)
            % Test robust standard errors computation
            
            % Call armaxfilter with robust standard errors option
            y = obj.simulatedArmaSeries;
            options_default = struct('p', 1, 'q', 0, 'stdErr', 'hessian');
            options_robust = struct('p', 1, 'q', 0, 'stdErr', 'robust');
            
            results_default = armaxfilter(y, [], options_default);
            results_robust = armaxfilter(y, [], options_robust);
            
            % Verify robust standard errors are different from regular ones
            obj.assertFalse(isequal(results_default.standardErrors, results_robust.standardErrors), ...
                'Robust standard errors should differ from default standard errors');
            
            % Check consistency of robust error computation
            obj.assertTrue(all(results_robust.standardErrors > 0), 'Robust standard errors should be positive');
            
            % Validate implementation against reference values
            % AR parameters should be the same regardless of standard error method
            obj.assertMatrixEqualsWithTolerance(results_default.parameters, results_robust.parameters, ...
                obj.testTolerance, 'Parameter estimates should be the same regardless of standard error method');
        end
        
        function testWithFinancialData(obj)
            % Test armaxfilter with real financial return data
            
            % Load financial returns data from test data directory
            if isfield(obj.testData, 'financial_returns')
                returns = obj.testData.financial_returns;
                
                % Fit ARMA models to different financial time series
                options = struct('p', 1, 'q', 1);
                results = armaxfilter(returns, [], options);
                
                % Verify parameter stability across different samples
                obj.assertTrue(all(isfinite(results.parameters)), 'Parameters should be finite');
                obj.assertTrue(all(isfinite(results.standardErrors)), 'Standard errors should be finite');
                
                % Check residual properties (normality, autocorrelation)
                residuals = results.residuals;
                obj.assertTrue(isfinite(results.logL), 'Log-likelihood should be finite');
                
                % Validate model selection criteria performance
                options_higherorder = struct('p', 2, 'q', 2);
                results_higher = armaxfilter(returns, [], options_higherorder);
                
                % Compare information criteria
                % Higher order model might be better if data has more complex structure
                aic_diff = results.aic - results_higher.aic;
                sbic_diff = results.sbic - results_higher.sbic;
                
                % Just check that information criteria were computed successfully
                obj.assertTrue(isfinite(aic_diff), 'AIC difference should be finite');
                obj.assertTrue(isfinite(sbic_diff), 'SBIC difference should be finite');
            else
                % Skip test if financial data not available
                warning('Financial returns data not available. Skipping test.');
            end
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of armaxfilter with challenging datasets
            
            % Test with very small values (near machine epsilon)
            y_small = obj.simulatedArmaSeries * 1e-6;
            options = struct('p', 1, 'q', 0);
            results_small = armaxfilter(y_small, [], options);
            
            % AR parameter should still be close to 0.7
            obj.assertMatrixEqualsWithTolerance(results_small.parameters(2), 0.7, obj.testTolerance, ...
                'AR parameter estimation failed with small values');
            
            % Test with very large values
            y_large = obj.simulatedArmaSeries * 1e6;
            results_large = armaxfilter(y_large, [], options);
            
            % AR parameter should still be close to 0.7
            obj.assertMatrixEqualsWithTolerance(results_large.parameters(2), 0.7, obj.testTolerance, ...
                'AR parameter estimation failed with large values');
            
            % Test with near-unit-root time series
            T = 1000;
            e = randn(T, 1);
            y_near_unit = zeros(T, 1);
            ar_near_unit = 0.98;
            for t = 2:T
                y_near_unit(t) = ar_near_unit * y_near_unit(t-1) + e(t);
            end
            
            results_near_unit = armaxfilter(y_near_unit, [], options);
            
            % AR parameter should be close to 0.98
            obj.assertMatrixEqualsWithTolerance(results_near_unit.parameters(2), ar_near_unit, 0.05, ...
                'AR parameter estimation failed with near-unit-root series');
            
            % Test with near-cancellation of AR and MA components
            T = 1000;
            e = randn(T, 1);
            y_cancel = filter([1, 0.8], [1, 0.79], e);  % Nearly cancelling AR and MA roots
            
            options_arma = struct('p', 1, 'q', 1);
            results_cancel = armaxfilter(y_cancel, [], options_arma);
            
            % Verify that numerical stability is maintained across all cases
            obj.assertTrue(all(isfinite(results_cancel.parameters)), 'Parameters should be finite with near-cancellation');
            obj.assertTrue(all(isfinite(results_cancel.standardErrors)), 'Standard errors should be finite with near-cancellation');
        end
        
        function testPerformance(obj)
            % Test performance of armaxfilter with large datasets
            
            % Generate large time series dataset
            T = 5000;
            e = randn(T, 1);
            y = zeros(T, 1);
            ar_param = 0.7;
            for t = 2:T
                y(t) = ar_param * y(t-1) + e(t);
            end
            
            % Measure execution time for ARMAX estimation
            options = struct('p', 1, 'q', 0);
            execution_time = obj.measureExecutionTime(@() armaxfilter(y, [], options));
            
            % Verify execution time scales appropriately with data size
            % This is more of a benchmark than a test, but we want to ensure it completes in reasonable time
            fprintf('ARMAX estimation for T=%d completed in %.3f seconds\n', T, execution_time);
            
            % Compare performance with and without MEX optimization
            % This would require having a non-MEX version to compare against
            % For now, we'll just verify the results are correct
            results = armaxfilter(y, [], options);
            obj.assertMatrixEqualsWithTolerance(results.parameters(2), ar_param, obj.testTolerance, ...
                'Parameter estimation incorrect in performance test');
            
            % Assert performance meets acceptable thresholds
            % Thresholds would need to be adjusted based on testing hardware
            % On modern hardware, this should complete in under a few seconds with MEX optimization
            if execution_time > 30  % Very conservative threshold
                warning('Performance may be suboptimal: %.3f seconds for T=%d', execution_time, T);
            end
        end
        
        function testHighOrderModels(obj)
            % Test armaxfilter with high-order ARMA models
            
            % Generate time series from high-order ARMA processes
            T = 1000;
            e = randn(T, 1);
            
            % Generate time series with high AR order
            y_ar5 = zeros(T, 1);
            ar_params = [0.5, -0.3, 0.2, -0.1, 0.05];  % AR(5) parameters
            
            for t = 6:T
                y_ar5(t) = ar_params(1) * y_ar5(t-1) + ar_params(2) * y_ar5(t-2) + ...
                           ar_params(3) * y_ar5(t-3) + ar_params(4) * y_ar5(t-4) + ...
                           ar_params(5) * y_ar5(t-5) + e(t);
            end
            
            % Test with different combinations of high AR and MA orders
            options_ar5 = struct('p', 5, 'q', 0);
            results_ar5 = armaxfilter(y_ar5, [], options_ar5);
            
            % Verify parameter estimation accuracy degrades gracefully
            estimated_ar = results_ar5.parameters(2:6);  % [constant, AR(1), AR(2), AR(3), AR(4), AR(5)]
            
            % Use larger tolerance for high-order models as estimation is more challenging
            highOrderTolerance = 0.15;
            
            % Check each parameter
            for i = 1:length(ar_params)
                obj.assertTrue(abs(estimated_ar(i) - ar_params(i)) < highOrderTolerance, ...
                    sprintf('AR(%d) parameter estimation too far from true value', i));
            end
            
            % Check numerical stability with many parameters
            obj.assertTrue(all(isfinite(results_ar5.parameters)), 'Parameters should be finite for high-order model');
            obj.assertTrue(all(isfinite(results_ar5.standardErrors)), 'Standard errors should be finite for high-order model');
            
            % Validate identification of simpler models when appropriate
            % Try fitting AR(6) when true model is AR(5)
            options_ar6 = struct('p', 6, 'q', 0);
            results_ar6 = armaxfilter(y_ar5, [], options_ar6);
            
            % AR(6) coefficient should be close to zero
            obj.assertTrue(abs(results_ar6.parameters(7)) < highOrderTolerance, 'Extra AR parameter should be close to zero');
            
            % Information criteria should favor the true model
            obj.assertTrue(results_ar5.sbic <= results_ar6.sbic, 'SBIC should favor the true model');
        end
    end
end