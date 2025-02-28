classdef PpTestTest < BaseTest
    % PpTestTest Unit test class for the Phillips-Perron (PP) test implementation
    %
    % This class tests the functionality, accuracy, and robustness of the PP test
    % function for unit root testing in time series analysis. It verifies proper
    % handling of different regression specifications, lag selection, and various
    % data conditions including stationary and non-stationary series.
    
    properties
        testData        % General test data matrix
        stationaryData  % Known stationary time series
        integratedData  % Series with unit root (non-stationary)
        financialData   % Real financial returns data
        macroData       % Macroeconomic time series data
    end
    
    methods
        function obj = PpTestTest()
            % Initialize the PpTestTest class with default configurations
            obj@BaseTest(); % Call parent constructor
            obj.testName = 'Phillips-Perron Test Unit Tests';
            
            % Initialize properties as empty
            obj.testData = [];
            obj.stationaryData = [];
            obj.integratedData = [];
            obj.financialData = [];
            obj.macroData = [];
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj); % Call parent setUp
            
            % Load test data from MAT files
            obj.financialData = obj.loadTestData('financial_returns.mat');
            obj.macroData = obj.loadTestData('macroeconomic_data.mat');
            
            % Set random number generator seed for reproducibility
            rng(123); 
            
            % Generate stationary test series (random data)
            obj.stationaryData = randn(100, 1);
            
            % Generate integrated series (unit root present)
            obj.integratedData = cumsum(randn(100, 1));
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj); % Call parent tearDown
            
            % Clean up any temporary test data
            % (No specific cleanup needed for this test class)
        end
        
        function testPpBasic(obj)
            % Test basic PP test functionality with default parameters
            
            % Test on stationary data
            results_stat = pp_test(obj.stationaryData);
            
            % Verify results for stationary series (should reject unit root)
            obj.assertTrue(results_stat.stat_tau < -2.5, 'Test statistic should be highly negative for stationary series');
            obj.assertTrue(results_stat.pval < 0.05, 'P-value should be small for stationary series');
            
            % Test on integrated data
            results_int = pp_test(obj.integratedData);
            
            % Verify results for integrated series (should not reject unit root)
            obj.assertTrue(results_int.stat_tau > -2.0, 'Test statistic should be close to zero for integrated series');
            obj.assertTrue(results_int.pval > 0.05, 'P-value should be large for integrated series');
            
            % Verify null hypothesis decisions match expectations
            obj.assertFalse(results_stat.pval > 0.05, 'Should reject null for stationary data');
            obj.assertTrue(results_int.pval > 0.05, 'Should not reject null for integrated data');
        end
        
        function testRegressionTypes(obj)
            % Test PP test with different regression specifications
            
            % Test no constant regression type
            options = struct();
            options.regression_type = 'n';
            results_n = pp_test(obj.stationaryData, options);
            
            % Test constant only regression type
            options.regression_type = 'c';
            results_c = pp_test(obj.stationaryData, options);
            
            % Test constant and trend regression type
            options.regression_type = 'ct';
            results_ct = pp_test(obj.stationaryData, options);
            
            % Verify different regression types give different results
            obj.assertFalse(abs(results_n.stat_tau - results_c.stat_tau) < 0.01, 'Regression types n and c should give different results');
            obj.assertFalse(abs(results_c.stat_tau - results_ct.stat_tau) < 0.01, 'Regression types c and ct should give different results');
            
            % Verify critical values are different for each regression type
            obj.assertFalse(results_n.cv_5pct == results_c.cv_5pct, 'Critical values should differ between regression types n and c');
            obj.assertFalse(results_c.cv_5pct == results_ct.cv_5pct, 'Critical values should differ between regression types c and ct');
            
            % Verify results consistency with econometric theory
            obj.assertTrue(results_n.cv_5pct > results_c.cv_5pct, 'No constant model should have less negative critical values than constant model');
            obj.assertTrue(results_c.cv_5pct > results_ct.cv_5pct, 'Constant model should have less negative critical values than trend model');
        end
        
        function testLagSpecification(obj)
            % Test PP test with different lag specifications
            
            % Test with specific lag
            options = struct();
            options.lags = 5;
            results_lag5 = pp_test(obj.stationaryData, options);
            
            % Test with different lag
            options.lags = 10;
            results_lag10 = pp_test(obj.stationaryData, options);
            
            % Verify lag value is correctly recorded in results
            obj.assertEqual(results_lag5.lags, 5, 'Lag value should be 5');
            obj.assertEqual(results_lag10.lags, 10, 'Lag value should be 10');
            
            % Verify that changing lag affects results but not drastically
            obj.assertFalse(results_lag5.stat_tau == results_lag10.stat_tau, 'Different lags should give different test statistics');
            obj.assertTrue(abs(results_lag5.stat_tau - results_lag10.stat_tau) < 1.0, 'Difference in test statistics with different lags should be moderate');
            
            % Verify automatic lag selection works
            default_results = pp_test(obj.stationaryData);
            expected_auto_lag = floor(4 * (length(obj.stationaryData)/100)^0.25);
            obj.assertEqual(default_results.lags, expected_auto_lag, 'Default lag should follow automatic selection rule');
        end
        
        function testFinancialData(obj)
            % Test PP test with real financial returns data
            
            % Get financial returns series
            returns = obj.financialData.returns;
            
            % Run PP test with default settings
            results = pp_test(returns);
            
            % Financial returns should be stationary
            obj.assertTrue(results.pval < 0.05, 'Financial returns should be stationary (reject unit root)');
            
            % Test with different regression types
            options = struct();
            options.regression_type = 'ct';
            results_ct = pp_test(returns, options);
            
            % Financial returns should still be stationary with trend
            obj.assertTrue(results_ct.pval < 0.05, 'Financial returns should be stationary even with trend term');
            
            % Verify results are consistent with financial theory
            obj.assertTrue(results.stat_tau < results.cv_5pct, 'Returns should have test statistic below 5% critical value');
        end
        
        function testMacroData(obj)
            % Test PP test with macroeconomic data
            
            % Test GDP (likely non-stationary)
            gdp = obj.macroData.gdp;
            results_gdp = pp_test(gdp);
            
            % Test interest rates
            rates = obj.macroData.interest_rates;
            results_rates = pp_test(rates);
            
            % Test inflation data
            inflation = obj.macroData.inflation;
            results_inflation = pp_test(inflation);
            
            % GDP should typically have a unit root
            obj.assertTrue(results_gdp.pval > 0.05, 'GDP series should have unit root (not reject null)');
            
            % Test first differences of GDP (should be stationary)
            dgdp = diff(gdp);
            results_dgdp = pp_test(dgdp);
            obj.assertTrue(results_dgdp.pval < 0.05, 'Differenced GDP should be stationary');
            
            % Verify results match expected macroeconomic properties
            obj.assertTrue(results_gdp.stat_tau > results_gdp.cv_5pct, 'GDP should have test statistic above 5% critical value');
            obj.assertTrue(results_dgdp.stat_tau < results_dgdp.cv_5pct, 'Differenced GDP should have test statistic below 5% critical value');
        end
        
        function testCompareWithAdf(obj)
            % Compare PP test results with ADF test results
            % Note: This assumes adf_test function exists in the codebase
            
            try
                % Try to load the ADF test function
                adf_test_exists = exist('adf_test', 'file') == 2;
            catch
                adf_test_exists = false;
            end
            
            % Skip the test if adf_test doesn't exist
            if ~adf_test_exists
                warning('Skipping PP vs ADF comparison - adf_test function not available');
                return;
            end
            
            % For stationary data, both should reject null
            pp_results = pp_test(obj.stationaryData);
            adf_results = adf_test(obj.stationaryData);
            
            % For integrated data, both should fail to reject null
            pp_int_results = pp_test(obj.integratedData);
            adf_int_results = adf_test(obj.integratedData);
            
            % Both tests should agree on clear cases
            obj.assertTrue((pp_results.pval < 0.05) == (adf_results.pval < 0.05), 'PP and ADF should agree on stationary data');
            obj.assertTrue((pp_int_results.pval > 0.05) == (adf_int_results.pval > 0.05), 'PP and ADF should agree on integrated data');
            
            % Compare results under heteroskedasticity
            hetero_data = obj.stationaryData .* linspace(1, 3, length(obj.stationaryData))';
            pp_hetero = pp_test(hetero_data);
            adf_hetero = adf_test(hetero_data);
            
            % PP should be more robust to heteroskedasticity
            pp_better = abs(pp_hetero.pval - pp_results.pval) < abs(adf_hetero.pval - adf_results.pval);
            obj.assertTrue(pp_better, 'PP test should be more robust to heteroskedasticity than ADF');
        end
        
        function testEdgeCases(obj)
            % Test PP test with edge cases and boundary conditions
            
            % Test with small sample
            small_sample = randn(20, 1);
            results_small = pp_test(small_sample);
            
            % Test with data containing outliers
            outlier_data = obj.stationaryData;
            outlier_data(10) = 10 * max(abs(outlier_data)); % Add outlier
            results_outlier = pp_test(outlier_data);
            
            % Test with heteroskedastic data
            hetero_data = randn(100, 1) .* linspace(1, 5, 100)';
            results_hetero = pp_test(hetero_data);
            
            % Test with constant data
            try
                constant_data = ones(100, 1);
                results_constant = pp_test(constant_data);
                has_constant_error = false;
            catch
                has_constant_error = true;
            end
            
            % Verify function handles edge cases appropriately
            obj.assertTrue(isfield(results_small, 'stat_tau'), 'Should handle small sample sizes');
            obj.assertTrue(isfield(results_outlier, 'stat_tau'), 'Should handle outliers');
            obj.assertTrue(isfield(results_hetero, 'stat_tau'), 'Should handle heteroskedasticity');
            
            % Constant data may cause singular matrix in regression - either should
            % handle gracefully or throw appropriate error
            if ~has_constant_error
                obj.assertTrue(isfield(results_constant, 'stat_tau'), 'Should handle constant data series');
            end
        end
        
        function testInvalidInputs(obj)
            % Test PP test with invalid inputs
            
            % Test with empty data
            obj.assertThrows(@() pp_test([]), 'y cannot be empty', 'Empty data should throw error');
            
            % Test with invalid regression type
            options = struct();
            options.regression_type = 'invalid';
            obj.assertThrows(@() pp_test(obj.stationaryData, options), 'options.regression_type must be one of', 'Invalid regression type should throw error');
            
            % Test with invalid lag specification
            options = struct();
            options.regression_type = 'c';
            options.lags = -1;
            obj.assertThrows(@() pp_test(obj.stationaryData, options), 'must contain only non-negative values', 'Negative lag should throw error');
            
            % Test with invalid data types
            obj.assertThrows(@() pp_test('string'), 'must be numeric', 'Non-numeric data should throw error');
        end
        
        function testPerformance(obj)
            % Test performance characteristics of the PP test implementation
            
            % Measure execution time for different data sizes
            small_data = obj.stationaryData(1:50);
            large_data = obj.stationaryData;
            
            % Execute with small data
            t_start = tic;
            pp_test(small_data);
            small_time = toc(t_start);
            
            % Execute with large data
            t_start = tic;
            pp_test(large_data);
            large_time = toc(t_start);
            
            % Measure execution time for different lag values
            options_small_lag = struct('lags', 2);
            options_large_lag = struct('lags', 10);
            
            % Execute with small lag
            t_start = tic;
            pp_test(obj.stationaryData, options_small_lag);
            small_lag_time = toc(t_start);
            
            % Execute with large lag
            t_start = tic;
            pp_test(obj.stationaryData, options_large_lag);
            large_lag_time = toc(t_start);
            
            % Verify performance scales appropriately
            expected_ratio = length(large_data) / length(small_data);
            actual_ratio = large_time / small_time;
            
            obj.assertTrue(large_time > small_time, 'Larger data should take longer to process');
            obj.assertTrue(actual_ratio < expected_ratio * 2, 'Execution time should scale reasonably with data size');
            obj.assertTrue(large_lag_time > small_lag_time, 'Larger lag should take longer to process');
        end
        
        function testResultsStructure(obj)
            % Test that the results structure contains all expected fields
            
            % Run PP test
            results = pp_test(obj.stationaryData);
            
            % Verify all expected fields are present
            obj.assertTrue(isfield(results, 'stat_alpha'), 'Results should contain stat_alpha field');
            obj.assertTrue(isfield(results, 'stat_tau'), 'Results should contain stat_tau field');
            obj.assertTrue(isfield(results, 'pval'), 'Results should contain pval field');
            obj.assertTrue(isfield(results, 'cv_1pct'), 'Results should contain cv_1pct field');
            obj.assertTrue(isfield(results, 'cv_5pct'), 'Results should contain cv_5pct field');
            obj.assertTrue(isfield(results, 'cv_10pct'), 'Results should contain cv_10pct field');
            obj.assertTrue(isfield(results, 'regression_type'), 'Results should contain regression_type field');
            obj.assertTrue(isfield(results, 'lags'), 'Results should contain lags field');
            obj.assertTrue(isfield(results, 'nobs'), 'Results should contain nobs field');
            
            % Verify field types and values are reasonable
            obj.assertTrue(isnumeric(results.stat_alpha), 'stat_alpha should be numeric');
            obj.assertTrue(isnumeric(results.stat_tau), 'stat_tau should be numeric');
            obj.assertTrue(isnumeric(results.pval) && results.pval >= 0 && results.pval <= 1, 'pval should be between 0 and 1');
            obj.assertTrue(ischar(results.regression_type), 'regression_type should be character');
            obj.assertTrue(isnumeric(results.lags) && results.lags >= 0, 'lags should be non-negative numeric');
        end
        
        function result = compareWithReference(obj, results, reference, tolerance)
            % Helper method to compare test results with reference values
            
            if nargin < 4
                tolerance = 1e-8;
            end
            
            % Compare test statistics
            stat_tau_match = abs(results.stat_tau - reference.stat_tau) < tolerance;
            stat_alpha_match = abs(results.stat_alpha - reference.stat_alpha) < tolerance;
            pval_match = abs(results.pval - reference.pval) < tolerance;
            
            % Compare critical values
            cv_match = all(abs([results.cv_1pct, results.cv_5pct, results.cv_10pct] - ...
                              [reference.cv_1pct, reference.cv_5pct, reference.cv_10pct]) < tolerance);
            
            % Return true if all comparisons pass
            result = stat_tau_match && stat_alpha_match && pval_match && cv_match;
        end
        
        function series = generateSeriesWithFractionalIntegration(obj, numObservations, integrationOrder)
            % Helper method to generate test time series with fractional integration
            
            % Generate random white noise innovations
            innovations = randn(numObservations, 1);
            
            % Apply fractional differencing (simplified implementation)
            if integrationOrder == 0
                series = innovations;
            elseif integrationOrder == 1
                series = cumsum(innovations);
            else
                % For other values, would need a more complex implementation
                % This is a simplified approach for testing purposes
                series = innovations;
                for i = 2:numObservations
                    series(i) = series(i) + integrationOrder * sum(series(1:i-1));
                end
                
                % Normalize to prevent explosion
                series = series / max(abs(series)) * max(abs(innovations)) * 5;
            end
        end
    end
end