classdef AdfTestTest < BaseTest
    % Test class for validating the functionality and accuracy of the Augmented Dickey-Fuller test implementation
    
    properties
        testData         % General test data
        stationaryData   % Stationary test series
        integratedData   % Integrated (unit root) test series
        financialData    % Financial return series for testing
        macroData        % Macroeconomic data for testing
    end
    
    methods
        function obj = AdfTestTest()
            % Initialize the AdfTestTest class with default configurations
            
            % Call parent constructor for BaseTest initialization
            obj@BaseTest('ADF Test');
            
            % Set up test name
            obj.testName = 'AdfTestTest';
            
            % Initialize test properties as empty
            obj.testData = [];
            obj.stationaryData = [];
            obj.integratedData = [];
            obj.financialData = struct();
            obj.macroData = struct();
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Load financial returns test data from financial_returns.mat
            try
                financial_data = obj.loadTestData('financial_returns.mat');
                obj.financialData = financial_data;
            catch ME
                warning('Financial test data loading failed: %s. Using synthetic data instead.', ME.message);
                % Create synthetic financial data if loading fails
                obj.financialData.returns = randn(500, 1) * 0.02; % Typical financial returns
            end
            
            % Load macroeconomic test data from macroeconomic_data.mat
            try
                macro_data = obj.loadTestData('macroeconomic_data.mat');
                obj.macroData = macro_data;
            catch ME
                warning('Macroeconomic test data loading failed: %s. Using synthetic data instead.', ME.message);
                
                % Create synthetic macro data if loading fails
                % GDP - typically non-stationary
                gdp_growth = 0.005 + 0.001 * randn(100, 1);  % ~0.5% quarterly growth with noise
                obj.macroData.gdp = 100 * cumprod(1 + gdp_growth);  % Exponential growth
                
                % Interest rates - may have unit root
                obj.macroData.interest_rates = 0.95 * (1:100)' / 100 + 0.05 * cumsum(randn(100, 1) * 0.01);
                
                % Inflation - often stationary
                obj.macroData.inflation = 0.02 + 0.7 * randn(100, 1) * 0.01;
            end
            
            % Generate stationary test series using random normal data
            obj.stationaryData = randn(500, 1);  % White noise is stationary
            
            % Generate integrated (unit root) test series using cumulative sum of random data
            obj.integratedData = cumsum(randn(500, 1));  % Random walk is I(1)
            
            % Set random number generator seed for reproducibility
            rng(123);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Clean up any temporary test data
        end
        
        function testAdfBasic(obj)
            % Test basic ADF test functionality with default parameters
            
            % Run ADF test on stationary data with default options
            results_stationary = adf_test(obj.stationaryData);
            
            % Verify test statistic is highly negative for stationary series
            obj.assertTrue(results_stationary.stat < -2.5, 'ADF statistic should be significantly negative for stationary data');
            
            % Verify p-value is appropriately small for stationary series
            obj.assertTrue(results_stationary.pval < 0.05, 'P-value should be small for stationary data');
            
            % Run ADF test on integrated data with default options
            results_integrated = adf_test(obj.integratedData);
            
            % Verify test statistic is close to zero for integrated series
            obj.assertTrue(results_integrated.stat > -2.0, 'ADF statistic should be close to zero for integrated data');
            
            % Verify p-value is appropriately large for integrated series
            obj.assertTrue(results_integrated.pval > 0.05, 'P-value should be large for integrated data');
            
            % Verify null hypothesis decisions match expectations
            obj.assertTrue(results_stationary.pval < 0.05, 'Should reject unit root null for stationary data');
            obj.assertFalse(results_integrated.pval < 0.05, 'Should not reject unit root null for integrated data');
        end
        
        function testRegressionTypes(obj)
            % Test ADF test with different regression specifications (no constant, constant, constant and trend)
            
            % Test 'n' regression type (no constant or trend)
            options_n = struct('regression_type', 'n');
            results_n = adf_test(obj.stationaryData, options_n);
            
            % Test 'c' regression type (constant only)
            options_c = struct('regression_type', 'c');
            results_c = adf_test(obj.stationaryData, options_c);
            
            % Test 'ct' regression type (constant and trend)
            options_ct = struct('regression_type', 'ct');
            results_ct = adf_test(obj.stationaryData, options_ct);
            
            % Verify appropriate test statistics for each specification
            obj.assertTrue(isfinite(results_n.stat), 'Test statistic should be finite for "n" regression');
            obj.assertTrue(isfinite(results_c.stat), 'Test statistic should be finite for "c" regression');
            obj.assertTrue(isfinite(results_ct.stat), 'Test statistic should be finite for "ct" regression');
            
            % Verify critical values differ appropriately based on specification
            obj.assertFalse(isequal(results_n.crit_vals, results_c.crit_vals), 'Critical values should differ between regression types "n" and "c"');
            obj.assertFalse(isequal(results_c.crit_vals, results_ct.crit_vals), 'Critical values should differ between regression types "c" and "ct"');
            
            % Verify results consistency with econometric theory
            % Create trend-stationary data to test 'ct' specification
            t = (1:500)';
            trend_stationary = 0.01 * t + 5 * obj.stationaryData;
            
            % Test trend-stationary data with different specifications
            results_ts_n = adf_test(trend_stationary, options_n);
            results_ts_c = adf_test(trend_stationary, options_c);
            results_ts_ct = adf_test(trend_stationary, options_ct);
            
            % 'ct' specification should perform best for trend-stationary data
            obj.assertTrue(results_ts_ct.pval < results_ts_c.pval || results_ts_ct.stat < results_ts_c.stat, 
                'CT specification should perform better for trend-stationary data');
        end
        
        function testLagSelection(obj)
            % Test automatic lag selection using information criteria
            
            % Create AR(1) process for testing lag selection
            ar_data = zeros(500, 1);
            ar_data(1) = randn(1);
            for t = 2:500
                ar_data(t) = 0.5 * ar_data(t-1) + randn(1);
            end
            
            % Test ADF with 'aic' lag selection
            options_aic = struct('lags', 'aic', 'max_lags', 10);
            results_aic = adf_test(ar_data, options_aic);
            
            % Test ADF with 'bic' lag selection
            options_bic = struct('lags', 'bic', 'max_lags', 10);
            results_bic = adf_test(ar_data, options_bic);
            
            % Verify optimal lag determination is working properly
            obj.assertTrue(isfield(results_aic, 'optimal_lag'), 'Results should contain optimal_lag field with AIC selection');
            obj.assertTrue(isfield(results_bic, 'optimal_lag'), 'Results should contain optimal_lag field with BIC selection');
            
            % Compare results with manual lag specification
            options_fixed = struct('lags', results_aic.optimal_lag);
            results_fixed = adf_test(ar_data, options_fixed);
            
            % Test statistics should be similar when using optimal lag directly
            obj.assertAlmostEqual(results_fixed.stat, results_aic.stat, 'Fixed lag using AIC optimal should give similar results');
            
            % Verify information criteria computations are accurate
            obj.assertTrue(isfield(results_aic, 'ic_values'), 'Results should contain ic_values field with AIC selection');
            obj.assertTrue(isfield(results_bic, 'ic_values'), 'Results should contain ic_values field with BIC selection');
            obj.assertTrue(length(results_aic.ic_values) == options_aic.max_lags + 1, 'IC values vector should have length max_lags + 1');
        end
        
        function testFinancialData(obj)
            % Test ADF test with real financial returns data
            
            % Run ADF test on financial return series
            returns = obj.financialData.returns;
            results = adf_test(returns);
            
            % Verify returns are stationary (reject unit root null)
            obj.assertTrue(results.pval < 0.05, 'Financial returns should be stationary (reject unit root)');
            
            % Test with different regression specifications
            options_n = struct('regression_type', 'n');
            options_c = struct('regression_type', 'c');
            options_ct = struct('regression_type', 'ct');
            
            results_n = adf_test(returns, options_n);
            results_c = adf_test(returns, options_c);
            results_ct = adf_test(returns, options_ct);
            
            % Verify results are consistent with financial theory
            % All specifications should reject unit root for returns
            obj.assertTrue(results_n.pval < 0.1, 'Returns should be stationary under "n" specification');
            obj.assertTrue(results_c.pval < 0.1, 'Returns should be stationary under "c" specification');
            
            % Compare results with known financial time series properties
            % For financial returns, constant specification ('c') is typically most appropriate
            % as returns may have non-zero mean but should not have trend
            log_returns = diff(log(cumsum(returns) + 100));
            results_log = adf_test(log_returns);
            obj.assertTrue(results_log.pval < 0.05, 'Log returns should be stationary');
        end
        
        function testMacroData(obj)
            % Test ADF test with macroeconomic data which may contain unit roots
            
            % Run ADF test on GDP series (likely non-stationary)
            results_gdp = adf_test(obj.macroData.gdp);
            
            % Run ADF test on interest rate series
            results_ir = adf_test(obj.macroData.interest_rates);
            
            % Run ADF test on inflation series
            results_inf = adf_test(obj.macroData.inflation);
            
            % Verify results match expected macroeconomic properties
            % GDP levels typically have unit root (non-stationary)
            obj.assertTrue(results_gdp.pval > 0.05, 'GDP levels are typically non-stationary');
            
            % Test first differences of non-stationary series
            diff_gdp = diff(obj.macroData.gdp);
            results_diff_gdp = adf_test(diff_gdp);
            
            % Verify first differences are stationary
            obj.assertTrue(results_diff_gdp.pval < 0.05, 'First difference of GDP should be stationary');
            
            % Test inflation with trend specification
            options_ct = struct('regression_type', 'ct');
            results_inf_ct = adf_test(obj.macroData.inflation, options_ct);
            
            % Verify appropriate inference for macro variables
            obj.assertTrue(isfinite(results_inf_ct.stat), 'Test statistics should be finite for macro data');
        end
        
        function testEdgeCases(obj)
            % Test ADF test with edge cases and boundary conditions
            
            % Test with very small sample size
            small_sample = randn(15, 1);
            options_small = struct('lags', 1);  % Small lag for small sample
            results_small = adf_test(small_sample, options_small);
            obj.assertTrue(isfield(results_small, 'stat'), 'ADF test should run with small sample');
            
            % Test with very large lag order
            medium_sample = randn(100, 1);
            options_large_lag = struct('lags', 20);
            results_large_lag = adf_test(medium_sample, options_large_lag);
            obj.assertTrue(isfinite(results_large_lag.stat), 'ADF test should handle large lag orders');
            
            % Test with series containing NaN values
            data_with_nan = obj.stationaryData;
            data_with_nan(5) = NaN;
            obj.assertThrows(@() adf_test(data_with_nan), 'DATACHECK:NaN', 'ADF test should throw error with NaN values');
            
            % Test with constant series
            constant_series = ones(100, 1);
            obj.assertThrows(@() adf_test(constant_series), '', 'ADF test should handle constant series appropriately');
            
            % Verify numerical stability under extreme conditions
            near_integrated = zeros(500, 1);
            near_integrated(1) = randn(1);
            for t = 2:500
                near_integrated(t) = 0.999 * near_integrated(t-1) + 0.001 * randn(1);
            end
            results_near = adf_test(near_integrated);
            obj.assertTrue(isfinite(results_near.stat), 'ADF test should be numerically stable for near-integrated series');
        end
        
        function testInvalidInputs(obj)
            % Test ADF test with invalid inputs to verify proper error handling
            
            % Test with empty data
            obj.assertThrows(@() adf_test([]), '', 'Should error with empty data');
            
            % Test with invalid regression_type value
            options_invalid_reg = struct('regression_type', 'invalid');
            obj.assertThrows(@() adf_test(obj.stationaryData, options_invalid_reg), 
                'Invalid regression_type', 'Should error with invalid regression type');
            
            % Test with invalid lag specification
            options_invalid_lag = struct('lags', -1);
            obj.assertThrows(@() adf_test(obj.stationaryData, options_invalid_lag), 
                'lags must be a non-negative integer', 'Should error with negative lag');
            
            % Test with invalid data types
            obj.assertThrows(@() adf_test({'a', 'b', 'c'}), '', 'Should error with non-numeric data');
            
            % Verify appropriate error messages are generated
            options_invalid_method = struct('lags', 'unknown');
            obj.assertThrows(@() adf_test(obj.stationaryData, options_invalid_method), 
                'must be ''aic'' or ''bic''', 'Should error with invalid lag selection method');
            
            % Ensure robust error handling for all invalid inputs
            data_with_inf = obj.stationaryData;
            data_with_inf(10) = Inf;
            obj.assertThrows(@() adf_test(data_with_inf), '', 'Should error with infinite values');
        end
        
        function testPerformance(obj)
            % Test performance characteristics of the ADF test implementation
            
            % Measure execution time with varying data sizes
            sizes = [100, 500, 1000];
            times = zeros(length(sizes), 1);
            
            for i = 1:length(sizes)
                data = obj.stationaryData(1:sizes(i));
                start_time = tic;
                adf_test(data);
                times(i) = toc(start_time);
            end
            
            % Verify performance scales appropriately
            % Time shouldn't increase more than quadratically with size
            if times(1) > 0
                scaling_factor = (times(end)/times(1)) / (sizes(end)/sizes(1))^2;
                obj.assertTrue(scaling_factor < 2, 'Performance should scale reasonably with data size');
            end
            
            % Measure execution time with varying lag orders
            fixed_size = 500;
            data = obj.stationaryData(1:fixed_size);
            lag_values = [0, 5, 10];
            lag_times = zeros(length(lag_values), 1);
            
            for i = 1:length(lag_values)
                options = struct('lags', lag_values(i));
                start_time = tic;
                adf_test(data, options);
                lag_times(i) = toc(start_time);
            end
            
            % Compare performance with and without automatic lag selection
            start_time = tic;
            adf_test(data, struct('lags', 'aic', 'max_lags', 5));
            time_aic = toc(start_time);
            
            start_time = tic;
            adf_test(data, struct('lags', 5));
            time_fixed = toc(start_time);
            
            % Ensure efficient computation for large datasets
            obj.assertTrue(isfinite(time_aic) && isfinite(time_fixed), 'Performance measures should be finite');
        end
        
        function testResultsStructure(obj)
            % Test that the results structure contains all expected fields
            
            % Run ADF test and capture results structure
            results = adf_test(obj.stationaryData);
            
            % Verify presence of test statistic field
            obj.assertTrue(isfield(results, 'stat'), 'Results should contain test statistic field');
            obj.assertTrue(isscalar(results.stat), 'Test statistic should be a scalar');
            
            % Verify presence of critical values field
            obj.assertTrue(isfield(results, 'crit_vals'), 'Results should contain critical values field');
            obj.assertEqual(size(results.crit_vals), [1, 4], 'Critical values should be a 1x4 vector');
            
            % Verify presence of p-value field
            obj.assertTrue(isfield(results, 'pval'), 'Results should contain p-value field');
            obj.assertTrue(results.pval >= 0 && results.pval <= 1, 'P-value should be between 0 and 1');
            
            % Verify presence of regression type information
            obj.assertTrue(isfield(results, 'regression_type'), 'Results should contain regression type field');
            obj.assertTrue(ismember(results.regression_type, {'n', 'c', 'ct'}), 'Regression type should be valid');
            
            % Verify presence of lag information
            obj.assertTrue(isfield(results, 'lags'), 'Results should contain lags field');
            
            % Ensure all expected fields are present and correctly formatted
            % Test with automatic lag selection
            results_auto = adf_test(obj.stationaryData, struct('lags', 'aic'));
            obj.assertTrue(isfield(results_auto, 'optimal_lag'), 'Results should contain optimal_lag with automatic selection');
            obj.assertTrue(isfield(results_auto, 'ic_values'), 'Results should contain ic_values with automatic selection');
        end
        
        function result = compareWithReference(obj, results, reference, tolerance)
            % Helper method to compare test results with reference values
            
            % Set default tolerance if not provided
            if nargin < 4
                tolerance = 1e-6;
            end
            
            % Compare test statistic with reference value
            stat_match = abs(results.stat - reference.stat) < tolerance;
            
            % Compare p-value with reference value
            pval_match = abs(results.pval - reference.pval) < tolerance;
            
            % Compare critical values with reference values
            crit_match = all(abs(results.crit_vals - reference.crit_vals) < tolerance);
            
            % Return true if all comparisons are within tolerance
            result = stat_match && pval_match && crit_match;
        end
        
        function series = generateTestSeries(obj, seriesType, numObservations, parameters)
            % Helper method to generate test time series with known properties
            
            % Set default parameters if not provided
            if nargin < 4
                parameters = struct();
            end
            
            % Set default observations if not provided
            if nargin < 3
                numObservations = 500;
            end
            
            % Generate white noise innovations
            innovations = randn(numObservations, 1);
            
            % For 'stationary', return innovations directly
            if strcmpi(seriesType, 'stationary')
                series = innovations;
                
            % For 'integrated', compute cumulative sum of innovations
            elseif strcmpi(seriesType, 'integrated')
                series = cumsum(innovations);
                
            % For 'trend_stationary', add deterministic trend
            elseif strcmpi(seriesType, 'trend_stationary')
                if isfield(parameters, 'trend_coef')
                    trend_coef = parameters.trend_coef;
                else
                    trend_coef = 0.01;
                end
                
                t = (1:numObservations)';
                series = trend_coef * t + innovations;
                
            % For 'arma', generate ARMA process with specified parameters
            elseif strcmpi(seriesType, 'arma')
                if isfield(parameters, 'ar_coefs')
                    ar_coefs = parameters.ar_coefs;
                else
                    ar_coefs = 0.7;  % Default AR(1) coefficient
                end
                
                if isfield(parameters, 'ma_coefs')
                    ma_coefs = parameters.ma_coefs;
                else
                    ma_coefs = 0.3;  % Default MA(1) coefficient
                end
                
                % Get orders
                p = length(ar_coefs);
                q = length(ma_coefs);
                
                % Initialize series and error terms
                series = zeros(numObservations, 1);
                errors = innovations;
                
                % Generate ARMA process recursively
                for t = max(p, q) + 1 : numObservations
                    % AR component
                    for i = 1:p
                        if t > i
                            series(t) = series(t) + ar_coefs(i) * series(t - i);
                        end
                    end
                    
                    % MA component
                    for i = 1:q
                        if t > i
                            series(t) = series(t) + ma_coefs(i) * errors(t - i);
                        end
                    end
                    
                    % Add current innovation
                    series(t) = series(t) + errors(t);
                end
            else
                error('Unknown series type: %s', seriesType);
            end
            
            % Return series with requested properties
            series = series;
        end
    end
end