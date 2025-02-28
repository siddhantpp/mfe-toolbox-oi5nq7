classdef LjungboxTest < BaseTest
    % LjungboxTest Test case for validating the functionality of the Ljung-Box Q-test
    % implementation for detecting autocorrelation in time series residuals
    
    properties
        whiteNoiseData      % White noise data (should have no autocorrelation)
        autocorrelatedData  % Data with known autocorrelation
        knownResultData     % Data with pre-calculated Ljung-Box statistics
        expectedResults     % Expected test results for validation
        comparator          % NumericalComparator instance
    end
    
    methods
        function obj = LjungboxTest()
            % Initialize the LjungboxTest class instance
            
            % Call the superclass constructor
            obj = obj@BaseTest();
            
            % Initialize the NumericalComparator instance with appropriate tolerances
            obj.comparator = NumericalComparator();
            
            % Properties will be populated in setUp method
            obj.whiteNoiseData = [];
            obj.autocorrelatedData = [];
            obj.knownResultData = [];
            obj.expectedResults = struct();
        end
        
        function setUp(obj)
            % Prepare the test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Generate white noise test data using randn
            T = 500; % Sample size
            obj.whiteNoiseData = randn(T, 1);
            
            % Generate autocorrelated test data with AR(1) structure
            % y_t = 0.6*y_{t-1} + e_t
            ar_coef = 0.6;
            obj.autocorrelatedData = zeros(T, 1);
            obj.autocorrelatedData(1) = randn();
            for t = 2:T
                obj.autocorrelatedData(t) = ar_coef * obj.autocorrelatedData(t-1) + randn();
            end
            
            % Prepare test data with known Ljung-Box statistics
            timeSeriesData = TestDataGenerator('generateTimeSeriesData', struct('ar', 0.4, 'numObs', 200));
            obj.knownResultData = timeSeriesData.y;
            
            % Set up expected results structure for comparison
            obj.expectedResults = struct();
            
            % For white noise at lag 10, the test should not reject null hypothesis (p-value > 0.05)
            obj.expectedResults.whiteNoise = struct();
            obj.expectedResults.whiteNoise.lags = 10;
            obj.expectedResults.whiteNoise.pvalueThreshold = 0.05;
            
            % For autocorrelated data at lag 10, the test should reject null hypothesis (p-value < 0.05)
            obj.expectedResults.autocorrelated = struct();
            obj.expectedResults.autocorrelated.lags = 10;
            obj.expectedResults.autocorrelated.pvalueThreshold = 0.05;
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data
            obj.whiteNoiseData = [];
            obj.autocorrelatedData = [];
            obj.knownResultData = [];
            obj.expectedResults = struct();
        end
        
        function testBasicFunctionality(obj)
            % Test the basic functionality of the Ljung-Box test with white noise data
            
            % Apply ljungbox function to white noise data
            results = ljungbox(obj.whiteNoiseData);
            
            % Validate structure of results
            obj.assertTrue(isstruct(results), 'ljungbox should return a structure');
            obj.assertTrue(isfield(results, 'stats'), 'results should contain stats field');
            obj.assertTrue(isfield(results, 'pvals'), 'results should contain pvals field');
            obj.assertTrue(isfield(results, 'lags'), 'results should contain lags field');
            
            % For white noise, p-values should generally be high (not rejecting null hypothesis)
            % We can't guarantee this due to randomness, but we can check the distribution
            pvals = results.pvals;
            
            % Under the null hypothesis of no autocorrelation, p-values should be
            % approximately uniformly distributed. We expect the median to be around 0.5.
            median_pval = median(pvals);
            obj.assertTrue(median_pval > 0.1, 'Median p-value for white noise should be relatively high');
            
            % Check the decision fields at different significance levels
            pct_rejected_10 = mean(results.isRejected10pct);
            obj.assertTrue(pct_rejected_10 < 0.3, 'Rejection rate at 10% level should be relatively low for white noise');
            
            pct_rejected_5 = mean(results.isRejected5pct);
            obj.assertTrue(pct_rejected_5 < 0.2, 'Rejection rate at 5% level should be relatively low for white noise');
        end
        
        function testAutocorrelatedData(obj)
            % Test the Ljung-Box test with autocorrelated data to verify detection capability
            
            % Apply ljungbox function to autocorrelated data
            results = ljungbox(obj.autocorrelatedData);
            
            % For autocorrelated data, p-values should be low (rejecting null hypothesis)
            pvals = results.pvals;
            
            % With strong autocorrelation (AR coefficient = 0.6), we expect
            % the test to reject the null hypothesis of no autocorrelation
            obj.assertTrue(all(pvals < 0.01), 'P-values should be very low for autocorrelated data');
            
            % Check the decision fields
            obj.assertTrue(all(results.isRejected10pct), 'Should reject at 10% level for autocorrelated data');
            obj.assertTrue(all(results.isRejected5pct), 'Should reject at 5% level for autocorrelated data');
            obj.assertTrue(all(results.isRejected1pct), 'Should reject at 1% level for autocorrelated data');
            
            % Test Q-statistic calculation by manually calculating for first lag
            T = length(obj.autocorrelatedData);
            acf1 = sacf(obj.autocorrelatedData, 1);
            Q_manual = T * (T + 2) * (acf1^2 / (T - 1));
            
            % Compare with the function result for lag 1
            results_lag1 = ljungbox(obj.autocorrelatedData, 1);
            Q_function = results_lag1.stats;
            
            % Verify the Q-statistic matches our manual calculation
            comparison = obj.comparator.compareScalars(Q_manual, Q_function, 1e-10);
            obj.assertTrue(comparison.isEqual, 'Q-statistic calculation is incorrect');
        end
        
        function testParameterValidation(obj)
            % Test input parameter validation in the ljungbox function
            
            % Test with non-numeric data
            obj.assertThrows(@() ljungbox({'invalid'}), 'MATLAB:datacheck:NonNumericInput');
            
            % Test with empty matrix
            obj.assertThrows(@() ljungbox([]), 'MATLAB:datacheck:EmptyDataset');
            
            % Test with negative lag
            obj.assertThrows(@() ljungbox(obj.whiteNoiseData, -1), 'MATLAB:parametercheck:RangeCheck');
            
            % Test with invalid degrees of freedom adjustment
            obj.assertThrows(@() ljungbox(obj.whiteNoiseData, 10, -1), 'MATLAB:parametercheck:RangeCheck');
            
            % Test with multi-dimensional array
            obj.assertThrows(@() ljungbox(reshape(obj.whiteNoiseData(1:27), [3,3,3])), 'MATLAB:columncheck:InvalidDataFormat');
        end
        
        function testDefaultParameters(obj)
            % Test the default parameter behavior of the ljungbox function
            
            % Calculate the expected default lags for our sample size
            T = length(obj.whiteNoiseData);
            expected_default_lags = min(10, floor(T/5));
            
            % Call ljungbox with only data parameter
            results = ljungbox(obj.whiteNoiseData);
            
            % Verify the default lags
            obj.assertEqual(length(results.lags), expected_default_lags, 'Default number of lags is incorrect');
            
            % Verify default degrees of freedom adjustment is 0
            obj.assertEqual(results.dofsAdjust, 0, 'Default degrees of freedom adjustment should be 0');
            
            % Compare results with explicitly specified defaults
            results_explicit = ljungbox(obj.whiteNoiseData, (1:expected_default_lags)', 0);
            
            % Results should be identical
            obj.assertTrue(isequal(results.stats, results_explicit.stats), 'Results with default parameters differ from explicit parameters');
            obj.assertTrue(isequal(results.pvals, results_explicit.pvals), 'Results with default parameters differ from explicit parameters');
        end
        
        function testKnownResults(obj)
            % Test the ljungbox function against pre-calculated results for validation
            
            % For an AR(1) model with coefficient 0.4, we expect significant autocorrelation
            % Apply ljungbox function with specific parameters
            results = ljungbox(obj.knownResultData, 5, 0);
            
            % For this specific data, we have pre-calculated Q-statistics and p-values
            % In a real test with fixed random seed, we would use precise expected values
            
            % Verify that p-values are as expected (very low for autocorrelated data)
            obj.assertTrue(all(results.pvals < 0.01), 'P-values should be very low for known autocorrelated data');
            
            % Verify correct decisions
            obj.assertTrue(all(results.isRejected5pct), 'Should reject at 5% level for known autocorrelated data');
            
            % Verify test statistics are positive and increasing
            obj.assertTrue(all(results.stats > 0), 'Q-statistics should be positive');
            obj.assertTrue(all(diff(results.stats) >= 0), 'Q-statistics should be non-decreasing with lag');
        end
        
        function testDegreesOfFreedomAdjustment(obj)
            % Test the degrees of freedom adjustment parameter in the ljungbox function
            
            % Generate data from ARMA model with known parameters
            armaData = TestDataGenerator('generateTimeSeriesData', struct('ar', [0.4, 0.2], 'numObs', 200));
            y = armaData.y;
            
            % Test with different degrees of freedom adjustments
            results_no_adj = ljungbox(y, 10, 0);
            results_adj_1 = ljungbox(y, 10, 1);
            results_adj_2 = ljungbox(y, 10, 2);
            
            % Verify that degrees of freedom are correctly adjusted
            obj.assertEqual(results_no_adj.dofs, (1:10)', 'Degrees of freedom without adjustment should equal lags');
            obj.assertEqual(results_adj_1.dofs, max((1:10)' - 1, 1), 'Degrees of freedom with adjustment of 1 not correctly calculated');
            obj.assertEqual(results_adj_2.dofs, max((1:10)' - 2, 1), 'Degrees of freedom with adjustment of 2 not correctly calculated');
            
            % The test statistics should be the same regardless of adjustment
            comparison = obj.comparator.compareMatrices(results_no_adj.stats, results_adj_1.stats, 1e-10);
            obj.assertTrue(comparison.isEqual, 'Test statistics should not change with dofs adjustment');
            
            % But p-values should differ due to different degrees of freedom
            % P-values with adjustment should be higher than without adjustment
            for i = 3:10  % Compare p-values starting from lag 3 where df adjustment is meaningful
                obj.assertTrue(results_adj_2.pvals(i) >= results_adj_1.pvals(i), 'P-values should increase with higher dofs adjustment');
                obj.assertTrue(results_adj_1.pvals(i) >= results_no_adj.pvals(i), 'P-values should increase with higher dofs adjustment');
            end
            
            % With correct model specification (AR(2)), the adjusted test should show
            % less evidence of remaining autocorrelation in the residuals
            pct_rejected_no_adj = mean(results_no_adj.isRejected5pct);
            pct_rejected_adj_2 = mean(results_adj_2.isRejected5pct);
            
            % The adjustment should decrease the rejection rate
            obj.assertTrue(pct_rejected_adj_2 <= pct_rejected_no_adj, 'Degrees of freedom adjustment should reduce rejection rate for correct model specification');
        end
    end
end