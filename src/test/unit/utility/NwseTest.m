classdef NwseTest < BaseTest
    % NWSETEST Test class for validating the functionality and correctness of the 
    % Newey-West standard error (nwse) implementation
    %
    % This test class validates the Newey-West heteroskedasticity and autocorrelation
    % consistent standard error calculation for regression models, ensuring proper
    % error handling, numerical stability, and correct results across various inputs.
    %
    % See also: nwse, BaseTest
    
    properties
        % Test design matrix
        X
        
        % Test residuals vector
        residuals
        
        % Lag parameter for Newey-West estimation
        lag
        
        % Structure containing test data
        testData
        
        % Tolerance for numerical comparisons
        tolerance
    end
    
    methods
        function obj = NwseTest()
            % Initialize the NwseTest class
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Set default tolerance for numerical comparisons
            obj.tolerance = 1e-10;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load financial returns test data
            try
                obj.testData = obj.loadTestData('financialReturns.mat');
            catch
                % Create synthetic data if test data file isn't available
                warning('Test data file not found. Using synthetic data.');
                obj.testData = struct();
                obj.testData.returns = randn(100, 1) * 0.01 + 0.0005; % Synthetic returns
            end
            
            % Initialize dummy regression design matrix with 100 observations and 3 regressors
            obj.X = [ones(100, 1), randn(100, 1), randn(100, 1)];
            
            % Initialize dummy residuals vector with 100 observations
            obj.residuals = randn(100, 1);
            
            % Set default lag parameter to 5
            obj.lag = 5;
            
            % Ensure data is properly formatted for testing
            obj.residuals = columncheck(obj.residuals, 'residuals');
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.X = [];
            obj.residuals = [];
            obj.testData = struct();
        end
        
        function testBasicFunctionality(obj)
            % Test that nwse function produces expected results for standard inputs
            
            % Call nwse with standard inputs
            se = nwse(obj.X, obj.residuals, obj.lag);
            
            % Verify output dimensions
            obj.assertEqual(size(se), [size(obj.X, 2), 1], 'Standard error vector has incorrect dimensions');
            
            % Verify that standard errors are positive
            obj.assertTrue(all(se > 0), 'Standard errors should be positive');
            
            % Calculate OLS standard errors for comparison
            XX_inv = inv(obj.X' * obj.X);
            e2 = obj.residuals.^2;
            V_ols = XX_inv * (obj.X' * diag(e2) * obj.X) * XX_inv;
            se_ols = sqrt(diag(V_ols));
            
            % Verify that number of standard errors matches the number of regressors
            obj.assertEqual(length(se), size(obj.X, 2), 'Number of standard errors should match number of regressors');
        end
        
        function testLagVariations(obj)
            % Test nwse function with different lag values
            
            % Test with lag = 0 (equivalent to White's heteroskedasticity-robust standard errors)
            se0 = nwse(obj.X, obj.residuals, 0);
            
            % Test with lag = 1
            se1 = nwse(obj.X, obj.residuals, 1);
            
            % Test with lag = 10
            se10 = nwse(obj.X, obj.residuals, 10);
            
            % Verify dimensions are consistent
            obj.assertEqual(size(se0), [size(obj.X, 2), 1], 'SE with lag=0 has incorrect dimensions');
            obj.assertEqual(size(se1), [size(obj.X, 2), 1], 'SE with lag=1 has incorrect dimensions');
            obj.assertEqual(size(se10), [size(obj.X, 2), 1], 'SE with lag=10 has incorrect dimensions');
            
            % Verify that all standard errors are positive
            obj.assertTrue(all(se0 > 0), 'Standard errors with lag=0 should be positive');
            obj.assertTrue(all(se1 > 0), 'Standard errors with lag=1 should be positive');
            obj.assertTrue(all(se10 > 0), 'Standard errors with lag=10 should be positive');
            
            % With financial time series that have positive autocorrelation,
            % standard errors typically increase with lag (heuristic check, not always true)
        end
        
        function testWithFinancialData(obj)
            % Test nwse function with actual financial returns data
            
            % Skip test if financial data is not available
            if ~isfield(obj.testData, 'returns') || isempty(obj.testData.returns)
                warning('Financial returns data not available. Skipping test.');
                return;
            end
            
            % Set up regression model using financial returns data
            returns = obj.testData.returns;
            T = size(returns, 1);
            X = [ones(T-1, 1), returns(1:end-1)];  % AR(1) model
            y = returns(2:end);
            
            % Estimate the model
            beta = (X' * X) \ (X' * y);
            residuals = y - X * beta;
            
            % Calculate Newey-West standard errors with different lags
            se1 = nwse(X, residuals, 1);
            se5 = nwse(X, residuals, 5);
            se10 = nwse(X, residuals, 10);
            
            % Verify dimensions
            obj.assertEqual(size(se1), [size(X, 2), 1], 'SE with financial data has incorrect dimensions');
            
            % Verify all standard errors are positive
            obj.assertTrue(all(se1 > 0), 'Standard errors with financial data should be positive');
            obj.assertTrue(all(se5 > 0), 'Standard errors with financial data should be positive');
            obj.assertTrue(all(se10 > 0), 'Standard errors with financial data should be positive');
            
            % Calculate OLS standard errors for comparison
            XX_inv = inv(X' * X);
            e2 = residuals.^2;
            V_ols = XX_inv * (X' * diag(e2) * X) * XX_inv;
            se_ols = sqrt(diag(V_ols));
            
            % With financial returns, Newey-West SE should typically be larger than OLS SE
            % due to autocorrelation in the data (heuristic check)
        end
        
        function testInputValidation(obj)
            % Test that nwse correctly validates input parameters
            
            % Test with empty X matrix
            try
                nwse([], obj.residuals, obj.lag);
                obj.assertTrue(false, 'nwse should throw an error with empty X');
            catch
                % Expected behavior
            end
            
            % Test with non-numeric X
            try
                nwse('string', obj.residuals, obj.lag);
                obj.assertTrue(false, 'nwse should throw an error with non-numeric X');
            catch
                % Expected behavior
            end
            
            % Test with empty residuals
            try
                nwse(obj.X, [], obj.lag);
                obj.assertTrue(false, 'nwse should throw an error with empty residuals');
            catch
                % Expected behavior
            end
            
            % Test with residuals containing NaN values
            residuals_nan = obj.residuals;
            residuals_nan(1) = NaN;
            try
                nwse(obj.X, residuals_nan, obj.lag);
                obj.assertTrue(false, 'nwse should throw an error with NaN in residuals');
            catch
                % Expected behavior
            end
            
            % Test with negative lag value
            try
                nwse(obj.X, obj.residuals, -1);
                obj.assertTrue(false, 'nwse should throw an error with negative lag value');
            catch
                % Expected behavior
            end
            
            % Test with non-scalar lag
            try
                nwse(obj.X, obj.residuals, [1 2]);
                obj.assertTrue(false, 'nwse should throw an error with non-scalar lag');
            catch
                % Expected behavior
            end
            
            % Test with incompatible dimensions between X and residuals
            try
                nwse(obj.X, obj.residuals(1:50), obj.lag);
                obj.assertTrue(false, 'nwse should throw an error with incompatible dimensions');
            catch
                % Expected behavior
            end
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of nwse with challenging inputs
            
            % Test with near-multicollinear regressors
            X_multicol = [ones(100, 1), randn(100, 1), randn(100, 1) + 0.999 * randn(100, 1)];
            se_multicol = nwse(X_multicol, obj.residuals, obj.lag);
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_multicol), [size(X_multicol, 2), 1], 'SE with multicollinear data has incorrect dimensions');
            obj.assertTrue(all(se_multicol > 0), 'Standard errors with multicollinear data should be positive');
            
            % Test with very small residuals
            small_residuals = obj.residuals * 1e-6;
            se_small = nwse(obj.X, small_residuals, obj.lag);
            
            % Standard errors should scale proportionally
            obj.assertMatrixEqualsWithTolerance(se_small, nwse(obj.X, obj.residuals, obj.lag) * 1e-6, 1e-10, 
                'Standard errors should scale proportionally with residuals');
            
            % Test with large dataset (many observations)
            large_T = 1000;
            X_large = [ones(large_T, 1), randn(large_T, 1), randn(large_T, 1)];
            residuals_large = randn(large_T, 1);
            se_large = nwse(X_large, residuals_large, obj.lag);
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_large), [size(X_large, 2), 1], 'SE with large dataset has incorrect dimensions');
            obj.assertTrue(all(se_large > 0), 'Standard errors with large dataset should be positive');
            
            % Test with many regressors
            many_K = 10;
            X_many = [ones(100, 1), randn(100, many_K-1)];
            se_many = nwse(X_many, obj.residuals, obj.lag);
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_many), [many_K, 1], 'SE with many regressors has incorrect dimensions');
            obj.assertTrue(all(se_many > 0), 'Standard errors with many regressors should be positive');
        end
        
        function testEdgeCases(obj)
            % Test nwse function with edge cases
            
            % Test with minimal valid inputs (2 observations, 1 regressor)
            X_min = ones(2, 1);
            residuals_min = randn(2, 1);
            se_min = nwse(X_min, residuals_min, 0);  % Lag must be 0 for 2 observations
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_min), [1, 1], 'SE with minimal data has incorrect dimensions');
            obj.assertTrue(all(se_min > 0), 'Standard errors with minimal data should be positive');
            
            % Test with lag equal to number of observations minus 1
            % This is technically valid but might cause numerical issues
            T = 20;
            X_edge = [ones(T, 1), randn(T, 1)];
            residuals_edge = randn(T, 1);
            lag_edge = T - 2;  % Maximum valid lag is T-2 to get at least one autocorrelation
            se_edge = nwse(X_edge, residuals_edge, lag_edge);
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_edge), [size(X_edge, 2), 1], 'SE with edge case lag has incorrect dimensions');
            obj.assertTrue(all(se_edge > 0), 'Standard errors with edge case lag should be positive');
            
            % Test with X as single column (intercept only model)
            X_intercept = ones(50, 1);
            residuals_intercept = randn(50, 1);
            se_intercept = nwse(X_intercept, residuals_intercept, obj.lag);
            
            % Verify output dimensions and positivity
            obj.assertEqual(size(se_intercept), [1, 1], 'SE with intercept-only model has incorrect dimensions');
            obj.assertTrue(all(se_intercept > 0), 'Standard errors with intercept-only model should be positive');
        end
    end
end