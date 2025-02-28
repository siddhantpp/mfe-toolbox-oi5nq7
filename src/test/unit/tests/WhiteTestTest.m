classdef WhiteTestTest < BaseTest
    % WhiteTestTest Test class for validating the functionality of White's 
    % heteroskedasticity test implementation for detecting non-constant variance 
    % in regression residuals
    %
    % This test class validates the implementation of White's heteroskedasticity test,
    % which is essential for valid statistical inference in cross-sectional analysis.
    % It tests proper detection of heteroskedasticity, handling of homoskedastic data,
    % output format, error handling, and numerical accuracy.
    
    properties
        homoskedasticData   % Residuals with constant variance
        heteroskedasticData % Residuals with non-constant variance
        regressors          % Matrix of predictor variables
        significanceLevel   % Threshold for hypothesis testing
        tolerance           % Numerical tolerance for floating-point comparisons
    end
    
    methods
        function obj = WhiteTestTest()
            % Initialize the WhiteTestTest class with default configuration
            obj = obj@BaseTest(); % Call superclass constructor
            obj.tolerance = 1e-8; % Set numerical tolerance for floating-point comparisons
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method execution
            setUp@BaseTest(obj); % Call superclass setUp method
            
            % Generate test data
            n = 200; % Sample size
            k = 3;   % Number of regressors
            
            % Create regressor matrix with multiple predictors
            obj.regressors = randn(n, k);
            
            % Create homoskedastic regression residuals with constant variance
            obj.homoskedasticData = randn(n, 1);
            
            % Create heteroskedastic regression residuals with variance related to regressors
            obj.heteroskedasticData = obj.generateHeteroskedasticData(obj.regressors, n);
            
            % Set significanceLevel for hypothesis testing (typically 0.05)
            obj.significanceLevel = 0.05;
        end
        
        function tearDown(obj)
            % Cleans up resources after each test method execution
            tearDown@BaseTest(obj); % Call superclass tearDown method
            
            % Clear test data variables
            obj.homoskedasticData = [];
            obj.heteroskedasticData = [];
            obj.regressors = [];
            % Reset any modified global states
        end
        
        function testHeteroskedasticityDetection(obj)
            % Tests that White's test correctly identifies heteroskedasticity in regression residuals
            
            % Run white_test on heteroskedasticData with regressors
            results = white_test(obj.heteroskedasticData, obj.regressors);
            
            % Verify p-values are below significance level (strong rejection of homoskedasticity)
            obj.assertTrue(results.pval < obj.significanceLevel, ...
                'White test should detect heteroskedasticity in heteroskedastic data');
                
            % Verify test statistics are positive and statistically significant
            obj.assertTrue(results.stat > 0, 'Test statistic should be positive');
            
            % Confirm rejection of null hypothesis at various significance levels
            obj.assertTrue(any(results.rej), 'Should reject null hypothesis at some significance level');
            obj.assertTrue(results.rej(2), 'Should reject null hypothesis at 5% significance level');
        end
        
        function testHomoskedasticDataNoFalsePositives(obj)
            % Tests that White's test correctly fails to reject null hypothesis in homoskedastic residuals
            
            % Run white_test on homoskedasticData with regressors
            results = white_test(obj.homoskedasticData, obj.regressors);
            
            % Note: This is a probabilistic test. Even with homoskedastic data,
            % we expect rejection at rate alpha (Type I error).
            % We don't assert p-value > alpha, as this could fail randomly.
            
            % Verify Type I error rate is consistent with significance level
            if results.pval < obj.significanceLevel
                fprintf('Note: p-value = %g < %g (expected for ~%g%% of runs)\n', ...
                    results.pval, obj.significanceLevel, obj.significanceLevel * 100);
            end
            
            % Check if test statistic is reasonable for homoskedastic data
            % For chi-square distribution, statistic/df ratio should be near 1 for homoskedastic data
            stat_per_df = results.stat / results.df;
            obj.assertTrue(stat_per_df < 3, ...
                'Test statistic per degree of freedom should be reasonable for homoskedastic data');
        end
        
        function testOutputFormat(obj)
            % Tests the output format and structure of White's test results
            
            % Run white_test and verify structure of output
            results = white_test(obj.homoskedasticData, obj.regressors);
            
            % Confirm presence of test statistic, p-value, critical values, and rejection fields
            expectedFields = {'stat', 'pval', 'df', 'crit', 'rej'};
            actualFields = fieldnames(results);
            
            for i = 1:length(expectedFields)
                obj.assertTrue(ismember(expectedFields{i}, actualFields), ...
                    ['Results structure missing expected field: ' expectedFields{i}]);
            end
            
            % Verify field names match expectations
            obj.assertEqual(length(actualFields), length(expectedFields), ...
                'Results structure should have exactly the expected number of fields');
            
            % Verify dimensions of output fields are correct
            obj.assertTrue(isscalar(results.stat), 'Test statistic should be a scalar');
            obj.assertTrue(isscalar(results.pval), 'P-value should be a scalar');
            obj.assertTrue(isscalar(results.df), 'Degrees of freedom should be a scalar');
            obj.assertTrue(isequal(size(results.crit), [1, 3]), 'Critical values should be a 1x3 vector');
            obj.assertTrue(isequal(size(results.rej), [1, 3]), 'Rejection indicators should be a 1x3 vector');
        end
        
        function testInvalidInputs(obj)
            % Tests error handling with invalid inputs to White's test function
            
            % Test with empty residuals, verifying appropriate error
            obj.assertThrows(@() white_test([], obj.regressors), ...
                'DATACHECK:InvalidInput', 'Should throw error for empty residuals');
            
            % Test with empty regressors, verifying appropriate error
            obj.assertThrows(@() white_test(obj.homoskedasticData, []), ...
                'DATACHECK:InvalidInput', 'Should throw error for empty regressors');
            
            % Test with non-numeric data, verifying appropriate error
            obj.assertThrows(@() white_test('string', obj.regressors), ...
                'DATACHECK:InvalidInput', 'Should throw error for non-numeric residuals');
            obj.assertThrows(@() white_test(obj.homoskedasticData, 'string'), ...
                'DATACHECK:InvalidInput', 'Should throw error for non-numeric regressors');
            
            % Test with NaN/Inf values, verifying appropriate error
            nanResiduals = obj.homoskedasticData;
            nanResiduals(1) = NaN;
            obj.assertThrows(@() white_test(nanResiduals, obj.regressors), ...
                'DATACHECK:InvalidInput', 'Should throw error for NaN in residuals');
            
            infResiduals = obj.homoskedasticData;
            infResiduals(1) = Inf;
            obj.assertThrows(@() white_test(infResiduals, obj.regressors), ...
                'DATACHECK:InvalidInput', 'Should throw error for Inf in residuals');
            
            % Test with mismatched dimensions between residuals and regressors, verifying appropriate error
            mismatchedResiduals = obj.homoskedasticData(1:end-1);
            obj.assertThrows(@() white_test(mismatchedResiduals, obj.regressors), ...
                'WHITE_TEST:InvalidInput', 'Should throw error for mismatched dimensions');
        end
        
        function testCrossSectionalData(obj)
            % Tests White's test on cross-sectional financial data
            
            try
                % Load cross_sectional_data.mat using loadTestData method
                data = obj.loadTestData('cross_sectional_data.mat');
                
                % Extract returns and characteristics for regression
                returns = data.returns;
                characteristics = data.characteristics;
                
                % Run preliminary regression to obtain residuals
                X = [ones(size(characteristics, 1), 1), characteristics];
                b = (X'*X)\(X'*returns);
                residuals = returns - X*b;
                
                % Perform white_test on regression residuals
                results = white_test(residuals, characteristics);
                
                % Verify results are consistent with expected heteroskedasticity patterns in financial data
                obj.assertTrue(isfield(results, 'stat'), 'Results should contain test statistic');
                obj.assertTrue(isfield(results, 'pval'), 'Results should contain p-value');
                obj.assertTrue(results.stat > 0, 'Test statistic should be positive');
                
                % Log results for inspection (when verbose mode is enabled)
                if obj.verbose
                    fprintf('Cross-sectional data White test results:\n');
                    fprintf('  Test statistic: %g\n', results.stat);
                    fprintf('  P-value: %g\n', results.pval);
                    fprintf('  Degrees of freedom: %d\n', results.df);
                    fprintf('  Reject H0 at 5%%: %d\n', results.rej(2));
                end
                
            catch ME
                if strcmp(ME.identifier, 'BaseTest:FileNotFound')
                    warning('Skipping cross-sectional data test: Test data file not found');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testManualWhiteTestCalculation(obj)
            % Validates White's test implementation against manually calculated values
            
            % Implement manual White test calculation for verification
            n = 100;
            X = [ones(n, 1), (1:n)' / n, ((1:n)' / n).^2];
            residuals = (1:n)' .* randn(n, 1) / 10; % Heteroskedastic residuals
            
            % Calculate test statistic using squared residuals regression on augmented regressors
            results = white_test(residuals, X(:, 2:end)); % Exclude constant term
            
            % Calculate p-value using chi-square distribution
            manual = obj.calculateManualWhiteStatistic(residuals, X(:, 2:end));
            
            % Compare manual calculation with white_test results
            obj.assertEqualsWithTolerance(manual.stat, results.stat, obj.tolerance, ...
                'Manual and function test statistics should match');
                
            % Verify numerical accuracy within specified tolerance
            obj.assertEqualsWithTolerance(manual.pval, results.pval, obj.tolerance, ...
                'Manual and function p-values should match');
                
            obj.assertEqual(manual.df, results.df, ...
                'Manual and function degrees of freedom should match');
        end
        
        function manual = calculateManualWhiteStatistic(obj, residuals, X)
            % Helper method that manually calculates White's test statistic for verification
            
            % Square the residuals to get dependent variable for auxiliary regression
            squared_residuals = residuals.^2;
            
            % Get dimensions
            [T, K] = size(X);
            
            % Create auxiliary regressors including original X, squared terms, and cross products
            X_aux = ones(T, 1);
            X_aux = [X_aux, X];
            
            % Add squared terms
            for i = 1:K
                X_aux = [X_aux, X(:, i).^2];
            end
            
            % Add cross products
            for i = 1:K
                for j = (i+1):K
                    X_aux = [X_aux, X(:, i) .* X(:, j)];
                end
            end
            
            % Add constant term (column of ones) to auxiliary regressors
            [~, num_aux_regressors] = size(X_aux);
            
            % Perform auxiliary regression of squared residuals on augmented regressors
            b_aux = (X_aux'*X_aux)\(X_aux'*squared_residuals);
            fitted = X_aux * b_aux;
            aux_residuals = squared_residuals - fitted;
            
            % Calculate R² from auxiliary regression
            mean_squared_residuals = mean(squared_residuals);
            TSS = sum((squared_residuals - mean_squared_residuals).^2);
            RSS = sum(aux_residuals.^2);
            R2 = 1 - (RSS/TSS);
            
            % Compute test statistic as n*R²
            stat = T * R2;
            
            % Calculate degrees of freedom as number of regressors in auxiliary regression minus 1
            df = num_aux_regressors - 1;
            
            % Calculate p-value using chi-square distribution
            pval = 1 - chi2cdf(stat, df);
            
            % Return structure with test statistic, p-value, and degrees of freedom
            manual = struct('stat', stat, 'pval', pval, 'df', df);
        end
        
        function heteroskedastic = generateHeteroskedasticData(obj, X, n)
            % Helper method to generate heteroskedastic regression residuals for testing
            
            % Check X dimensions to determine number of observations
            if nargin < 3
                n = size(X, 1);
            end
            
            % Generate base random errors using randn
            base_errors = randn(n, 1);
            
            % Create variance multiplier that depends on values in X
            variance_multiplier = 1 + 2*sum(X.^2, 2);
            
            % Scale errors by variance multiplier to create heteroskedasticity
            heteroskedastic = base_errors .* sqrt(variance_multiplier);
            
            % Return residuals with non-constant variance
            return;
        end
    end
end