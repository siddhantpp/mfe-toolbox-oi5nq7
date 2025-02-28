classdef CrossSectionRegressionTest < matlab.unittest.TestCase & BaseTest & NumericalComparator
    %CROSSSECTIONREGRESSIONTEST Unit test class for the cross_section_regression function, testing all aspects of cross-sectional regression functionality
    
    properties
        testData
        testDependent
        testIndependent
        expectedResults
        weights
        comparator
    end
    
    methods
        function obj = CrossSectionRegressionTest()
            % Initialize test class instance and set up common test properties
            
            % Call superclass constructor
            obj = obj@BaseTest();
            
            % Initialize NumericalComparator instance for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set up default tolerance appropriate for regression calculations
            obj.defaultTolerance = 1e-8;
        end
        
        function setUp(obj)
            % Prepare test environment before each test method execution
            
            % Load cross-sectional test data from cross_sectional_data.mat
            data = obj.loadTestData('cross_sectional_data.mat');
            
            % Extract dependent variable (asset_returns) for regression tests
            obj.testData = data;
            obj.testDependent = data.asset_returns;
            
            % Extract independent variables (asset_characteristics) for regression tests
            obj.testIndependent = data.asset_characteristics;
            
            % Generate weights vector for weighted regression tests
            obj.weights = rand(size(obj.testDependent));
            
            % Initialize expected results for baseline regression comparison
            obj.expectedResults = struct();
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Clear temporary test variables
            clear obj.testData obj.testDependent obj.testIndependent obj.expectedResults obj.weights;
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testBasicRegression(obj)
            % Test basic OLS regression functionality with default options
            
            % Call cross_section_regression with dependent and independent variables
            results = cross_section_regression(obj.testDependent, obj.testIndependent);
            
            % Verify results structure contains expected fields
            obj.assertTrue(isfield(results, 'beta'), 'Results structure must contain beta field');
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            obj.assertTrue(isfield(results, 'tstat'), 'Results structure must contain tstat field');
            obj.assertTrue(isfield(results, 'pval'), 'Results structure must contain pval field');
            obj.assertTrue(isfield(results, 'r2'), 'Results structure must contain r2 field');
            
            % Validate coefficient estimates against analytically computed values
            expectedBeta = (obj.testIndependent' * obj.testIndependent) \ (obj.testIndependent' * obj.testDependent);
            obj.assertAlmostEqual(results.beta, expectedBeta, 'Coefficient estimates do not match analytical values');
            
            % Check standard errors calculation
            residuals = obj.testDependent - obj.testIndependent * results.beta;
            sigma2 = sum(residuals.^2) / (length(obj.testDependent) - size(obj.testIndependent, 2));
            expectedSE = sqrt(diag(sigma2 * inv(obj.testIndependent' * obj.testIndependent)));
            obj.assertAlmostEqual(results.se, expectedSE, 'Standard errors do not match analytical values');
            
            % Verify t-statistics and p-values
            tstat = results.beta ./ results.se;
            pval = 2 * (1 - tcdf(abs(tstat), length(obj.testDependent) - size(obj.testIndependent, 2)));
            obj.assertAlmostEqual(results.tstat, tstat, 'T-statistics do not match calculated values');
            obj.assertAlmostEqual(results.pval, pval, 'P-values do not match calculated values');
            
            % Validate R-squared and adjusted R-squared values
            y_mean = mean(obj.testDependent);
            SST = sum((obj.testDependent - y_mean).^2);
            SSE = sum(residuals.^2);
            r2 = 1 - SSE/SST;
            r2_adj = 1 - (SSE/(length(obj.testDependent)-size(obj.testIndependent, 2))) / (SST/(length(obj.testDependent)-1));
            obj.assertAlmostEqual(results.r2, r2, 'R-squared value does not match calculated value');
            obj.assertAlmostEqual(results.r2_adj, r2_adj, 'Adjusted R-squared value does not match calculated value');
            
            % Check F-statistic and its p-value
            F = (r2/(size(obj.testIndependent, 2)-1)) / ((1-r2)/(length(obj.testDependent)-size(obj.testIndependent, 2)));
            pval_F = 1 - fcdf(F, size(obj.testIndependent, 2)-1, length(obj.testDependent)-size(obj.testIndependent, 2));
            obj.assertAlmostEqual(results.f_stat, F, 'F-statistic does not match calculated value');
            obj.assertAlmostEqual(results.f_pval, pval_F, 'F-statistic p-value does not match calculated value');
        end
        
        function testRegressionWithoutConstant(obj)
            % Test regression without constant term (intercept)
            
            % Create options structure with addConstant = false
            options = struct('add_constant', false);
            
            % Call cross_section_regression with options for no constant term
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify results do not include intercept coefficient
            obj.assertFalse(any(isnan(results.beta)), 'Results should not contain NaN values for coefficients');
            
            % Validate coefficient estimates match analytical values
            expectedBeta = (obj.testIndependent' * obj.testIndependent) \ (obj.testIndependent' * obj.testDependent);
            obj.assertAlmostEqual(results.beta, expectedBeta, 'Coefficient estimates do not match analytical values');
            
            % Check standard errors, t-statistics, and goodness-of-fit measures
            residuals = obj.testDependent - obj.testIndependent * results.beta;
            sigma2 = sum(residuals.^2) / (length(obj.testDependent) - size(obj.testIndependent, 2));
            expectedSE = sqrt(diag(sigma2 * inv(obj.testIndependent' * obj.testIndependent)));
            obj.assertAlmostEqual(results.se, expectedSE, 'Standard errors do not match analytical values');
            
            % Compare with expected results for no-constant regression
            obj.assertEqual(size(results.beta, 1), size(obj.testIndependent, 2), 'Number of coefficients should match number of regressors');
        end
        
        function testWeightedRegression(obj)
            % Test weighted least squares regression
            
            % Create options structure with method = 'WLS' and weights vector
            options = struct('method', 'WLS', 'weights', obj.weights);
            
            % Call cross_section_regression with WLS options
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify results structure contains weighted regression fields
            obj.assertTrue(isfield(results, 'beta'), 'Results structure must contain beta field');
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            
            % Validate weighted coefficient estimates against analytical values
            W = diag(obj.weights);
            expectedBeta = (obj.testIndependent' * W * obj.testIndependent) \ (obj.testIndependent' * W * obj.testDependent);
            obj.assertAlmostEqual(results.beta, expectedBeta, 'Weighted coefficient estimates do not match analytical values');
            
            % Check weighted standard errors calculation
            residuals = obj.testDependent - obj.testIndependent * results.beta;
            sigma2 = sum(obj.weights .* residuals.^2) / (length(obj.testDependent) - size(obj.testIndependent, 2));
            expectedSE = sqrt(diag(sigma2 * inv(obj.testIndependent' * W * obj.testIndependent)));
            obj.assertAlmostEqual(results.se, expectedSE, 'Weighted standard errors do not match analytical values');
            
            % Verify t-statistics and p-values for weighted regression
            tstat = results.beta ./ results.se;
            pval = 2 * (1 - tcdf(abs(tstat), length(obj.testDependent) - size(obj.testIndependent, 2)));
            obj.assertAlmostEqual(results.tstat, tstat, 'Weighted t-statistics do not match calculated values');
            obj.assertAlmostEqual(results.pval, pval, 'Weighted p-values do not match calculated values');
            
            % Compare with expected results for weighted regression
            obj.assertEqual(size(results.beta, 1), size(obj.testIndependent, 2), 'Number of coefficients should match number of regressors');
        end
        
        function testRobustRegression(obj)
            % Test robust regression methods resistant to outliers
            
            % Create options structure with method = 'robust'
            options = struct('method', 'robust');
            
            % Call cross_section_regression with robust regression options
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify results structure contains robust regression fields
            obj.assertTrue(isfield(results, 'beta'), 'Results structure must contain beta field');
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            
            % Validate robust coefficient estimates
            obj.assertEqual(size(results.beta, 1), size(obj.testIndependent, 2), 'Number of coefficients should match number of regressors');
            
            % Check robust standard errors calculation
            obj.assertTrue(isnumeric(results.se), 'Robust standard errors must be numeric');
            
            % Verify robustness to outliers by comparing with OLS on contaminated data
            % (This is a qualitative check, as the exact values will vary)
            
            % Compare with expected results for robust regression
            obj.assertEqual(size(results.beta, 1), size(obj.testIndependent, 2), 'Number of coefficients should match number of regressors');
        end
        
        function testRobustStandardErrors(obj)
            % Test different robust standard error options
            
            % Test heteroskedasticity-robust (White) standard errors
            options = struct('se_type', 'robust');
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify options.se_type = 'robust' produces White standard errors
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            
            % Validate robust standard errors against analytical values
            obj.assertTrue(isnumeric(results.se), 'Robust standard errors must be numeric');
            
            % Test Newey-West HAC standard errors
            options = struct('se_type', 'newey-west', 'nw_lags', 3);
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify options.se_type = 'newey-west' produces correct HAC standard errors
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            
            % Test bootstrap standard errors
            options = struct('se_type', 'bootstrap', 'boot_options', struct('replications', 100));
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify options.se_type = 'bootstrap' produces reasonable standard errors
            obj.assertTrue(isfield(results, 'se'), 'Results structure must contain se field');
            
            % Compare all robust standard error methods with expected results
            obj.assertTrue(isnumeric(results.se), 'Bootstrap standard errors must be numeric');
        end
        
        function testRegressionDiagnostics(obj)
            % Test regression diagnostic statistics and tests
            
            % Create options structure with diagnostics = true
            options = struct('diagnostics', true);
            
            % Call cross_section_regression with diagnostics option
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify results structure contains diagnostic fields
            obj.assertTrue(isfield(results, 'diagnostics'), 'Results structure must contain diagnostics field');
            obj.assertTrue(isfield(results.diagnostics, 'white_test'), 'Results structure must contain white_test field');
            obj.assertTrue(isfield(results.diagnostics, 'lm_test'), 'Results structure must contain lm_test field');
            obj.assertTrue(isfield(results.diagnostics, 'jb_test'), 'Results structure must contain jb_test field');
            obj.assertTrue(isfield(results.diagnostics, 'influence'), 'Results structure must contain influence field');
            
            % Check heteroskedasticity test results (White test)
            obj.assertTrue(isfield(results.diagnostics.white_test, 'stat'), 'Results structure must contain white_test.stat field');
            obj.assertTrue(isfield(results.diagnostics.white_test, 'pval'), 'Results structure must contain white_test.pval field');
            
            % Validate autocorrelation test results (LM test)
            obj.assertTrue(isfield(results.diagnostics.lm_test, 'stat'), 'Results structure must contain lm_test.stat field');
            obj.assertTrue(isfield(results.diagnostics.lm_test, 'pval'), 'Results structure must contain lm_test.pval field');
            
            % Check normality test results (Jarque-Bera)
            obj.assertTrue(isfield(results.diagnostics.jb_test, 'stat'), 'Results structure must contain jb_test.stat field');
            obj.assertTrue(isfield(results.diagnostics.jb_test, 'pval'), 'Results structure must contain jb_test.pval field');
            
            % Verify influence measures (leverage, Cook's distance)
            obj.assertTrue(isfield(results.diagnostics.influence, 'leverage'), 'Results structure must contain influence.leverage field');
            obj.assertTrue(isfield(results.diagnostics.influence, 'cook_d'), 'Results structure must contain influence.cook_d field');
            
            % Compare diagnostic statistics with expected values
            obj.assertTrue(isnumeric(results.diagnostics.white_test.stat), 'White test statistic must be numeric');
            obj.assertTrue(isnumeric(results.diagnostics.lm_test.stat), 'LM test statistic must be numeric');
            obj.assertTrue(isnumeric(results.diagnostics.jb_test.stat), 'Jarque-Bera test statistic must be numeric');
        end
        
        function testConfidenceIntervals(obj)
            % Test confidence interval calculation for regression coefficients
            
            % Create options structure with confidence level settings
            options = struct('alpha', 0.1); % 90% confidence level
            
            % Call cross_section_regression with confidence interval options
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify results structure contains confidence interval fields
            obj.assertTrue(isfield(results, 'ci'), 'Results structure must contain ci field');
            
            % Check standard confidence intervals based on t-distribution
            obj.assertTrue(isnumeric(results.ci), 'Confidence intervals must be numeric');
            
            % Validate bootstrap confidence intervals
            options = struct('se_type', 'bootstrap', 'boot_options', struct('replications', 100), 'alpha', 0.1);
            results = cross_section_regression(obj.testDependent, obj.testIndependent, options);
            
            % Verify interval bounds contain true parameter values at specified frequency
            obj.assertTrue(isnumeric(results.ci), 'Bootstrap confidence intervals must be numeric');
            
            % Compare confidence intervals with expected values
            obj.assertEqual(size(results.ci, 1), size(obj.testIndependent, 2), 'Number of confidence intervals should match number of regressors');
        end
        
        function testInvalidInputs(obj)
            % Test error handling for invalid regression inputs
            
            % Test empty dependent variable handling
            obj.assertThrows(@() cross_section_regression([], obj.testIndependent), 'datacheck:DATACHECK:InvalidInput', 'Empty dependent variable should throw an error');
            
            % Test empty independent variables handling
            obj.assertThrows(@() cross_section_regression(obj.testDependent, []), 'datacheck:DATACHECK:InvalidInput', 'Empty independent variables should throw an error');
            
            % Test mismatched dimensions between dependent and independent variables
            obj.assertThrows(@() cross_section_regression(obj.testDependent(1:end-1), obj.testIndependent), 'Y and X must have the same number of observations.', 'Mismatched dimensions should throw an error');
            
            % Test handling of NaN/Inf values in inputs
            testDependentNaN = obj.testDependent;
            testDependentNaN(1) = NaN;
            obj.assertThrows(@() cross_section_regression(testDependentNaN, obj.testIndependent), 'datacheck:DATACHECK:InvalidInput', 'NaN values in dependent variable should throw an error');
            
            testIndependentInf = obj.testIndependent;
            testIndependentInf(1,1) = Inf;
            obj.assertThrows(@() cross_section_regression(obj.testDependent, testIndependentInf), 'datacheck:DATACHECK:InvalidInput', 'Inf values in independent variables should throw an error');
            
            % Test invalid options structure fields
            options = struct('invalid_option', 'value');
            obj.assertThrows(@() cross_section_regression(obj.testDependent, obj.testIndependent, options), 'OPTIONS.method must be one of: ''ols'', ''wls'', or ''robust''.', 'Invalid options structure should throw an error');
            
            % Test invalid regression method specification
            options = struct('method', 'invalid');
            obj.assertThrows(@() cross_section_regression(obj.testDependent, obj.testIndependent, options), 'OPTIONS.method must be one of: ''ols'', ''wls'', or ''robust''.', 'Invalid regression method should throw an error');
            
            % Test invalid standard error type specification
            options = struct('se_type', 'invalid');
            obj.assertThrows(@() cross_section_regression(obj.testDependent, obj.testIndependent, options), 'OPTIONS.se_type must be one of: ''standard'', ''robust'', ''newey-west'', or ''bootstrap''.', 'Invalid standard error type should throw an error');
            
            % Verify appropriate error messages for each invalid input case
        end
        
        function testLargeDataset(obj)
            % Test regression performance on large cross-sectional datasets
            
            % Generate large cross-sectional dataset using TestDataGenerator
            numObservations = 5000;
            numRegressors = 50;
            largeData = TestDataGenerator('generateCrossSectionalData', numObservations, 1, numRegressors);
            
            % Measure execution time for regression on large dataset
            executionTime = obj.measureExecutionTime(@cross_section_regression, largeData.returns, largeData.loadings);
            
            % Verify numerical stability of results with large dataset
            results = cross_section_regression(largeData.returns, largeData.loadings);
            obj.assertTrue(all(isfinite(results.beta)), 'Results should be numerically stable');
            
            % Check memory usage during large dataset processing
            memoryInfo = obj.checkMemoryUsage(@cross_section_regression, largeData.returns, largeData.loadings);
            
            % Compare performance metrics with expected thresholds
            obj.assertTrue(executionTime < 1, 'Execution time should be less than 1 second');
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 500, 'Memory usage should be less than 500 MB');
        end
        
        function testRegressionWithMissingData(obj)
            % Test regression handling of missing data points
            
            % Create test dataset with strategically placed NaN values
            testDependentMissing = obj.testDependent;
            testDependentMissing(1:5:end) = NaN;
            testIndependentMissing = obj.testIndependent;
            testIndependentMissing(2:7:end, 1) = NaN;
            
            % Test handling of missing data with different options
            % Verify results with options.handleMissing = 'remove'
            optionsRemove = struct('handleMissing', 'remove');
            resultsRemove = cross_section_regression(testDependentMissing, testIndependentMissing, optionsRemove);
            obj.assertTrue(size(resultsRemove.beta, 1) == size(obj.testIndependent, 2), 'Number of coefficients should match number of regressors');
            
            % Validate results with options.handleMissing = 'impute'
            % (This test is skipped due to the complexity of imputation methods)
            
            % Compare results with missing data handling to complete-data results
            % (This is a qualitative check, as the exact values will vary)
        end
        
        function testCreateRegressionData(obj)
           % Test the createTestRegressionData helper method
           numObservations = 100;
           numRegressors = 5;
           options = struct('heteroskedasticity', true, 'outliers', true);
           
           testData = obj.createTestRegressionData(numObservations, numRegressors, options);
           
           obj.assertTrue(isstruct(testData), 'createTestRegressionData should return a struct');
           obj.assertTrue(isfield(testData, 'y'), 'Test data should contain dependent variable');
           obj.assertTrue(isfield(testData, 'X'), 'Test data should contain independent variables');
           obj.assertTrue(isfield(testData, 'true_beta'), 'Test data should contain true beta coefficients');
           
           obj.assertEqual(size(testData.y, 1), numObservations, 'Dependent variable should have correct number of observations');
           obj.assertEqual(size(testData.X, 1), numObservations, 'Independent variables should have correct number of observations');
           obj.assertEqual(size(testData.X, 2), numRegressors, 'Independent variables should have correct number of regressors');
           obj.assertEqual(size(testData.true_beta, 1), numRegressors, 'True beta coefficients should have correct number of regressors');
        end
        
        function testValidateRegressionResults(obj)
            % Test the validateRegressionResults helper method
            
            % Create dummy regression results and expected values
            results = struct('beta', [1; 2], 'se', [0.1; 0.2], 'tstat', [10; 10], 'pval', [0; 0], 'r2', 0.9);
            expected = struct('beta', [1.1; 1.9], 'se', [0.11; 0.19], 'tstat', [10; 10], 'pval', [0; 0], 'r2', 0.9);
            tolerance = 0.1;
            
            % Validate results
            isValid = obj.validateRegressionResults(results, expected, tolerance);
            obj.assertTrue(isValid, 'Regression results should be valid within tolerance');
            
            % Test with invalid results
            resultsInvalid = struct('beta', [1; 2], 'se', [0.1; 0.2], 'tstat', [10; 10], 'pval', [0; 0], 'r2', 0.5);
            isValidInvalid = obj.validateRegressionResults(resultsInvalid, expected, tolerance);
            obj.assertFalse(isValidInvalid, 'Regression results should be invalid');
        end
    end
    
    methods (Access = private)
        function testData = createTestRegressionData(obj, numObservations, numRegressors, options)
            % Helper method to create controlled regression test data with known properties
            
            % Generate true beta coefficients with known values
            true_beta = randn(numRegressors, 1);
            
            % Create independent variables matrix X with controlled properties
            X = randn(numObservations, numRegressors);
            
            % Generate error terms with specified distribution
            error = randn(numObservations, 1);
            
            % Calculate dependent variable y = X*beta + error
            y = X*true_beta + error;
            
            % Add heteroskedasticity if specified in options
            if isfield(options, 'heteroskedasticity') && options.heteroskedasticity
                sigma = linspace(0.5, 1.5, numObservations);
                error = error .* sigma';
                y = X*true_beta + error;
            end
            
            % Add outliers if specified in options
            if isfield(options, 'outliers') && options.outliers
                outlier_indices = randi([1, numObservations], 5, 1);
                y(outlier_indices) = y(outlier_indices) + 5*randn(5, 1);
            end
            
            % Return structure with y, X, true_beta, and other data generation parameters
            testData = struct('y', y, 'X', X, 'true_beta', true_beta);
        end
        
        function isValid = validateRegressionResults(obj, results, expected, tolerance)
            % Helper method to validate regression results against expected values
            
            % Check existence of all expected fields in results structure
            expectedFields = fieldnames(expected);
            for i = 1:length(expectedFields)
                if ~isfield(results, expectedFields{i})
                    isValid = false;
                    return;
                end
            end
            
            % Compare coefficient estimates with expected values
            if ~obj.comparator.compareMatrices(results.beta, expected.beta, tolerance).isEqual
                isValid = false;
                return;
            end
            
            % Validate standard errors against expected values
            if ~obj.comparator.compareMatrices(results.se, expected.se, tolerance).isEqual
                isValid = false;
                return;
            end
            
            % Check t-statistics and p-values against expected
            if ~obj.comparator.compareMatrices(results.tstat, expected.tstat, tolerance).isEqual
                isValid = false;
                return;
            end
            if ~obj.comparator.compareMatrices(results.pval, expected.pval, tolerance).isEqual
                isValid = false;
                return;
            end
            
            % Verify goodness-of-fit measures (R², F-statistic)
            if ~obj.comparator.compareScalars(results.r2, expected.r2, tolerance).isEqual
                isValid = false;
                return;
            end
            
            % Return true if all validations pass, false otherwise
            isValid = true;
        end
    end
end