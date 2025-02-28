classdef SkewtloglikTest < BaseTest
    % SKEWTLOGLIKTEST Unit test class for Hansen's skewed t-distribution log-likelihood function
    %
    % This class tests the skewtloglik function, which calculates the log-likelihood
    % for Hansen's skewed t-distribution. The tests validate numerical accuracy,
    % parameter handling, and error conditions for the log-likelihood calculation
    % function essential for maximum likelihood estimation with the skewed t-distribution.
    %
    % See also: skewtloglik, skewtpdf, BaseTest
    
    properties
        % Test data structure containing reference values
        testData
        
        % Numerical tolerance for floating-point comparisons
        tolerance
    end
    
    methods
        function obj = SkewtloglikTest()
            % Initialize the SkewtloglikTest class with test data and configurations
            
            % Call superclass constructor
            obj@BaseTest();
            
            % Set default numerical tolerance for floating-point comparisons
            obj.tolerance = 1e-10;
            
            % Load reference data for distribution tests
            obj.testData = obj.loadTestData('known_distributions.mat');
        end
        
        function setUp(obj)
            % Prepare test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Test the basic functionality of the log-likelihood calculation for known cases
            
            % Extract test parameters from test data
            parameters = obj.testData.skewt_parameters;
            expectedLogLik = obj.testData.skewt_loglik_values;
            
            % Generate sample data
            rng(42, 'twister'); % For reproducibility
            n = 1000;
            data = randn(n, 1);
            
            % Calculate log-likelihood for sample parameters
            [nLogLik, logLik] = skewtloglik(data, parameters(1,:));
            
            % Verify negative log-likelihood is the negative sum of individual log-likelihoods
            obj.assertAlmostEqual(-sum(logLik), nLogLik, 'Negative log-likelihood should equal negative sum of individual log-likelihoods');
            
            % Compare with expected values for known parameters
            [knownNLogLik, ~] = skewtloglik(obj.testData.skewt_test_data, parameters(1,:));
            obj.assertAlmostEqual(knownNLogLik, expectedLogLik(1), 'Log-likelihood value does not match expected value');
        end
        
        function testParameterHandling(obj)
            % Test correct handling of different parameter inputs including edge cases
            
            % Generate sample data
            rng(42, 'twister'); % For reproducibility
            n = 500;
            data = randn(n, 1);
            
            % Test with different valid parameter combinations
            
            % 1. Test with minimal valid nu (degrees of freedom)
            params1 = [2.0001, 0, 0, 1]; % nu just above 2
            [nLogLik1, ~] = skewtloglik(data, params1);
            obj.assertTrue(isfinite(nLogLik1), 'Log-likelihood should be finite for nu just above 2');
            
            % 2. Test with boundary valid lambda (skewness)
            params2 = [5, -0.999, 0, 1]; % lambda near -1
            [nLogLik2, ~] = skewtloglik(data, params2);
            obj.assertTrue(isfinite(nLogLik2), 'Log-likelihood should be finite for lambda near -1');
            
            params3 = [5, 0.999, 0, 1]; % lambda near 1
            [nLogLik3, ~] = skewtloglik(data, params3);
            obj.assertTrue(isfinite(nLogLik3), 'Log-likelihood should be finite for lambda near 1');
            
            % 3. Test parameter reordering is handled correctly
            params4 = [5, 0, 1, 2]; % [nu, lambda, mu, sigma]
            [nLogLik4a, logLik4a] = skewtloglik(data, params4);
            
            % Create same parameters but with different order in the code (should be identical)
            nu = params4(1);
            lambda = params4(2);
            mu = params4(3);
            sigma = params4(4);
            [nLogLik4b, logLik4b] = skewtloglik(data, [nu, lambda, mu, sigma]);
            
            obj.assertAlmostEqual(nLogLik4a, nLogLik4b, 'Parameter ordering should not affect results');
            obj.assertAlmostEqual(logLik4a, logLik4b, 'Parameter ordering should not affect individual log-likelihoods');
        end
        
        function testInputValidation(obj)
            % Test input validation and error handling
            
            % Generate sample data
            data = randn(100, 1);
            
            % 1. Test invalid degrees of freedom (nu ≤ 2)
            invalidNu = [2, 0, 0, 1]; % nu exactly 2 (should be > 2)
            obj.assertThrows(@() skewtloglik(data, invalidNu), 'BaseTest:AssertionFailed', 'Should throw error for nu ≤ 2');
            
            % 2. Test invalid skewness (lambda outside [-1,1])
            invalidLambdaHigh = [5, 1.1, 0, 1]; % lambda > 1
            obj.assertThrows(@() skewtloglik(data, invalidLambdaHigh), 'BaseTest:AssertionFailed', 'Should throw error for lambda > 1');
            
            invalidLambdaLow = [5, -1.1, 0, 1]; % lambda < -1
            obj.assertThrows(@() skewtloglik(data, invalidLambdaLow), 'BaseTest:AssertionFailed', 'Should throw error for lambda < -1');
            
            % 3. Test invalid scale (sigma ≤ 0)
            invalidSigmaZero = [5, 0, 0, 0]; % sigma = 0
            obj.assertThrows(@() skewtloglik(data, invalidSigmaZero), 'BaseTest:AssertionFailed', 'Should throw error for sigma = 0');
            
            invalidSigmaNeg = [5, 0, 0, -1]; % sigma < 0
            obj.assertThrows(@() skewtloglik(data, invalidSigmaNeg), 'BaseTest:AssertionFailed', 'Should throw error for sigma < 0');
            
            % 4. Test with non-numeric input data
            nonNumericData = {'a', 'b', 'c'};
            obj.assertThrows(@() skewtloglik(nonNumericData, [5, 0, 0, 1]), 'BaseTest:AssertionFailed', 'Should throw error for non-numeric data');
            
            % 5. Test with empty input data
            emptyData = [];
            [nLogL, logL] = skewtloglik(emptyData, [5, 0, 0, 1]);
            obj.assertTrue(isempty(nLogL), 'Should return empty for empty data');
            obj.assertTrue(isempty(logL), 'Should return empty for empty data');
            
            % 6. Test with NaN or Inf in data
            nanData = [1; 2; NaN; 4];
            obj.assertThrows(@() skewtloglik(nanData, [5, 0, 0, 1]), 'BaseTest:AssertionFailed', 'Should throw error for NaN in data');
            
            infData = [1; 2; Inf; 4];
            obj.assertThrows(@() skewtloglik(infData, [5, 0, 0, 1]), 'BaseTest:AssertionFailed', 'Should throw error for Inf in data');
        end
        
        function testVectorization(obj)
            % Test behavior with vectorized inputs
            
            % Generate sample data
            rng(42, 'twister'); % For reproducibility
            
            % Test with row vector
            rowData = randn(1, 100); % Row vector
            params = [5, 0.5, 0, 1]; % [nu, lambda, mu, sigma]
            
            % This should work - columncheck will convert row to column
            [nLogLikRow, logLikRow] = skewtloglik(rowData, params);
            
            % Compare with column vector
            colData = rowData'; % Convert to column
            [nLogLikCol, logLikCol] = skewtloglik(colData, params);
            
            % Results should be the same
            obj.assertAlmostEqual(nLogLikRow, nLogLikCol, 'Row and column vector should give same result');
            obj.assertAlmostEqual(logLikRow, logLikCol, 'Row and column vector should give same individual log-likelihoods');
            
            % Test with matrix input (should fail)
            matrixData = reshape(randn(300), 100, 3); % 100x3 matrix
            obj.assertThrows(@() skewtloglik(matrixData, params), 'BaseTest:AssertionFailed', 'Should throw error for matrix input');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability with extreme parameter values
            
            % Generate sample data
            rng(42, 'twister'); % For reproducibility
            n = 100;
            data = randn(n, 1);
            
            % 1. Test with large degrees of freedom (approaching normal distribution)
            largeNu = [1000, 0, 0, 1]; % large nu
            [nLogLik1, logLik1] = skewtloglik(data, largeNu);
            obj.assertTrue(isfinite(nLogLik1), 'Log-likelihood should be finite for large nu');
            obj.assertTrue(all(isfinite(logLik1)), 'Individual log-likelihoods should be finite for large nu');
            
            % 2. Test with extreme skewness values
            extremeLambdaPos = [5, 0.9999, 0, 1]; % lambda very close to 1
            [nLogLik2, logLik2] = skewtloglik(data, extremeLambdaPos);
            obj.assertTrue(isfinite(nLogLik2), 'Log-likelihood should be finite for lambda near 1');
            obj.assertTrue(all(isfinite(logLik2)), 'Individual log-likelihoods should be finite for lambda near 1');
            
            extremeLambdaNeg = [5, -0.9999, 0, 1]; % lambda very close to -1
            [nLogLik3, logLik3] = skewtloglik(data, extremeLambdaNeg);
            obj.assertTrue(isfinite(nLogLik3), 'Log-likelihood should be finite for lambda near -1');
            obj.assertTrue(all(isfinite(logLik3)), 'Individual log-likelihoods should be finite for lambda near -1');
            
            % 3. Test with very small and very large scale parameters
            smallSigma = [5, 0, 0, 1e-5]; % very small sigma
            [nLogLik4, logLik4] = skewtloglik(data, smallSigma);
            obj.assertTrue(isfinite(nLogLik4), 'Log-likelihood should be finite for small sigma');
            obj.assertTrue(all(isfinite(logLik4)), 'Individual log-likelihoods should be finite for small sigma');
            
            largeSigma = [5, 0, 0, 1e5]; % very large sigma
            [nLogLik5, logLik5] = skewtloglik(data, largeSigma);
            obj.assertTrue(isfinite(nLogLik5), 'Log-likelihood should be finite for large sigma');
            obj.assertTrue(all(isfinite(logLik5)), 'Individual log-likelihoods should be finite for large sigma');
            
            % 4. Test with outlier data points
            outlierData = [data; 10; -10]; % Add some outliers
            [nLogLik6, logLik6] = skewtloglik(outlierData, [5, 0, 0, 1]);
            obj.assertTrue(isfinite(nLogLik6), 'Log-likelihood should be finite with outliers');
            obj.assertTrue(all(isfinite(logLik6)), 'Individual log-likelihoods should be finite with outliers');
            
            % 5. Check for -Inf replacement (log of very small values)
            % Create data that will result in very small PDF values
            extremeData = 1000 * ones(10, 1); % Will be very far in the tail
            [nLogLik7, logLik7] = skewtloglik(extremeData, [3, 0, 0, 1]);
            obj.assertTrue(isfinite(nLogLik7), 'Log-likelihood should handle extreme tail values');
            obj.assertTrue(all(isfinite(logLik7)), 'Individual log-likelihoods should handle extreme tail values');
        end
        
        function testComparisonWithPDF(obj)
            % Test consistency between log-likelihood and PDF calculations
            
            % Generate sample data
            rng(42, 'twister'); % For reproducibility
            n = 100;
            data = randn(n, 1);
            
            % Parameters
            nu = 5;
            lambda = 0.5;
            mu = 0;
            sigma = 2;
            parameters = [nu, lambda, mu, sigma];
            
            % Calculate log-likelihood using skewtloglik
            [~, logLik] = skewtloglik(data, parameters);
            
            % Calculate log-likelihood manually from PDF
            % First standardize the data as done in skewtloglik
            standardized_data = (data - mu) / sigma;
            
            % Calculate PDF values
            pdf_values = skewtpdf(standardized_data, nu, lambda);
            
            % Calculate log-likelihood manually
            manual_logLik = log(pdf_values) - log(sigma);
            
            % Compare the two log-likelihood calculations
            obj.assertAlmostEqual(logLik, manual_logLik, 'Log-likelihood from skewtloglik should match manual calculation from PDF');
        end
    end
end