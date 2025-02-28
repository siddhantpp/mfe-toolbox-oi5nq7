classdef StdtloglikTest < BaseTest
    % StdtloglikTest Unit test class for standardized Student's t-distribution log-likelihood function
    %
    % This class tests the stdtloglik function which computes the log-likelihood
    % of data under the standardized Student's t-distribution. Tests cover 
    % numerical accuracy, parameter validation, vectorized operation, 
    % numerical stability, and performance characteristics.
    %
    % The test suite validates three key requirements:
    % 1. Distribution Analysis Engine - Tests standardized T distribution implementation
    % 2. Parameter estimation accuracy - Validates log-likelihood calculations
    % 3. Numerical stability - Ensures reliable computation across all input ranges
    
    properties
        testData    % Structure containing test data
        tolerance   % Numerical tolerance for comparisons
    end
    
    methods
        function obj = StdtloglikTest()
            % Initialize the test class with test data and tolerance settings
            
            % Call superclass constructor
            obj = obj@BaseTest();
            
            % Set numerical tolerance for high-precision financial calculations
            obj.tolerance = 1e-10;
            
            % Load test data for distribution tests
            obj.testData = obj.loadTestData('known_distributions.mat');
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Ensure test data is properly loaded and accessible
            if ~isfield(obj.testData, 'stdt_loglik_values') || ~isfield(obj.testData, 'stdt_parameters')
                error('Test data is missing required fields for stdtloglik tests');
            end
        end
        
        function testBasicFunctionality(obj)
            % Tests basic functionality of stdtloglik with standard inputs
            
            % Generate test data vector with randn
            data = randn(100, 1);
            
            % Calculate log-likelihood with standard parameters
            nu = 5;      % Degrees of freedom
            mu = 0;      % Location (mean)
            sigma = 1;   % Scale (std-like)
            
            % Call the function
            ll = stdtloglik(data, nu, mu, sigma);
            
            % Verify log-likelihood is a finite scalar value
            obj.assertTrue(isscalar(ll), 'Log-likelihood should be a scalar value');
            obj.assertTrue(isfinite(ll), 'Log-likelihood should be finite');
            
            % Verify log-likelihood is negative (expected for continuous distributions)
            % Note: stdtloglik returns the negative log-likelihood, so result should be positive
            obj.assertTrue(ll > 0, 'Negative log-likelihood should be positive');
            
            % Verify consistency of results when called multiple times with same inputs
            ll2 = stdtloglik(data, nu, mu, sigma);
            obj.assertEqual(ll, ll2, 'Function should return consistent results');
        end
        
        function testKnownValues(obj)
            % Tests stdtloglik against pre-computed values for known inputs
            
            % Extract pre-computed test cases from testData
            testCases = obj.testData.stdt_loglik_values;
            params = obj.testData.stdt_parameters;
            
            % For each test case, run stdtloglik with specified parameters
            for i = 1:size(testCases, 1)
                data = testCases(i).data;
                nu = params(i).nu;
                mu = params(i).mu;
                sigma = params(i).sigma;
                expectedLL = testCases(i).loglik;
                
                % Calculate log-likelihood
                actualLL = stdtloglik(data, nu, mu, sigma);
                
                % Compare results with pre-computed expected values using assertAlmostEqual
                obj.assertAlmostEqual(expectedLL, actualLL, ...
                    sprintf('Log-likelihood mismatch for test case %d', i));
            end
        end
        
        function testParameterValidation(obj)
            % Tests parameter validation and error handling in stdtloglik
            
            % Generate test data
            data = randn(10, 1);
            
            % Test with invalid nu parameter (nu < 2) and verify -Inf return value
            ll = stdtloglik(data, 1.9, 0, 1);
            obj.assertEqual(-Inf, ll, 'Invalid nu should return -Inf');
            
            % Test with invalid sigma parameter (sigma <= 0) and verify -Inf return value
            ll = stdtloglik(data, 5, 0, 0);
            obj.assertEqual(-Inf, ll, 'Invalid sigma should return -Inf');
            
            ll = stdtloglik(data, 5, 0, -1);
            obj.assertEqual(-Inf, ll, 'Negative sigma should return -Inf');
            
            % Test with non-numeric inputs and verify appropriate error handling
            obj.assertThrows(@() stdtloglik('string', 5, 0, 1), ...
                'MATLAB:parametercheck:invalidParameter', 'Should throw error for non-numeric data');
            
            % Test with NaN/Inf inputs and verify appropriate error handling
            data_with_nan = [1; 2; NaN; 4];
            obj.assertThrows(@() stdtloglik(data_with_nan, 5, 0, 1), ...
                'MATLAB:datacheck:containsNaN', 'Should throw error for NaN in data');
            
            data_with_inf = [1; 2; Inf; 4];
            obj.assertThrows(@() stdtloglik(data_with_inf, 5, 0, 1), ...
                'MATLAB:datacheck:containsInf', 'Should throw error for Inf in data');
            
            % Verify parameter dimension validation with inconsistent vector sizes
            mu_vec = [0; 1; 2];
            sigma_vec = [1; 1; 1];
            data_short = [1; 2];
            
            % Vector mu/sigma with incompatible data length should throw error
            obj.assertThrows(@() stdtloglik(data_short, 5, mu_vec, sigma_vec), ...
                'MATLAB:sizeDimensionsMustMatch', 'Should throw error for dimension mismatch');
        end
        
        function testVectorizedOperation(obj)
            % Tests vectorized operation with vector inputs for mu and sigma
            
            % Generate test data vector
            n = 5;
            data = randn(n, 1);
            nu = 5;
            
            % Test with vector mu matching data dimension
            mu_vec = linspace(-1, 1, n)';
            sigma_scalar = 1;
            
            ll_vec_mu = stdtloglik(data, nu, mu_vec, sigma_scalar);
            
            % Calculate element-wise for verification
            ll_element_wise = 0;
            for i = 1:n
                ll_element_wise = ll_element_wise + stdtloglik(data(i), nu, mu_vec(i), sigma_scalar);
            end
            
            obj.assertAlmostEqual(ll_vec_mu, ll_element_wise, ...
                'Vectorized operation with vector mu should match element-wise calculation');
            
            % Test with vector sigma matching data dimension
            mu_scalar = 0;
            sigma_vec = linspace(0.5, 1.5, n)';
            
            ll_vec_sigma = stdtloglik(data, nu, mu_scalar, sigma_vec);
            
            % Calculate element-wise for verification
            ll_element_wise = 0;
            for i = 1:n
                ll_element_wise = ll_element_wise + stdtloglik(data(i), nu, mu_scalar, sigma_vec(i));
            end
            
            obj.assertAlmostEqual(ll_vec_sigma, ll_element_wise, ...
                'Vectorized operation with vector sigma should match element-wise calculation');
            
            % Test with both vector mu and sigma matching data dimension
            ll_vec_both = stdtloglik(data, nu, mu_vec, sigma_vec);
            
            % Calculate element-wise for verification
            ll_element_wise = 0;
            for i = 1:n
                ll_element_wise = ll_element_wise + stdtloglik(data(i), nu, mu_vec(i), sigma_vec(i));
            end
            
            obj.assertAlmostEqual(ll_vec_both, ll_element_wise, ...
                'Vectorized operation with vector mu and sigma should match element-wise calculation');
            
            % Test performance with large vectors to ensure vectorization efficiency
            large_n = 10000;
            large_data = randn(large_n, 1);
            large_mu = zeros(large_n, 1);
            large_sigma = ones(large_n, 1);
            
            tic;
            stdtloglik(large_data, nu, large_mu, large_sigma);
            vectorized_time = toc;
            
            % Verify execution completes within reasonable time
            obj.assertTrue(vectorized_time < 1, ...
                'Vectorized operation should be efficient for large vectors');
        end
        
        function testNumericalStability(obj)
            % Tests numerical stability with extreme parameter values
            
            % Generate test data
            data = randn(10, 1);
            
            % Test with very large nu values (approaching normal distribution)
            ll_large_nu = stdtloglik(data, 1e6, 0, 1);
            obj.assertTrue(isfinite(ll_large_nu), 'Log-likelihood should be finite for large nu');
            
            % Test with nu very close to 2 (heavy tails)
            ll_heavy_tail = stdtloglik(data, 2.001, 0, 1);
            obj.assertTrue(isfinite(ll_heavy_tail), 'Log-likelihood should be finite for nu near 2');
            
            % Test with very small sigma values
            ll_small_sigma = stdtloglik(data, 5, 0, 1e-6);
            obj.assertTrue(isfinite(ll_small_sigma), 'Log-likelihood should be finite for small sigma');
            
            % Test with very large sigma values
            ll_large_sigma = stdtloglik(data, 5, 0, 1e6);
            obj.assertTrue(isfinite(ll_large_sigma), 'Log-likelihood should be finite for large sigma');
            
            % Test with extreme data values
            extreme_data = 1e6 * ones(10, 1);
            ll_extreme_data = stdtloglik(extreme_data, 5, 0, 1);
            obj.assertTrue(isfinite(ll_extreme_data), 'Log-likelihood should be finite for extreme data');
            
            % Verify increasing nu approaches normal distribution behavior
            % As nu increases, standardized t approaches normal distribution
            data_point = [0];  % At mean
            ll_t_large_nu = stdtloglik(data_point, 1e6, 0, 1);
            ll_normal = -log(normpdf(data_point, 0, 1));
            
            obj.assertAlmostEqual(ll_t_large_nu, ll_normal, 1e-3, ...
                'Large nu should approach normal distribution behavior');
        end
        
        function testPerformance(obj)
            % Tests performance with large datasets
            
            % Generate large dataset (10,000+ observations)
            n = 10000;
            large_data = randn(n, 1);
            nu = 5;
            mu = 0;
            sigma = 1;
            
            % Measure execution time using measureExecutionTime method
            executionTime = obj.measureExecutionTime(@() stdtloglik(large_data, nu, mu, sigma));
            
            % Verify reasonable performance for financial applications
            obj.assertTrue(executionTime < 1, ...
                sprintf('Processing %d observations should complete in under 1 second (actual: %.4f)', ...
                n, executionTime));
            
            % Test memory usage with large inputs
            memoryInfo = obj.checkMemoryUsage(@() stdtloglik(large_data, nu, mu, sigma));
            
            % Verify memory usage is reasonable
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 10, ...
                sprintf('Memory usage should be reasonable (actual: %.2f MB)', ...
                memoryInfo.memoryDifferenceMB));
        end
    end
end