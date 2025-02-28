classdef GedloglikTest < BaseTest
    % Test class for validating the gedloglik function which computes the log-likelihood 
    % of data under the Generalized Error Distribution
    
    properties
        testData
        testTolerance
    end
    
    methods
        function obj = GedloglikTest()
            % Initialize the GedloglikTest class
            obj@BaseTest();
            obj.testTolerance = 1e-10;
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepare the test environment before each test
            setUp@BaseTest(obj);
            obj.testData = obj.loadTestData('known_distributions.mat');
        end
        
        function tearDown(obj)
            % Clean up test environment after each test
            tearDown@BaseTest(obj);
            % Clear test data
            obj.testData = struct();
        end
        
        function testBasicCalculation(obj)
            % Test basic log-likelihood calculation with known values
            if isfield(obj.testData, 'ged_loglik_values')
                testCases = obj.testData.ged_loglik_values;
                
                % Test each case
                for i = 1:size(testCases, 1)
                    data = testCases(i).data;
                    nu = testCases(i).nu;
                    mu = testCases(i).mu;
                    sigma = testCases(i).sigma;
                    expected = testCases(i).loglik;
                    
                    % Compute log-likelihood
                    actual = gedloglik(data, nu, mu, sigma);
                    
                    % Assert almost equal with custom message
                    obj.assertAlmostEqual(expected, actual, ...
                        sprintf('Log-likelihood mismatch for case %d: nu=%.2f, mu=%.2f, sigma=%.2f', ...
                        i, nu, mu, sigma));
                end
            else
                % If test data doesn't have the expected field, create simple test case
                data = randn(10, 1);
                nu = 2; % Normal distribution
                mu = 0;
                sigma = 1;
                
                % For normal distribution, we can compare with standard normal log-likelihood
                expected = sum(-0.5*log(2*pi) - 0.5 * data.^2);
                actual = gedloglik(data, nu, mu, sigma);
                
                obj.assertAlmostEqual(expected, actual, 'Basic calculation failed for normal case');
            end
        end
        
        function testParameterValidation(obj)
            % Test parameter validation in the gedloglik function
            % Create valid test data
            data = randn(10, 1);
            nu = 1.5;
            mu = 0;
            sigma = 1;
            
            % Test invalid nu parameter (negative)
            obj.assertThrows(@() gedloglik(data, -1, mu, sigma), ...
                             'parametercheck:InvalidParameter', ...
                             'Should reject negative nu parameter');
                         
            % Test invalid nu parameter (zero)
            % Note: in gedloglik, zero nu returns -inf loglikelihood rather than error
            loglik_zero_nu = gedloglik(data, 0, mu, sigma);
            obj.assertTrue(isinf(loglik_zero_nu) && loglik_zero_nu < 0, ...
                          'Zero nu should return -Inf log-likelihood');
                         
            % Test invalid sigma parameter (negative)
            obj.assertThrows(@() gedloglik(data, nu, mu, -1), ...
                             'parametercheck:InvalidParameter', ...
                             'Should reject negative sigma parameter');
                         
            % Test invalid sigma parameter (zero)
            % Note: in gedloglik, zero sigma returns -inf loglikelihood rather than error
            loglik_zero_sigma = gedloglik(data, nu, mu, 0);
            obj.assertTrue(isinf(loglik_zero_sigma) && loglik_zero_sigma < 0, ...
                          'Zero sigma should return -Inf log-likelihood');
                         
            % Test invalid data (NaN)
            data_nan = [data; NaN];
            obj.assertThrows(@() gedloglik(data_nan, nu, mu, sigma), ...
                             'datacheck:InvalidData', ...
                             'Should reject data with NaN values');
                         
            % Test invalid data (Inf)
            data_inf = [data; Inf];
            obj.assertThrows(@() gedloglik(data_inf, nu, mu, sigma), ...
                             'datacheck:InvalidData', ...
                             'Should reject data with Inf values');
        end
        
        function testVectorizedOperation(obj)
            % Test vectorized operation with matrix inputs
            % Create test data
            data = randn(5, 1);
            nu = 1.5;
            mu = 0;
            sigma = 1;
            
            % Test scalar parameters with vector data
            loglik_vector = gedloglik(data, nu, mu, sigma);
            
            % Calculate individual log-likelihoods and sum
            loglik_sum = 0;
            for i = 1:length(data)
                loglik_sum = loglik_sum + gedloglik(data(i), nu, mu, sigma);
            end
            
            % Verify vectorized calculation equals sum of individual calculations
            obj.assertAlmostEqual(loglik_vector, loglik_sum, ...
                                 'Vectorized operation with scalar parameters failed');
            
            % Test vector parameters with compatible data
            mu_vector = linspace(-1, 1, length(data))';
            sigma_vector = linspace(0.5, 1.5, length(data))';
            
            % Vectorized calculation
            loglik_vec_params = gedloglik(data, nu, mu_vector, sigma_vector);
            
            % Individual calculations
            loglik_ind_sum = 0;
            for i = 1:length(data)
                loglik_ind_sum = loglik_ind_sum + gedloglik(data(i), nu, mu_vector(i), sigma_vector(i));
            end
            
            % Verify vectorized calculation with vector parameters
            obj.assertAlmostEqual(loglik_vec_params, loglik_ind_sum, ...
                                 'Vectorized operation with vector parameters failed');
            
            % Test matrix data
            data_matrix = reshape(randn(6, 1), [2, 3]);
            loglik_matrix = gedloglik(data_matrix, nu, mu, sigma);
            
            % Calculate using flattened data
            loglik_flat = gedloglik(data_matrix(:), nu, mu, sigma);
            
            % Verify matrix handling
            obj.assertAlmostEqual(loglik_matrix, loglik_flat, ...
                                 'Matrix data handling failed');
        end
        
        function testEdgeCases(obj)
            % Test edge cases and boundary conditions
            % Small dataset for edge case testing
            data = randn(5, 1);
            
            % Test with small nu (approaching 0)
            small_nu = 1e-5;
            loglik_small_nu = gedloglik(data, small_nu, 0, 1);
            
            % Test with large nu (approaching infinity)
            large_nu = 1e5;
            loglik_large_nu = gedloglik(data, large_nu, 0, 1);
            
            % Both should be finite
            obj.assertTrue(isfinite(loglik_small_nu), 'Log-likelihood with small nu should be finite');
            obj.assertTrue(isfinite(loglik_large_nu), 'Log-likelihood with large nu should be finite');
            
            % Test with small sigma (approaching 0, but still valid)
            small_sigma = 1e-5;
            loglik_small_sigma = gedloglik(data, 2, 0, small_sigma);
            
            % Test with large sigma
            large_sigma = 1e5;
            loglik_large_sigma = gedloglik(data, 2, 0, large_sigma);
            
            % Both should be finite
            obj.assertTrue(isfinite(loglik_small_sigma), 'Log-likelihood with small sigma should be finite');
            obj.assertTrue(isfinite(loglik_large_sigma), 'Log-likelihood with large sigma should be finite');
            
            % Test with extreme data values
            extreme_data = 1e5 * ones(5, 1);
            loglik_extreme = gedloglik(extreme_data, 2, 0, 1);
            
            % Should be finite but very negative
            obj.assertTrue(isfinite(loglik_extreme), 'Log-likelihood with extreme data should be finite');
            obj.assertTrue(loglik_extreme < -1e5, 'Log-likelihood with extreme data should be very negative');
        end
        
        function testSpecialCases(obj)
            % Test special cases of the GED distribution
            % Test data
            data = randn(100, 1);
            mu = 0;
            sigma = 1;
            
            % Case 1: nu=2 (Normal distribution)
            ged_normal = gedloglik(data, 2, mu, sigma);
            
            % Calculate normal log-likelihood directly
            normal_loglik = sum(-0.5*log(2*pi) - 0.5 * ((data - mu) ./ sigma).^2);
            
            % Verify GED with nu=2 matches normal distribution
            obj.assertAlmostEqual(ged_normal, normal_loglik, ...
                                 'GED with nu=2 should equal normal distribution log-likelihood');
            
            % Case 2: nu=1 (Laplace distribution)
            ged_laplace = gedloglik(data, 1, mu, sigma);
            
            % Calculate Laplace log-likelihood directly
            laplace_loglik = sum(-log(2*sigma) - abs((data - mu) ./ sigma));
            
            % Verify GED with nu=1 approximates Laplace distribution
            obj.assertAlmostEqual(ged_laplace, laplace_loglik, obj.testTolerance, ...
                                 'GED with nu=1 should approximate Laplace distribution log-likelihood');
            
            % Case 3: nu approaches infinity (uniform-like)
            % For very large nu, the tails become extremely thin
            ged_large_nu_1 = gedloglik(data, 10, mu, sigma);
            ged_large_nu_2 = gedloglik(data, 20, mu, sigma);
            
            % As nu increases, log-likelihood should decrease
            obj.assertTrue(ged_large_nu_2 < ged_large_nu_1, ...
                          'Log-likelihood should decrease as nu increases beyond normal distribution');
        end
        
        function testNumericalAccuracy(obj)
            % Test numerical accuracy against alternative calculation methods
            % Small dataset for accuracy testing
            data = randn(10, 1);
            nu_values = [0.5, 1, 1.5, 2, 2.5, 3];
            mu = 0;
            sigma = 1;
            
            for i = 1:length(nu_values)
                nu = nu_values(i);
                
                % Calculate using gedloglik
                loglik1 = gedloglik(data, nu, mu, sigma);
                
                % Calculate using formula directly
                z = abs((data - mu) ./ sigma);
                const = log(nu) - log(2) - log(sigma) - gammaln(1/nu);
                loglik2 = sum(const - 0.5 * (z.^nu));
                
                % Verify results match
                obj.assertAlmostEqual(loglik1, loglik2, ...
                    sprintf('Numerical calculation mismatch for nu=%.1f', nu));
            end
            
            % Test with non-zero mean
            mu = 1.5;
            loglik3 = gedloglik(data, 2, mu, sigma);
            
            % Normal distribution log-likelihood with mu=1.5
            loglik4 = sum(-0.5*log(2*pi) - 0.5 * ((data - mu) ./ sigma).^2);
            
            % Verify results match
            obj.assertAlmostEqual(loglik3, loglik4, ...
                'Numerical calculation mismatch for non-zero mean');
            
            % Test with different scale
            mu = 0;
            sigma = 2.5;
            loglik5 = gedloglik(data, 2, mu, sigma);
            
            % Normal distribution log-likelihood with sigma=2.5
            loglik6 = sum(-0.5*log(2*pi) - log(sigma) - 0.5 * ((data - mu) ./ sigma).^2);
            
            % Verify results match
            obj.assertAlmostEqual(loglik5, loglik6, ...
                'Numerical calculation mismatch for non-unit scale');
        end
        
        function testPerformance(obj)
            % Test performance with large datasets
            % Generate large dataset
            n = 10000;
            data = randn(n, 1);
            nu = 1.5;
            mu = 0;
            sigma = 1;
            
            % Measure execution time
            executionTime = obj.measureExecutionTime(@() gedloglik(data, nu, mu, sigma));
            
            % Verify execution completes in reasonable time
            % This is a soft check since performance depends on hardware
            obj.assertTrue(executionTime < 1.0, ...
                sprintf('Performance test failed: execution time %.4f seconds exceeds threshold', executionTime));
            
            % Compare vectorized vs. loop performance
            n_small = 1000;
            data_small = randn(n_small, 1);
            
            % Time vectorized calculation
            vec_time = obj.measureExecutionTime(@() gedloglik(data_small, nu, mu, sigma));
            
            % Time loop-based calculation
            loop_time = obj.measureExecutionTime(@() obj.loopCalculation(data_small, nu, mu, sigma));
            
            % Vectorized should be faster than loop
            obj.assertTrue(vec_time < loop_time, ...
                sprintf('Vectorized calculation (%.4fs) should be faster than loop-based calculation (%.4fs)', ...
                vec_time, loop_time));
        end
        
        function result = loopCalculation(obj, data, nu, mu, sigma)
            % Helper method for loop-based calculation
            result = 0;
            for i = 1:length(data)
                result = result + gedloglik(data(i), nu, mu, sigma);
            end
        end
    end
end