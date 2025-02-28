classdef SkewtrndTest < BaseTest
    % Test class for the skewtrnd function that generates random numbers from Hansen's skewed t-distribution
    
    properties
        testData      % Structure containing test data
        testTolerance % Tolerance for numerical comparisons
    end
    
    methods
        function obj = SkewtrndTest()
            % Initialize the SkewtrndTest class with test data
            obj = obj@BaseTest();
            obj.testTolerance = 1e-10;
            obj.testData = obj.loadTestData('known_distributions.mat');
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            % Set fixed random seed for reproducible tests
            rng(1);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            tearDown@BaseTest(obj);
            % Reset random number generator state
            rng('default');
        end
        
        function testBasicFunctionality(obj)
            % Test that skewtrnd generates matrix of correct size
            nu = 5;
            lambda = 0.5;
            rows = 5;
            cols = 10;
            
            % Generate random numbers
            result = skewtrnd(nu, lambda, rows, cols);
            
            % Check dimensions
            obj.assertEqual(size(result), [rows, cols], 'Matrix dimensions are incorrect');
            
            % Check that all values are finite
            obj.assertTrue(all(isfinite(result(:))), 'Generated values should be finite');
        end
        
        function testOutputDistribution(obj)
            % Test that output values follow the skewed t-distribution
            % Use fixed random seed for reproducibility
            rng(123);
            
            % Generate large sample for statistical tests
            nu = 5;
            lambda = 0.3;
            n = 10000;
            sample = skewtrnd(nu, lambda, n, 1);
            
            % Calculate empirical moments
            mean_sample = mean(sample);
            var_sample = var(sample);
            skewness_sample = skewness(sample);
            kurtosis_sample = kurtosis(sample);
            
            % Check that moments are within reasonable bounds
            % Note: For Hansen's skewed t, exact theoretical moments are complex
            % but we can check that they're in reasonable ranges
            
            % Mean should be close to 0 for small lambda
            obj.assertTrue(abs(mean_sample) < 0.1, 'Sample mean should be close to 0 for small lambda');
            
            % Variance should be close to 1 for standardized distribution
            obj.assertTrue(abs(var_sample - 1) < 0.1, 'Sample variance should be close to 1');
            
            % Skewness should match sign of lambda
            if lambda > 0
                obj.assertTrue(skewness_sample > 0, 'Skewness should be positive for positive lambda');
            elseif lambda < 0
                obj.assertTrue(skewness_sample < 0, 'Skewness should be negative for negative lambda');
            end
            
            % Perform chi-square goodness-of-fit test
            [counts, edges] = histcounts(sample, 50);
            centers = (edges(1:end-1) + edges(2:end))/2;
            expected_probs = skewtpdf(centers, nu, lambda);
            expected_counts = n * expected_probs * (edges(2) - edges(1));
            
            % Chi-square test
            [h, p] = chi2gof(sample, 'Edges', edges, 'Expected', expected_counts(1:end));
            
            % Assert that the test doesn't reject at 1% significance level
            obj.assertTrue(p > 0.01, 'Chi-square test rejected the skewed t-distribution hypothesis');
        end
        
        function testParameterEffects(obj)
            % Test that different parameter combinations produce appropriate distributions
            rng(234);
            
            % Test degrees of freedom effect on kurtosis
            nu_values = [3, 5, 10, 30];
            kurtosis_values = zeros(size(nu_values));
            
            for i = 1:length(nu_values)
                sample = skewtrnd(nu_values(i), 0, 5000, 1);
                kurtosis_values(i) = kurtosis(sample);
            end
            
            % Verify kurtosis decreases as nu increases
            for i = 1:(length(nu_values)-1)
                obj.assertTrue(kurtosis_values(i) > kurtosis_values(i+1), ...
                    'Kurtosis should decrease as degrees of freedom increases');
            end
            
            % Test skewness parameter effect
            lambda_values = [-0.8, -0.3, 0, 0.3, 0.8];
            skewness_values = zeros(size(lambda_values));
            
            for i = 1:length(lambda_values)
                sample = skewtrnd(5, lambda_values(i), 5000, 1);
                skewness_values(i) = skewness(sample);
            end
            
            % Verify skewness increases with lambda
            for i = 1:(length(lambda_values)-1)
                obj.assertTrue(skewness_values(i) < skewness_values(i+1), ...
                    'Skewness should increase with lambda');
            end
            
            % Check symmetry for lambda=0 (should be close to zero skewness)
            zero_idx = find(lambda_values == 0);
            obj.assertTrue(abs(skewness_values(zero_idx)) < 0.1, ...
                'Zero lambda should produce approximately symmetric distribution');
        end
        
        function testInvalidInputs(obj)
            % Test error handling for invalid input parameters
            
            % Invalid degrees of freedom
            obj.assertThrows(@() skewtrnd(2, 0), 'MATLAB:parametercheck:lowerBound', ...
                'Should throw error for nu = 2');
            obj.assertThrows(@() skewtrnd(1, 0), 'MATLAB:parametercheck:lowerBound', ...
                'Should throw error for nu < 2');
            obj.assertThrows(@() skewtrnd(-5, 0), 'MATLAB:parametercheck:lowerBound', ...
                'Should throw error for negative nu');
            
            % Invalid skewness parameter
            obj.assertThrows(@() skewtrnd(5, -1.1), 'MATLAB:parametercheck:upperBound', ...
                'Should throw error for lambda < -1');
            obj.assertThrows(@() skewtrnd(5, 1.1), 'MATLAB:parametercheck:upperBound', ...
                'Should throw error for lambda > 1');
            
            % Invalid size parameters
            obj.assertThrows(@() skewtrnd(5, 0, -1), 'MATLAB:parametercheck:isPositive', ...
                'Should throw error for negative n');
            obj.assertThrows(@() skewtrnd(5, 0, 1, -1), 'MATLAB:parametercheck:isPositive', ...
                'Should throw error for negative m');
            
            % Non-numeric inputs
            obj.assertThrows(@() skewtrnd('string', 0), 'MATLAB:parametercheck:invalidType', ...
                'Should throw error for non-numeric nu');
            obj.assertThrows(@() skewtrnd(5, 'string'), 'MATLAB:parametercheck:invalidType', ...
                'Should throw error for non-numeric lambda');
            obj.assertThrows(@() skewtrnd(5, 0, 'string'), 'MATLAB:parametercheck:invalidType', ...
                'Should throw error for non-numeric n');
            obj.assertThrows(@() skewtrnd(5, 0, 1, 'string'), 'MATLAB:parametercheck:invalidType', ...
                'Should throw error for non-numeric m');
        end
        
        function testReproducibility(obj)
            % Test that results are reproducible with fixed random seed
            
            % Set specific random seed
            rng(42);
            sample1 = skewtrnd(5, 0.3, 10, 10);
            
            % Reset to same seed
            rng(42);
            sample2 = skewtrnd(5, 0.3, 10, 10);
            
            % Check that both samples are identical
            obj.assertEqual(sample1, sample2, 'Samples should be identical with the same random seed');
        end
        
        function testKnownValues(obj)
            % Test against pre-computed known values from test data
            
            % Extract expected values from test data
            if isfield(obj.testData, 'skewt_rnd_seeds')
                knownData = obj.testData.skewt_rnd_seeds;
                
                for i = 1:length(knownData)
                    % Set random seed according to test data
                    rng(knownData(i).seed);
                    
                    % Generate sample
                    result = skewtrnd(knownData(i).nu, knownData(i).lambda, knownData(i).n, knownData(i).m);
                    
                    % Compare with expected values
                    obj.assertMatrixEqualsWithTolerance(result, knownData(i).expected, obj.testTolerance, ...
                        sprintf('Random values do not match expected for seed %d', knownData(i).seed));
                end
            else
                % Skip this test if test data doesn't contain skewt_rnd_seeds
                warning('Test data does not contain skewt_rnd_seeds field. Skipping known values test.');
            end
        end
        
        function testSizeParameters(obj)
            % Test different size parameter combinations
            
            % Test default size (1x1)
            result = skewtrnd(5, 0.3);
            obj.assertEqual(size(result), [1, 1], 'Default size should be 1x1');
            
            % Test scalar output
            result = skewtrnd(5, 0.3, 1, 1);
            obj.assertEqual(size(result), [1, 1], 'Size should be 1x1');
            
            % Test vector output
            result = skewtrnd(5, 0.3, 10, 1);
            obj.assertEqual(size(result), [10, 1], 'Size should be 10x1');
            
            % Test matrix output
            result = skewtrnd(5, 0.3, 5, 10);
            obj.assertEqual(size(result), [5, 10], 'Size should be 5x10');
        end
        
        function testExtremeCases(obj)
            % Test behavior with extreme parameter values
            
            % Test with degrees of freedom very close to lower bound
            result = skewtrnd(2.001, 0, 5, 5);
            obj.assertTrue(all(isfinite(result(:))), 'Result should be finite even with nu close to 2');
            
            % Test with extreme skewness
            result_neg = skewtrnd(5, -0.999, 5, 5);
            result_pos = skewtrnd(5, 0.999, 5, 5);
            obj.assertTrue(all(isfinite(result_neg(:))), 'Result should be finite with extreme negative skewness');
            obj.assertTrue(all(isfinite(result_pos(:))), 'Result should be finite with extreme positive skewness');
            
            % Test with large output size
            try
                result = skewtrnd(5, 0.3, 1000, 100);
                obj.assertTrue(true, 'Function should handle large output sizes');
            catch e
                obj.assertTrue(false, sprintf('Function failed with large output size: %s', e.message));
            end
            
            % Check numerical stability near boundaries
            nu_near_bound = 2.0001;
            lambda_near_bound = 0.9999;
            result = skewtrnd(nu_near_bound, lambda_near_bound, 10, 1);
            obj.assertTrue(all(isfinite(result)), 'Function should be numerically stable near parameter bounds');
        end
    end
end