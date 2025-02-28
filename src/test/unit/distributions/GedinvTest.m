classdef GedinvTest < BaseTest
    % Test class for the gedinv function, providing comprehensive test coverage for Generalized Error Distribution inverse CDF calculations
    
    properties
        comparator
        defaultTolerance
        testData
        testProbabilities
        nuValues
        expectedResults
        referenceData
    end
    
    methods
        function obj = GedinvTest()
            % Initializes a new GedinvTest instance with numerical comparator
            obj = obj@BaseTest('GedinvTest');
            obj.testData = struct();
            obj.comparator = NumericalComparator();
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            setUp@BaseTest(obj);
            
            % Load reference data if available
            obj.referenceData = obj.loadTestData('known_distributions.mat');
            
            % Initialize test probabilities
            obj.testProbabilities = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]';
            
            % Initialize nu values to test
            obj.nuValues = [0.5, 1, 1.5, 2, 2.5, 5]';
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            tearDown@BaseTest(obj);
            
            % Clear test data
            obj.testData = struct();
        end
        
        function testBasicInv(obj)
            % Tests gedinv with basic parameter values
            
            % Test median value for standard normal case (nu=2)
            x = gedinv(0.5, 2);
            obj.assertAlmostEqual(x, 0, 'Median value should be 0 for standard GED with nu=2');
            
            % Test that the result is a double
            obj.assertTrue(isa(x, 'double'), 'Output should be of type double');
            
            % Test that the dimensions are correct
            obj.assertEqual(size(x), [1, 1], 'Output should be a scalar for scalar input');
            
            % Test symmetry around p=0.5 (should have x(0.5-a) = -x(0.5+a) due to symmetry)
            delta = 0.2;
            x1 = gedinv(0.5-delta, 2);
            x2 = gedinv(0.5+delta, 2);
            obj.assertAlmostEqual(x1, -x2, 'GED inverse CDF should be symmetric around p=0.5');
        end
        
        function testVectorInput(obj)
            % Tests gedinv with vectorized inputs
            
            % Test with a vector of probabilities
            probs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]';
            x = gedinv(probs, 2);
            
            % Check dimensions
            obj.assertEqual(size(x), size(probs), 'Output dimensions should match input dimensions');
            
            % For nu=2 (normal), we can compare with well-known values
            % Normal quantiles for these probabilities
            normal_quantiles = [-2.326, -1.645, -1.282, -0.674, 0, 0.674, 1.282, 1.645, 2.326]';
            
            % Compare with normal quantiles (with some tolerance)
            obj.assertMatrixEqualsWithTolerance(x, normal_quantiles, 1e-3, 'GED with nu=2 should match normal quantiles');
            
            % Test with multiple nu values
            for nu = [0.5, 1, 1.5, 2, 2.5, 5]
                x = gedinv(probs, nu);
                obj.assertEqual(size(x), size(probs), 'Output dimensions should match input dimensions for all nu values');
            end
        end
        
        function testLaplaceCase(obj)
            % Tests gedinv with nu=1 (Laplace distribution case)
            probs = obj.testProbabilities;
            
            % Calculate expected Laplace inverse CDF values
            expected = zeros(size(probs));
            for i = 1:length(probs)
                p = probs(i);
                if p < 0.5
                    expected(i) = log(2*p);
                else
                    expected(i) = -log(2*(1-p));
                end
            end
            
            % Test gedinv with nu=1
            actual = gedinv(probs, 1);
            
            % Compare with manually calculated values
            obj.assertMatrixEqualsWithTolerance(actual, expected, 1e-8, 'GED with nu=1 should match Laplace distribution');
        end
        
        function testNormalCase(obj)
            % Tests gedinv with nu=2 (Normal distribution case)
            probs = obj.testProbabilities;
            
            % Calculate expected values using standard normal quantiles
            expected = zeros(size(probs));
            for i = 1:length(probs)
                p = probs(i);
                if p == 0
                    expected(i) = -Inf;
                elseif p == 1
                    expected(i) = Inf;
                elseif p == 0.5
                    expected(i) = 0;
                else
                    % Calculate using standard normal approximation
                    if p < 0.5
                        q = sqrt(-2 * log(2 * p));
                        expected(i) = -q;
                    else
                        q = sqrt(-2 * log(2 * (1 - p)));
                        expected(i) = q;
                    end
                end
            end
            
            % Test gedinv with nu=2
            actual = gedinv(probs, 2);
            
            % Compare with expected values
            obj.assertMatrixEqualsWithTolerance(actual, expected, 0.01, 'GED with nu=2 should match normal distribution');
        end
        
        function testInvBoundary(obj)
            % Tests gedinv at boundary values (p=0, p=1)
            
            % Test p=0 for various nu values
            for nu = obj.nuValues
                x = gedinv(0, nu);
                obj.assertEqual(x, -Inf, ['gedinv(0, ', num2str(nu), ') should return -Inf']);
            end
            
            % Test p=1 for various nu values
            for nu = obj.nuValues
                x = gedinv(1, nu);
                obj.assertEqual(x, Inf, ['gedinv(1, ', num2str(nu), ') should return Inf']);
            end
            
            % Test values very close to boundaries
            small_p = 1e-10;
            large_p = 1-small_p;
            
            % These should be very large negative/positive but finite values
            for nu = obj.nuValues
                x_small = gedinv(small_p, nu);
                x_large = gedinv(large_p, nu);
                
                obj.assertTrue(x_small < -10, ['gedinv(', num2str(small_p), ', ', num2str(nu), ') should be a large negative value']);
                obj.assertTrue(x_large > 10, ['gedinv(', num2str(large_p), ', ', num2str(nu), ') should be a large positive value']);
                obj.assertTrue(isfinite(x_small), ['gedinv(', num2str(small_p), ', ', num2str(nu), ') should be finite']);
                obj.assertTrue(isfinite(x_large), ['gedinv(', num2str(large_p), ', ', num2str(nu), ') should be finite']);
            end
        end
        
        function testParameterValidation(obj)
            % Tests gedinv error handling for invalid parameters
            
            % Test invalid parameters
            obj.assertThrows(@() gedinv(0.5, 0), 'parametercheck:isPositive', 'Should throw error for nu = 0');
            obj.assertThrows(@() gedinv(0.5, -1), 'parametercheck:isPositive', 'Should throw error for nu < 0');
            obj.assertThrows(@() gedinv(-0.1, 2), 'gedinv:invalidProbability', 'Should throw error for p < 0');
            obj.assertThrows(@() gedinv(1.1, 2), 'gedinv:invalidProbability', 'Should throw error for p > 1');
            obj.assertThrows(@() gedinv(0.5, 2, 0, 0), 'parametercheck:isPositive', 'Should throw error for sigma = 0');
            obj.assertThrows(@() gedinv(0.5, 2, 0, -1), 'parametercheck:isPositive', 'Should throw error for sigma < 0');
            obj.assertThrows(@() gedinv(0.5, NaN), 'parametercheck:isNaN', 'Should throw error for NaN nu');
            obj.assertThrows(@() gedinv(0.5, Inf), 'parametercheck:isInf', 'Should throw error for Inf nu');
            obj.assertThrows(@() gedinv(NaN, 2), 'datacheck:isNaN', 'Should throw error for NaN p');
            obj.assertThrows(@() gedinv(Inf, 2), 'datacheck:isInf', 'Should throw error for Inf p');
        end
        
        function testInvCdfConsistency(obj)
            % Tests that gedinv is the inverse function of gedcdf
            
            % Test for various nu values
            for nu = obj.nuValues
                % Generate test values
                x_test = linspace(-5, 5, 11)';
                
                % Apply CDF then inverse CDF
                p = gedcdf(x_test, nu);
                x_recovered = gedinv(p, nu);
                
                % The recovered x should match the original x
                obj.assertMatrixEqualsWithTolerance(x_recovered, x_test, 1e-8, ['Inverse consistency failed for nu=', num2str(nu)]);
                
                % Also test the reverse: apply inverse CDF then CDF
                p_test = linspace(0.01, 0.99, 11)';
                x = gedinv(p_test, nu);
                p_recovered = gedcdf(x, nu);
                
                obj.assertMatrixEqualsWithTolerance(p_recovered, p_test, 1e-8, ['Forward consistency failed for nu=', num2str(nu)]);
            end
        end
        
        function testReferenceValues(obj)
            % Tests gedinv against pre-computed reference values
            
            % Skip test if reference data not available
            if ~isfield(obj.referenceData, 'ged_inv_values') || ~isfield(obj.referenceData, 'ged_parameters')
                warning('Reference data not available, skipping testReferenceValues');
                return;
            end
            
            % Load reference values
            refValues = obj.referenceData.ged_inv_values;
            refParams = obj.referenceData.ged_parameters;
            
            % Test against reference data
            for i = 1:size(refParams, 1)
                nu = refParams(i, 1);
                mu = refParams(i, 2);
                sigma = refParams(i, 3);
                
                % Get reference quantiles for this parameter set
                refQuantiles = refValues(i, :)';
                
                % Compute quantiles using gedinv
                actualQuantiles = gedinv(obj.testProbabilities, nu, mu, sigma);
                
                % Compare with reference values
                obj.assertMatrixEqualsWithTolerance(actualQuantiles, refQuantiles, 1e-8, ...
                    ['Reference value test failed for nu=', num2str(nu), ', mu=', num2str(mu), ', sigma=', num2str(sigma)]);
            end
        end
        
        function testLocationScaleParameters(obj)
            % Tests gedinv with location and scale parameters (mu and sigma)
            
            probs = obj.testProbabilities;
            nu = 1.5; % Example shape parameter
            
            % Test with different location parameters
            mu_values = [-2, 0, 3];
            for mu = mu_values
                % Compute quantiles with location parameter
                x_mu = gedinv(probs, nu, mu);
                
                % Compute quantiles without location parameter and add mu
                x_0 = gedinv(probs, nu);
                expected = x_0 + mu;
                
                % Quantiles should be shifted by mu
                obj.assertMatrixEqualsWithTolerance(x_mu, expected, obj.defaultTolerance, ...
                    ['Location parameter test failed for mu=', num2str(mu)]);
            end
            
            % Test with different scale parameters
            sigma_values = [0.5, 1, 2];
            for sigma = sigma_values
                % Compute quantiles with scale parameter
                x_sigma = gedinv(probs, nu, 0, sigma);
                
                % Compute quantiles without scale parameter and multiply by sigma
                x_1 = gedinv(probs, nu);
                
                % Note: need to account for the lambda term in the transformation
                lambda = sqrt(gamma(3/nu)/gamma(1/nu));
                expected = lambda * sigma * x_1;
                
                % Quantiles should be scaled by sigma * lambda
                obj.assertMatrixEqualsWithTolerance(x_sigma, expected, 1e-8, ...
                    ['Scale parameter test failed for sigma=', num2str(sigma)]);
            end
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of gedinv for extreme values
            
            % Test with small nu values (heavy tails)
            small_nu = 0.1;
            x_small_nu = gedinv(obj.testProbabilities, small_nu);
            
            % Verify that values are finite and ordered correctly
            obj.assertTrue(all(isfinite(x_small_nu)), 'gedinv should return finite values for small nu');
            obj.assertTrue(all(diff(x_small_nu) > 0), 'gedinv values should be strictly increasing for small nu');
            
            % Test with large nu values (approaching normal)
            large_nu = 100;
            x_large_nu = gedinv(obj.testProbabilities, large_nu);
            
            % For large nu, should approach normal quantiles
            normal_quantiles = norminv(obj.testProbabilities);
            obj.assertMatrixEqualsWithTolerance(x_large_nu, normal_quantiles, 0.1, 'For large nu, GED should approach normal distribution');
            
            % Test extreme probability values
            extreme_probs = [1e-10, 1e-5, 1 - 1e-5, 1 - 1e-10]';
            
            % For multiple nu values
            for nu = [1, 2, 5]
                x_extreme = gedinv(extreme_probs, nu);
                obj.assertTrue(all(isfinite(x_extreme)), ['gedinv should return finite values for extreme probabilities with nu=', num2str(nu)]);
                obj.assertTrue(all(diff(x_extreme) > 0), ['gedinv values should be strictly increasing for extreme probabilities with nu=', num2str(nu)]);
            end
        end
        
        function expected = calculateManualGedinv(obj, p, nu)
            % Helper method to manually calculate GED inverse CDF for validation
            
            % Calculate lambda parameter
            lambda = sqrt(gamma(3/nu)/gamma(1/nu));
            
            % Initialize output
            expected = zeros(size(p));
            
            % For p=0.5, the quantile is 0 due to symmetry
            expected(p == 0.5) = 0;
            
            % For p=0, the quantile is -Inf
            expected(p == 0) = -Inf;
            
            % For p=1, the quantile is Inf
            expected(p == 1) = Inf;
            
            % Process remaining values
            idx = (p > 0 & p < 0.5);
            if any(idx)
                % For left tail (p < 0.5)
                % This is a simplified approximation for the left tail
                p_idx = p(idx);
                expected(idx) = -(-2 * log(p_idx)).^(1/nu) / lambda;
            end
            
            idx = (p > 0.5 & p < 1);
            if any(idx)
                % For right tail (p > 0.5)
                % This is a simplified approximation for the right tail
                p_idx = p(idx);
                expected(idx) = (-2 * log(1 - p_idx)).^(1/nu) / lambda;
            end
        end
    end
end