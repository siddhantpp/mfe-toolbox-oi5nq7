classdef SkewtcdfTest < BaseTest
    % SkewtcdfTest Test class for verifying the skewed t-distribution CDF (skewtcdf) implementation
    
    properties
        NumericalComparator comparator
        defaultTolerance
        testPoints
        testParameters
        knownValues
    end
    
    methods
        function obj = SkewtcdfTest()
            % Constructor - setup test environment with appropriate tolerance settings
            obj = obj@BaseTest();
            obj.defaultTolerance = 1e-10;
            obj.NumericalComparator = NumericalComparator();
        end
        
        function setUp(obj)
            % Test setup method that executes before each test method
            setUp@BaseTest(obj);
            
            % Generate test points spanning the range [-10, 10]
            obj.testPoints = linspace(-10, 10, 100)';
            
            % Initialize test parameters for multiple test scenarios
            obj.testParameters = struct();
            obj.testParameters.standard = struct('nu', 5, 'lambda', 0);
            obj.testParameters.skewedPositive = struct('nu', 5, 'lambda', 0.5);
            obj.testParameters.skewedNegative = struct('nu', 5, 'lambda', -0.5);
            obj.testParameters.heavyTail = struct('nu', 2.1, 'lambda', 0);
            obj.testParameters.lightTail = struct('nu', 100, 'lambda', 0);
            obj.testParameters.extremePositive = struct('nu', 5, 'lambda', 0.9);
            obj.testParameters.extremeNegative = struct('nu', 5, 'lambda', -0.9);
            
            % Set up known reference values for validation
            obj.knownValues = struct();
            obj.knownValues.symmetricMedian = struct('x', 0, 'nu', 5, 'lambda', 0, 'p', 0.5);
            obj.knownValues.quarterPoint = struct('x', -0.7265, 'nu', 5, 'lambda', 0, 'p', 0.25);
            obj.knownValues.threeQuarterPoint = struct('x', 0.7265, 'nu', 5, 'lambda', 0, 'p', 0.75);
        end
        
        function tearDown(obj)
            % Test teardown method that executes after each test method
            tearDown@BaseTest(obj);
        end
        
        function testBasicFunctionality(obj)
            % Tests basic functionality of the skewtcdf function with standard parameters
            
            % Define standard parameters (nu=5, lambda=0)
            nu = obj.testParameters.standard.nu;
            lambda = obj.testParameters.standard.lambda;
            
            % Compute CDF values for test points using skewtcdf
            p = skewtcdf(obj.testPoints, nu, lambda);
            
            % Verify CDF values are within [0,1] range
            obj.assertTrue(all(p >= 0 & p <= 1), 'CDF values must be in range [0,1]');
            
            % Verify CDF is monotonically increasing using diff function
            obj.assertTrue(all(diff(p) >= 0), 'CDF must be monotonically increasing');
            
            % Verify CDF approaches 0 at extreme negative values
            obj.assertEqualsWithTolerance(0, skewtcdf(-1000, nu, lambda), 1e-6);
            
            % Verify CDF approaches 1 at extreme positive values
            obj.assertEqualsWithTolerance(1, skewtcdf(1000, nu, lambda), 1e-6);
            
            % Verify special values (median, quartiles) match expected probabilities
            obj.assertEqualsWithTolerance(obj.knownValues.symmetricMedian.p, 
                skewtcdf(obj.knownValues.symmetricMedian.x, nu, lambda), 
                obj.defaultTolerance);
                
            obj.assertEqualsWithTolerance(obj.knownValues.quarterPoint.p, 
                skewtcdf(obj.knownValues.quarterPoint.x, nu, lambda), 
                obj.defaultTolerance);
                
            obj.assertEqualsWithTolerance(obj.knownValues.threeQuarterPoint.p, 
                skewtcdf(obj.knownValues.threeQuarterPoint.x, nu, lambda), 
                obj.defaultTolerance);
        end
        
        function testSymmetricCase(obj)
            % Tests that skewtcdf reduces to standard t-distribution when lambda=0
            
            % Define symmetric case parameters (nu=5, lambda=0)
            nu = obj.testParameters.standard.nu;
            lambda = 0;
            
            % Compute CDF values using skewtcdf
            p_skewt = skewtcdf(obj.testPoints, nu, lambda);
            
            % Compute reference values using stdtcdf
            p_stdt = stdtcdf(obj.testPoints, nu);
            
            % Compare results using NumericalComparator
            result = obj.NumericalComparator.compareMatrices(p_skewt, p_stdt, obj.defaultTolerance);
            obj.assertTrue(result.isEqual, 'Skewed t-CDF should match standard t-CDF when lambda=0');
            
            % Test with several degrees of freedom values
            nu_values = [3, 10, 30];
            for i = 1:length(nu_values)
                p_skewt = skewtcdf(obj.testPoints, nu_values(i), lambda);
                p_stdt = stdtcdf(obj.testPoints, nu_values(i));
                
                result = obj.NumericalComparator.compareMatrices(p_skewt, p_stdt, obj.defaultTolerance);
                obj.assertTrue(result.isEqual, sprintf('Symmetric case failed for nu=%g', nu_values(i)));
            end
        end
        
        function testSkewedCases(obj)
            % Tests skewed t-distribution CDF with various asymmetry parameters
            
            % Test with positive skewness (lambda=0.5)
            nu = obj.testParameters.standard.nu;
            lambda_pos = obj.testParameters.skewedPositive.lambda;
            p_pos = skewtcdf(obj.testPoints, nu, lambda_pos);
            
            % Test with negative skewness (lambda=-0.5)
            lambda_neg = obj.testParameters.skewedNegative.lambda;
            p_neg = skewtcdf(obj.testPoints, nu, lambda_neg);
            
            % Compute symmetric case for comparison
            p_sym = skewtcdf(obj.testPoints, nu, 0);
            
            % For positive lambda (right-skewed), CDF values should be lower in right tail
            right_tail_indices = obj.testPoints > 2;
            obj.assertTrue(all(p_pos(right_tail_indices) < p_sym(right_tail_indices)), 
                'With positive lambda, CDF should be lower in right tail');
            
            % For positive lambda, CDF values should be higher in left tail
            left_tail_indices = obj.testPoints < -2;
            obj.assertTrue(all(p_pos(left_tail_indices) > p_sym(left_tail_indices)), 
                'With positive lambda, CDF should be higher in left tail');
            
            % For negative lambda (left-skewed), the opposite should be true
            obj.assertTrue(all(p_neg(right_tail_indices) > p_sym(right_tail_indices)), 
                'With negative lambda, CDF should be higher in right tail');
            obj.assertTrue(all(p_neg(left_tail_indices) < p_sym(left_tail_indices)), 
                'With negative lambda, CDF should be lower in left tail');
            
            % Test with extreme positive skewness (lambda=0.9)
            lambda_extreme_pos = obj.testParameters.extremePositive.lambda;
            p_extreme_pos = skewtcdf(obj.testPoints, nu, lambda_extreme_pos);
            
            % Test with extreme negative skewness (lambda=-0.9)
            lambda_extreme_neg = obj.testParameters.extremeNegative.lambda;
            p_extreme_neg = skewtcdf(obj.testPoints, nu, lambda_extreme_neg);
            
            % Verify expected asymmetry properties are more pronounced with extreme skewness
            obj.assertTrue(all(p_extreme_pos(right_tail_indices) < p_pos(right_tail_indices)), 
                'Extreme positive skewness should have more pronounced effect in right tail');
            obj.assertTrue(all(p_extreme_neg(left_tail_indices) < p_neg(left_tail_indices)), 
                'Extreme negative skewness should have more pronounced effect in left tail');
        end
        
        function testDegreesOfFreedom(obj)
            % Tests behavior with different degrees of freedom parameters
            
            % Test with small degrees of freedom (nu=2.1) for heavy tails
            nu_heavy = obj.testParameters.heavyTail.nu;
            lambda = 0; % Using symmetric case for simplicity
            p_heavy = skewtcdf(obj.testPoints, nu_heavy, lambda);
            
            % Test with moderate degrees of freedom (nu=5)
            nu_moderate = obj.testParameters.standard.nu;
            p_moderate = skewtcdf(obj.testPoints, nu_moderate, lambda);
            
            % Test with large degrees of freedom (nu=100) approaching normal
            nu_light = obj.testParameters.lightTail.nu;
            p_light = skewtcdf(obj.testPoints, nu_light, lambda);
            
            % For positive x > 2, heavy tails should have lower CDF values
            pos_indices = obj.testPoints > 2;
            obj.assertTrue(all(p_heavy(pos_indices) < p_moderate(pos_indices)), 
                'Heavy tails should have lower CDF values in right tail');
            
            % For negative x < -2, heavy tails should have higher CDF values
            neg_indices = obj.testPoints < -2;
            obj.assertTrue(all(p_heavy(neg_indices) > p_moderate(neg_indices)), 
                'Heavy tails should have higher CDF values in left tail');
            
            % Light tails should be closer to normal distribution
            obj.assertTrue(all(p_light(pos_indices) > p_moderate(pos_indices)), 
                'Light tails should have higher CDF values in right tail');
            obj.assertTrue(all(p_light(neg_indices) < p_moderate(neg_indices)), 
                'Light tails should have lower CDF values in left tail');
        end
        
        function testVectorization(obj)
            % Tests vectorized behavior of skewtcdf function
            
            % Test with vector of x values and scalar parameters
            nu = obj.testParameters.standard.nu;
            lambda = obj.testParameters.standard.lambda;
            x_vector = linspace(-5, 5, 20)';
            
            % Compute CDF using vectorized call
            p_vector = skewtcdf(x_vector, nu, lambda);
            
            % Compute element-by-element for comparison
            p_element = zeros(size(x_vector));
            for i = 1:length(x_vector)
                p_element(i) = skewtcdf(x_vector(i), nu, lambda);
            end
            
            % Compare results
            result = obj.NumericalComparator.compareMatrices(p_vector, p_element, obj.defaultTolerance);
            obj.assertTrue(result.isEqual, 'Vectorized calculation should match element-wise calculation for x vector');
            
            % Test with scalar x and vector parameters
            x_scalar = 0;
            nu_vector = [3; 5; 10];
            lambda_vector = [0; 0.2; -0.2];
            
            % Compute CDF using vectorized call
            p_vector_params = skewtcdf(x_scalar, nu_vector, lambda_vector);
            
            % Compute element-by-element for comparison
            p_element_params = zeros(size(nu_vector));
            for i = 1:length(nu_vector)
                p_element_params(i) = skewtcdf(x_scalar, nu_vector(i), lambda_vector(i));
            end
            
            % Compare results
            result = obj.NumericalComparator.compareMatrices(p_vector_params, p_element_params, obj.defaultTolerance);
            obj.assertTrue(result.isEqual, 'Vectorized calculation should match element-wise calculation for parameter vectors');
            
            % Test with compatible vector inputs for all arguments
            x_small = [0; 1; 2];
            nu_small = [3; 5; 10];
            lambda_small = [0; 0.2; -0.2];
            
            % Compute CDF using vectorized call
            p_compatible = skewtcdf(x_small, nu_small, lambda_small);
            
            % Compute element-by-element for comparison
            p_element_compatible = zeros(size(x_small));
            for i = 1:length(x_small)
                p_element_compatible(i) = skewtcdf(x_small(i), nu_small(i), lambda_small(i));
            end
            
            % Compare results
            result = obj.NumericalComparator.compareMatrices(p_compatible, p_element_compatible, obj.defaultTolerance);
            obj.assertTrue(result.isEqual, 'Vectorized calculation should match element-wise calculation for compatible inputs');
        end
        
        function testErrorHandling(obj)
            % Tests error handling for invalid inputs
            
            % Test with nu ≤ 2 (invalid degrees of freedom)
            obj.assertThrows(@() skewtcdf(0, 1.5, 0), 'parametercheck:lowerBound');
            
            % Test with |lambda| ≥ 1 (invalid skewness)
            obj.assertThrows(@() skewtcdf(0, 5, 1.2), 'parametercheck:upperBound');
            obj.assertThrows(@() skewtcdf(0, 5, -1.2), 'parametercheck:lowerBound');
            
            % Test with incompatible size inputs
            x_vector = [0; 1; 2];
            nu_matrix = ones(2, 2);
            lambda_scalar = 0;
            
            obj.assertThrows(@() skewtcdf(x_vector, nu_matrix, lambda_scalar), 'MATLAB:sizeDimensionsMustMatch');
            
            % Test with NaN and Inf inputs
            obj.assertThrows(@() skewtcdf(NaN, 5, 0), 'datacheck:.*');
            obj.assertThrows(@() skewtcdf(Inf, 5, 0), 'datacheck:.*');
            obj.assertThrows(@() skewtcdf(0, NaN, 0), 'parametercheck:.*');
            obj.assertThrows(@() skewtcdf(0, 5, NaN), 'parametercheck:.*');
        end
        
        function testNumericalStability(obj)
            % Tests numerical stability for extreme parameter values
            
            % Test with nu very close to 2 (e.g., 2.0001)
            nu_edge = 2.0001;
            lambda = 0;
            x_test = [-2; -1; 0; 1; 2];
            
            p_edge_nu = skewtcdf(x_test, nu_edge, lambda);
            
            % Results should be finite and within [0,1] range
            obj.assertTrue(all(isfinite(p_edge_nu)), 'CDF should return finite values for nu close to 2');
            obj.assertTrue(all(p_edge_nu >= 0 & p_edge_nu <= 1), 'CDF values must be in range [0,1] for nu close to 2');
            
            % Test with lambda very close to ±1 (e.g., ±0.9999)
            lambda_pos_edge = 0.9999;
            lambda_neg_edge = -0.9999;
            nu = 5;
            
            p_edge_lambda_pos = skewtcdf(x_test, nu, lambda_pos_edge);
            p_edge_lambda_neg = skewtcdf(x_test, nu, lambda_neg_edge);
            
            % Results should be finite and within [0,1] range
            obj.assertTrue(all(isfinite(p_edge_lambda_pos)), 'CDF should return finite values for lambda close to 1');
            obj.assertTrue(all(p_edge_lambda_pos >= 0 & p_edge_lambda_pos <= 1), 'CDF values must be in range [0,1] for lambda close to 1');
            
            obj.assertTrue(all(isfinite(p_edge_lambda_neg)), 'CDF should return finite values for lambda close to -1');
            obj.assertTrue(all(p_edge_lambda_neg >= 0 & p_edge_lambda_neg <= 1), 'CDF values must be in range [0,1] for lambda close to -1');
            
            % Test with extreme x values
            x_extreme = [-1e6; 1e6];
            p_extreme = skewtcdf(x_extreme, nu, lambda);
            
            % CDF should approach 0 for extreme negative values and 1 for extreme positive values
            obj.assertEqualsWithTolerance(0, p_extreme(1), 1e-10);
            obj.assertEqualsWithTolerance(1, p_extreme(2), 1e-10);
        end
        
        function testEmpirical(obj)
            % Tests CDF values against empirical distribution from random samples
            
            % Generate large sample from skewed t-distribution using skewtrnd
            nu = 5;
            lambda = 0.3;
            n_samples = 100000;
            
            rng(123456);  % Set seed for reproducibility
            samples = skewtrnd(nu, lambda, n_samples, 1);
            
            % Compute empirical CDF from the sample
            sorted_samples = sort(samples);
            empirical_cdf = (1:n_samples)' / n_samples;
            
            % Compute theoretical CDF using skewtcdf at same points
            theoretical_cdf = skewtcdf(sorted_samples, nu, lambda);
            
            % Compare empirical and theoretical CDFs at key quantiles
            quantile_levels = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
            quantile_indices = round(quantile_levels * n_samples);
            
            for i = 1:length(quantile_levels)
                idx = quantile_indices(i);
                empirical_prob = empirical_cdf(idx);
                theoretical_prob = theoretical_cdf(idx);
                
                % Use a looser tolerance due to sampling variability
                obj.assertEqualsWithTolerance(empirical_prob, theoretical_prob, 0.01, 
                    sprintf('Empirical and theoretical CDFs should match at quantile %g', quantile_levels(i)));
            end
        end
        
        function testContinuityProperty(obj)
            % Tests continuity properties of the CDF implementation
            
            % Parameters
            nu = 5;
            lambda = 0.5;
            
            % Calculate threshold -a/b in the implementation
            c = gamma((nu+1)/2) / (gamma(nu/2) * sqrt(pi*(nu-2)));
            a = 4 * lambda * c * (nu-2) / nu;
            b = sqrt(1 + 3*lambda^2 - a^2);
            threshold = -a/b;
            
            % Test CDF values at closely spaced points around the threshold
            delta = 1e-6;
            x_around_threshold = threshold + (-10:10)' * delta;
            p_around_threshold = skewtcdf(x_around_threshold, nu, lambda);
            
            % Verify smooth transition and continuity at threshold boundary
            diffs = diff(p_around_threshold);
            obj.assertTrue(all(isfinite(diffs)), 'CDF should be finite around threshold');
            obj.assertTrue(all(diffs >= 0), 'CDF should be monotonically increasing around threshold');
            
            % Test continuity across the entire support using fine grid
            x_fine = linspace(-10, 10, 1000)';
            p_fine = skewtcdf(x_fine, nu, lambda);
            diffs_fine = diff(p_fine);
            
            obj.assertTrue(all(isfinite(p_fine)), 'CDF should be finite across entire support');
            obj.assertTrue(all(diffs_fine >= 0), 'CDF should be monotonically increasing across entire support');
        end
        
        function result = verifyMonotonicity(obj, x, nu, lambda)
            % Helper method to verify CDF monotonicity property
            
            % Sort x input
            x_sorted = sort(x);
            
            % Compute CDF values for sorted x input
            p = skewtcdf(x_sorted, nu, lambda);
            
            % Calculate differences between consecutive CDF values
            diffs = diff(p);
            
            % Verify all differences are non-negative (monotonically increasing)
            result = all(diffs >= 0);
        end
        
        function result = compareWithNumericalIntegration(obj, x, nu, lambda)
            % Compares CDF implementation with numerical integration of PDF
            
            % Compute CDF values using skewtcdf
            p_cdf = skewtcdf(x, nu, lambda);
            
            % Compute reference CDF by numerically integrating skewtpdf
            p_ref = zeros(size(x));
            for i = 1:length(x)
                integration_points = linspace(-100, x(i), 10000);
                integration_step = integration_points(2) - integration_points(1);
                pdf_values = skewtpdf(integration_points, nu, lambda);
                p_ref(i) = sum(pdf_values) * integration_step;
            end
            
            % Compare results using NumericalComparator
            comparison = obj.NumericalComparator.compareMatrices(p_cdf, p_ref, 1e-4);
            
            % Calculate maximum deviation between methods
            result = struct(...
                'isEqual', comparison.isEqual, ...
                'maxDeviation', comparison.maxAbsoluteDifference, ...
                'cdfValues', p_cdf, ...
                'referenceValues', p_ref ...
            );
        end
    end
end