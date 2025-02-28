classdef StdtinvTest < BaseTest
    % STDTINVTEST Test class for the stdtinv function
    %
    % This class provides comprehensive unit tests for the stdtinv function,
    % which computes the inverse CDF (quantile function) of the standardized
    % Student's t-distribution.
    %
    % The tests validate the function's behavior with various parameter
    % combinations, ensure numerical stability for edge cases, and verify
    % the inverse relationship with stdtcdf.
    %
    % See also: BaseTest, NumericalComparator, stdtinv, stdtcdf
    
    properties
        % Numerical comparator for floating-point comparisons
        comparator
        
        % Default tolerance for numerical comparisons
        defaultTolerance
        
        % Test data structure
        testData
        
        % Test probability values
        testProbabilities
        
        % Degrees of freedom values for testing
        nuValues
        
        % Expected results from reference calculations
        expectedResults
    end
    
    methods
        function obj = StdtinvTest()
            % Initialize a new StdtinvTest instance
            
            % Call the superclass constructor with test name
            obj@BaseTest('StdtinvTest');
            
            % Initialize test data structure
            obj.testData = struct();
            
            % Create numerical comparator for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for high-precision numeric comparisons
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Set up test environment before each test method runs
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Initialize array of test probability values
            obj.testProbabilities = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99];
            
            % Initialize array of test degrees of freedom values
            obj.nuValues = [3, 4, 5, 6, 8, 10, 20, 30];
            
            % Load reference data if available
            try
                refData = obj.loadTestData('known_distributions.mat');
                obj.testData.referenceValues = refData.stdt_inv_values;
            catch
                % If reference data not available, continue without it
                warning('Reference data for stdtinv not available. Some tests may be skipped.');
            end
            
            % Configure numerical comparator with appropriate tolerance
            obj.comparator.setDefaultTolerances(obj.defaultTolerance, obj.defaultTolerance);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method completes
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary test data to free memory
            obj.testData = struct();
        end
        
        function testBasicInv(obj)
            % Tests stdtinv with basic parameter values
            
            % Test median (probability 0.5)
            p = 0.5;
            nu = 5;
            expectedMedian = 0; % t-distribution is symmetric around 0
            actualMedian = stdtinv(p, nu);
            
            % Verify result equals expected value
            obj.assertEqual(expectedMedian, actualMedian, 'Median of t-distribution should be 0');
            
            % Verify result is a double
            obj.assertTrue(isa(actualMedian, 'double'), 'Result should be a double');
            
            % Test common quantiles
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];
            result = stdtinv(quantiles, 5);
            
            % Verify dimensions
            obj.assertEqual(size(quantiles), size(result), 'Output dimensions should match input');
            
            % Test symmetry property
            obj.assertAlmostEqual(-stdtinv(0.25, 5), stdtinv(0.75, 5), 't-distribution should be symmetric');
        end
        
        function testVectorInput(obj)
            % Tests stdtinv with vectorized inputs
            
            % Test vector of probabilities
            probs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
            nu = 5;
            
            % Calculate inverse CDF values
            result = stdtinv(probs, nu);
            
            % Verify dimensions
            obj.assertEqual(size(probs), size(result), 'Output dimensions should match input');
            
            % Verify each value individually for clarity in case of failure
            obj.assertAlmostEqual(stdtinv(probs(1), nu), result(1), 'Vector element 1 mismatch');
            obj.assertAlmostEqual(stdtinv(probs(4), nu), result(4), 'Vector element 4 mismatch');
            obj.assertAlmostEqual(stdtinv(probs(7), nu), result(7), 'Vector element 7 mismatch');
            
            % Test with multiple nu values
            nuValues = [5, 10, 30];
            for i = 1:length(nuValues)
                currentNu = nuValues(i);
                for j = 1:length(probs)
                    singleResult = stdtinv(probs(j), currentNu);
                    vectorResult = stdtinv(probs, currentNu);
                    obj.assertAlmostEqual(singleResult, vectorResult(j), ...
                        sprintf('Vector result mismatch with nu=%d, p=%g', currentNu, probs(j)));
                end
            end
        end
        
        function testNormalApproximation(obj)
            % Tests that stdtinv approaches normal inverse CDF as degrees of freedom increase
            
            % Set up test probabilities
            probs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
            
            % Large degrees of freedom should approximate normal distribution
            highNu = 100;
            
            % Get t-distribution quantiles
            tQuantiles = stdtinv(probs, highNu);
            
            % Get normal distribution quantiles for comparison
            normQuantiles = norminv(probs);
            
            % For high degrees of freedom, should be very close to normal
            for i = 1:length(probs)
                diff = abs(tQuantiles(i) - normQuantiles(i));
                % Tolerance increases with smaller p (tail regions)
                tolerance = probs(i) < 0.05 || probs(i) > 0.95 ? 1e-2 : 1e-3;
                obj.assertTrue(diff < tolerance, ...
                    sprintf('t-distribution with high df should approach normal at p=%g', probs(i)));
            end
            
            % Test convergence rate - should be roughly proportional to 1/nu
            nu1 = 30;
            nu2 = 100;
            
            tq1 = stdtinv(0.975, nu1);
            tq2 = stdtinv(0.975, nu2);
            normq = norminv(0.975);
            
            diff1 = abs(tq1 - normq);
            diff2 = abs(tq2 - normq);
            
            % Convergence ratio should be approximately nu1/nu2
            expectedRatio = nu2/nu1;
            actualRatio = diff1/diff2;
            
            % Allow some flexibility in the ratio comparison
            obj.assertTrue(abs(actualRatio/expectedRatio - 1) < 0.5, ...
                'Convergence to normal should be approximately proportional to 1/nu');
        end
        
        function testTailBehavior(obj)
            % Tests stdtinv tail behavior with different degrees of freedom
            
            % Test with different degrees of freedom
            nuValues = [3, 10, 30];
            
            % Small probability (left tail)
            p = 0.001;
            
            % For heavier tails (smaller nu), absolute value should be larger
            leftTails = zeros(size(nuValues));
            for i = 1:length(nuValues)
                leftTails(i) = stdtinv(p, nuValues(i));
            end
            
            % Check that values are in descending order (ascending when taking absolute value)
            obj.assertTrue(all(diff(leftTails) > 0), 'Left tail values should decrease with higher nu');
            
            % High probability (right tail)
            p = 0.999;
            
            % For heavier tails (smaller nu), value should be larger
            rightTails = zeros(size(nuValues));
            for i = 1:length(nuValues)
                rightTails(i) = stdtinv(p, nuValues(i));
            end
            
            % Check that values are in ascending order
            obj.assertTrue(all(diff(rightTails) < 0), 'Right tail values should decrease with higher nu');
        end
        
        function testParameterValidation(obj)
            % Tests stdtinv error handling for invalid parameters
            
            % Test invalid degrees of freedom (nu must be > 2 for standardized t)
            obj.assertThrows(@() stdtinv(0.5, 0), 'parametercheck:lowerBound', 'Should reject nu = 0');
            obj.assertThrows(@() stdtinv(0.5, 1), 'parametercheck:lowerBound', 'Should reject nu = 1');
            obj.assertThrows(@() stdtinv(0.5, 1.5), 'parametercheck:lowerBound', 'Should reject nu < 2');
            obj.assertThrows(@() stdtinv(0.5, NaN), 'datacheck:isnan', 'Should reject NaN nu');
            obj.assertThrows(@() stdtinv(0.5, Inf), 'datacheck:isfinite', 'Should reject Inf nu');
            
            % Test invalid probabilities
            obj.assertThrows(@() stdtinv(-0.1, 5), 'parametercheck:lowerBound', 'Should reject p < 0');
            obj.assertThrows(@() stdtinv(1.1, 5), 'parametercheck:upperBound', 'Should reject p > 1');
            obj.assertThrows(@() stdtinv(NaN, 5), 'datacheck:isnan', 'Should reject NaN p');
            
            % Test invalid input dimensions (incompatible vector sizes)
            obj.assertThrows(@() stdtinv([0.5; 0.6], [5, 6]), 'parametercheck:isscalar', ...
                'Should reject non-scalar nu with vector p');
        end
        
        function testBoundaryValues(obj)
            % Tests stdtinv with boundary probability values
            
            % Test p = 0
            result = stdtinv(0, 5);
            obj.assertTrue(isinf(result) && result < 0, 'p = 0 should return -Inf');
            
            % Test p = 1
            result = stdtinv(1, 5);
            obj.assertTrue(isinf(result) && result > 0, 'p = 1 should return Inf');
            
            % Test values very close to 0 and 1
            p_small = 1e-10;
            p_large = 1 - 1e-10;
            
            % These should return very large negative and positive values, respectively
            result_small = stdtinv(p_small, 5);
            result_large = stdtinv(p_large, 5);
            
            obj.assertTrue(result_small < -100, 'Very small p should return large negative value');
            obj.assertTrue(result_large > 100, 'Very large p should return large positive value');
            
            % Test symmetry at extreme values
            obj.assertAlmostEqual(result_small, -result_large, 'Extreme values should be symmetric');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of stdtinv for extreme values
            
            % Test with nu just above the lower bound
            nu_small = 2.01;
            
            % Get quantiles at standard test points
            quantiles = stdtinv([0.025, 0.975], nu_small);
            
            % These should be finite and have large absolute values
            obj.assertTrue(all(isfinite(quantiles)), 'Quantiles with small nu should be finite');
            obj.assertTrue(all(abs(quantiles) > 10), 'Quantiles with small nu should be large');
            
            % Test with very large nu
            nu_large = 1e6;
            
            % Should be very close to normal quantiles
            t_quantiles = stdtinv([0.025, 0.975], nu_large);
            norm_quantiles = norminv([0.025, 0.975]);
            
            % Compare with very tight tolerance
            obj.assertMatrixEqualsWithTolerance(norm_quantiles, t_quantiles, 1e-4, ...
                'Very large nu should give results extremely close to normal');
            
            % Test with nu=3 (close to boundary of existence of moments) at extreme probabilities
            p_extreme = [0.001, 0.999];
            quantiles = stdtinv(p_extreme, 3);
            
            % These should be finite but large
            obj.assertTrue(all(isfinite(quantiles)), 'Extreme quantiles with nu=3 should be finite');
            obj.assertTrue(all(abs(quantiles) > 10), 'Extreme quantiles with nu=3 should be large');
        end
        
        function testInverseFunction(obj)
            % Tests that stdtinv is truly the inverse function of stdtcdf
            
            % Create array of x values for testing
            xValues = linspace(-5, 5, 20);
            
            % Test with different degrees of freedom
            for nu = [3, 5, 10, 30]
                for i = 1:length(xValues)
                    x = xValues(i);
                    
                    % Calculate probability with stdtcdf
                    p = stdtcdf(x, nu);
                    
                    % Calculate inverse with stdtinv
                    x_inv = stdtinv(p, nu);
                    
                    % They should be equal (within tolerance)
                    obj.assertAlmostEqual(x, x_inv, ...
                        sprintf('stdtinv should be inverse of stdtcdf at x=%g, nu=%d', x, nu));
                end
            end
        end
        
        function testAgainstReferenceData(obj)
            % Tests stdtinv against pre-computed reference values from known_distributions.mat
            
            % Skip test if reference data is not available
            if ~isfield(obj.testData, 'referenceValues')
                warning('Skipping reference data test: data not available');
                return;
            end
            
            % Test against reference values
            referenceValues = obj.testData.referenceValues;
            
            % Calculate stdtinv values for each combination of probability and nu
            for i = 1:length(obj.testProbabilities)
                for j = 1:length(obj.nuValues)
                    p = obj.testProbabilities(i);
                    nu = obj.nuValues(j);
                    
                    % Get reference value
                    expectedValue = referenceValues(i, j);
                    
                    % Calculate value using stdtinv
                    actualValue = stdtinv(p, nu);
                    
                    % Compare with appropriate tolerance
                    obj.assertAlmostEqual(expectedValue, actualValue, ...
                        sprintf('Reference test failed for p=%g, nu=%d', p, nu));
                end
            end
        end
        
        function x = calculateManualStdtInv(obj, p, nu)
            % Helper method to manually calculate standardized Student's t-distribution inverse CDF
            % for validation against the implementation
            
            % For p = 0.5, return 0 directly (from symmetry)
            if p == 0.5
                x = 0;
                return;
            end
            
            % For p = 0, return -Inf
            if p == 0
                x = -Inf;
                return;
            end
            
            % For p = 1, return Inf
            if p == 1
                x = Inf;
                return;
            end
            
            % Calculate using relationship with inverse incomplete beta function
            if p < 0.5
                % For left tail
                sign_factor = -1;
                p_adj = 2 * p;
            else
                % For right tail
                sign_factor = 1;
                p_adj = 2 * (1 - p);
            end
            
            % Calculate using the beta function relationship
            beta_inv = betaincinv(p_adj, 0.5, nu/2);
            
            % Convert to regular t quantile
            t_quantile = sign_factor * sqrt(nu * (1/beta_inv - 1));
            
            % Convert to standardized t quantile
            x = t_quantile * sqrt((nu-2)/nu);
        end
    end
end