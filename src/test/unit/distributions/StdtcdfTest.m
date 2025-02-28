classdef StdtcdfTest < BaseTest
    % STDTCDFTEST Test class for the stdtcdf function
    %
    % This class provides comprehensive test coverage for the standardized
    % Student's t-distribution CDF calculations. It tests various parameter
    % combinations, edge cases, and ensures numerical stability and accuracy.
    
    properties
        comparator          % NumericalComparator instance
        defaultTolerance    % Default tolerance for numerical comparisons
        testData            % Structure for test data
        testValues          % Array of test x values
        nuValues            % Array of test nu values
        expectedResults     % Array of expected results
    end
    
    methods
        function obj = StdtcdfTest()
            % Initialize a new StdtcdfTest instance
            
            % Call superclass constructor
            obj = obj@BaseTest('StdtcdfTest');
            
            % Initialize test data structure
            obj.testData = struct();
            
            % Create numerical comparator for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set default tolerance for high-precision numeric comparisons
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Prepare the test environment before each test
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Initialize array of test x values
            obj.testValues = [-3, -2, -1, 0, 1, 2, 3];
            
            % Initialize array of test nu values
            obj.nuValues = [3, 4, 5, 6, 8, 10, 20, 30];
            
            % Load reference data from known_distributions.mat
            try
                data = obj.loadTestData('known_distributions.mat');
                obj.testData.stdt_cdf_values = data.stdt_cdf_values;
                obj.testData.stdt_parameters = data.stdt_parameters;
            catch ME
                % If file not found, we'll compute expected results in the test methods
                if obj.verbose
                    warning('Reference data file not found. Tests will use computed values.');
                end
            end
            
            % Configure numerical comparator with appropriate tolerance
            obj.comparator.setDefaultTolerances(obj.defaultTolerance, obj.defaultTolerance/10);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary test data to free memory
            obj.testData = struct();
        end
        
        function testBasicCdf(obj)
            % Tests stdtcdf with basic parameter values
            
            % Test CDF at x=0, which should be 0.5 by symmetry
            result = stdtcdf(0, 5);
            obj.assertEqual(0.5, result, 'CDF at x=0 should be 0.5');
            
            % Verify result is a double with correct dimensions
            obj.assertTrue(isscalar(result), 'Result should be a scalar for scalar input');
            obj.assertTrue(isa(result, 'double'), 'Result should be a double');
            
            % Verify CDF value is in the range [0,1]
            obj.assertTrue(result >= 0 && result <= 1, 'CDF values must be in [0,1] range');
            
            % Test a simple case with known value
            result = stdtcdf(1, 5);
            expected = 0.81742; % Approximate value for t(5) at x=1
            obj.assertAlmostEqual(expected, result, 'CDF value at x=1, nu=5 is incorrect');
        end
        
        function testVectorInput(obj)
            % Tests stdtcdf with vectorized inputs
            
            % Test with vector of x values
            x = [-3, -2, -1, 0, 1, 2, 3];
            nu = 5;
            results = stdtcdf(x, nu);
            
            % Verify output dimensions match input
            obj.assertEqual(size(x), size(results), 'Output size should match input size');
            
            % Expected results (approximate values for standardized t-distribution with nu=5)
            expected = [0.0152, 0.0506, 0.1853, 0.5, 0.8147, 0.9494, 0.9848];
            
            % Compare with expected values using appropriate tolerance
            obj.assertMatrixEqualsWithTolerance(expected, results, 1e-4, ...
                'CDF values do not match expected values');
            
            % Test with multiple nu values
            nu = 10;
            results2 = stdtcdf(x, nu);
            
            % Expected results for nu=10 (higher degrees of freedom)
            expected2 = [0.0065, 0.0372, 0.1711, 0.5, 0.8289, 0.9628, 0.9935];
            
            % Compare with expected values
            obj.assertMatrixEqualsWithTolerance(expected2, results2, 1e-4, ...
                'CDF values do not match expected values for nu=10');
            
            % Test with matrix input
            x_matrix = reshape(x, [1, 7]);
            results_matrix = stdtcdf(x_matrix, nu);
            obj.assertEqual(size(x_matrix), size(results_matrix), 'Output size should match input size for matrix');
            obj.assertMatrixEqualsWithTolerance(expected2, results_matrix, 1e-4, ...
                'CDF values do not match expected values for matrix input');
        end
        
        function testNormalApproximation(obj)
            % Tests that stdtcdf approaches normal CDF as degrees of freedom increase
            
            % Create array of test points
            x = -3:0.5:3;
            
            % Calculate stdtcdf with large degrees of freedom
            largeNu = 100;
            tResults = stdtcdf(x, largeNu);
            
            % Calculate normal CDF values
            normalResults = normcdf(x);
            
            % Verify t-distribution approaches normal distribution
            obj.assertMatrixEqualsWithTolerance(normalResults, tResults, 1e-3, ...
                'Large-nu t-distribution should approximate normal distribution');
            
            % Check convergence rate (error should be proportional to 1/nu)
            largerNu = 1000;
            tResults2 = stdtcdf(x, largerNu);
            
            % Errors should decrease as nu increases
            err1 = max(abs(tResults - normalResults));
            err2 = max(abs(tResults2 - normalResults));
            
            % Error for nu=1000 should be smaller than for nu=100
            obj.assertTrue(err2 < err1, 'Convergence rate does not match expectations');
            
            % Very large nu should be almost identical to normal
            veryLargeNu = 10000;
            tResults3 = stdtcdf(x, veryLargeNu);
            obj.assertMatrixEqualsWithTolerance(normalResults, tResults3, 1e-4, ...
                'Very large-nu t-distribution should be practically indistinguishable from normal');
        end
        
        function testTailBehavior(obj)
            % Tests stdtcdf tail behavior with different degrees of freedom
            
            % Test for small degrees of freedom (heavier tails)
            x = 5; % Far in the tail
            nu_small = 3;
            p_small = stdtcdf(x, nu_small);
            
            % Test for moderate degrees of freedom
            nu_moderate = 10;
            p_moderate = stdtcdf(x, nu_moderate);
            
            % Test for large degrees of freedom (lighter tails)
            nu_large = 30;
            p_large = stdtcdf(x, nu_large);
            
            % For the t-distribution, smaller nu means heavier tails
            % So CDF should approach 1 more slowly with smaller nu
            obj.assertTrue(p_small < p_moderate, 'Smaller nu should have heavier tails');
            obj.assertTrue(p_moderate < p_large, 'Larger nu should have lighter tails');
            
            % Test extreme tail behavior
            x_extreme = 10;
            p_extreme_small = stdtcdf(x_extreme, nu_small);
            p_extreme_large = stdtcdf(x_extreme, nu_large);
            
            % Still under 1 even for extreme values
            obj.assertTrue(p_extreme_small < 1, 'Even extreme values should not reach exactly 1');
            obj.assertTrue(p_extreme_large < 1, 'Even extreme values should not reach exactly 1');
            
            % Test negative extreme tail behavior
            x_neg_extreme = -10;
            p_neg_extreme_small = stdtcdf(x_neg_extreme, nu_small);
            p_neg_extreme_large = stdtcdf(x_neg_extreme, nu_large);
            
            % Still above 0 even for extreme negative values
            obj.assertTrue(p_neg_extreme_small > 0, 'Even extreme negative values should not reach exactly 0');
            obj.assertTrue(p_neg_extreme_large > 0, 'Even extreme negative values should not reach exactly 0');
            
            % Verify symmetry in tails
            obj.assertAlmostEqual(1 - p_extreme_small, p_neg_extreme_small, 'Symmetry property violated in tails');
            obj.assertAlmostEqual(1 - p_extreme_large, p_neg_extreme_large, 'Symmetry property violated in tails');
        end
        
        function testParameterValidation(obj)
            % Tests stdtcdf error handling for invalid parameters
            
            % Test with invalid degrees of freedom (nu ≤ 2)
            obj.assertThrows(@() stdtcdf(0, 2), 'parametercheck:lowerBound', ...
                'Should throw error for nu = 2');
            obj.assertThrows(@() stdtcdf(0, 1.5), 'parametercheck:lowerBound', ...
                'Should throw error for nu < 2');
            obj.assertThrows(@() stdtcdf(0, -1), 'parametercheck:lowerBound', ...
                'Should throw error for negative nu');
            
            % Test with NaN values
            obj.assertThrows(@() stdtcdf(NaN, 5), 'datacheck:NaN', ...
                'Should throw error for NaN in x');
            obj.assertThrows(@() stdtcdf(0, NaN), 'parametercheck:NaN', ...
                'Should throw error for NaN in nu');
            
            % Test with Inf values
            obj.assertThrows(@() stdtcdf(Inf, 5), 'datacheck:Inf', ...
                'Should throw error for Inf in x');
            obj.assertThrows(@() stdtcdf(0, Inf), 'parametercheck:Inf', ...
                'Should throw error for Inf in nu');
            
            % Test with non-numeric values
            obj.assertThrows(@() stdtcdf('string', 5), 'datacheck:numeric', ...
                'Should throw error for non-numeric x');
            obj.assertThrows(@() stdtcdf(0, 'string'), 'parametercheck:numeric', ...
                'Should throw error for non-numeric nu');
            
            % Test with empty values
            obj.assertThrows(@() stdtcdf([], 5), 'datacheck:empty', ...
                'Should throw error for empty x');
            obj.assertThrows(@() stdtcdf(0, []), 'parametercheck:empty', ...
                'Should throw error for empty nu');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of stdtcdf for extreme values
            
            % Test with degrees of freedom very close to 2 (boundary case)
            nu_boundary = 2.0001;
            result_boundary = stdtcdf(1, nu_boundary);
            obj.assertTrue(isfinite(result_boundary), 'Result should be finite for nu close to 2');
            obj.assertTrue(result_boundary > 0.5 && result_boundary < 1, 'Result should be in valid range');
            
            % Test with very large degrees of freedom
            nu_large = 1e6;
            result_large_nu = stdtcdf(1, nu_large);
            % Should be very close to normal CDF
            normal_result = normcdf(1);
            obj.assertAlmostEqual(normal_result, result_large_nu, ...
                'Large nu should approach normal distribution');
            
            % Test with extreme x values (far from mean)
            x_extreme = 20;
            result_extreme_x = stdtcdf(x_extreme, 5);
            % Should be very close to 1 but not exactly 1
            obj.assertTrue(result_extreme_x < 1, 'Extreme x should be less than 1');
            obj.assertTrue(result_extreme_x > 0.9999, 'Extreme x should be very close to 1');
            
            % Test with extremely large negative value
            x_extreme_neg = -50;
            result_extreme_neg = stdtcdf(x_extreme_neg, 5);
            % Should be very close to 0 but not exactly 0
            obj.assertTrue(result_extreme_neg > 0, 'Extreme negative x should be greater than 0');
            obj.assertTrue(result_extreme_neg < 0.0001, 'Extreme negative x should be very close to 0');
            
            % Test case where the standardization factor approaches 1
            nu_very_large = 1e10;
            standardization_factor = sqrt((nu_very_large-2)/nu_very_large);
            obj.assertAlmostEqual(1, standardization_factor, 'Standardization factor should approach 1 for large nu');
        end
        
        function testCdfProperties(obj)
            % Tests that the CDF satisfies mathematical properties of a valid CDF
            
            % Generate a sequence of points to check monotonicity
            x = -5:0.5:5;
            nu = 5;
            
            % Calculate CDF values
            cdf_values = stdtcdf(x, nu);
            
            % Verify CDF is monotonically increasing
            diffs = diff(cdf_values);
            obj.assertTrue(all(diffs >= 0), 'CDF must be monotonically increasing');
            
            % Verify CDF values are within [0,1] range
            obj.assertTrue(all(cdf_values >= 0 & cdf_values <= 1), ...
                'CDF values must be within [0,1] range');
            
            % Verify symmetry property: F(-x) = 1 - F(x)
            for i = 1:length(x)
                if x(i) ~= 0 % Skip x=0 case
                    left = stdtcdf(-x(i), nu);
                    right = 1 - stdtcdf(x(i), nu);
                    obj.assertAlmostEqual(left, right, ...
                        sprintf('Symmetry property violated at x = %g', x(i)));
                end
            end
            
            % Verify F(0) = 0.5 for all nu values (symmetry)
            for nu_val = obj.nuValues
                obj.assertEqual(0.5, stdtcdf(0, nu_val), ...
                    sprintf('CDF at x=0 should be 0.5 for all nu (tested nu = %g)', nu_val));
            end
            
            % Verify CDF approaches 0 as x -> -∞ and 1 as x -> +∞
            x_large_neg = -1000;
            x_large_pos = 1000;
            p_neg = stdtcdf(x_large_neg, nu);
            p_pos = stdtcdf(x_large_pos, nu);
            
            obj.assertTrue(p_neg < 1e-10, 'CDF should approach 0 as x -> -∞');
            obj.assertTrue(p_pos > 1-1e-10, 'CDF should approach 1 as x -> +∞');
        end
        
        function testAgainstReferenceData(obj)
            % Tests stdtcdf against pre-computed reference values
            
            try
                % Load reference data or use test values if not available
                if ~isfield(obj.testData, 'stdt_cdf_values')
                    % If no reference data, compute values using manual calculation
                    x_values = [-3, -2, -1, 0, 1, 2, 3];
                    nu_values = [3, 5, 10, 30];
                    
                    cdf_values = zeros(length(x_values), length(nu_values));
                    for i = 1:length(nu_values)
                        for j = 1:length(x_values)
                            % Calculate reference CDF manually using formula
                            cdf_values(j, i) = obj.calculateManualStdtCdf(x_values(j), nu_values(i));
                        end
                    end
                    
                    % Store these for comparison
                    obj.testData.stdt_cdf_values = cdf_values;
                    obj.testData.stdt_parameters = struct('x', x_values, 'nu', nu_values);
                end
                
                % Extract reference data
                cdf_values = obj.testData.stdt_cdf_values;
                if isfield(obj.testData, 'stdt_parameters')
                    x_values = obj.testData.stdt_parameters.x;
                    nu_values = obj.testData.stdt_parameters.nu;
                else
                    % Fallback if not structured as expected
                    x_values = [-3, -2, -1, 0, 1, 2, 3];
                    nu_values = [3, 5, 10, 30];
                end
                
                % Calculate results for each parameter combination
                for i = 1:length(nu_values)
                    nu = nu_values(i);
                    computed = stdtcdf(x_values, nu);
                    expected = cdf_values(:, i);
                    
                    obj.assertMatrixEqualsWithTolerance(expected, computed, obj.defaultTolerance, ...
                        sprintf('CDF mismatch for nu=%g', nu));
                end
                
            catch ME
                if strcmp(ME.identifier, 'BaseTest:FileNotFound')
                    % Skip with warning if no reference data
                    warning('Reference data file not found. Using calculated values for comparison.');
                else
                    % Re-throw other errors
                    rethrow(ME);
                end
            end
        end
        
        function testInverseRelationship(obj)
            % Tests the inverse relationship between stdtcdf and stdtinv
            
            try
                % Try to load stdtinv function dynamically
                if exist('stdtinv', 'file') == 2
                    % Generate array of probability values
                    p_values = 0.05:0.1:0.95;
                    nu = 5;
                    
                    % For each probability, check that stdtcdf(stdtinv(p)) ≈ p
                    for p = p_values
                        % Calculate quantile for probability p
                        x = stdtinv(p, nu);
                        
                        % Calculate CDF at x
                        p_computed = stdtcdf(x, nu);
                        
                        % Verify inverse relationship
                        obj.assertAlmostEqual(p, p_computed, ...
                            sprintf('Inverse relationship failed at p=%g', p));
                    end
                    
                    % Test across multiple degrees of freedom
                    for nu_val = obj.nuValues
                        p = 0.75; % Test at 75th percentile
                        x = stdtinv(p, nu_val);
                        p_computed = stdtcdf(x, nu_val);
                        
                        obj.assertAlmostEqual(p, p_computed, ...
                            sprintf('Inverse relationship failed at nu=%g', nu_val));
                    end
                else
                    % Skip test if stdtinv is not available
                    warning('stdtinv function not available. Skipping inverse relationship test.');
                end
            catch ME
                if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
                    % Skip test if stdtinv is not available
                    warning('stdtinv function not available. Skipping inverse relationship test.');
                else
                    % Re-throw other errors
                    rethrow(ME);
                end
            end
        end
        
        function p = calculateManualStdtCdf(obj, x, nu)
            % Helper method to manually calculate standardized Student's t-distribution CDF
            %
            % INPUTS:
            %   x  - Point(s) at which to evaluate the CDF
            %   nu - Degrees of freedom
            %
            % OUTPUTS:
            %   p  - Manually calculated CDF value(s)
            
            % Adjust x for standardized t-distribution
            x_adjusted = x / sqrt((nu-2)/nu);
            
            % Initialize output
            p = zeros(size(x_adjusted));
            
            % Handle special cases
            p(x_adjusted == 0) = 0.5;
            p(x_adjusted > 1e10) = 1;
            p(x_adjusted < -1e10) = 0;
            
            % Process remaining values
            regular_vals = (x_adjusted ~= 0) & (x_adjusted <= 1e10) & (x_adjusted >= -1e10);
            x_reg = x_adjusted(regular_vals);
            
            % Calculate using the relationship with incomplete beta function
            neg_idx = (x_reg < 0);
            pos_idx = ~neg_idx;
            
            if any(neg_idx)
                p_neg = 0.5 * betainc(nu./(nu + x_reg(neg_idx).^2), nu/2, 0.5);
                p(regular_vals & (x_adjusted < 0)) = p_neg;
            end
            
            if any(pos_idx)
                p_pos = 1 - 0.5 * betainc(nu./(nu + x_reg(pos_idx).^2), nu/2, 0.5);
                p(regular_vals & (x_adjusted >= 0)) = p_pos;
            end
        end
    end
end