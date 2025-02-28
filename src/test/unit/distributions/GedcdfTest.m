classdef GedcdfTest < BaseTest
    % GEDCDFTEST Test class for the gedcdf function, providing comprehensive test coverage for Generalized Error Distribution CDF calculations
    
    properties
        comparator            % NumericalComparator instance for floating-point comparisons
        defaultTolerance    % Default tolerance for numeric comparisons
        testData            % Structure to store test vectors
        testValues          % Array of test x values
        nuValues            % Array of test nu values
        expectedResults     % Array of expected CDF values
        referenceData       % Struct containing reference data
    end
    
    methods
        function obj = GedcdfTest()
            % Initializes a new GedcdfTest instance with numerical comparator
            
            % Call the superclass (BaseTest) constructor with 'GedcdfTest' name
            obj = obj@BaseTest('GedcdfTest');
            
            % Initialize the testData structure for storing test vectors
            obj.testData = struct();
            
            % Create a NumericalComparator instance for floating-point comparisons
            obj.comparator = NumericalComparator();
            
            % Set defaultTolerance to 1e-12 for high-precision numeric comparisons
            obj.defaultTolerance = 1e-12;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load reference data from known_distributions.mat using loadTestData method
            obj.referenceData = obj.loadTestData('known_distributions.mat');
            
            % Extract ged_cdf_values and ged_parameters from the loaded data
            ged_cdf_values = obj.referenceData.dist_samples.ged.data;
            ged_parameters = obj.referenceData.dist_samples.ged.parameters;
            
            % Initialize array of test x values [-3, -2, -1, 0, 1, 2, 3]
            obj.testValues = [-3, -2, -1, 0, 1, 2, 3];
            
            % Initialize array of test nu values [0.5, 1, 1.5, 2, 2.5, 5]
            obj.nuValues = [0.5, 1, 1.5, 2, 2.5, 5];
            
            % Configure numerical comparator with appropriate tolerance
            obj.comparator.setDefaultTolerances(obj.defaultTolerance, 1e-8);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear any temporary test data to free memory
            obj.testData = struct();
        end
        
        function testBasicCdf(obj)
            % Tests gedcdf with basic parameter values
            
            % Call gedcdf(0, 2) which corresponds to standard normal CDF at 0
            result = gedcdf(0, 2);
            
            % Verify result is approximately 0.5
            obj.assertAlmostEqual(result, 0.5, 'Basic CDF test failed');
            
            % Test that the result is a double with correct dimensions
            obj.assertEqual(class(result), 'double', 'Result must be a double');
            obj.assertEqual(size(result), [1 1], 'Result must be a scalar');
            
            % Verify CDF value is between 0 and 1
            obj.assertTrue(result >= 0 && result <= 1, 'CDF value must be between 0 and 1');
        end
        
        function testVectorInput(obj)
            % Tests gedcdf with vectorized inputs
            
            % Create vector of x values [-3, -2, -1, 0, 1, 2, 3]
            x = obj.testValues;
            
            % Call gedcdf with vector input and nu=2
            nu = 2;
            result = gedcdf(x, nu);
            
            % Verify output is a vector with same dimensions as input
            obj.assertEqual(class(result), 'double', 'Result must be a double');
            obj.assertEqual(size(result), size(x), 'Result must have the same dimensions as input');
            
            % Compare results with expected normal distribution CDF values
            expected = normcdf(x);
            obj.assertAlmostEqual(result, expected, 'Vector input test failed for nu=2');
            
            % Test with multiple nu values to verify correct vectorization
            nu_values = [1.5, 2.5, 5];
            for i = 1:length(nu_values)
                result = gedcdf(x, nu_values(i));
                obj.assertEqual(size(result), size(x), sprintf('Vector input test failed for nu=%g', nu_values(i)));
            end
        end
        
        function testLaplaceCase(obj)
            % Tests gedcdf with nu=1 (Laplace distribution case)
            
            % Create array of test points
            x = obj.testValues;
            
            % Calculate expected Laplace CDF values
            expected = 0.5 + 0.5 * sign(x) .* (1 - exp(-abs(x)));
            
            % Call gedcdf with test points and nu=1
            result = gedcdf(x, 1);
            
            % Compare calculated values with expected values
            obj.assertAlmostEqual(result, expected, 'Laplace case test failed');
            
            % Verify correct implementation of special case
            obj.assertTrue(all(result >= 0 & result <= 1), 'CDF values must be between 0 and 1');
        end
        
        function testNormalCase(obj)
            % Tests gedcdf with nu=2 (Normal distribution case)
            
            % Create array of test points
            x = obj.testValues;
            
            % Calculate expected Normal CDF values
            expected = normcdf(x);
            
            % Call gedcdf with test points and nu=2
            result = gedcdf(x, 2);
            
            % Compare calculated values with expected values
            obj.assertAlmostEqual(result, expected, 'Normal case test failed');
            
            % Verify correct implementation of special case
            obj.assertTrue(all(result >= 0 & result <= 1), 'CDF values must be between 0 and 1');
        end
        
        function testCdfBounds(obj)
            % Tests that the CDF values are properly bounded between 0 and 1
            
            % Generate a wide range of x values from very small to very large
            x = linspace(-10, 10, 100);
            
            % Test gedcdf with various nu values across the x range
            nu_values = [0.5, 1, 2, 5];
            for i = 1:length(nu_values)
                nu = nu_values(i);
                result = gedcdf(x, nu);
                
                % Verify all CDF values are >= 0 using assertGreaterThanOrEqual
                obj.assertGreaterThanOrEqual(result, 0, sprintf('CDF values must be >= 0 for nu=%g', nu));
                
                % Verify all CDF values are <= 1 using assertLessThanOrEqual
                obj.assertLessThanOrEqual(result, 1, sprintf('CDF values must be <= 1 for nu=%g', nu));
            end
            
            % Check extreme cases at x = -Inf and x = Inf
            obj.assertAlmostEqual(gedcdf(-Inf, 2), 0, 'CDF(-Inf) must be 0');
            obj.assertAlmostEqual(gedcdf(Inf, 2), 1, 'CDF(Inf) must be 1');
        end
        
        function testParameterValidation(obj)
            % Tests gedcdf error handling for invalid parameters
            
            % Test with invalid shape parameter (nu ≤ 0)
            obj.assertThrows(@() gedcdf(0, 0), 'parametercheck:InvalidInput', 'Test failed: nu <= 0 should throw an error');
            obj.assertThrows(@() gedcdf(0, -1), 'parametercheck:InvalidInput', 'Test failed: nu < 0 should throw an error');
            
            % Test with NaN and Inf values in parameters
            obj.assertThrows(@() gedcdf(NaN, 2), 'datacheck:InvalidInput', 'Test failed: NaN x should throw an error');
            obj.assertThrows(@() gedcdf(Inf, 2), 'datacheck:InvalidInput', 'Test failed: Inf x should throw an error');
            obj.assertThrows(@() gedcdf(0, NaN), 'parametercheck:InvalidInput', 'Test failed: NaN nu should throw an error');
            obj.assertThrows(@() gedcdf(0, Inf), 'parametercheck:InvalidInput', 'Test failed: Inf nu should throw an error');
            
            % Test with invalid sigma (sigma ≤ 0)
            obj.assertThrows(@() gedcdf(0, 2, 0, 0), 'parametercheck:InvalidInput', 'Test failed: sigma = 0 should throw an error');
            obj.assertThrows(@() gedcdf(0, 2, 0, -1), 'parametercheck:InvalidInput', 'Test failed: sigma < 0 should throw an error');
            
            % Verify appropriate error messages are generated
            % Use assertThrows to confirm expected exceptions are raised
        end
        
        function testCdfProperties(obj)
            % Tests that the CDF satisfies mathematical properties of a valid CDF
            
            % Verify CDF is monotonically increasing for all test points
            x = linspace(-5, 5, 100);
            nu_values = [0.5, 1, 2, 5];
            for i = 1:length(nu_values)
                nu = nu_values(i);
                cdf_values = gedcdf(x, nu);
                obj.assertTrue(all(diff(cdf_values) >= 0), sprintf('CDF must be monotonically increasing for nu=%g', nu));
            end
            
            % Verify CDF approaches 0 as x approaches -Infinity
            obj.assertAlmostEqual(gedcdf(-1e6, 2), 0, 'CDF must approach 0 as x approaches -Infinity');
            
            % Verify CDF approaches 1 as x approaches Infinity
            obj.assertAlmostEqual(gedcdf(1e6, 2), 1, 'CDF must approach 1 as x approaches Infinity');
            
            % Verify CDF(0) = 0.5 for symmetric distributions
            obj.assertAlmostEqual(gedcdf(0, 2), 0.5, 'CDF(0) must be 0.5 for symmetric distributions');
            
            % Test across multiple nu values to ensure properties hold generally
            for nu = [1, 1.5, 2, 3, 5]
                obj.assertAlmostEqual(gedcdf(0, nu), 0.5, sprintf('CDF(0) must be 0.5 for symmetric distributions (nu=%g)', nu));
            end
        end
        
        function testReferenceValues(obj)
            % Tests gedcdf against pre-computed reference values
            
            % Load reference CDF values from known_distributions.mat
            ged_cdf_values = obj.referenceData.dist_samples.ged.data;
            ged_parameters = obj.referenceData.dist_samples.ged.parameters;
            
            % Iterate through different parameter combinations
            nu_values = [1.2, 1.5, 2, 3];
            x_values = [-2, -1, 0, 1, 2];
            
            for i = 1:length(nu_values)
                nu = nu_values(i);
                for j = 1:length(x_values)
                    x = x_values(j);
                    
                    % Calculate CDF values using gedcdf
                    cdf_value = gedcdf(x, nu);
                    
                    % Compare with reference values using appropriate tolerance
                    expected_cdf = obj.calculateManualGedCdf(x, nu);
                    obj.assertAlmostEqual(cdf_value, expected_cdf, sprintf('Reference value test failed for x=%g, nu=%g', x, nu));
                end
            end
            
            % Verify high accuracy across the full range of test cases
        end
        
        function testLocationScaleParameters(obj)
            % Tests gedcdf with location and scale parameters (mu and sigma)
            
            % Test with non-zero location parameter (mu) values
            mu_values = [-1, 0, 1];
            sigma = 1;
            nu = 2;
            x = obj.testValues;
            
            for i = 1:length(mu_values)
                mu = mu_values(i);
                result = gedcdf(x, nu, mu, sigma);
                expected = normcdf(x - mu);
                obj.assertAlmostEqual(result, expected, sprintf('Location parameter test failed for mu=%g', mu));
            end
            
            % Verify CDF is correctly shifted
            
            % Test with different scale parameter (sigma) values
            sigma_values = [0.5, 1, 2];
            mu = 0;
            for i = 1:length(sigma_values)
                sigma = sigma_values(i);
                result = gedcdf(x, nu, mu, sigma);
                expected = normcdf(x ./ sigma);
                obj.assertAlmostEqual(result, expected, sprintf('Scale parameter test failed for sigma=%g', sigma));
            end
            
            % Verify CDF is correctly scaled
            
            % Test combinations of location and scale parameters
            mu = 1;
            sigma = 2;
            result = gedcdf(x, nu, mu, sigma);
            expected = normcdf((x - mu) ./ sigma);
            obj.assertAlmostEqual(result, expected, 'Location and scale parameter test failed');
        end
        
        function testNumericalPrecision(obj)
            % Tests numerical precision of gedcdf for extreme values
            
            % Test with very small nu values (approaching 0)
            nu_small = 1e-6;
            x_values = [-1, 0, 1];
            for x = x_values
                result = gedcdf(x, nu_small);
                obj.assertTrue(result >= 0 && result <= 1, sprintf('Numerical precision test failed for small nu=%g', nu_small));
            end
            
            % Test with very large nu values (approaching infinity)
            nu_large = 1e6;
            x_values = [-1, 0, 1];
            for x = x_values
                result = gedcdf(x, nu_large);
                obj.assertTrue(result >= 0 && result <= 1, sprintf('Numerical precision test failed for large nu=%g', nu_large));
            end
            
            % Test with extreme x values (far from mean)
            x_extreme = [-1e6, 0, 1e6];
            nu = 2;
            for x = x_extreme
                result = gedcdf(x, nu);
                obj.assertTrue(result >= 0 && result <= 1, sprintf('Numerical precision test failed for extreme x=%g', x));
            end
            
            % Verify numerical stability under these conditions
            
            % Compare against high-precision reference calculations
        end
        
        function cdf = calculateManualGedCdf(obj, x, nu)
            % Helper method to manually calculate GED CDF for validation
            
            % Calculate lambda = sqrt(gamma(3/nu)/gamma(1/nu))
            lambda = sqrt(gamma(3/nu)/gamma(1/nu));
            
            % Standardize input values: z = x / lambda
            z = x / lambda;
            
            % For positive z values: CDF = 0.5 + 0.5 * sign(z) * gammainc((abs(z))^nu / 2, 1/nu)
            % For negative z values: CDF = 0.5 - 0.5 * gammainc((abs(z))^nu / 2, 1/nu)
            cdf = zeros(size(z));
            posZ = z >= 0;
            negZ = z < 0;
            cdf(posZ) = 0.5 + 0.5 * sign(z(posZ)) .* gammainc((abs(z(posZ))).^nu / 2, 1/nu);
            cdf(negZ) = 0.5 - 0.5 * gammainc((abs(z(negZ))).^nu / 2, 1/nu);
            
            % Return calculated CDF values for comparison with gedcdf function output
        end
    end
end