classdef GarchinitTest < BaseTest
    % GarchinitTest Unit test class for the garchinit function
    %
    % This class tests the parameter initialization functionality for GARCH models
    % in the MFE Toolbox, ensuring proper initialization across different model
    % types, orders, and error distributions.
    %
    % The garchinit function provides intelligent starting values for GARCH model
    % estimation, which is critical for convergence of maximum likelihood estimation
    % in volatility modeling.
    
    properties
        testData        % Test data structure
        comparator      % Numerical comparator for floating-point comparisons
        tolerance       % Tolerance for numerical comparisons
    end
    
    methods
        function obj = GarchinitTest()
            % Initialize the GarchinitTest class with appropriate test configuration
            obj@BaseTest();
            obj.comparator = NumericalComparator();
            obj.tolerance = 1e-8;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Load test data if available
            try
                obj.testData = obj.loadTestData('volatility_test_data.mat');
            catch
                % If test data file not found, we'll generate data in the tests
            end
            
            % Set random seed for reproducibility
            rng(1234);
            
            % Initialize numerical comparator with appropriate tolerance
            obj.comparator = NumericalComparator();
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method
            tearDown@BaseTest(obj);
            % Additional cleanup if needed
        end
        
        function testStandardGarchInit(obj)
            % Test parameter initialization for standard GARCH(p,q) model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % Standard GARCH(1,1) with normal distribution
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % Verify number of parameters (omega, alpha, beta)
            expectedNumParams = 1 + options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for GARCH(1,1)');
            
            % Extract components for verification
            omega = params(1);
            alpha = params(2:(options.p+1));
            beta = params((options.p+2):end);
            
            % omega should be positive and based on unconditional variance
            obj.assertTrue(omega > 0, 'omega should be positive');
            obj.assertTrue(omega < var(data), 'omega should be less than unconditional variance');
            
            % alpha should be reasonable (typically small)
            obj.assertTrue(all(alpha > 0) && sum(alpha) < 0.3, 'alpha parameters should be positive and reasonably sized');
            
            % beta should be reasonable (typically large)
            obj.assertTrue(all(beta > 0) && sum(beta) < 1, 'beta parameters should be positive and less than 1');
            
            % stationarity constraint: alpha + beta < 1
            obj.assertTrue(sum(alpha) + sum(beta) < 1, 'Stationarity constraint should be satisfied');
            
            % Test with different model orders
            
            % GARCH(2,1)
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for GARCH(2,1)');
            
            % GARCH(1,2)
            options.p = 1;
            options.q = 2;
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for GARCH(1,2)');
            
            % Verify parameter values decrease with lag order
            options.p = 2;
            options.q = 2;
            params = garchinit(data, options);
            alpha = params(2:3);
            beta = params(4:5);
            obj.assertTrue(alpha(1) >= alpha(2), 'ARCH parameters should decrease with lag');
            obj.assertTrue(beta(1) >= beta(2), 'GARCH parameters should decrease with lag');
        end
        
        function testEgarchInit(obj)
            % Test parameter initialization for EGARCH model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % EGARCH(1,1) with normal distribution
            options = struct('model', 'EGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % Verify number of parameters (omega, alpha, gamma, beta)
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for EGARCH(1,1)');
            
            % Verify basic parameter properties
            omega = params(1);
            alpha = params(2:(options.p+1));
            gamma = params((options.p+2):(2*options.p+1));
            beta = params((2*options.p+2):end);
            
            % omega should be related to log variance
            obj.assertTrue(abs(omega - log(var(data))*0.25) < 2, 'omega should be related to log variance');
            
            % alpha should be reasonable
            obj.assertTrue(all(alpha >= 0) && sum(alpha) < 0.5, 'alpha parameters should be non-negative and reasonable');
            
            % gamma typically reflects asymmetry/leverage effect
            negativeReturns = data(data < 0);
            positiveReturns = data(data > 0);
            if mean(negativeReturns.^2) > mean(positiveReturns.^2)
                obj.assertTrue(any(gamma < 0), 'gamma should reflect leverage effect');
            end
            
            % beta should indicate persistence
            obj.assertTrue(all(beta > 0) && sum(beta) < 1, 'beta parameters should indicate persistence');
            
            % Test with different model orders
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for EGARCH(2,1)');
        end
        
        function testTarchInit(obj)
            % Test parameter initialization for Threshold ARCH (TARCH) model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % TARCH(1,1) with normal distribution
            options = struct('model', 'TARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % Verify number of parameters (omega, alpha, gamma, beta)
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for TARCH(1,1)');
            
            % Verify basic parameter properties
            omega = params(1);
            alpha = params(2:(options.p+1));
            gamma = params((options.p+2):(2*options.p+1));
            beta = params((2*options.p+2):end);
            
            % omega should be positive
            obj.assertTrue(omega > 0, 'omega should be positive');
            
            % alpha should be reasonable
            obj.assertTrue(all(alpha >= 0) && sum(alpha) < 0.3, 'alpha parameters should be non-negative and reasonable');
            
            % gamma should be positive for leverage effect
            obj.assertTrue(all(gamma >= 0), 'gamma parameters should be non-negative for leverage effect');
            
            % beta should be reasonable
            obj.assertTrue(all(beta > 0) && sum(beta) < 1, 'beta parameters should be positive and reasonable');
            
            % stationarity constraint: alpha + 0.5*gamma + beta < 1
            obj.assertTrue(sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1, 'Stationarity constraint should be satisfied');
            
            % Test with different model orders
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for TARCH(2,1)');
            
            % Test with GJR model alias
            options.model = 'GJR';
            options.p = 1;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for GJR(1,1)');
        end
        
        function testAgarchInit(obj)
            % Test parameter initialization for Asymmetric GARCH (AGARCH) model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % AGARCH(1,1) with normal distribution
            options = struct('model', 'AGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % Verify number of parameters (omega, alpha, gamma, beta)
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for AGARCH(1,1)');
            
            % Verify basic parameter properties
            omega = params(1);
            alpha = params(2:(options.p+1));
            gamma = params((options.p+2):(2*options.p+1));
            beta = params((2*options.p+2):end);
            
            % omega should be positive
            obj.assertTrue(omega > 0, 'omega should be positive');
            
            % alpha should be reasonable
            obj.assertTrue(all(alpha >= 0) && sum(alpha) < 0.3, 'alpha parameters should be non-negative and reasonable');
            
            % gamma should reflect asymmetry
            obj.assertTrue(all(abs(gamma) < 0.5), 'gamma parameters should be reasonable for asymmetry');
            
            % beta should be reasonable
            obj.assertTrue(all(beta > 0) && sum(beta) < 1, 'beta parameters should be positive and reasonable');
            
            % stationarity constraint: alpha + beta < 1
            obj.assertTrue(sum(alpha) + sum(beta) < 1, 'Stationarity constraint should be satisfied');
            
            % Test with different model orders
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for AGARCH(2,1)');
        end
        
        function testIgarchInit(obj)
            % Test parameter initialization for Integrated GARCH (IGARCH) model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % IGARCH(1,1) with normal distribution
            options = struct('model', 'IGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % For IGARCH, we only get omega and alpha parameters
            % since beta is constrained by sum(alpha) + sum(beta) = 1
            expectedNumParams = 1 + options.p;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for IGARCH(1,1)');
            
            % Extract parameters for verification
            omega = params(1);
            alpha = params(2:end);
            
            % omega should be positive but small
            obj.assertTrue(omega > 0 && omega < var(data), 'omega should be positive but small for IGARCH');
            
            % alpha should be reasonable and less than 1 (to allow for beta)
            obj.assertTrue(all(alpha >= 0) && sum(alpha) < 1, 'alpha parameters should be reasonable for IGARCH');
            
            % Test with different model orders
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for IGARCH(2,1)');
        end
        
        function testNagarchInit(obj)
            % Test parameter initialization for Nonlinear Asymmetric GARCH (NAGARCH) model
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % NAGARCH(1,1) with normal distribution
            options = struct('model', 'NAGARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            
            % Get initialization
            params = garchinit(data, options);
            
            % Verify number of parameters (omega, alpha, gamma, beta)
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for NAGARCH(1,1)');
            
            % Verify basic parameter properties
            omega = params(1);
            alpha = params(2:(options.p+1));
            gamma = params((options.p+2):(2*options.p+1));
            beta = params((2*options.p+2):end);
            
            % omega should be positive
            obj.assertTrue(omega > 0, 'omega should be positive');
            
            % alpha should be reasonable
            obj.assertTrue(all(alpha >= 0) && sum(alpha) < 0.3, 'alpha parameters should be reasonable');
            
            % gamma should reflect asymmetry
            obj.assertTrue(all(abs(gamma) < 0.5), 'gamma parameters should be reasonable for asymmetry');
            
            % beta should be reasonable
            obj.assertTrue(all(beta > 0) && sum(beta) < 1, 'beta parameters should be positive and reasonable');
            
            % stationarity constraint: alpha + beta < 1
            obj.assertTrue(sum(alpha) + sum(beta) < 1, 'Stationarity constraint should be satisfied');
            
            % Test with different model orders
            options.p = 2;
            options.q = 1;
            params = garchinit(data, options);
            expectedNumParams = 1 + 2*options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for NAGARCH(2,1)');
        end
        
        function testDistributionParameters(obj)
            % Test initialization with different error distributions
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % Test with normal distribution (no additional parameters)
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'distribution', 'NORMAL');
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for normal distribution');
            
            % Test with t distribution (adds 1 parameter: degrees of freedom)
            options.distribution = 'T';
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q + 1;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for t distribution');
            obj.assertTrue(params(end) > 2.1, 'Degrees of freedom should be > 2 for t distribution');
            
            % Test with GED distribution (adds 1 parameter: shape parameter)
            options.distribution = 'GED';
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q + 1;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for GED distribution');
            obj.assertTrue(params(end) > 0, 'Shape parameter should be positive for GED distribution');
            
            % Test with skewed t distribution (adds 2 parameters: df and skew)
            options.distribution = 'SKEWT';
            params = garchinit(data, options);
            expectedNumParams = 1 + options.p + options.q + 2;
            obj.assertEqual(length(params), expectedNumParams, 'Parameter count mismatch for skewed t distribution');
            obj.assertTrue(params(end-1) > 2.1, 'Degrees of freedom should be > 2 for skewed t distribution');
            obj.assertTrue(abs(params(end)) < 1, 'Skew parameter should be between -1 and 1');
        end
        
        function testInvalidInputs(obj)
            % Test garchinit function's handling of invalid inputs
            
            % Test with empty data
            obj.assertThrows(@() garchinit([], []), 'MATLAB:rowVector:NotRowVector', ...
                'Should throw error for empty data');
            
            % Test with non-numeric data
            obj.assertThrows(@() garchinit('abc', []), 'MATLAB:rowVector:NotNumeric', ...
                'Should throw error for non-numeric data');
            
            % Test with invalid model type
            data = obj.createTestData(100, 1);
            options = struct('model', 'INVALID_MODEL');
            obj.assertThrows(@() garchinit(data, options), 'MATLAB:UndefinedFunction', ...
                'Should throw error for invalid model type');
            
            % Test with invalid p or q values
            options = struct('model', 'GARCH', 'p', -1);
            obj.assertThrows(@() garchinit(data, options), 'MATLAB:parametercheck:OutOfRange', ...
                'Should throw error for negative p value');
            
            options = struct('model', 'GARCH', 'p', 1.5);
            obj.assertThrows(@() garchinit(data, options), 'MATLAB:parametercheck:isInteger', ...
                'Should throw error for non-integer p value');
            
            % Test with invalid distribution
            options = struct('model', 'GARCH', 'distribution', 'INVALID_DIST');
            obj.assertThrows(@() garchinit(data, options), 'MATLAB:UndefinedFunction', ...
                'Should throw error for invalid distribution');
        end
        
        function testCustomInitialization(obj)
            % Test initialization with user-provided starting values
            
            % Create test data
            data = obj.createTestData(1000, 1);
            
            % Test with custom starting values
            customParams = [0.05; 0.1; 0.8];  % omega, alpha, beta for GARCH(1,1)
            options = struct('model', 'GARCH', 'p', 1, 'q', 1, 'startingvals', customParams);
            
            params = garchinit(data, options);
            
            % Verify parameters match custom values
            obj.assertAlmostEqual(params, customParams, 'Custom starting values not respected');
            
            % Test with different model types
            % EGARCH with custom values
            customEGARCH = [0.1; 0.05; -0.1; 0.85];  % omega, alpha, gamma, beta
            options = struct('model', 'EGARCH', 'p', 1, 'q', 1, 'startingvals', customEGARCH);
            params = garchinit(data, options);
            obj.assertAlmostEqual(params, customEGARCH, 'Custom EGARCH values not respected');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of garchinit with extreme data
            
            % Test with very small variance data
            smallVarData = obj.createTestData(1000, 0.00001);
            options = struct('model', 'GARCH', 'p', 1, 'q', 1);
            smallParams = garchinit(smallVarData, options);
            
            % Omega should still be positive
            obj.assertTrue(smallParams(1) > 0, 'Omega should remain positive for small variance data');
            
            % Test with very large variance data
            largeVarData = obj.createTestData(1000, 10000);
            largeParams = garchinit(largeVarData, options);
            
            % Parameters should maintain proper proportions
            obj.assertTrue(largeParams(1) > 0, 'Omega should be positive for large variance data');
            obj.assertTrue(sum(largeParams(2:end)) < 1, 'Sum of alpha+beta should be less than 1');
            
            % Test with data containing outliers
            outlierData = obj.createTestData(1000, 1);
            outlierData(100) = outlierData(100) * 10;  % Add an outlier
            outlierParams = garchinit(outlierData, options);
            
            % Parameters should still maintain proper constraints
            obj.assertTrue(outlierParams(1) > 0, 'Omega should be positive with outliers');
            obj.assertTrue(sum(outlierParams(2:end)) < 1, 'Sum of alpha+beta should be less than 1 with outliers');
        end
        
        function data = createTestData(obj, sampleSize, variance)
            % Helper method to create test data for GARCH initialization testing
            %
            % INPUTS:
            %   sampleSize - Size of the test dataset
            %   variance - Variance of the random data
            %
            % OUTPUTS:
            %   data - Test returns data with specified properties
            
            % Set random seed for reproducibility
            rng(2345);
            
            % Generate random normal data with specified variance
            data = sqrt(variance) * randn(sampleSize, 1);
        end
        
        function isValid = verifyParameterProperties(obj, parameters, modelType, p, q)
            % Helper method to verify GARCH parameter properties
            %
            % INPUTS:
            %   parameters - Parameter vector to validate
            %   modelType - Type of GARCH model
            %   p - ARCH order
            %   q - GARCH order
            %
            % OUTPUTS:
            %   isValid - True if parameters satisfy model constraints
            
            modelType = upper(modelType);
            isValid = true;
            
            switch modelType
                case 'GARCH'
                    % Extract components
                    omega = parameters(1);
                    alpha = parameters(2:(p+1));
                    beta = parameters((p+2):(p+q+1));
                    
                    % Check constraints
                    if omega <= 0 || any(alpha < 0) || any(beta < 0) || (sum(alpha) + sum(beta) >= 1)
                        isValid = false;
                    end
                    
                case {'GJR', 'TARCH'}
                    % Extract components
                    omega = parameters(1);
                    alpha = parameters(2:(p+1));
                    gamma = parameters((p+2):(2*p+1));
                    beta = parameters((2*p+2):(2*p+q+1));
                    
                    % Check constraints
                    if omega <= 0 || any(alpha < 0) || any(gamma < 0) || any(beta < 0) || ...
                       (sum(alpha) + 0.5*sum(gamma) + sum(beta) >= 1)
                        isValid = false;
                    end
                    
                case 'EGARCH'
                    % For EGARCH, constraints are different
                    omega = parameters(1);
                    beta = parameters((2*p+2):(2*p+q+1));
                    
                    % For EGARCH, main constraint is on beta for stationarity
                    if sum(beta) >= 1
                        isValid = false;
                    end
                    
                case 'IGARCH'
                    % For IGARCH, we only have omega and alpha
                    omega = parameters(1);
                    alpha = parameters(2:end);
                    
                    if omega <= 0 || any(alpha < 0) || sum(alpha) >= 1
                        isValid = false;
                    end
                    
                case {'AGARCH', 'NAGARCH'}
                    % Extract components
                    omega = parameters(1);
                    alpha = parameters(2:(p+1));
                    beta = parameters((p+2):(p+q+1));
                    
                    if omega <= 0 || any(alpha < 0) || any(beta < 0) || (sum(alpha) + sum(beta) >= 1)
                        isValid = false;
                    end
            end
        end
    end
end