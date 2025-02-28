classdef StdtfitTest < BaseTest
    % STDTFITTEST Test class for validating the standardized Student's t-distribution 
    % parameter estimation functionality in stdtfit.m
    %
    % This class tests the accuracy, numerical stability, and convergence of
    % the standardized Student's t-distribution parameter estimation algorithm
    % implemented in stdtfit.m. It validates the function against reference data
    % sets and checks for robustness in various challenging scenarios.
    
    properties
        testData    % Structure to hold test data
        tolerance   % Numerical tolerance for parameter comparison
    end
    
    methods
        function obj = StdtfitTest()
            % Initialize the StdtfitTest with proper setup for distribution testing
            obj@BaseTest();
            
            % Set appropriate numerical tolerance for parameter comparisons
            % Distribution parameter estimation can be sensitive, especially for df
            obj.tolerance = 1e-4;
            
            % Initialize empty testData property
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Prepare test environment before each test method execution
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Load distribution test data from known_distributions.mat
            try
                obj.testData = obj.loadTestData('known_distributions.mat');
            catch ME
                % If test data file doesn't exist, create minimal test data structure
                obj.testData.tDist.dof = [3, 5, 8, 15, 30];
                warning('Test data file not found. Using synthesized test data.');
            end
            
            % Set random number generator seed for reproducible results
            rng(123);
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % No additional cleanup needed for this test class
        end
        
        function testBasicFunctionality(obj)
            % Test basic functionality of stdtfit with standard inputs
            
            % Generate random data with known t-distribution
            nu = 5; % Degrees of freedom
            n = 1000; % Sample size
            data = trnd(nu, n, 1);
            
            % Call stdtfit with basic parameters
            result = stdtfit(data);
            
            % Verify function returns expected structure with all required fields
            expectedFields = {'nu', 'nuSE', 'mu', 'muSE', 'sigma', 'sigmaSE', ...
                             'logL', 'AIC', 'BIC', 'convCode', 'optOutput'};
            actualFields = fieldnames(result);
            
            % Check all expected fields are present
            for i = 1:length(expectedFields)
                obj.assertTrue(ismember(expectedFields{i}, actualFields), ...
                    ['Result missing expected field: ' expectedFields{i}]);
            end
            
            % Verify degrees of freedom parameter is reasonable
            obj.assertTrue(result.nu > 2, 'Degrees of freedom (nu) must be greater than 2');
            obj.assertTrue(result.nu < 100, 'Degrees of freedom (nu) should be less than 100 for this test');
            
            % Verify convergence code indicates success
            obj.assertEqual(1, result.convCode, 'Optimization should converge successfully');
        end
        
        function testParameterEstimation(obj)
            % Test parameter estimation accuracy using known distribution data
            
            % Get reference data for standardized t distribution
            tDistData = struct();
            if isfield(obj.testData, 'tDist')
                tDistData = obj.testData.tDist;
            else
                % If no test data available, use default test values
                tDistData.dof = [3, 5, 8, 15, 30];
            end
            
            % Test estimation for various degrees of freedom
            for i = 1:length(tDistData.dof)
                nu = tDistData.dof(i);
                
                % Generate sample from t distribution with known parameters
                n = 2000; % Large sample for better estimation accuracy
                data = trnd(nu, n, 1);
                
                % Estimate parameters
                result = stdtfit(data);
                
                % Assert estimated nu is close to true value within tolerance
                % We use a wider tolerance for df estimation as it's a challenging parameter
                obj.assertAlmostEqual(nu, result.nu, ['Parameter estimation failed for nu = ' num2str(nu)]);
                
                % Verify mu is close to 0 (theoretical mean for standardized t)
                obj.assertAlmostEqual(0, result.mu, 0.1, ...
                    ['Mean should be close to 0 for standardized t with nu = ' num2str(nu)]);
                
                % For nu > 2, verify variance is close to 1 (std dev close to 1)
                if nu > 2
                    obj.assertAlmostEqual(1, result.sigma, 0.1, ...
                        ['Std dev should be close to 1 for standardized t with nu = ' num2str(nu)]);
                end
                
                % Verify standard errors are calculated and reasonable
                obj.assertTrue(~isnan(result.nuSE) || nu > 50, 'Standard error for nu should be calculated');
                obj.assertTrue(result.nuSE > 0 || isnan(result.nuSE), 'Standard error for nu should be positive when available');
            end
        end
        
        function testLogLikelihoodConsistency(obj)
            % Test consistency between log-likelihood calculation in stdtfit and stdtloglik
            
            % Generate random data
            nu = 6;
            n = 1000;
            data = trnd(nu, n, 1);
            
            % Estimate parameters using stdtfit
            result = stdtfit(data);
            
            % Calculate log-likelihood directly using stdtloglik
            directLogL = -stdtloglik(data, result.nu, result.mu, result.sigma);
            
            % Assert log-likelihood values are consistent
            obj.assertAlmostEqual(result.logL, directLogL, ...
                'Log-likelihood values should be consistent between stdtfit and stdtloglik');
        end
        
        function testOptionsParsing(obj)
            % Test proper parsing and application of options struct
            
            % Generate random data
            nu = 4;
            n = 1000;
            data = trnd(nu, n, 1);
            
            % Create options with custom settings
            options = struct();
            options.startingVal = 10;
            options.display = 'off';
            
            % Run stdtfit with options
            result = stdtfit(data, options);
            
            % Verify estimation still works correctly with custom options
            obj.assertTrue(result.nu > 2, 'Degrees of freedom should be greater than 2');
            obj.assertTrue(result.convCode == 1, 'Optimization should converge with custom options');
            
            % Test with extreme starting value (very high)
            optionsHigh = struct();
            optionsHigh.startingVal = 50;
            resultHigh = stdtfit(data, optionsHigh);
            
            % Verify convergence despite extreme starting value
            obj.assertTrue(resultHigh.convCode == 1, 'Should converge even with high starting value');
            
            % Test that results are similar regardless of starting value
            obj.assertAlmostEqual(result.nu, resultHigh.nu, 0.5, ...
                'Parameter estimates should be similar regardless of starting value');
            
            % Test with boundary case (starting value <= 2)
            optionsLow = struct();
            optionsLow.startingVal = 1.5; % Below minimum threshold
            
            % This should issue a warning and reset to default value, but continue
            warning('off', 'all'); % Suppress warning for test
            resultLow = stdtfit(data, optionsLow);
            warning('on', 'all');
            
            % Verify algorithm still converges despite invalid starting value
            obj.assertTrue(resultLow.convCode == 1, 'Should converge despite invalid starting value');
        end
        
        function testInputValidation(obj)
            % Test input validation for incorrect or problematic inputs
            
            % Test with empty data
            try
                stdtfit([]);
                obj.assertTrue(false, 'Empty data should cause an error');
            catch ME
                % Expected behavior: should throw an error for empty data
                obj.assertTrue(~isempty(ME.message), 'Error message should not be empty');
            end
            
            % Test with non-numeric data
            try
                stdtfit('string');
                obj.assertTrue(false, 'Non-numeric data should cause an error');
            catch ME
                % Expected behavior: should throw an error for non-numeric data
                obj.assertTrue(~isempty(ME.message), 'Error message should not be empty');
            end
            
            % Test with data containing NaN
            data = randn(100, 1);
            data(5) = NaN;
            try
                stdtfit(data);
                obj.assertTrue(false, 'Data with NaN should cause an error');
            catch ME
                % Expected behavior: should throw an error for data with NaN
                obj.assertTrue(~isempty(ME.message), 'Error message should not be empty');
            end
            
            % Test with data containing Inf
            data = randn(100, 1);
            data(10) = Inf;
            try
                stdtfit(data);
                obj.assertTrue(false, 'Data with Inf should cause an error');
            catch ME
                % Expected behavior: should throw an error for data with Inf
                obj.assertTrue(~isempty(ME.message), 'Error message should not be empty');
            end
            
            % Test with matrix rather than vector
            data = randn(10, 10);
            try
                stdtfit(data);
                obj.assertTrue(false, 'Matrix data should cause an error');
            catch ME
                % Expected behavior: should throw an error for matrix input
                obj.assertTrue(~isempty(ME.message), 'Error message should not be empty');
            end
        end
        
        function testNumericalStability(obj)
            % Test numerical stability with challenging data sets
            
            % Test with very heavy tails (small nu)
            nuHeavy = 2.1; % Just above the minimum threshold
            nHeavy = 1000;
            dataHeavy = trnd(nuHeavy, nHeavy, 1);
            resultHeavy = stdtfit(dataHeavy);
            
            % Verify reasonable estimate for heavy tails
            obj.assertTrue(resultHeavy.nu > 2, 'Nu estimate should be greater than 2');
            obj.assertTrue(resultHeavy.nu < 10, 'Nu estimate for heavy tails should be small');
            
            % Test with light tails (large nu)
            nuLight = 50;
            nLight = 1000;
            dataLight = trnd(nuLight, nLight, 1);
            resultLight = stdtfit(dataLight);
            
            % Verify reasonable estimate for light tails
            % Note: Estimating high dof is challenging as t approaches normal
            obj.assertTrue(resultLight.nu > 5, 'Nu estimate should be reasonably high for light tails');
            
            % Test with small sample
            nuSmall = 5;
            nSmall = 50; % Small sample size
            dataSmall = trnd(nuSmall, nSmall, 1);
            resultSmall = stdtfit(dataSmall);
            
            % Verify convergence with small sample
            obj.assertTrue(resultSmall.convCode == 1, 'Should converge even with small sample');
            
            % Test with large sample
            nuLarge = 5;
            nLarge = 10000; % Large sample size
            dataLarge = trnd(nuLarge, nLarge, 1);
            resultLarge = stdtfit(dataLarge);
            
            % Verify accurate estimation with large sample
            obj.assertAlmostEqual(nuLarge, resultLarge.nu, 1.0, 'Should estimate accurately with large sample');
            
            % Test stability through multiple runs with same data
            nuStable = 6;
            nStable = 2000;
            dataStable = trnd(nuStable, nStable, 1);
            
            result1 = stdtfit(dataStable);
            result2 = stdtfit(dataStable);
            
            % Verify consistent results across multiple runs
            obj.assertEqual(result1.nu, result2.nu, 'Results should be identical for multiple runs on same data');
            obj.assertEqual(result1.logL, result2.logL, 'Log-likelihood should be identical for multiple runs');
        end
        
        function testInfoCriteria(obj)
            % Test information criteria (AIC/BIC) calculation
            
            % Generate data from t distribution
            nu = 5;
            n = 1000;
            data = trnd(nu, n, 1);
            
            % Estimate parameters
            result = stdtfit(data);
            
            % Manually calculate expected AIC/BIC
            logL = result.logL;
            k = 3; % Number of parameters (nu, mu, sigma)
            expectedAIC = -2 * logL + 2 * k;
            expectedBIC = -2 * logL + k * log(n);
            
            % Assert AIC/BIC values match expected
            obj.assertAlmostEqual(expectedAIC, result.AIC, 'AIC calculation should match expected formula');
            obj.assertAlmostEqual(expectedBIC, result.BIC, 'BIC calculation should match expected formula');
            
            % Verify consistency with separate calculation using aicsbic function
            ic = aicsbic(logL, k, n);
            obj.assertAlmostEqual(ic.aic, result.AIC, 'AIC should match aicsbic calculation');
            obj.assertAlmostEqual(ic.sbic, result.BIC, 'BIC should match aicsbic calculation');
        end
        
        function results = runAllTests(obj)
            % Convenience method to run all test methods in the class
            results = runAllTests@BaseTest(obj);
        end
    end
end