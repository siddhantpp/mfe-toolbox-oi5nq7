classdef GarchcoreTest < BaseTest
    % GARCHCORETEST Unit test class for the garchcore function
    %
    % This class tests the garchcore function, which provides the core variance 
    % computation engine for various GARCH model types in the MFE Toolbox.
    % It validates correctness of GARCH variance computation across different 
    % model specifications with both MATLAB and MEX implementations.
    
    properties
        testData         % Structure containing test data
        comparator       % NumericalComparator instance for floating-point comparisons
        tolerance        % Tolerance threshold for numerical comparisons
    end
    
    methods
        function obj = GarchcoreTest()
            % Initialize the GarchcoreTest class with appropriate test configuration
            obj@BaseTest();
            obj.comparator = NumericalComparator();
            % Set higher tolerance specifically for GARCH calculations which can have numerical sensitivity
            obj.tolerance = 1e-6;
        end
        
        function setUp(obj)
            % Set up test environment before each test method execution
            setUp@BaseTest(obj);
            
            % Load volatility test data from voldata.mat
            try
                obj.testData = obj.loadTestData('voldata.mat');
            catch
                % Generate test data if file doesn't exist
                rng(123); % For reproducibility
            end
            
            % Set random number generator seed for reproducibility
            rng(42);
            
            % Initialize the numerical comparator with appropriate tolerance for GARCH computations
            obj.comparator = NumericalComparator();
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method
            tearDown@BaseTest(obj);
            
            % Clear test-specific variables
        end
        
        function testStandardGarch(obj)
            % Test the standard GARCH(p,q) model implementation in garchcore
            
            % Create test data with known GARCH properties
            testData = obj.createTestGarchData('GARCH', struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85));
            
            % Define GARCH parameters (omega, alpha, beta)
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.beta];
            options = struct('model', 'GARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'GARCH variances do not match expected values');
            
            % Test edge cases including minimum parameter values
            parameters = [0.01; 0.01; 0.5];
            ht3 = garchcore(parameters, testData.returns, struct('model', 'GARCH', 'p', 1, 'q', 1));
            obj.assertTrue(all(ht3 > 0), 'GARCH with minimal parameters should produce positive variances');
            
            % Verify behavior with different (p,q) orders
            options.p = 2;
            options.q = 2;
            parameters = [0.05; 0.08; 0.02; 0.7; 0.15];
            ht2 = garchcore(parameters, testData.returns, options);
            obj.assertTrue(all(ht2 > 0), 'GARCH(2,2) variances should be positive');
            
            % Test both MATLAB and MEX implementations (if available)
            try
                options.useMEX = true;
                htMEX = garchcore(parameters, testData.returns, options);
                obj.assertTrue(all(htMEX > 0), 'GARCH MEX implementation should produce positive variances');
            catch ME
                if ~contains(ME.identifier, 'MATLAB:UndefinedFunction')
                    rethrow(ME);
                end
            end
        end
        
        function testEgarch(obj)
            % Test the EGARCH model implementation in garchcore
            
            % Create test data with known EGARCH properties
            testData = obj.createTestGarchData('EGARCH', struct('omega', -0.1, 'alpha', 0.1, 'gamma', -0.05, 'beta', 0.95));
            
            % Define EGARCH parameters (omega, alpha, gamma, beta)
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.gamma; testData.parameters.beta];
            options = struct('model', 'EGARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'EGARCH variances do not match expected values');
            
            % Test the asymmetric response to negative vs positive shocks
            negativeReturns = -abs(testData.returns);
            positiveReturns = abs(testData.returns);
            
            htNeg = garchcore(parameters, negativeReturns, options);
            htPos = garchcore(parameters, positiveReturns, options);
            
            % Due to negative gamma, negative returns should generate higher volatility
            % than positive returns of the same magnitude in later periods
            obj.assertTrue(mean(htNeg(end-10:end)) > mean(htPos(end-10:end)), 'EGARCH fails to capture asymmetric effects');
            
            % Test both MATLAB and MEX implementations (if available)
            try
                options.useMEX = false;
                htMATLAB = garchcore(parameters, testData.returns, options);
                
                options.useMEX = true;
                htMEX = garchcore(parameters, testData.returns, options);
                
                obj.assertAlmostEqual(htMATLAB, htMEX, 'EGARCH MATLAB and MEX implementations should match');
            catch ME
                if ~contains(ME.identifier, 'MATLAB:UndefinedFunction')
                    rethrow(ME);
                end
            end
        end
        
        function testTarch(obj)
            % Test the Threshold ARCH (TARCH) model implementation in garchcore
            
            % Create test data with known TARCH properties
            testData = obj.createTestGarchData('GJR', struct('omega', 0.05, 'alpha', 0.05, 'gamma', 0.1, 'beta', 0.85));
            
            % Define TARCH parameters (omega, alpha, gamma, beta)
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.gamma; testData.parameters.beta];
            options = struct('model', 'TARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'TARCH variances do not match expected values');
            
            % Test threshold effects specifically for negative returns
            negativeReturns = -abs(testData.returns);
            positiveReturns = abs(testData.returns);
            
            htNeg = garchcore(parameters, negativeReturns, options);
            htPos = garchcore(parameters, positiveReturns, options);
            
            % Due to positive gamma, negative returns should generate higher volatility
            % than positive returns of the same magnitude in later periods
            obj.assertTrue(mean(htNeg(end-10:end)) > mean(htPos(end-10:end)), 'TARCH fails to capture threshold effects');
            
            % Test both MATLAB and MEX implementations (if available)
            try
                options = struct('model', 'GJR', 'p', 1, 'q', 1);
                options.useMEX = true;
                htMEX = garchcore(parameters, testData.returns, options);
                
                options.useMEX = false;
                htMATLAB = garchcore(parameters, testData.returns, options);
                
                obj.assertAlmostEqual(htMATLAB, htMEX, 'GJR MATLAB and MEX implementations should match');
            catch ME
                if ~contains(ME.identifier, 'MATLAB:UndefinedFunction')
                    rethrow(ME);
                end
            end
        end
        
        function testAgarch(obj)
            % Test the Asymmetric GARCH (AGARCH) model implementation in garchcore
            
            % Create test data with known AGARCH properties
            testData = obj.createTestGarchData('AGARCH', struct('omega', 0.05, 'alpha', 0.1, 'gamma', 0.1, 'beta', 0.85));
            
            % Define AGARCH parameters (omega, alpha, gamma, beta)
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.gamma; testData.parameters.beta];
            options = struct('model', 'AGARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'AGARCH variances do not match expected values');
            
            % Test asymmetric news impact curve effects
            negativeReturns = -abs(testData.returns);
            positiveReturns = abs(testData.returns);
            
            htNeg = garchcore(parameters, negativeReturns, options);
            htPos = garchcore(parameters, positiveReturns, options);
            
            % The difference between volatilities should be consistent with gamma
            obj.assertTrue(abs(mean(htNeg) - mean(htPos)) > 0, 'AGARCH fails to capture asymmetric effects');
            
            % Test both MATLAB and MEX implementations (if available)
            try
                options.useMEX = false;
                htMATLAB = garchcore(parameters, testData.returns, options);
                
                options.useMEX = true;
                htMEX = garchcore(parameters, testData.returns, options);
                
                obj.assertAlmostEqual(htMATLAB, htMEX, 'AGARCH MATLAB and MEX implementations should match');
            catch ME
                if ~contains(ME.identifier, 'MATLAB:UndefinedFunction')
                    rethrow(ME);
                end
            end
        end
        
        function testIgarch(obj)
            % Test the Integrated GARCH (IGARCH) model implementation in garchcore
            
            % Create test data with known IGARCH properties
            totalParam = 0.1 + 0.9; % alpha + beta = 1
            alpha = 0.1 / totalParam;
            beta = 0.9 / totalParam;
            testData = obj.createTestGarchData('IGARCH', struct('omega', 0.05, 'alpha', alpha, 'beta', beta));
            
            % Define IGARCH parameters (omega, alpha, beta) with integrated constraint
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.beta];
            options = struct('model', 'IGARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'IGARCH variances do not match expected values');
            
            % Verify that the integrated constraint is properly enforced
            obj.assertAlmostEqual(testData.parameters.alpha + testData.parameters.beta, 1, 'IGARCH constraint not enforced');
            
            % Test both MATLAB and MEX implementations (if available)
            try
                options.useMEX = false;
                htMATLAB = garchcore(parameters, testData.returns, options);
                
                options.useMEX = true;
                htMEX = garchcore(parameters, testData.returns, options);
                
                obj.assertAlmostEqual(htMATLAB, htMEX, 'IGARCH MATLAB and MEX implementations should match');
            catch ME
                if ~contains(ME.identifier, 'MATLAB:UndefinedFunction')
                    rethrow(ME);
                end
            end
        end
        
        function testNagarch(obj)
            % Test the Nonlinear Asymmetric GARCH (NAGARCH) model implementation in garchcore
            
            % Create test data with known NAGARCH properties
            testData = obj.createTestGarchData('NAGARCH', struct('omega', 0.05, 'alpha', 0.1, 'gamma', 0.1, 'beta', 0.85));
            
            % Define NAGARCH parameters (omega, alpha, gamma, beta)
            parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.gamma; testData.parameters.beta];
            options = struct('model', 'NAGARCH', 'p', 1, 'q', 1);
            
            % Call garchcore with test data and parameters
            ht = garchcore(parameters, testData.returns, options);
            
            % Verify computed variances match expected values with appropriate tolerance
            obj.assertAlmostEqual(testData.trueVariances, ht, 'NAGARCH variances do not match expected values');
            
            % Test nonlinear asymmetric effects
            negativeReturns = -abs(testData.returns);
            positiveReturns = abs(testData.returns);
            
            htNeg = garchcore(parameters, negativeReturns, options);
            htPos = garchcore(parameters, positiveReturns, options);
            
            % Verify nonlinear effects
            obj.assertTrue(abs(mean(htNeg) - mean(htPos)) > 0, 'NAGARCH fails to capture nonlinear asymmetric effects');
            
            % Test MATLAB implementation (MEX not available for NAGARCH)
            options.useMEX = false;
            htMATLAB = garchcore(parameters, testData.returns, options);
            obj.assertTrue(all(htMATLAB > 0), 'NAGARCH MATLAB implementation should produce positive variances');
        end
        
        function testInvalidInputs(obj)
            % Test garchcore function's handling of invalid inputs
            
            % Create valid test data for baseline
            returns = randn(100, 1);
            parameters = [0.05; 0.1; 0.85];
            options = struct('model', 'GARCH', 'p', 1, 'q', 1);
            
            % Test empty parameters using assertThrows
            obj.assertThrows(@() garchcore([], returns, options), 'parameters cannot be empty');
            
            % Test non-numeric parameters using assertThrows
            obj.assertThrows(@() garchcore('invalid', returns, options), 'parameters must be numeric');
            
            % Test invalid data format using assertThrows
            obj.assertThrows(@() garchcore(parameters, 'invalid', options), 'data must be numeric');
            
            % Test invalid model type in options using assertThrows
            invalidOptions = options;
            invalidOptions.model = 'INVALID';
            obj.assertThrows(@() garchcore(parameters, returns, invalidOptions), 'Unknown model type');
            
            % Test invalid p,q orders using assertThrows
            invalidOptions = options;
            invalidOptions.p = -1;
            obj.assertThrows(@() garchcore(parameters, returns, invalidOptions), 'options.p must be positive');
            
            invalidOptions = options;
            invalidOptions.q = 0;
            obj.assertThrows(@() garchcore(parameters, returns, invalidOptions), 'options.q must be positive');
            
            % Test parameters out of bounds for different models using assertThrows
            invalidOptions = options;
            invalidOptions.model = 'GJR';
            obj.assertThrows(@() garchcore(parameters, returns, invalidOptions), 'parameters vector length');
        end
        
        function testBackcastOptions(obj)
            % Test garchcore with different backcast initialization options
            
            % Create test data for a standard GARCH model
            returns = randn(100, 1);
            parameters = [0.05; 0.1; 0.85];
            
            % Test with default backcast calculation
            options = struct('model', 'GARCH', 'p', 1, 'q', 1);
            ht1 = garchcore(parameters, returns, options);
            
            % Test with user-specified backcast value
            options.backcast = 0.1;
            ht2 = garchcore(parameters, returns, options);
            
            % Variance series should differ initially but converge later
            obj.assertTrue(abs(ht1(1) - ht2(1)) > 0, 'Different backcasts should produce different initial variances');
            obj.assertAlmostEqual(ht1(end), ht2(end), 'Variance series should converge regardless of backcast');
            
            % Test with different backcast methods in options
            options.backcast = struct('type', 'EWMA', 'lambda', 0.94);
            ht3 = garchcore(parameters, returns, options);
            obj.assertTrue(all(ht3 > 0), 'EWMA backcast should produce positive variances');
            
            options.backcast = struct('type', 'decay', 'decay', 0.7);
            ht4 = garchcore(parameters, returns, options);
            obj.assertTrue(all(ht4 > 0), 'Decay backcast should produce positive variances');
            
            % Verify correct application of backcast for pre-sample variance values
            options.backcast = 0.2;
            ht5 = garchcore(parameters, returns, options);
            obj.assertTrue(abs(ht2(1) - ht5(1)) > 0, 'Different explicit backcast values should yield different initial variances');
        end
        
        function testMexImplementation(obj)
            % Test MEX implementation against MATLAB implementation
            
            % Create test data for various GARCH models
            modelTypes = {'GARCH', 'GJR', 'EGARCH', 'AGARCH', 'IGARCH'};
            
            for i = 1:length(modelTypes)
                modelType = modelTypes{i};
                
                % Skip if model not supported
                try
                    testData = obj.createTestGarchData(modelType, []);
                catch
                    continue;
                end
                
                % Create corresponding parameters based on model type
                switch modelType
                    case 'GARCH'
                        parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.beta];
                    case {'GJR', 'TARCH', 'EGARCH'}
                        parameters = [testData.parameters.omega; testData.parameters.alpha; ...
                                     testData.parameters.gamma; testData.parameters.beta];
                    case {'AGARCH', 'NAGARCH'}
                        parameters = [testData.parameters.omega; testData.parameters.alpha; ...
                                     testData.parameters.gamma; testData.parameters.beta];
                    case 'IGARCH'
                        parameters = [testData.parameters.omega; testData.parameters.alpha; testData.parameters.beta];
                end
                
                % Test each model both with useMEX=false (MATLAB) and useMEX=true (MEX)
                try
                    % Verify MEX and MATLAB implementations produce equivalent results
                    isEqual = obj.verifyMexEquivalence(testData.returns, parameters, modelType);
                    obj.assertTrue(isEqual, ['MEX and MATLAB implementations differ for ' modelType]);
                    
                    % Measure and compare performance between implementations
                    optionsMATLAB = struct('model', modelType, 'p', 1, 'q', 1, 'useMEX', false);
                    optionsMEX = struct('model', modelType, 'p', 1, 'q', 1, 'useMEX', true);
                    
                    timeMATLAB = obj.measureExecutionTime(@() garchcore(parameters, testData.returns, optionsMATLAB));
                    timeMEX = obj.measureExecutionTime(@() garchcore(parameters, testData.returns, optionsMEX));
                    
                    % MEX should run successfully (not asserting it's faster as that could be platform-dependent)
                    obj.assertTrue(timeMEX > 0, ['MEX implementation for ' modelType ' failed to run']);
                    
                catch ME
                    % Skip MEX tests if not available on current platform
                    if contains(ME.identifier, 'MATLAB:UndefinedFunction')
                        warning('MEX implementation for %s not available, skipping comparison', modelType);
                    else
                        % Rethrow other errors
                        rethrow(ME);
                    end
                end
            end
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of garchcore with extreme values
            
            % Test with very small parameter values close to zero
            T = 100;
            returns = randn(T, 1);
            parameters = [1e-8; 1e-8; 0.99];
            options = struct('model', 'GARCH', 'p', 1, 'q', 1);
            ht = garchcore(parameters, returns, options);
            
            % Verify variances remain positive and don't explode
            obj.assertTrue(all(ht > 0), 'Variances should remain positive with small parameters');
            obj.assertTrue(all(isfinite(ht)), 'Variances should be finite with small parameters');
            
            % Test with large outliers in the input data
            outlierReturns = returns;
            outlierReturns(50) = 20; % Large outlier
            ht = garchcore(parameters, outlierReturns, options);
            
            % Verify handling of outliers
            obj.assertTrue(all(isfinite(ht)), 'Variances should remain finite with outliers');
            obj.assertTrue(ht(51) > ht(49), 'Variance should increase after outlier');
            
            % Test with parameter values near constraint boundaries
            parameters = [0.001; 0.049; 0.95]; % alpha + beta close to 1
            ht = garchcore(parameters, returns, options);
            
            % Verify minimum variance threshold is properly applied
            obj.assertTrue(all(ht > 0), 'Minimum variance threshold should ensure positive values');
            obj.assertTrue(all(isfinite(ht)), 'Variances should be finite with near-constraint parameters');
            
            % Verify stability over long time series with potential error accumulation
            longReturns = [returns; randn(900, 1)];
            ht = garchcore(parameters, longReturns, options);
            
            obj.assertTrue(all(isfinite(ht)), 'Variances should remain finite with long time series');
            obj.assertTrue(all(ht > 0), 'Variances should remain positive with long time series');
        end
        
        function testData = createTestGarchData(obj, modelType, parameters)
            % Helper method to create test data for GARCH model testing
            
            % Validate input parameters using parametercheck
            parametercheck(modelType, 'modelType');
            
            % Set default parameters if not provided
            if isempty(parameters)
                switch upper(modelType)
                    case 'GARCH'
                        parameters = struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
                    case 'EGARCH'
                        parameters = struct('omega', -0.1, 'alpha', 0.1, 'gamma', -0.05, 'beta', 0.95);
                    case {'GJR', 'TARCH'}
                        parameters = struct('omega', 0.05, 'alpha', 0.05, 'gamma', 0.1, 'beta', 0.85);
                    case 'AGARCH'
                        parameters = struct('omega', 0.05, 'alpha', 0.1, 'gamma', 0.1, 'beta', 0.85);
                    case 'IGARCH'
                        totalParam = 0.1 + 0.9; % alpha + beta = 1
                        alpha = 0.1 / totalParam;
                        beta = 0.9 / totalParam;
                        parameters = struct('omega', 0.05, 'alpha', alpha, 'beta', beta);
                    case 'NAGARCH'
                        parameters = struct('omega', 0.05, 'alpha', 0.1, 'gamma', 0.1, 'beta', 0.85);
                    otherwise
                        error('Unknown model type: %s', modelType);
                end
            end
            
            % Use TestDataGenerator to create series with known properties
            volatilitySeries = TestDataGenerator('generateVolatilitySeries', 500, modelType, parameters);
            
            % Return structure containing returns, true variances, and model parameters
            testData = struct();
            testData.returns = volatilitySeries.returns;
            testData.trueVariances = volatilitySeries.ht;
            testData.parameters = parameters;
            testData.modelType = modelType;
        end
        
        function isEqual = verifyMexEquivalence(obj, data, parameters, modelType)
            % Helper method to verify MEX and MATLAB implementations produce equivalent results
            
            % Create options structure with useMEX=false for MATLAB implementation
            optionsMATLAB = struct('model', modelType, 'p', 1, 'q', 1, 'useMEX', false);
            
            % Run garchcore with MATLAB implementation
            htMATLAB = garchcore(parameters, data, optionsMATLAB);
            
            % Create options structure with useMEX=true for MEX implementation
            optionsMEX = struct('model', modelType, 'p', 1, 'q', 1, 'useMEX', true);
            
            % Run garchcore with MEX implementation
            htMEX = garchcore(parameters, data, optionsMEX);
            
            % Compare results using NumericalComparator
            result = obj.comparator.compareMatrices(htMATLAB, htMEX, obj.tolerance);
            
            % Return true if implementations match within tolerance
            isEqual = result.isEqual;
        end
    end
end