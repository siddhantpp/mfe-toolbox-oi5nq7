classdef BackcastTest < BaseTest
    % BACKCASTTEST Test class for the backcast utility function that validates initial variance estimation for GARCH models
    %
    % This class provides comprehensive unit tests for the backcast function which is used to 
    % calculate initial variance estimates (backcasts) for GARCH model estimation. It tests 
    % various backcasting methods including default variance, EWMA, decay factor, and fixed 
    % value approaches.
    %
    % The test suite validates:
    % - Correctness of backcast calculations for different methods
    % - Proper handling of multi-column input data
    % - Error handling for invalid inputs and options
    % - Numerical stability with extreme values and zero-variance cases
    % - Consistency of results with real financial data
    
    properties
        simulatedData      % Random data for testing with known properties
        financialReturns   % Real financial returns data for realistic testing
        comparator         % NumericalComparator for comparing results
        defaultTolerance   % Default tolerance for numerical comparisons
    end
    
    methods
        function obj = BackcastTest(testName)
            % Initialize a new BackcastTest instance with test name
            %
            % INPUTS:
            %   testName - Name for the test case (char)
            
            % Call superclass constructor
            obj = obj@BaseTest(testName);
            
            % Set default tolerance for numerical comparisons
            obj.defaultTolerance = 1e-10;
            
            % Create a NumericalComparator instance
            obj.comparator = NumericalComparator();
        end
        
        function setUp(obj)
            % Prepare test environment before each test method runs
            
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Generate simulated data for testing with known properties
            rng(123); % Set random seed for reproducibility
            obj.simulatedData = randn(1000, 1) * 0.02; % Simulated returns
            
            % Load financial returns data from test file
            try
                % Attempt to load real financial data
                loadedData = load('voldata.mat');
                if isfield(loadedData, 'returns')
                    obj.financialReturns = loadedData.returns;
                else
                    % Create synthetic returns if field not found
                    obj.financialReturns = obj.simulatedData;
                end
            catch
                % If loading fails, use simulated data
                obj.financialReturns = obj.simulatedData;
            end
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method completes
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
            
            % Clear test data variables
            obj.simulatedData = [];
            obj.financialReturns = [];
        end
        
        function testDefaultBackcast(obj)
            % Tests the default backcast method (unconditional variance)
            
            % Call backcast function with default options
            result = backcast(obj.simulatedData);
            
            % Calculate expected backcast value (sample variance)
            expected = var(obj.simulatedData);
            
            % Compare result with expected value
            obj.assertAlmostEqual(result, expected, 'Default backcast should equal sample variance');
            
            % Verify result is positive and numerically stable
            obj.assertTrue(result > 0, 'Backcast value should be positive');
            obj.assertTrue(isfinite(result), 'Backcast value should be finite');
        end
        
        function testEWMABackcast(obj)
            % Tests the EWMA (Exponentially Weighted Moving Average) backcast method
            
            % Create options structure for EWMA
            options = struct('type', 'ewma', 'lambda', 0.94);
            
            % Call backcast function with EWMA options
            result = backcast(obj.simulatedData, options);
            
            % Manually calculate expected EWMA backcast
            T = length(obj.simulatedData);
            lambda = options.lambda;
            squaredData = obj.simulatedData.^2;
            weights = (1-lambda) * lambda.^(T-1:-1:0)';
            expected = weights' * squaredData;
            
            % Compare result with expected value
            obj.assertAlmostEqual(result, expected, 'EWMA backcast calculation is incorrect');
            
            % Test with different lambda values
            lambdaValues = [0.85, 0.90, 0.95];
            for i = 1:length(lambdaValues)
                options.lambda = lambdaValues(i);
                result = backcast(obj.simulatedData, options);
                
                % Recalculate expected value with new lambda
                weights = (1-options.lambda) * options.lambda.^(T-1:-1:0)';
                expected = weights' * squaredData;
                
                obj.assertAlmostEqual(result, expected, ...
                    sprintf('EWMA backcast with lambda=%.2f is incorrect', options.lambda));
            end
        end
        
        function testDecayBackcast(obj)
            % Tests the decay factor backcast method
            
            % Create options structure for decay method
            options = struct('type', 'decay', 'decay', 0.7);
            
            % Call backcast function with decay options
            result = backcast(obj.simulatedData, options);
            
            % Manually calculate expected decay backcast
            T = length(obj.simulatedData);
            decayFactor = options.decay;
            squaredData = obj.simulatedData.^2;
            weights = decayFactor.^((T-1):-1:0)';
            weights = weights / sum(weights); % Normalize weights
            expected = weights' * squaredData;
            
            % Compare result with expected value
            obj.assertAlmostEqual(result, expected, 'Decay backcast calculation is incorrect');
            
            % Test with different decay factors
            decayValues = [0.7, 0.8, 0.9];
            for i = 1:length(decayValues)
                options.decay = decayValues(i);
                result = backcast(obj.simulatedData, options);
                
                % Recalculate expected value with new decay factor
                weights = options.decay.^((T-1):-1:0)';
                weights = weights / sum(weights); % Normalize weights
                expected = weights' * squaredData;
                
                obj.assertAlmostEqual(result, expected, ...
                    sprintf('Decay backcast with decay=%.1f is incorrect', options.decay));
            end
        end
        
        function testFixedBackcast(obj)
            % Tests fixed value backcast method
            
            % Create options structure for fixed value
            fixedValue = 0.05;
            options = struct('type', 'fixed', 'value', fixedValue);
            
            % Call backcast function with fixed options
            result = backcast(obj.simulatedData, options);
            
            % Verify result exactly equals the fixed value specified
            obj.assertEqual(result, fixedValue, 'Fixed backcast should return the specified value');
            
            % Test with various fixed values
            testValues = [0.01, 1.0, 10.0];
            for i = 1:length(testValues)
                options.value = testValues(i);
                result = backcast(obj.simulatedData, options);
                
                obj.assertEqual(result, testValues(i), ...
                    sprintf('Fixed backcast with value=%.2f should return the specified value', testValues(i)));
            end
            
            % Test with zero or negative values (should handle appropriately)
            options.value = 0;
            obj.assertThrows(@() backcast(obj.simulatedData, options), ...
                'OPTIONS.value must be a positive scalar');
            
            options.value = -1;
            obj.assertThrows(@() backcast(obj.simulatedData, options), ...
                'OPTIONS.value must be a positive scalar');
        end
        
        function testMultiColumnInput(obj)
            % Tests backcast function with multi-column input data
            
            % Create a multi-column data matrix
            numColumns = 3;
            multiData = randn(500, numColumns) * 0.02;
            
            % Call backcast function with multi-column input
            result = backcast(multiData);
            
            % Verify result has same number of columns as input
            obj.assertEqual(length(result), numColumns, 'Result should have same number of columns as input');
            
            % Check each column result against individual column computation
            for i = 1:numColumns
                columnResult = backcast(multiData(:, i));
                obj.assertAlmostEqual(result(i), columnResult, ...
                    sprintf('Column %d backcast should match individual calculation', i));
            end
            
            % Verify vectorized processing works correctly with different methods
            % Test EWMA with multi-column data
            options = struct('type', 'ewma', 'lambda', 0.94);
            multiResult = backcast(multiData, options);
            
            for i = 1:numColumns
                singleResult = backcast(multiData(:, i), options);
                obj.assertAlmostEqual(multiResult(i), singleResult, ...
                    sprintf('EWMA: Column %d should match individual calculation', i));
            end
        end
        
        function testZeroVarianceHandling(obj)
            % Tests handling of zero variance input data
            
            % Create constant value data (zero variance)
            constantData = zeros(100, 1) + 1;  % All values are 1
            
            % Call backcast function with constant data
            result = backcast(constantData);
            
            % Verify appropriate small non-zero variance is returned
            obj.assertTrue(result > 0, 'Backcast should return a positive value for zero variance data');
            
            % Calculate expected fallback value (mean absolute value squared)
            expected = mean(abs(constantData))^2;
            obj.assertAlmostEqual(result, expected, 'Fallback for zero variance should be mean absolute value squared');
            
            % Test with near-zero variance data
            nearConstantData = ones(100, 1) + 1e-10 * randn(100, 1);
            result = backcast(nearConstantData);
            
            % Verify result is positive and reasonable
            obj.assertTrue(result > 0, 'Backcast should return a positive value for near-zero variance data');
            obj.assertTrue(isfinite(result), 'Backcast should be finite for near-zero variance data');
        end
        
        function testInvalidInputs(obj)
            % Tests error handling for invalid inputs
            
            % Test with empty array
            obj.assertThrows(@() backcast([]), 'data cannot be empty');
            
            % Test with non-numeric data
            obj.assertThrows(@() backcast('invalid'), 'data must be numeric');
            
            % Test with NaN values
            nanData = [1; 2; NaN; 4];
            obj.assertThrows(@() backcast(nanData), 'data cannot contain NaN');
            
            % Test with Inf values
            infData = [1; 2; Inf; 4];
            obj.assertThrows(@() backcast(infData), 'data cannot contain Inf');
        end
        
        function testInvalidOptions(obj)
            % Tests error handling for invalid options
            
            % Test with invalid backcast type
            invalidOptions = struct('type', 'invalidType');
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'Unknown backcast type');
            
            % Test with missing value for fixed type
            invalidOptions = struct('type', 'fixed');
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'OPTIONS.value must be provided');
            
            % Test with invalid fixed value (not a scalar)
            invalidOptions = struct('type', 'fixed', 'value', [1, 2]);
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'OPTIONS.value must be a positive scalar');
            
            % Test with invalid lambda for EWMA (out of range)
            invalidOptions = struct('type', 'ewma', 'lambda', 1.5);
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'LAMBDA must be between 0 and 1');
            
            invalidOptions = struct('type', 'ewma', 'lambda', -0.5);
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'LAMBDA must be between 0 and 1');
            
            % Test with invalid decay factor (out of range)
            invalidOptions = struct('type', 'decay', 'decay', -0.5);
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'DECAY must be between 0 and 1');
            
            invalidOptions = struct('type', 'decay', 'decay', 1.5);
            obj.assertThrows(@() backcast(obj.simulatedData, invalidOptions), ...
                'DECAY must be between 0 and 1');
        end
        
        function testBackcastConsistency(obj)
            % Tests consistency of backcast results with real financial data
            
            % Use real financial returns data from test data set
            data = obj.financialReturns;
            
            % Call backcast with different methods on same data
            defaultResult = backcast(data);
            ewmaOptions = struct('type', 'ewma', 'lambda', 0.94);
            ewmaResult = backcast(data, ewmaOptions);
            decayOptions = struct('type', 'decay', 'decay', 0.7);
            decayResult = backcast(data, decayOptions);
            
            % Verify results are within expected ranges for financial data
            % (positive and reasonable magnitude)
            obj.assertTrue(defaultResult > 0, 'Default backcast should be positive');
            obj.assertTrue(ewmaResult > 0, 'EWMA backcast should be positive');
            obj.assertTrue(decayResult > 0, 'Decay backcast should be positive');
            
            % Calculate sample variance for reference
            sampleVar = var(data);
            
            % Compare relationships between different method results
            % EWMA with lambda=0.94 typically gives more weight to recent observations
            % than simple variance, and should be in a reasonable range
            obj.assertTrue(abs(ewmaResult/sampleVar - 1) < 1, ...
                'EWMA backcast should be within reasonable range of sample variance');
            
            % Decay method should also be in reasonable range
            obj.assertTrue(abs(decayResult/sampleVar - 1) < 1, ...
                'Decay backcast should be within reasonable range of sample variance');
            
            % Verify stability across different decay factor values
            decayFactors = [0.5, 0.7, 0.9];
            prevResult = 0;
            
            for i = 1:length(decayFactors)
                options = struct('type', 'decay', 'decay', decayFactors(i));
                result = backcast(data, options);
                
                % Each result should be positive
                obj.assertTrue(result > 0, 'Backcast should be positive for all decay factors');
                
                % Higher decay factors give more weight to older observations
                if i > 1
                    % This relationship can vary based on data patterns
                    % Just verify results change with different decay factors
                    obj.assertTrue(abs(result - prevResult) > 0, ...
                        'Results should differ with different decay factors');
                end
                prevResult = result;
            end
        end
        
        function testBackcastNumericalStability(obj)
            % Tests numerical stability of backcast function with extreme values
            
            % Create test data with very small values
            smallData = randn(100, 1) * 1e-6;
            smallResult = backcast(smallData);
            
            % Verify function handles small values appropriately
            obj.assertTrue(smallResult > 0, 'Backcast should be positive for very small data');
            obj.assertTrue(isfinite(smallResult), 'Backcast should be finite for very small data');
            
            % Create test data with very large values
            largeData = randn(100, 1) * 1e6;
            largeResult = backcast(largeData);
            
            % Verify function handles large values appropriately
            obj.assertTrue(largeResult > 0, 'Backcast should be positive for very large data');
            obj.assertTrue(isfinite(largeResult), 'Backcast should be finite for very large data');
            
            % Check scaling relationship
            scaleFactor = 1e6;
            scaledSmallData = smallData * scaleFactor;
            
            % Default method (variance) should scale with square of data scaling
            scaledResult = backcast(scaledSmallData);
            expectedScaledResult = smallResult * scaleFactor^2;
            
            % Use comparator with appropriate tolerance for numerical precision
            result = obj.comparator.compareScalars(scaledResult / expectedScaledResult, 1, 1e-6);
            obj.assertTrue(result.isEqual, 'Backcast should scale proportionally to variance of data');
            
            % Test with alternating large and small values (numerical stability test)
            mixedData = zeros(100, 1);
            mixedData(1:2:end) = 1e6;
            mixedData(2:2:end) = 1e-6;
            
            mixedResult = backcast(mixedData);
            obj.assertTrue(mixedResult > 0, 'Backcast should be positive for mixed scale data');
            obj.assertTrue(isfinite(mixedResult), 'Backcast should be finite for mixed scale data');
        end
    end
end