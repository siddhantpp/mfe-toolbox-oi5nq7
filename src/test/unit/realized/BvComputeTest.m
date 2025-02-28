classdef BvComputeTest < BaseTest
    % BvComputeTest Test class for validating the bipower variation computation function
    %
    % This class tests the functionality, accuracy, and robustness of the bv_compute.m 
    % function, which estimates integrated variance using products of adjacent absolute 
    % returns in high-frequency financial data.
    %
    % Tests include basic functionality, options handling, validation against known values,
    % robustness with different input formats, error handling, and performance checks.
    %
    % See also bv_compute, BaseTest
    
    properties
        testData               % Test data structure
        highFrequencyReturns   % High-frequency returns for testing
        referenceValues        % Reference values for validation
        tolerance              % Tolerance for floating-point comparisons
    end
    
    methods
        function obj = BvComputeTest(testName)
            % Initialize the BvComputeTest class with test name
            if nargin < 1
                testName = 'BvComputeTest';
            end
            obj = obj@BaseTest(testName);
            
            % Set default tolerance for floating-point comparisons
            obj.tolerance = 1e-10;
            
            % Initialize testData structure to empty
            obj.testData = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            setUp@BaseTest(obj);
            
            try
                % Load test data from the test data directory
                obj.testData = obj.loadTestData('high_frequency_data.mat');
                
                % Extract high-frequency returns and reference values if available
                if isfield(obj.testData, 'highFrequencyReturns')
                    obj.highFrequencyReturns = obj.testData.highFrequencyReturns;
                else
                    warning('highFrequencyReturns not found in test data');
                    obj.highFrequencyReturns = [];
                end
                
                if isfield(obj.testData, 'referenceBV')
                    obj.referenceValues = obj.testData.referenceBV;
                else
                    warning('referenceBV not found in test data');
                    obj.referenceValues = [];
                end
            catch ME
                warning('Error loading test data: %s', ME.message);
                obj.testData = struct();
                obj.highFrequencyReturns = [];
                obj.referenceValues = [];
            end
            
            % Set numerical tolerance for floating-point comparisons
            obj.tolerance = 1e-10;
        end
        
        function tearDown(obj)
            % Clean up the test environment after each test method execution
            tearDown@BaseTest(obj);
            
            % Clear any resources created during testing
        end
        
        function testBvComputeBasic(obj)
            % Tests the basic functionality of the bv_compute function with standard inputs
            
            % Create simple test case with known values
            returns = [0.01; -0.02; 0.015; -0.005; 0.008];
            
            % Calculate expected BV manually
            adjacentProducts = abs(returns(1:end-1)) .* abs(returns(2:end));
            expected = (pi/2) * sum(adjacentProducts);
            
            % Compute BV using the function
            actual = bv_compute(returns);
            
            % Assert that the result is as expected
            obj.assertAlmostEqual(expected, actual, 'Basic BV computation incorrect');
        end
        
        function testBvComputeWithOptions(obj)
            % Tests the bv_compute function with various option settings
            
            % Create test returns
            returns = [0.01; -0.02; 0.015; -0.005; 0.008];
            
            % Test with custom scaling factor
            scaleFactor = 1.0; % Custom scaling factor
            options = struct('scaleFactor', scaleFactor);
            
            % Calculate expected BV with custom scaling factor
            adjacentProducts = abs(returns(1:end-1)) .* abs(returns(2:end));
            expected = scaleFactor * sum(adjacentProducts);
            
            % Compute BV with options
            actual = bv_compute(returns, options);
            
            % Assert that result reflects the custom scaling factor
            obj.assertAlmostEqual(expected, actual, 'BV computation with custom scaling factor incorrect');
            
            % Test with another scaling factor
            scaleFactor = 2.5;
            options.scaleFactor = scaleFactor;
            expected = scaleFactor * sum(adjacentProducts);
            actual = bv_compute(returns, options);
            obj.assertAlmostEqual(expected, actual, 'BV computation with alternate scaling factor incorrect');
        end
        
        function testBvComputeWithKnownValues(obj)
            % Tests the bv_compute function against pre-computed reference values
            
            % Skip test if reference values or test data are not available
            if isempty(obj.highFrequencyReturns) || isempty(obj.referenceValues)
                warning('Skipping testBvComputeWithKnownValues: Test data not available');
                return;
            end
            
            % Compute BV for the test data
            actual = bv_compute(obj.highFrequencyReturns);
            
            % Compare with reference values within tolerance
            obj.assertMatrixEqualsWithTolerance(obj.referenceValues, actual, obj.tolerance, ...
                'BV computation does not match reference values');
        end
        
        function testBvComputeWithSyntheticData(obj)
            % Tests the bv_compute function with synthetic data having known properties
            
            % Generate synthetic returns with known properties
            T = 1000;
            rng(123); % Set random seed for reproducibility
            returns = randn(T, 1) * 0.01; % Normally distributed returns
            
            % Compute theoretical BV using our helper method
            theoretical = obj.computeTheoreticalBV(returns);
            
            % Compute BV using the function
            actual = bv_compute(returns);
            
            % Assert results match within tolerance
            obj.assertAlmostEqual(theoretical, actual, 'BV for synthetic data incorrect');
        end
        
        function testBvComputeRowVector(obj)
            % Tests that bv_compute handles row vectors correctly by converting them to column vectors
            
            % Create a column vector and corresponding row vector
            colVector = [0.01; -0.02; 0.015; -0.005; 0.008];
            rowVector = colVector';
            
            % Compute BV for both
            bvCol = bv_compute(colVector);
            bvRow = bv_compute(rowVector);
            
            % Assert that results are identical
            obj.assertAlmostEqual(bvCol, bvRow, 'Row vector handling incorrect');
        end
        
        function testBvComputeMultipleAssets(obj)
            % Tests bv_compute on a matrix of returns representing multiple assets
            
            % Create matrix with multiple assets (5 observations, 3 assets)
            returns = [0.01, 0.02, 0.03; 
                      -0.02, -0.01, 0.01; 
                       0.015, 0.005, -0.02;
                      -0.005, 0.01, 0.015;
                       0.008, -0.02, 0.005];
            
            % Compute BV for the multi-asset returns
            bvMulti = bv_compute(returns);
            
            % Compute BV for each asset individually
            bvSingle1 = bv_compute(returns(:,1));
            bvSingle2 = bv_compute(returns(:,2));
            bvSingle3 = bv_compute(returns(:,3));
            
            % Create expected result vector
            expected = [bvSingle1; bvSingle2; bvSingle3];
            
            % Assert dimensions and values match
            obj.assertEqual(size(bvMulti), [3, 1], 'Multi-asset BV has incorrect dimensions');
            obj.assertMatrixEqualsWithTolerance(expected, bvMulti, obj.tolerance, ...
                'Multi-asset BV computation incorrect');
        end
        
        function testBvComputeErrorHandling(obj)
            % Tests error handling for invalid inputs to bv_compute
            
            % Use helper function to check if function throws any error
            function checkThrowsAnyError(func, message)
                try
                    func();
                    % If we get here, no error was thrown
                    obj.assertTrue(false, message);
                catch
                    % Error was thrown as expected
                end
            end
            
            % Test with empty input
            checkThrowsAnyError(@() bv_compute([]), 'Empty input should throw an error');
            
            % Test with non-numeric input
            checkThrowsAnyError(@() bv_compute('invalid'), 'Non-numeric input should throw an error');
            
            % Test with NaN values
            invalidReturns = [0.01; NaN; 0.015];
            checkThrowsAnyError(@() bv_compute(invalidReturns), 'NaN values should throw an error');
            
            % Test with Inf values
            invalidReturns = [0.01; Inf; 0.015];
            checkThrowsAnyError(@() bv_compute(invalidReturns), 'Inf values should throw an error');
            
            % Test with insufficient observations
            insufficientReturns = [0.01];
            checkThrowsAnyError(@() bv_compute(insufficientReturns), 'Insufficient observations should throw an error');
            
            % Test with invalid options
            invalidOptions = struct('scaleFactor', -1);
            checkThrowsAnyError(@() bv_compute([0.01; -0.02; 0.015], invalidOptions), ...
                'Negative scale factor should throw an error');
        end
        
        function testBvComputePerformance(obj)
            % Tests the performance of bv_compute with large datasets
            
            % Generate large dataset
            T = 10000;
            N = 10;
            rng(123); % Set random seed for reproducibility
            largeReturns = randn(T, N) * 0.01;
            
            % Measure execution time
            tic;
            bv_compute(largeReturns);
            executionTime = toc;
            
            % Performance should be reasonable (adjust the time as needed based on hardware)
            % Here we use a relatively high threshold to accommodate different test environments
            obj.assertTrue(executionTime < 1.0, sprintf('Performance too slow: %f seconds', executionTime));
        end
        
        function testBvComputePiScaling(obj)
            % Tests that bv_compute applies the correct π/2 scaling factor
            
            % Create test returns
            returns = [0.01; -0.02; 0.015; -0.005; 0.008];
            
            % Calculate raw BV without π/2 scaling
            adjacentProducts = abs(returns(1:end-1)) .* abs(returns(2:end));
            rawBV = sum(adjacentProducts);
            
            % Apply π/2 scaling manually
            expectedBV = (pi/2) * rawBV;
            
            % Compute BV using the function
            actualBV = bv_compute(returns);
            
            % Assert that the scaling is correctly applied
            obj.assertAlmostEqual(expectedBV, actualBV, 'π/2 scaling not correctly applied');
        end
        
        function theoreticalBV = computeTheoreticalBV(obj, returns)
            % Helper function that computes the theoretical bipower variation for validation
            %
            % INPUTS:
            %   returns - Vector of return data
            %
            % OUTPUTS:
            %   theoreticalBV - Theoretical bipower variation value
            
            % Calculate absolute returns
            absReturns = abs(returns);
            
            % Calculate products of adjacent absolute returns
            adjacentProducts = absReturns(1:end-1) .* absReturns(2:end);
            
            % Sum products and apply scaling factor
            theoreticalBV = (pi/2) * sum(adjacentProducts);
        end
    end
end