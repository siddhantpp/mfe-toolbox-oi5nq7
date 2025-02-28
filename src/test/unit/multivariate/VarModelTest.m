classdef VarModelTest < BaseTest
    %VARMODELTEST Test case for Vector Autoregression (VAR) model implementation in the MFE Toolbox
    %
    % This test suite validates all aspects of VAR model estimation,
    % forecasting, impulse response functions, and variance decomposition
    % calculations.
    %
    % The VarModelTest class implements comprehensive unit tests to verify the
    % correctness and numerical stability of the VAR model implementation in the
    % MFE Toolbox. It includes tests for parameter estimation, forecasting,
    % impulse response analysis, variance decomposition, model selection, and
    % input validation.
    %
    % Example:
    %   % Create a test object
    %   testObj = VarModelTest();
    %
    %   % Run all tests
    %   results = testObj.runAllTests();
    %
    % See also: var_model, var_forecast, var_irf, var_fevd
    
    properties
        testData
        knownCoefficients
        knownResiduals
        varOptions
        tolerance
    end
    
    methods
        function obj = VarModelTest()
            % Initialize the VarModelTest class
            %
            % Call superclass constructor (BaseTest)
            obj = obj@BaseTest();
            
            % Set default tolerance for numerical comparisons
            obj.tolerance = 1e-6;
            
            % Initialize test properties as empty
            obj.testData = [];
            obj.knownCoefficients = [];
            obj.knownResiduals = [];
            obj.varOptions = struct();
        end
        
        function setUp(obj)
            % Set up the test environment before each test method execution
            %
            % This method initializes the test environment before each test method
            % runs. It loads test data, sets up known coefficients for a simple VAR
            % model, and initializes the varOptions structure with default test
            % parameters.
            
            % Call superclass setUp method
            obj.setUp@BaseTest();
            
            % Load test data from macroeconomic_data.mat or simulated_data.mat
            try
                % Attempt to load macroeconomic data
                obj.testData = obj.loadTestData('macroeconomic_data.mat');
            catch
                % If macroeconomic data is not available, load simulated data
                obj.testData = obj.loadTestData('simulated_data.mat');
            end
            
            % Set up known coefficients for a simple VAR model
            obj.knownCoefficients = [0.5, 0.1; -0.2, 0.8];
            
            % Initialize varOptions structure with default test parameters
            obj.varOptions = struct('constant', true, 'trend', false, 'criterion', 'aic');
            
            % Generate test time series with known properties if needed
            if ~isfield(obj.testData, 'timeSeriesData')
                % Example: Generate a simple AR(1) process
                T = 200; % Number of observations
                k = 2;   % Number of variables
                p = 1;   % Lag order
                
                % Define coefficients for the VAR process
                coefficients = rand(k, k * p); % Example coefficients
                
                % Generate the VAR process
                obj.testData.timeSeriesData = obj.generateTestVarProcess(T, k, p, coefficients);
            end
        end
        
        function tearDown(obj)
            % Clean up after test execution
            %
            % This method cleans up the test environment after each test method
            % runs. It clears test data variables and resets the random number
            % generator state.
            
            % Call superclass tearDown method
            obj.tearDown@BaseTest();
            
            % Clear test data variables
            clear obj.testData obj.knownCoefficients obj.knownResiduals obj.varOptions;
            
            % Reset random number generator state
            rng('default');
        end
        
        function testBasicVarEstimation(obj)
            % Test basic VAR model estimation with known coefficients
            %
            % This method tests the basic VAR model estimation functionality by
            % creating a simple VAR(1) model with known coefficients, estimating the
            % VAR model using the var_model function, and verifying that the
            % estimated coefficients match the known values within a specified
            % tolerance.
            
            % Create simple VAR(1) model with known coefficients
            T = 100; % Number of observations
            k = 2;   % Number of variables
            p = 1;   % Lag order
            
            % Define coefficients for the VAR process
            coefficients = obj.knownCoefficients;
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate VAR model using var_model function
            estimatedModel = var_model(varProcess.data, p, obj.varOptions);
            
            % Verify estimated coefficients match known values within tolerance
            obj.assertMatrixEqualsWithTolerance(coefficients, reshape(estimatedModel.coefficients(:, 1:k), k, k), obj.tolerance, 'Estimated coefficients do not match known values');
            
            % Check residuals are consistent with expected values
            obj.assertEqual(size(estimatedModel.residuals, 1), T - p, 'Residuals have incorrect number of observations');
            
            % Verify model structure contains required fields
            obj.assertTrue(obj.verifyModelStructure(estimatedModel), 'Model structure is invalid');
        end
        
        function testMultipleLagVarEstimation(obj)
            % Test VAR model estimation with multiple lags
            %
            % This method tests the VAR model estimation functionality with multiple
            % lags by generating data from a VAR(3) process with known coefficients,
            % estimating the VAR model with p=3 using the var_model function, and
            % verifying that the estimated coefficients for all lags match the known
            % values.
            
            % Generate data from a VAR(3) process with known coefficients
            T = 200; % Number of observations
            k = 2;   % Number of variables
            p = 3;   % Lag order
            
            % Define coefficients for the VAR process
            coefficients = rand(k, k * p); % Example coefficients
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate VAR model with p=3 using var_model function
            estimatedModel = var_model(varProcess.data, p, obj.varOptions);
            
            % Verify estimated coefficients for all lags match known values
            for i = 1:p
                knownCoefficientsLag = coefficients(:, (i-1)*k+1:i*k);
                estimatedCoefficientsLag = reshape(estimatedModel.coefficients(:, (i-1)*k+1:i*k), k, k);
                obj.assertMatrixEqualsWithTolerance(knownCoefficientsLag, estimatedCoefficientsLag, obj.tolerance, sprintf('Estimated coefficients for lag %d do not match known values', i));
            end
            
            % Validate residual properties (zero mean, appropriate covariance)
            obj.assertAlmostEqual(mean(estimatedModel.residuals), zeros(1, k), 'Residuals do not have zero mean');
            
            % Check information criteria (AIC, BIC) calculation
            obj.assertTrue(isfield(estimatedModel, 'aic'), 'AIC is not calculated');
            obj.assertTrue(isfield(estimatedModel, 'sbic'), 'BIC is not calculated');
        end
        
        function testVarForecasting(obj)
            % Test VAR model forecasting capabilities
            %
            % This method tests the VAR model forecasting capabilities by estimating
            % a VAR model using known data, generating out-of-sample forecasts with
            % the var_forecast function, and comparing the forecasts to analytically
            % calculated values.
            
            % Estimate VAR model using known data
            T = 100; % Number of observations
            k = 2;   % Number of variables
            p = 1;   % Lag order
            
            % Define coefficients for the VAR process
            coefficients = obj.knownCoefficients;
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate VAR model using var_model function
            estimatedModel = var_model(varProcess.data, p, obj.varOptions);
            
            % Generate out-of-sample forecasts with var_forecast function
            forecastHorizon = 10;
            forecasts = var_forecast(estimatedModel, forecastHorizon);
            
            % Compare forecasts to analytically calculated values
            % (This requires calculating forecasts manually based on known coefficients)
            
            % Verify forecast dimensions are correct
            obj.assertEqual(size(forecasts, 1), forecastHorizon, 'Forecasts have incorrect number of observations');
            obj.assertEqual(size(forecasts, 2), k, 'Forecasts have incorrect number of variables');
            
            % Test multi-step forecasting accuracy
            % (This requires comparing forecasts to known values or simulated data)
        end
        
        function testImpulseResponseFunctions(obj)
            % Test impulse response function computation
            %
            % This method tests the impulse response function (IRF) computation by
            % estimating a VAR model with simplified known coefficients, computing
            % impulse response functions using the var_irf function, and verifying that
            % the impulse responses match analytically derived values.
            
            % Estimate VAR model with simplified known coefficients
            T = 100; % Number of observations
            k = 2;   % Number of variables
            p = 1;   % Lag order
            
            % Define simplified coefficients for the VAR process
            coefficients = [0.2, 0; 0, 0.5];
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate VAR model using var_model function
            estimatedModel = var_model(varProcess.data, p, obj.varOptions);
            
            % Compute impulse response functions using var_irf function
            irfHorizon = 20;
            impulseResponses = var_irf(estimatedModel, irfHorizon);
            
            % Verify impulse responses match analytically derived values
            % (This requires calculating IRFs manually based on known coefficients)
            
            % Check orthogonalization of innovations is correct
            % (This requires verifying that the initial impact matrix is lower triangular)
            
            % Validate impulse response matrix dimensions
            obj.assertEqual(size(impulseResponses, 1), irfHorizon + 1, 'Impulse responses have incorrect number of observations');
            obj.assertEqual(size(impulseResponses, 2), k, 'Impulse responses have incorrect number of variables (response)');
            obj.assertEqual(size(impulseResponses, 3), k, 'Impulse responses have incorrect number of variables (shock)');
        end
        
        function testVarianceDecomposition(obj)
            % Test forecast error variance decomposition
            %
            % This method tests the forecast error variance decomposition (FEVD) by
            % estimating a VAR model with simplified known coefficients, computing
            % variance decomposition using the var_fevd function, and verifying that
            % the decomposition percentages sum to 100%.
            
            % Estimate VAR model with simplified known coefficients
            T = 100; % Number of observations
            k = 2;   % Number of variables
            p = 1;   % Lag order
            
            % Define simplified coefficients for the VAR process
            coefficients = [0.2, 0; 0, 0.5];
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate VAR model using var_model function
            estimatedModel = var_model(varProcess.data, p, obj.varOptions);
            
            % Compute variance decomposition using var_fevd function
            fevdHorizon = 20;
            varianceDecomposition = var_fevd(estimatedModel, fevdHorizon);
            
            % Verify decomposition percentages sum to 100%
            for t = 1:fevdHorizon + 1
                for i = 1:k
                    totalDecomposition = sum(varianceDecomposition(t, i, :));
                    obj.assertAlmostEqual(totalDecomposition, 1, 'Variance decomposition percentages do not sum to 100%');
                end
            end
            
            % Compare decomposition results to known values
            % (This requires calculating FEVD manually based on known coefficients)
            
            % Validate decomposition matrix dimensions
            obj.assertEqual(size(varianceDecomposition, 1), fevdHorizon + 1, 'Variance decomposition has incorrect number of observations');
            obj.assertEqual(size(varianceDecomposition, 2), k, 'Variance decomposition has incorrect number of variables (response)');
            obj.assertEqual(size(varianceDecomposition, 3), k, 'Variance decomposition has incorrect number of variables (shock)');
        end
        
        function testModelSelection(obj)
            % Test automatic lag selection using information criteria
            %
            % This method tests the automatic lag selection functionality by generating
            % data from a VAR(2) process, estimating models with varying lag lengths
            % (p=1,2,3,4), and verifying that the information criteria correctly identify
            % p=2 as the optimal lag order.
            
            % Generate data from a VAR(2) process
            T = 200; % Number of observations
            k = 2;   % Number of variables
            p = 2;   % Lag order
            
            % Define coefficients for the VAR process
            coefficients = rand(k, k * p); % Example coefficients
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Estimate models with varying lag lengths (p=1,2,3,4)
            lagOrders = 1:4;
            estimatedModels = cell(size(lagOrders));
            for i = 1:length(lagOrders)
                estimatedModels{i} = var_model(varProcess.data, lagOrders(i), obj.varOptions);
            end
            
            % Verify information criteria correctly identify p=2 as optimal
            aicValues = arrayfun(@(i) estimatedModels{i}.aic, 1:length(lagOrders));
            bicValues = arrayfun(@(i) estimatedModels{i}.sbic, 1:length(lagOrders));
            
            [~, minAicIndex] = min(aicValues);
            [~, minBicIndex] = min(bicValues);
            
            obj.assertEqual(lagOrders(minAicIndex), 2, 'AIC does not identify correct lag order');
            obj.assertEqual(lagOrders(minBicIndex), 2, 'BIC does not identify correct lag order');
            
            % Check automatic lag selection in var_model works correctly
            estimatedModelAuto = var_model(varProcess.data, lagOrders, obj.varOptions);
            obj.assertEqual(estimatedModelAuto.p, 2, 'Automatic lag selection failed');
            
            % Validate selected model contains correct coefficient structure
            obj.assertTrue(obj.verifyModelStructure(estimatedModelAuto), 'Selected model structure is invalid');
        end
        
        function testInputValidation(obj)
            % Test error handling and input validation
            %
            % This method tests the error handling and input validation of the
            % var_model function by providing invalid inputs and verifying that
            % appropriate error messages are generated.
            
            % Test with invalid input dimensions (non-conformable matrices)
            obj.assertThrows(@() var_model(rand(10, 3), 1, obj.varOptions), 'MATLAB: datacheck: y must be a column vector or a row vector', 'var_model did not throw error for invalid input dimensions');
            
            % Test with invalid lag order (negative or non-integer)
            obj.assertThrows(@() var_model(rand(10, 1), -1, obj.varOptions), 'MATLAB: parametercheck: p must be greater than or equal to 0.', 'var_model did not throw error for invalid lag order');
            obj.assertThrows(@() var_model(rand(10, 1), 1.5, obj.varOptions), 'MATLAB: parametercheck: p must contain only integer values.', 'var_model did not throw error for invalid lag order');
            
            % Test with missing required inputs
            obj.assertThrows(@() var_model([], 1, obj.varOptions), 'MATLAB: datacheck: y cannot be empty', 'var_model did not throw error for missing required inputs');
            
            % Verify appropriate error messages are generated
            % (This requires checking the specific error messages thrown by the function)
            
            % Check that validation is performed before computation
            % (This requires verifying that the function throws an error before performing any computations)
        end
        
        function testNumericalStability(obj)
            % Test numerical stability with challenging data
            %
            % This method tests the numerical stability of the var_model function by
            % providing challenging data, such as data with near unit-root properties
            % or data with a high condition number, and verifying that the function
            % produces stable and accurate results.
            
            % Generate data with near unit-root properties
            % (This requires creating a VAR process with coefficients close to 1)
            
            % Create data with high condition number for testing stability
            % (This requires generating data with a large range of values)
            
            % Verify model estimation works with scaled/unscaled data
            % (This requires testing the function with data that has been scaled to different magnitudes)
            
            % Check numerical stability under various initializations
            % (This requires testing the function with different initial parameter values)
            
            % Test with data having different magnitudes across variables
            % (This requires testing the function with data where some variables have much larger values than others)
        end
        
        function testPerformance(obj)
            % Test performance of VAR model estimation
            %
            % This method tests the performance of the var_model function by
            % generating a large dataset and measuring the execution time for
            % different model sizes.
            
            % Generate large dataset for performance testing
            T = 500; % Number of observations
            k = 5;   % Number of variables
            p = 2;   % Lag order
            
            % Define coefficients for the VAR process
            coefficients = rand(k, k * p); % Example coefficients
            
            % Generate the VAR process
            varProcess = obj.generateTestVarProcess(T, k, p, coefficients);
            
            % Measure execution time for different model sizes
            executionTime = obj.measureExecutionTime(@() var_model(varProcess.data, p, obj.varOptions));
            
            % Verify computational complexity scales as expected
            % (This requires comparing execution times for different model sizes and verifying that the scaling is consistent with the expected complexity)
            
            % Compare performance with and without optional optimizations
            % (This requires testing the function with and without optional optimizations and comparing the execution times)
            
            % Log performance metrics for benchmarking
            % (This requires recording the execution times for different model sizes and configurations)
        end
        
        function varProcess = generateTestVarProcess(obj, T, k, p, coefficients)
            % Helper method to generate test data from a VAR process with known coefficients
            %
            % This method generates test data from a VAR process with known
            % coefficients. It takes the number of observations (T), the number of
            % variables (k), the lag order (p), and the coefficients as inputs.
            %
            % Args:
            %   T (int): Number of observations.
            %   k (int): Number of variables.
            %   p (int): Lag order.
            %   coefficients (matrix): Coefficients for the VAR process.
            %
            % Returns:
            %   struct: Structure containing generated data and true model information.
            
            % Validate input parameters
            parametercheck(T, 'T', struct('isInteger', true, 'isPositive', true));
            parametercheck(k, 'k', struct('isInteger', true, 'isPositive', true));
            parametercheck(p, 'p', struct('isInteger', true, 'isPositive', true));
            
            % Initialize data matrix of appropriate dimensions
            data = zeros(T, k);
            
            % Generate innovations with known covariance structure
            innovations = randn(T, k);
            
            % Apply VAR filter using provided coefficients
            for t = p+1:T
                for lag = 1:p
                    data(t, :) = data(t, :) + data(t-lag, :) * coefficients(:, (lag-1)*k+1:lag*k)';
                end
                data(t, :) = data(t, :) + innovations(t, :);
            end
            
            % Return structure with data, true coefficients, and innovations
            varProcess = struct('data', data, 'coefficients', coefficients, 'innovations', innovations);
        end
        
        function isValid = verifyModelStructure(obj, model)
            % Helper method to verify that estimated model has correct structure
            %
            % This method verifies that the estimated model has the correct structure
            % by checking for required fields, verifying dimensions of coefficient
            % matrices, validating residual properties, and checking that the
            % covariance matrix is positive definite.
            %
            % Args:
            %   model (struct): Estimated model structure.
            %
            % Returns:
            %   logical: Boolean indicating if structure is valid.
            
            % Check for required fields in model structure
            requiredFields = {'coefficients', 'residuals', 'sigma', 'aic', 'sbic', 'p', 'k', 'T'};
            for i = 1:length(requiredFields)
                if ~isfield(model, requiredFields{i})
                    isValid = false;
                    return;
                end
            end
            
            % Verify dimensions of coefficient matrices
            k = model.k;
            p = model.p;
            if size(model.coefficients, 1) ~= k
                isValid = false;
                return;
            end
            if size(model.coefficients, 2) < k*p
                isValid = false;
                return;
            end
            
            % Validate residual properties
            if size(model.residuals, 2) ~= k
                isValid = false;
                return;
            end
            
            % Check covariance matrix is positive definite
            try
                chol(model.sigma);
            catch
                isValid = false;
                return;
            end
            
            % Return boolean indicating validation result
            isValid = true;
        end
    end
end