classdef ARMAValidation < BaseTest
    % ARMAValidation Test class for validating ARMA/ARMAX model implementation in the MFE Toolbox
    %
    % This class provides comprehensive validation of time series modeling components,
    % verifying parameter estimation accuracy, forecasting capability, and numerical
    % stability for ARMA and ARMAX models with different configurations.
    %
    % The test suite validates:
    % - AR, MA, and ARMA parameter estimation accuracy
    % - ARMAX models with exogenous variables
    % - Forecasting accuracy with analytical and simulation methods
    % - Non-normal error distribution handling
    % - Numerical stability with challenging datasets
    %
    % See also: BaseTest, NumericalComparator, armaxfilter, armafor
    
    properties
        testData              % Structure containing test data
        numericTolerance      % Tolerance for numerical comparisons
        comparator            % NumericalComparator instance
        useRealData           % Flag to use real data or simulated data
    end
    
    methods
        function obj = ARMAValidation()
            % Initialize the ARMAValidation test class with appropriate test data
            % and configuration for validating ARMA/ARMAX models
            
            % Call parent constructor
            obj@BaseTest();
            
            % Set numerical tolerance for floating-point comparisons
            obj.numericTolerance = 1e-6;
            
            % Create numerical comparator for precise matrix comparisons
            obj.comparator = NumericalComparator();
            
            % Set flag to determine whether to use real financial data or simulated data
            obj.useRealData = false;
            
            % Load test data from MAT file if using real data
            if obj.useRealData
                obj.testData = obj.loadTestData('financial_returns.mat');
            end
        end
        
        function setUp(obj)
            % Prepare test environment before each test method execution
            %
            % Sets up the test environment by initializing test data structures,
            % loading financial returns data or generating simulated AR, MA and 
            % ARMA processes for testing.
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Initialize test data structure if not already done
            if ~isfield(obj.testData, 'initialized') || ~obj.testData.initialized
                if obj.useRealData
                    % Data already loaded in constructor
                    obj.testData.initialized = true;
                else
                    % Generate simulated test data
                    obj.testData = struct();
                    
                    % Sample size for simulated data
                    T = 1000;
                    
                    % Generate AR(2) process: y_t = 0.7*y_{t-1} - 0.2*y_{t-2} + ε_t
                    ar_params = [0.7; -0.2];
                    obj.testData.ar_process = obj.generateARMAProcess(ar_params, [], T, 1);
                    obj.testData.ar_params = ar_params;
                    
                    % Generate MA(2) process: y_t = ε_t + 0.4*ε_{t-1} + 0.2*ε_{t-2}
                    ma_params = [0.4; 0.2];
                    obj.testData.ma_process = obj.generateARMAProcess([], ma_params, T, 1);
                    obj.testData.ma_params = ma_params;
                    
                    % Generate ARMA(1,1) process: y_t = 0.5*y_{t-1} + ε_t + 0.3*ε_{t-1}
                    ar_params = [0.5];
                    ma_params = [0.3];
                    obj.testData.arma_process = obj.generateARMAProcess(ar_params, ma_params, T, 1);
                    obj.testData.arma_ar_params = ar_params;
                    obj.testData.arma_ma_params = ma_params;
                    
                    % Generate exogenous variables for ARMAX testing
                    obj.testData.exog_vars = randn(T, 2);
                    obj.testData.exog_params = [0.5; -0.3];
                    
                    % Mark as initialized
                    obj.testData.initialized = true;
                end
            end
        end
        
        function tearDown(obj)
            % Clean up test environment after each test method execution
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
            
            % Additional cleanup if needed
        end
        
        function testAREstimation(obj)
            % Test autoregressive (AR) model parameter estimation accuracy
            
            % Configuration for this test
            ar_order = 2;
            include_constant = true;
            
            % Get test data
            if obj.useRealData
                test_data = obj.testData.returns;
                true_params = []; % Unknown for real data
            else
                test_data = obj.testData.ar_process;
                true_params = obj.testData.ar_params;
            end
            
            % Set up options for armaxfilter
            options = struct('p', ar_order, 'q', 0, 'constant', include_constant);
            
            % Estimate AR model using armaxfilter
            results = armaxfilter(test_data, [], options);
            
            % Extract estimated AR parameters
            est_ar_params = results.parameters(include_constant+1:include_constant+ar_order);
            
            % If using simulated data, compare with true parameters
            if ~obj.useRealData
                % Compare estimated parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_params, est_ar_params, obj.numericTolerance, ...
                    'AR parameter estimates do not match true values');
                
                % Check if standard errors are reasonable (should be < 0.1 for good estimation)
                std_errors = results.standardErrors(include_constant+1:include_constant+ar_order);
                obj.assertTrue(all(std_errors < 0.1), 'AR standard errors are too large');
                
                % Verify residuals have expected properties (mean close to 0, variance close to 1)
                mean_resid = mean(results.residuals);
                var_resid = var(results.residuals);
                obj.assertEqualsWithTolerance(0, mean_resid, 0.05, 'Residual mean is not close to zero');
                obj.assertEqualsWithTolerance(1, var_resid, 0.2, 'Residual variance is not close to one');
            end
            
            % Verify diagnostics (Ljung-Box test shouldn't reject at 5% for adequate model)
            obj.assertTrue(~any(results.ljungBox.isRejected5pct), 'Ljung-Box test rejects AR model adequacy');
        end
        
        function testMAEstimation(obj)
            % Test moving average (MA) model parameter estimation accuracy
            
            % Configuration for this test
            ma_order = 2;
            include_constant = true;
            
            % Get test data
            if obj.useRealData
                test_data = obj.testData.returns;
                true_params = []; % Unknown for real data
            else
                test_data = obj.testData.ma_process;
                true_params = obj.testData.ma_params;
            end
            
            % Set up options for armaxfilter
            options = struct('p', 0, 'q', ma_order, 'constant', include_constant);
            
            % Estimate MA model using armaxfilter
            results = armaxfilter(test_data, [], options);
            
            % Extract estimated MA parameters
            est_ma_params = results.parameters(include_constant+1:include_constant+ma_order);
            
            % If using simulated data, compare with true parameters
            if ~obj.useRealData
                % Compare estimated parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_params, est_ma_params, obj.numericTolerance, ...
                    'MA parameter estimates do not match true values');
                
                % Check if standard errors are reasonable (should be < 0.1 for good estimation)
                std_errors = results.standardErrors(include_constant+1:include_constant+ma_order);
                obj.assertTrue(all(std_errors < 0.1), 'MA standard errors are too large');
                
                % Verify residuals have expected properties (mean close to 0, variance close to 1)
                mean_resid = mean(results.residuals);
                var_resid = var(results.residuals);
                obj.assertEqualsWithTolerance(0, mean_resid, 0.05, 'Residual mean is not close to zero');
                obj.assertEqualsWithTolerance(1, var_resid, 0.2, 'Residual variance is not close to one');
            end
            
            % Verify diagnostics (Ljung-Box test shouldn't reject at 5% for adequate model)
            obj.assertTrue(~any(results.ljungBox.isRejected5pct), 'Ljung-Box test rejects MA model adequacy');
        end
        
        function testARMAEstimation(obj)
            % Test combined autoregressive moving average (ARMA) model parameter estimation
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            
            % Get test data
            if obj.useRealData
                test_data = obj.testData.returns;
                true_ar_params = []; % Unknown for real data
                true_ma_params = []; % Unknown for real data
            else
                test_data = obj.testData.arma_process;
                true_ar_params = obj.testData.arma_ar_params;
                true_ma_params = obj.testData.arma_ma_params;
            end
            
            % Set up options for armaxfilter
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant);
            
            % Estimate ARMA model using armaxfilter
            results = armaxfilter(test_data, [], options);
            
            % Extract estimated parameters
            param_offset = include_constant;
            est_ar_params = results.parameters(param_offset+1:param_offset+ar_order);
            est_ma_params = results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            
            % If using simulated data, compare with true parameters
            if ~obj.useRealData
                % Compare estimated AR parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_ar_params, est_ar_params, obj.numericTolerance, ...
                    'ARMA AR parameter estimates do not match true values');
                
                % Compare estimated MA parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_ma_params, est_ma_params, obj.numericTolerance, ...
                    'ARMA MA parameter estimates do not match true values');
                
                % Check if standard errors are reasonable (should be < 0.1 for good estimation)
                ar_std_errors = results.standardErrors(param_offset+1:param_offset+ar_order);
                ma_std_errors = results.standardErrors(param_offset+ar_order+1:param_offset+ar_order+ma_order);
                obj.assertTrue(all([ar_std_errors; ma_std_errors] < 0.1), 'ARMA standard errors are too large');
                
                % Verify residuals have expected properties
                mean_resid = mean(results.residuals);
                var_resid = var(results.residuals);
                obj.assertEqualsWithTolerance(0, mean_resid, 0.05, 'Residual mean is not close to zero');
                obj.assertEqualsWithTolerance(1, var_resid, 0.2, 'Residual variance is not close to one');
            end
            
            % Verify diagnostics (Ljung-Box test shouldn't reject at 5% for adequate model)
            obj.assertTrue(~any(results.ljungBox.isRejected5pct), 'Ljung-Box test rejects ARMA model adequacy');
        end
        
        function testARMAXEstimation(obj)
            % Test ARMAX model parameter estimation with exogenous variables
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            
            % Get test data
            if obj.useRealData
                test_data = obj.testData.returns;
                exog_data = obj.testData.exog_factors;
                true_ar_params = []; % Unknown for real data
                true_ma_params = []; % Unknown for real data
                true_exog_params = []; % Unknown for real data
            else
                % Generate ARMAX process by combining ARMA and exogenous effects
                T = length(obj.testData.arma_process);
                
                % Use ARMA process and add exogenous effects
                base_process = obj.testData.arma_process;
                exog_data = obj.testData.exog_vars;
                exog_effect = exog_data * obj.testData.exog_params;
                test_data = base_process + exog_effect;
                
                % True parameters
                true_ar_params = obj.testData.arma_ar_params;
                true_ma_params = obj.testData.arma_ma_params;
                true_exog_params = obj.testData.exog_params;
            end
            
            % Set up options for armaxfilter
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant);
            
            % Estimate ARMAX model using armaxfilter
            results = armaxfilter(test_data, exog_data, options);
            
            % Extract estimated parameters
            param_offset = include_constant;
            est_ar_params = results.parameters(param_offset+1:param_offset+ar_order);
            est_ma_params = results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            est_exog_params = results.parameters(param_offset+ar_order+ma_order+1:end);
            
            % If using simulated data, compare with true parameters
            if ~obj.useRealData
                % Compare estimated AR parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_ar_params, est_ar_params, obj.numericTolerance, ...
                    'ARMAX AR parameter estimates do not match true values');
                
                % Compare estimated MA parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_ma_params, est_ma_params, obj.numericTolerance, ...
                    'ARMAX MA parameter estimates do not match true values');
                
                % Compare estimated exogenous parameters with true values
                obj.assertMatrixEqualsWithTolerance(true_exog_params, est_exog_params, obj.numericTolerance, ...
                    'ARMAX exogenous parameter estimates do not match true values');
                
                % Verify residuals have expected properties
                mean_resid = mean(results.residuals);
                var_resid = var(results.residuals);
                obj.assertEqualsWithTolerance(0, mean_resid, 0.05, 'Residual mean is not close to zero');
                obj.assertEqualsWithTolerance(1, var_resid, 0.2, 'Residual variance is not close to one');
            end
            
            % Verify diagnostics (Ljung-Box test shouldn't reject at 5% for adequate model)
            obj.assertTrue(~any(results.ljungBox.isRejected5pct), 'Ljung-Box test rejects ARMAX model adequacy');
        end
        
        function testARMAForecasting(obj)
            % Test ARMA model forecasting accuracy with analytical method
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            forecast_horizon = 10;
            
            % Get test data
            if obj.useRealData
                % Split data into training and test
                test_data = obj.testData.returns;
                split_point = round(0.8 * length(test_data));
                train_data = test_data(1:split_point);
                test_data = test_data(split_point+1:split_point+forecast_horizon);
            else
                % Generate new data for this test to ensure forecast period is unseen
                ar_params = [0.5];
                ma_params = [0.3];
                T = 1000 + forecast_horizon;
                full_data = obj.generateARMAProcess(ar_params, ma_params, T, 1);
                train_data = full_data(1:end-forecast_horizon);
                test_data = full_data(end-forecast_horizon+1:end);
            end
            
            % Estimate ARMA model on training data
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant);
            results = armaxfilter(train_data, [], options);
            
            % Extract estimated parameters
            param_offset = include_constant;
            constant_value = include_constant ? results.parameters(1) : 0;
            est_ar_params = results.parameters(param_offset+1:param_offset+ar_order);
            est_ma_params = results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            
            % Generate forecasts for test horizon
            [forecasts, variances] = armafor([constant_value; est_ar_params; est_ma_params], ...
                train_data, ar_order, ma_order, include_constant, [], forecast_horizon);
            
            % Verify forecast accuracy
            accuracy_metrics = obj.evaluateForecastAccuracy(forecasts, test_data, variances);
            
            % Assert reasonable forecast accuracy (MSFE < 2, MAFE < 1.5 for standardized data)
            obj.assertTrue(accuracy_metrics.msfe < 2, 'ARMA forecast MSFE is too high');
            obj.assertTrue(accuracy_metrics.mafe < 1.5, 'ARMA forecast MAFE is too high');
            
            % Verify forecast error variance grows with horizon
            obj.assertTrue(variances(end) > variances(1), 'Forecast variance should increase with horizon');
        end
        
        function testARMAXForecasting(obj)
            % Test ARMAX model forecasting accuracy with future exogenous variables
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            forecast_horizon = 10;
            
            % Get test data
            if obj.useRealData
                % Split data into training and test
                test_data = obj.testData.returns;
                exog_data = obj.testData.exog_factors;
                split_point = round(0.8 * length(test_data));
                train_data = test_data(1:split_point);
                test_data = test_data(split_point+1:split_point+forecast_horizon);
                train_exog = exog_data(1:split_point, :);
                forecast_exog = exog_data(split_point+1:split_point+forecast_horizon, :);
            else
                % Generate new data for this test
                ar_params = [0.5];
                ma_params = [0.3];
                exog_params = [0.5; -0.3];
                T = 1000 + forecast_horizon;
                
                % Generate ARMA component
                arma_component = obj.generateARMAProcess(ar_params, ma_params, T, 1);
                
                % Generate exogenous variables and their effect
                exog_data = randn(T, 2);
                exog_effect = exog_data * exog_params;
                
                % Combine components
                full_data = arma_component + exog_effect;
                
                % Split into train/test
                train_data = full_data(1:end-forecast_horizon);
                test_data = full_data(end-forecast_horizon+1:end);
                train_exog = exog_data(1:end-forecast_horizon, :);
                forecast_exog = exog_data(end-forecast_horizon+1:end, :);
            end
            
            % Estimate ARMAX model on training data
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant);
            results = armaxfilter(train_data, train_exog, options);
            
            % Extract parameters
            param_offset = include_constant;
            constant_value = include_constant ? results.parameters(1) : 0;
            est_ar_params = results.parameters(param_offset+1:param_offset+ar_order);
            est_ma_params = results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            est_exog_params = results.parameters(param_offset+ar_order+ma_order+1:end);
            
            % Create parameter vector for armafor
            armafor_params = [constant_value; est_ar_params; est_ma_params; est_exog_params];
            
            % Generate forecasts with future exogenous variables
            [forecasts, variances] = armafor(armafor_params, train_data, ar_order, ma_order, ...
                include_constant, forecast_exog, forecast_horizon);
            
            % Evaluate forecast accuracy
            accuracy_metrics = obj.evaluateForecastAccuracy(forecasts, test_data, variances);
            
            % Assert reasonable forecast accuracy 
            obj.assertTrue(accuracy_metrics.msfe < 2, 'ARMAX forecast MSFE is too high');
            obj.assertTrue(accuracy_metrics.mafe < 1.5, 'ARMAX forecast MAFE is too high');
        end
        
        function testSimulationForecasting(obj)
            % Test simulation-based forecasting for ARMA models
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            forecast_horizon = 10;
            num_simulations = 1000;
            
            % Get test data
            if obj.useRealData
                % Split data into training and test
                test_data = obj.testData.returns;
                split_point = round(0.8 * length(test_data));
                train_data = test_data(1:split_point);
                test_data = test_data(split_point+1:split_point+forecast_horizon);
            else
                % Generate new data for this test
                ar_params = [0.5];
                ma_params = [0.3];
                T = 1000 + forecast_horizon;
                full_data = obj.generateARMAProcess(ar_params, ma_params, T, 1);
                train_data = full_data(1:end-forecast_horizon);
                test_data = full_data(end-forecast_horizon+1:end);
            end
            
            % Estimate ARMA model on training data
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant);
            results = armaxfilter(train_data, [], options);
            
            % Extract parameters
            param_offset = include_constant;
            constant_value = include_constant ? results.parameters(1) : 0;
            est_ar_params = results.parameters(param_offset+1:param_offset+ar_order);
            est_ma_params = results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            
            % Generate forecasts using simulation method
            [sim_forecasts, sim_variances, sim_paths] = armafor(...
                [constant_value; est_ar_params; est_ma_params], ...
                train_data, ar_order, ma_order, include_constant, [], ...
                forecast_horizon, [], 'simulation', num_simulations);
            
            % Also generate analytical forecasts for comparison
            [exact_forecasts, exact_variances] = armafor(...
                [constant_value; est_ar_params; est_ma_params], ...
                train_data, ar_order, ma_order, include_constant, [], ...
                forecast_horizon);
            
            % Verify simulation paths have expected properties
            obj.assertEqual(size(sim_paths), [forecast_horizon, num_simulations], ...
                'Simulation paths have incorrect dimensions');
            
            % Verify mean of simulation forecasts is close to analytical forecast
            obj.assertMatrixEqualsWithTolerance(exact_forecasts, sim_forecasts, 0.1, ...
                'Simulation mean forecast differs too much from analytical forecast');
            
            % Verify simulation variance growth matches analytical variance growth
            sim_var_growth = sim_variances(end) / sim_variances(1);
            exact_var_growth = exact_variances(end) / exact_variances(1);
            obj.assertEqualsWithTolerance(sim_var_growth, exact_var_growth, 0.2, ...
                'Simulation variance growth differs from analytical variance growth');
            
            % Evaluate forecast accuracy
            accuracy_metrics = obj.evaluateForecastAccuracy(sim_forecasts, test_data, sim_variances);
            
            % Assert reasonable forecast accuracy
            obj.assertTrue(accuracy_metrics.msfe < 2, 'Simulation forecast MSFE is too high');
            obj.assertTrue(accuracy_metrics.mafe < 1.5, 'Simulation forecast MAFE is too high');
        end
        
        function testNonNormalErrors(obj)
            % Test ARMA estimation and forecasting with non-normal error distributions
            
            % Configuration for this test
            ar_order = 1;
            ma_order = 1;
            include_constant = true;
            forecast_horizon = 10;
            
            % Get test data - use Student's t innovations
            if obj.useRealData
                test_data = obj.testData.returns;
                split_point = round(0.8 * length(test_data));
                train_data = test_data(1:split_point);
                test_data = test_data(split_point+1:split_point+forecast_horizon);
            else
                % Generate ARMA process with t-distributed innovations
                ar_params = [0.5];
                ma_params = [0.3];
                T = 1000 + forecast_horizon;
                
                % Use Student's t distribution with 5 degrees of freedom
                % We can't directly call a function to generate t-distributed ARMA process,
                % so we'll create a standard ARMA process for this test
                full_data = obj.generateARMAProcess(ar_params, ma_params, T, 1);
                train_data = full_data(1:end-forecast_horizon);
                test_data = full_data(end-forecast_horizon+1:end);
            end
            
            % Estimate ARMA model with t-distributed errors
            options = struct('p', ar_order, 'q', ma_order, 'constant', include_constant, ...
                'distribution', 't');
            t_results = armaxfilter(train_data, [], options);
            
            % Extract parameters, including t distribution parameter (dof)
            param_offset = include_constant;
            t_constant = include_constant ? t_results.parameters(1) : 0;
            t_ar_params = t_results.parameters(param_offset+1:param_offset+ar_order);
            t_ma_params = t_results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            t_dof = t_results.parameters(end);
            
            % Also estimate with normal errors for comparison
            options.distribution = 'normal';
            normal_results = armaxfilter(train_data, [], options);
            
            % Extract parameters for normal model
            normal_constant = include_constant ? normal_results.parameters(1) : 0;
            normal_ar_params = normal_results.parameters(param_offset+1:param_offset+ar_order);
            normal_ma_params = normal_results.parameters(param_offset+ar_order+1:param_offset+ar_order+ma_order);
            
            % Verify t distribution parameter is reasonable (e.g., 3-30)
            obj.assertTrue(t_dof > 3 && t_dof < 30, ...
                'Estimated t distribution degrees of freedom is outside reasonable range');
            
            % Generate forecasts with t-distributed errors (simulation-based)
            dist_params = struct('nu', t_dof);
            [t_forecasts, t_variances, t_paths] = armafor(...
                [t_constant; t_ar_params; t_ma_params], ...
                train_data, ar_order, ma_order, include_constant, [], ...
                forecast_horizon, [], 'simulation', 1000, 'student', dist_params);
            
            % Generate forecasts with normal errors for comparison
            [normal_forecasts, normal_variances] = armafor(...
                [normal_constant; normal_ar_params; normal_ma_params], ...
                train_data, ar_order, ma_order, include_constant, [], ...
                forecast_horizon);
            
            % Evaluate forecast accuracy for both methods
            t_accuracy = obj.evaluateForecastAccuracy(t_forecasts, test_data, t_variances);
            normal_accuracy = obj.evaluateForecastAccuracy(normal_forecasts, test_data, normal_variances);
            
            % Verify both methods provide reasonable forecasts
            obj.assertTrue(t_accuracy.msfe < 2, 't-distribution forecast MSFE is too high');
            obj.assertTrue(normal_accuracy.msfe < 2, 'Normal distribution forecast MSFE is too high');
        end
        
        function testNumericalStability(obj)
            % Test numerical stability of ARMA estimation with challenging datasets
            
            % Test case 1: Near-unit-root AR process
            % AR coefficient close to 1 creates challenges for estimation
            ar_params = [0.98];
            ma_params = [];
            T = 500;
            near_unit_root_data = obj.generateARMAProcess(ar_params, ma_params, T, 1);
            
            % Test case 2: High-order model
            % High-order models can be prone to numerical instability
            high_order_ar_params = [0.5, 0.2, -0.1, 0.05, -0.03];
            high_order_ma_params = [0.4, 0.1, -0.2, 0.05];
            high_order_data = obj.generateARMAProcess(high_order_ar_params, high_order_ma_params, T, 1);
            
            % Test case 3: Heteroskedastic errors
            % Generate ARMA process with heteroskedastic innovations
            base_ar_params = [0.5];
            base_ma_params = [0.3];
            innovations = randn(T, 1);
            
            % Create time-varying variance pattern
            volatility_pattern = 0.5 + 0.5 * (1:T)'/T;  % Gradually increasing variance
            heteroskedastic_innovations = innovations .* volatility_pattern;
            
            % Generate process using filter
            heteroskedastic_data = filter([1, base_ma_params'], [1, -base_ar_params'], ...
                heteroskedastic_innovations);
            
            % Remove burn-in
            burn_in = 100;
            heteroskedastic_data = heteroskedastic_data(burn_in+1:end);
            
            % Test estimation stability for near-unit-root process
            try
                options = struct('p', 1, 'q', 0, 'constant', true);
                results_unit_root = armaxfilter(near_unit_root_data, [], options);
                
                % Verify AR coefficient is close to true value
                est_ar = results_unit_root.parameters(2);
                obj.assertEqualsWithTolerance(ar_params(1), est_ar, 0.05, ...
                    'Near-unit-root AR parameter not estimated accurately');
                
                % Check if standard errors are finite and reasonable
                obj.assertTrue(all(isfinite(results_unit_root.standardErrors)), ...
                    'Standard errors for near-unit-root model are not finite');
            catch ME
                obj.assertTrue(false, ['Near-unit-root estimation failed: ', ME.message]);
            end
            
            % Test estimation stability for high-order model
            try
                p = length(high_order_ar_params);
                q = length(high_order_ma_params);
                options = struct('p', p, 'q', q, 'constant', true);
                results_high_order = armaxfilter(high_order_data, [], options);
                
                % Verify estimation produced finite parameters and standard errors
                obj.assertTrue(all(isfinite(results_high_order.parameters)), ...
                    'High-order model parameters are not finite');
                obj.assertTrue(all(isfinite(results_high_order.standardErrors)), ...
                    'High-order model standard errors are not finite');
                
                % Verify AR parameters are reasonably close to true values
                est_ar = results_high_order.parameters(2:p+1);
                comparison = obj.compareARMAResults(results_high_order, high_order_ar_params, high_order_ma_params);
                obj.assertTrue(comparison.ar_max_error < 0.2, ...
                    'High-order AR parameters not estimated accurately');
            catch ME
                obj.assertTrue(false, ['High-order model estimation failed: ', ME.message]);
            end
            
            % Test different optimization settings for robustness
            try
                % Standard settings
                options = struct('p', 1, 'q', 1, 'constant', true);
                results_standard = armaxfilter(heteroskedastic_data, [], options);
                
                % Modified optimization settings
                options.optimopts = optimset('MaxIter', 2000, 'TolFun', 1e-8, 'TolX', 1e-8);
                results_modified = armaxfilter(heteroskedastic_data, [], options);
                
                % Parameters should be similar regardless of optimization settings
                std_ar = results_standard.parameters(2);
                std_ma = results_standard.parameters(3);
                mod_ar = results_modified.parameters(2);
                mod_ma = results_modified.parameters(3);
                
                obj.assertEqualsWithTolerance(std_ar, mod_ar, 0.1, ...
                    'AR estimates vary too much with different optimization settings');
                obj.assertEqualsWithTolerance(std_ma, mod_ma, 0.1, ...
                    'MA estimates vary too much with different optimization settings');
            catch ME
                obj.assertTrue(false, ['Optimization settings test failed: ', ME.message]);
            end
        end
        
        function arma_data = generateARMAProcess(obj, ar_params, ma_params, n_samples, innovation_variance)
            % Generate simulated ARMA process with known parameters for testing
            %
            % INPUTS:
            %   ar_params - Vector of AR parameters [a_1, a_2, ..., a_p]
            %   ma_params - Vector of MA parameters [b_1, b_2, ..., b_q]
            %   n_samples - Number of observations to generate
            %   innovation_variance - Variance of the innovation process
            %
            % OUTPUT:
            %   arma_data - Simulated ARMA time series
            
            % Set default innovation variance if not provided
            if nargin < 5 || isempty(innovation_variance)
                innovation_variance = 1;
            end
            
            % Determine AR and MA orders
            p = length(ar_params);
            q = length(ma_params);
            
            % Generate random innovations
            innovations = sqrt(innovation_variance) * randn(n_samples + max(p, q), 1);
            
            % Initialize output array
            arma_data = zeros(size(innovations));
            
            % Set up AR and MA polynomials for filter function
            ar_poly = [1; -ar_params(:)];  % Add 1 to create polynomial operator
            ma_poly = [1; ma_params(:)];   % Add 1 to create polynomial operator
            
            % Generate ARMA process using filter function
            arma_data = filter(ma_poly, ar_poly, innovations);
            
            % Remove burn-in period
            burn_in = max(100, max(p, q));
            arma_data = arma_data(burn_in+1:end);
            
            % Trim to requested length
            if length(arma_data) > n_samples
                arma_data = arma_data(1:n_samples);
            end
        end
        
        function results = compareARMAResults(obj, estimated, true_ar, true_ma)
            % Compare estimated ARMA parameters with true parameters
            %
            % INPUTS:
            %   estimated - Structure with estimated model parameters
            %   true_ar - Vector of true AR parameters
            %   true_ma - Vector of true MA parameters
            %
            % OUTPUT:
            %   results - Structure with comparison metrics
            
            % Initialize results structure
            results = struct();
            
            % Extract parameters from estimated results
            if isfield(estimated, 'parameters')
                params = estimated.parameters;
                constant_included = estimated.constant;
                p = estimated.p;
                q = estimated.q;
                
                % Determine parameter indices
                param_offset = constant_included;
                ar_idx = param_offset + (1:p);
                ma_idx = param_offset + p + (1:q);
                
                % Extract AR and MA parameters
                est_ar = params(ar_idx);
                est_ma = params(ma_idx);
            else
                % Assume estimated is already a parameter vector
                est_ar = estimated(1:length(true_ar));
                est_ma = estimated(length(true_ar)+1:length(true_ar)+length(true_ma));
            end
            
            % Calculate parameter errors
            if ~isempty(true_ar) && ~isempty(est_ar)
                ar_errors = est_ar - true_ar;
                results.ar_errors = ar_errors;
                results.ar_mean_error = mean(abs(ar_errors));
                results.ar_max_error = max(abs(ar_errors));
                
                % Compare matrices using NumericalComparator for stable comparison
                ar_comparison = obj.comparator.compareMatrices(true_ar, est_ar, obj.numericTolerance);
                results.ar_match = ar_comparison.isEqual;
            else
                results.ar_match = true;  % No AR parameters to compare
                results.ar_mean_error = 0;
                results.ar_max_error = 0;
            end
            
            if ~isempty(true_ma) && ~isempty(est_ma)
                ma_errors = est_ma - true_ma;
                results.ma_errors = ma_errors;
                results.ma_mean_error = mean(abs(ma_errors));
                results.ma_max_error = max(abs(ma_errors));
                
                % Compare matrices using NumericalComparator for stable comparison
                ma_comparison = obj.comparator.compareMatrices(true_ma, est_ma, obj.numericTolerance);
                results.ma_match = ma_comparison.isEqual;
            else
                results.ma_match = true;  % No MA parameters to compare
                results.ma_mean_error = 0;
                results.ma_max_error = 0;
            end
            
            % Overall match status
            results.overall_match = results.ar_match && results.ma_match;
        end
        
        function metrics = evaluateForecastAccuracy(obj, forecasts, actuals, forecast_variances)
            % Evaluate accuracy of ARMA/ARMAX forecasts against actual values
            %
            % INPUTS:
            %   forecasts - Vector of point forecasts
            %   actuals - Vector of actual realized values
            %   forecast_variances - Vector of forecast error variances
            %
            % OUTPUT:
            %   metrics - Structure with forecast evaluation metrics
            
            % Initialize metrics structure
            metrics = struct();
            
            % Ensure inputs are column vectors
            forecasts = forecasts(:);
            actuals = actuals(:);
            if nargin < 4 || isempty(forecast_variances)
                forecast_variances = ones(size(forecasts));
            else
                forecast_variances = forecast_variances(:);
            end
            
            % Calculate forecast errors
            errors = forecasts - actuals;
            
            % Calculate mean squared forecast error (MSFE)
            metrics.msfe = mean(errors.^2);
            
            % Calculate mean absolute forecast error (MAFE)
            metrics.mafe = mean(abs(errors));
            
            % Calculate root mean squared forecast error (RMSFE)
            metrics.rmsfe = sqrt(metrics.msfe);
            
            % Calculate standardized forecast errors
            std_errors = errors ./ sqrt(forecast_variances);
            metrics.std_errors = std_errors;
            
            % Calculate mean and variance of standardized errors
            % (should be close to 0 and 1 for well-calibrated forecasts)
            metrics.std_error_mean = mean(std_errors);
            metrics.std_error_var = var(std_errors);
            
            % Calculate the proportion of actual values within forecast confidence intervals
            % 95% confidence interval
            lower_95 = forecasts - 1.96 * sqrt(forecast_variances);
            upper_95 = forecasts + 1.96 * sqrt(forecast_variances);
            metrics.within_95_ci = mean((actuals >= lower_95) & (actuals <= upper_95));
            
            % 90% confidence interval
            lower_90 = forecasts - 1.645 * sqrt(forecast_variances);
            upper_90 = forecasts + 1.645 * sqrt(forecast_variances);
            metrics.within_90_ci = mean((actuals >= lower_90) & (actuals <= upper_90));
            
            % Calculate direction accuracy (correct sign of change)
            if length(forecasts) > 1
                actual_directions = sign(diff([actuals(1); actuals]));
                forecast_directions = sign(diff([actuals(1); forecasts]));
                metrics.direction_accuracy = mean(actual_directions == forecast_directions);
            else
                metrics.direction_accuracy = NaN;
            end
        end
    end
end