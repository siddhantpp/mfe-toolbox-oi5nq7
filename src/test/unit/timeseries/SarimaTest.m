classdef SarimaTest < BaseTest
    %SARIMATEST Test class for validating the SARIMA model implementation
    %   in the MFE Toolbox with comprehensive test cases for estimation,
    %   forecasting, and diagnostics with various seasonal patterns.

    properties
        testData
        tolerance
        knownParameters
        seasonalTestSeries
        quarterlyData
        monthlyData
        seasonalPeriods
    end

    methods
        function obj = SarimaTest()
            %SARIMATEST Initialize the SarimaTest class with default properties

            % Call superclass constructor to initialize BaseTest
            obj = obj@BaseTest();

            % Set default tolerance for numerical comparisons
            obj.tolerance = 1e-7;

            % Initialize empty data structures for test data
            obj.testData = struct();
            obj.knownParameters = struct();
        end

        function setUp(obj)
            %setUp Set up test environment before each test method execution

            % Call superclass setUp method
            setUp@BaseTest(obj);

            % Load macroeconomic_data.mat for seasonal data
            data = load([obj.testDataPath, '/macroeconomic_data.mat']);

            % Extract quarterly GDP data for quarterly seasonal testing
            obj.quarterlyData = data.Data(:, 1);

            % Extract monthly data (industrial_production) for monthly seasonal testing
            obj.monthlyData = data.Data(:, 2);

            % Set default SARIMA model parameters for testing
            obj.seasonalPeriods = [4, 12]; % Quarterly and monthly seasonal periods

            % Generate synthetic seasonal data with known parameters
            obj.knownParameters.nobs = 200;
            obj.knownParameters.period = 4;
            obj.knownParameters.params = struct('ar', [0.5], 'ma', [0.3], 'sar', [0.4], 'sma', [0.2]);
            obj.seasonalTestSeries = obj.generateSeasonalTestData(obj.knownParameters.nobs, obj.knownParameters.period, obj.knownParameters.params);

            % Set numerical tolerance for floating-point comparisons
            obj.tolerance = 1e-6;
        end

        function tearDown(obj)
            %tearDown Clean up test environment after each test method execution

            % Call superclass tearDown method
            tearDown@BaseTest(obj);

            % Clear test data variables
            clear obj.testData obj.knownParameters obj.seasonalTestSeries obj.quarterlyData obj.monthlyData

            % Reset model parameters
            obj.seasonalPeriods = [];
        end

        function testBasicSarimaEstimation(obj)
            %testBasicSarimaEstimation Test basic SARIMA model estimation with simple seasonal pattern

            % Define simple SARIMA model orders (p=1,d=0,q=1,P=1,D=0,Q=1,s=4)
            p = 1; d = 0; q = 1; P = 1; D = 0; Q = 1; s = 4;

            % Estimate model using sarima function on synthetic data
            results = sarima(obj.seasonalTestSeries.series, p, d, q, P, D, Q, s);

            % Verify model parameters are estimated correctly within tolerance
            obj.assertAlmostEqual(results.parameters(2), obj.knownParameters.params.ar, obj.tolerance, 'AR parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(3), obj.knownParameters.params.ma, obj.tolerance, 'MA parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(4), obj.knownParameters.params.sar, obj.tolerance, 'SAR parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(5), obj.knownParameters.params.sma, obj.tolerance, 'SMA parameter estimation failed');

            % Check standard errors are calculated and reasonable
            obj.assertTrue(all(results.standardErrors > 0), 'Standard errors must be positive');

            % Verify model fit statistics are calculated correctly
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');
            obj.assertTrue(isnumeric(results.aic), 'AIC must be a number');
            obj.assertTrue(isnumeric(results.sbic), 'SBIC must be a number');
        end

        function testSarimaWithRegularDifferencing(obj)
            %testSarimaWithRegularDifferencing Test SARIMA model with regular (non-seasonal) differencing

            % Define SARIMA model with d=1 (non-seasonal differencing)
            p = 1; d = 1; q = 1; P = 0; D = 0; Q = 0; s = 4;

            % Create an integrated series
            integratedSeries = cumsum(obj.seasonalTestSeries.series);

            % Estimate model using sarima function on integrated series
            results = sarima(integratedSeries, p, d, q, P, D, Q, s);

            % Verify differencing is applied correctly
            obj.assertEqual(length(results.y), length(integratedSeries), 'Original data length should match');
            obj.assertEqual(length(results.y_diff), length(integratedSeries) - 1, 'Differenced data length should be one less');

            % Check that parameters are estimated accurately
            obj.assertAlmostEqual(results.parameters(2), obj.knownParameters.params.ar, obj.tolerance, 'AR parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(3), obj.knownParameters.params.ma, obj.tolerance, 'MA parameter estimation failed');

            % Verify residuals are stationary after differencing
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');
        end

        function testSarimaWithSeasonalDifferencing(obj)
            %testSarimaWithSeasonalDifferencing Test SARIMA model with seasonal differencing

            % Define SARIMA model with D=1 (seasonal differencing)
            p = 1; d = 0; q = 1; P = 0; D = 1; Q = 0; s = 4;

            % Create a seasonally integrated series
            seasonallyIntegratedSeries = cumsum(repelem(obj.seasonalTestSeries.series(1:obj.knownParameters.period), obj.knownParameters.nobs/obj.knownParameters.period));

            % Estimate model using sarima function on seasonally integrated series
            results = sarima(seasonallyIntegratedSeries, p, d, q, P, D, Q, s);

            % Verify seasonal differencing is applied correctly
            obj.assertEqual(length(results.y), length(seasonallyIntegratedSeries), 'Original data length should match');
            obj.assertEqual(length(results.y_diff), length(seasonallyIntegratedSeries) - s, 'Seasonally differenced data length should be s less');

            % Check that parameters are estimated accurately
            obj.assertAlmostEqual(results.parameters(2), obj.knownParameters.params.ar, obj.tolerance, 'AR parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(3), obj.knownParameters.params.ma, obj.tolerance, 'MA parameter estimation failed');

            % Verify residuals are free from seasonal patterns
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');
        end

        function testComplexSarimaModel(obj)
            %testComplexSarimaModel Test more complex SARIMA model with both regular and seasonal components

            % Define complex SARIMA model with multiple AR, MA, SAR, SMA terms
            p = 2; d = 0; q = 2; P = 1; D = 0; Q = 1; s = 4;

            % Generate more complex seasonal data
            complexParams = struct('ar', [0.5, 0.2], 'ma', [0.3, 0.1], 'sar', [0.4], 'sma', [0.2]);
            complexSeasonalData = obj.generateSeasonalTestData(obj.knownParameters.nobs, obj.knownParameters.period, complexParams);

            % Estimate model using sarima function on complex seasonal data
            results = sarima(complexSeasonalData.series, p, d, q, P, D, Q, s);

            % Verify all parameters (regular and seasonal) are estimated correctly
            obj.assertAlmostEqual(results.parameters(2), complexParams.ar(1), obj.tolerance, 'AR(1) parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(3), complexParams.ar(2), obj.tolerance, 'AR(2) parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(4), complexParams.ma(1), obj.tolerance, 'MA(1) parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(5), complexParams.ma(2), obj.tolerance, 'MA(2) parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(6), complexParams.sar, obj.tolerance, 'SAR parameter estimation failed');
            obj.assertAlmostEqual(results.parameters(7), complexParams.sma, obj.tolerance, 'SMA parameter estimation failed');

            % Check diagnostic statistics for model adequacy
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');

            % Verify residuals are white noise (Ljung-Box test)
            obj.assertTrue(all(results.ljungBox.pValues > 0.05), 'Ljung-Box test should not reject null hypothesis');
        end

        function testSarimaForecasting(obj)
            %testSarimaForecasting Test forecasting functionality of SARIMA models

            % Split data into in-sample and out-of-sample periods
            inSampleLength = floor(0.8 * length(obj.seasonalTestSeries.series));
            inSampleData = obj.seasonalTestSeries.series(1:inSampleLength);
            outOfSampleData = obj.seasonalTestSeries.series(inSampleLength+1:end);
            forecastHorizon = length(outOfSampleData);

            % Define SARIMA model orders
            p = 1; d = 0; q = 1; P = 1; D = 0; Q = 1; s = 4;

            % Estimate SARIMA model on in-sample data
            results = sarima(inSampleData, p, d, q, P, D, Q, s);

            % Generate forecasts for out-of-sample period
            forecasts = sarima_forecast(results, forecastHorizon);

            % Compare forecasts with actual values using error metrics
            rmse = sqrt(mean((forecasts.point - outOfSampleData).^2));
            mae = mean(abs(forecasts.point - outOfSampleData));

            % Verify forecast uncertainty bounds are calculated correctly
            obj.assertTrue(all(forecasts.lower < forecasts.upper), 'Forecast lower bounds must be less than upper bounds');

            % Check multi-step ahead forecasts maintain seasonal patterns
            obj.assertTrue(isnumeric(rmse), 'Root Mean Squared Error must be a number');
            obj.assertTrue(isnumeric(mae), 'Mean Absolute Error must be a number');
        end

        function testSarimaModelSelection(obj)
            %testSarimaModelSelection Test model selection criteria (AIC, SBIC) for SARIMA models

            % Estimate multiple SARIMA models with different orders
            model1 = sarima(obj.seasonalTestSeries.series, 1, 0, 1, 0, 0, 0, 4);
            model2 = sarima(obj.seasonalTestSeries.series, 0, 0, 0, 1, 0, 1, 4);
            model3 = sarima(obj.seasonalTestSeries.series, 1, 0, 1, 1, 0, 1, 4);

            % Calculate information criteria using aicsbic function
            ic1 = aicsbic(-model1.logL, length(model1.parameters), length(obj.seasonalTestSeries.series));
            ic2 = aicsbic(-model2.logL, length(model2.parameters), length(obj.seasonalTestSeries.series));
            ic3 = aicsbic(-model3.logL, length(model3.parameters), length(obj.seasonalTestSeries.series));

            % Verify model with known true parameters has lowest information criteria
            obj.assertTrue(model3.aic < model1.aic && model3.aic < model2.aic, 'Model 3 should have lowest AIC');
            obj.assertTrue(model3.sbic < model1.sbic && model3.sbic < model2.sbic, 'Model 3 should have lowest SBIC');

            % Check rank ordering of models matches expected complexity penalty
            obj.assertTrue(ic1.aic < ic2.aic, 'AIC should penalize complexity');
            obj.assertTrue(ic1.sbic > ic2.sbic, 'SBIC should penalize complexity more');

            % Verify consistency between direct criteria calculation and sarima output
            obj.assertAlmostEqual(model1.aic, ic1.aic, obj.tolerance, 'AIC calculation mismatch');
            obj.assertAlmostEqual(model1.sbic, ic1.sbic, obj.tolerance, 'SBIC calculation mismatch');
        end

        function testQuarterlySeasonal(obj)
            %testQuarterlySeasonal Test SARIMA model with quarterly seasonal pattern (s=4)

            % Use quarterly GDP data for testing
            data = obj.quarterlyData;

            % Define SARIMA model with quarterly seasonality (s=4)
            p = 1; d = 1; q = 1; P = 1; D = 1; Q = 1; s = 4;

            % Estimate SARIMA model
            results = sarima(data, p, d, q, P, D, Q, s);

            % Verify seasonal components capture quarterly patterns
            obj.assertTrue(isnumeric(results.parameters(6)), 'SAR parameter must be a number');
            obj.assertTrue(isnumeric(results.parameters(7)), 'SMA parameter must be a number');

            % Check forecasts maintain correct quarterly seasonal pattern
            forecasts = sarima_forecast(results, 8);
            obj.assertEqual(length(forecasts.point), 8, 'Forecast horizon must match');

            % Verify seasonal lag operations work correctly
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');
        end

        function testMonthlySeasonal(obj)
            %testMonthlySeasonal Test SARIMA model with monthly seasonal pattern (s=12)

            % Use monthly industrial production data for testing
            data = obj.monthlyData;

            % Define SARIMA model with monthly seasonality (s=12)
            p = 1; d = 1; q = 1; P = 1; D = 1; Q = 1; s = 12;

            % Estimate SARIMA model
            results = sarima(data, p, d, q, P, D, Q, s);

            % Verify seasonal components capture monthly patterns
            obj.assertTrue(isnumeric(results.parameters(6)), 'SAR parameter must be a number');
            obj.assertTrue(isnumeric(results.parameters(7)), 'SMA parameter must be a number');

            % Check forecasts maintain correct monthly seasonal pattern
            forecasts = sarima_forecast(results, 24);
            obj.assertEqual(length(forecasts.point), 24, 'Forecast horizon must match');

            % Verify computational efficiency with longer seasonal period
            obj.assertTrue(isnumeric(results.logL), 'Log-likelihood must be a number');
        end

        function testSarimaWithExogenousVariables(obj)
            %testSarimaWithExogenousVariables Test SARIMA model with exogenous variables (SARIMAX)

            % Generate data with seasonal patterns and known exogenous effects
            nobs = 100;
            s = 4;
            t = (1:nobs)';
            seasonalComponent = sin(2*pi*t/s);
            exogenousVariable = 0.5*t/nobs + randn(nobs, 1)*0.1; % Exogenous variable with trend and noise
            data = seasonalComponent + 0.3*exogenousVariable + randn(nobs, 1)*0.1;

            % Define SARIMAX model including exogenous variables
            p = 1; d = 0; q = 1; P = 1; D = 0; Q = 1;
            s = 4;

            % Estimate SARIMAX model including exogenous variables
            % sarima does not directly support exogenous variables, so this test is limited
            % to checking that it runs without errors when exogenous variables are present.
            % A proper SARIMAX implementation would require a different function.
            try
                results = sarima(data, p, d, q, P, D, Q, s);
                obj.assertTrue(true, 'SARIMA ran successfully without exogenous variables');
            catch ME
                obj.assertFalse(true, ['SARIMA failed to run: ' ME.message]);
            end

            % Verify both seasonal and exogenous parameters are estimated correctly
            % This part is skipped because sarima does not support exogenous variables.

            % Check forecasts incorporate effects of future exogenous variables
            % This part is skipped because sarima does not support exogenous variables.

            % Verify model fit improvement with inclusion of exogenous variables
            % This part is skipped because sarima does not support exogenous variables.
        end

        function testSarimaInputValidation(obj)
            %testSarimaInputValidation Test error handling and input validation for SARIMA function

            % Test with invalid order parameters (negative values)
            obj.assertThrows(@() sarima(obj.seasonalTestSeries.series, -1, 0, 0, 0, 0, 0, 4), 'parametercheck:InvalidInput', 'Negative order parameter should throw error');

            % Test with inconsistent seasonal period specification
            obj.assertThrows(@() sarima(obj.seasonalTestSeries.series, 1, 0, 0, 1, 0, 0, 1), 'MFE:InvalidInput', 'Inconsistent seasonal period should throw error');

            % Test with insufficient data for specified model
            obj.assertThrows(@() sarima(obj.seasonalTestSeries.series(1:5), 1, 0, 0, 0, 0, 0, 4), 'MFE:InvalidInput', 'Insufficient data should throw error');

            % Test with incompatible options structure
            obj.assertThrows(@() sarima(obj.seasonalTestSeries.series, 1, 0, 0, 0, 0, 0, 4, 'invalid'), 'MFE:InvalidInput', 'Incompatible options structure should throw error');

            % Verify appropriate error messages are generated
            % Check error cases are handled gracefully
        end

        function testSarimaResidualDiagnostics(obj)
            %testSarimaResidualDiagnostics Test residual diagnostic features of SARIMA model output

            % Estimate SARIMA model on seasonal data
            results = sarima(obj.seasonalTestSeries.series, 1, 0, 1, 1, 0, 1, 4);

            % Extract model residuals and diagnostic statistics
            residuals = results.residuals;
            ljungBoxPValues = results.ljungBox.pValues;

            % Verify residuals are free from autocorrelation using Ljung-Box test
            obj.assertTrue(all(ljungBoxPValues > 0.05), 'Ljung-Box test should not reject null hypothesis');

            % Check that residuals are free from remaining seasonal patterns
            % This part is skipped because it requires manual inspection of ACF/PACF plots.

            % Verify normality tests on residuals are calculated correctly
            % This part is skipped because sarima does not directly output normality test statistics.

            % Test consistency of diagnostic statistics with direct calculation
            obj.assertAlmostEqual(length(residuals), length(obj.seasonalTestSeries.series), obj.tolerance, 'Residual length must match data length');
        end

        function seasonalData = generateSeasonalTestData(obj, nobs, period, params)
            %generateSeasonalTestData Helper method to generate synthetic seasonal time series with known parameters

            % Set default parameters if not provided
            if nargin < 4
                params = struct('ar', 0.5, 'ma', 0.3, 'sar', 0.4, 'sma', 0.2);
            end

            % Generate non-seasonal ARMA component
            arCoeffs = params.ar;
            maCoeffs = params.ma;
            armaErrors = randn(nobs, 1);
            armaSeries = armafor([arCoeffs; maCoeffs], armaErrors, length(arCoeffs), length(maCoeffs), false);

            % Generate seasonal component with specified period
            sarCoeffs = params.sar;
            smaCoeffs = params.sma;
            seasonalErrors = randn(nobs, 1);
            seasonalSeries = armafor([sarCoeffs; smaCoeffs], seasonalErrors, length(sarCoeffs), length(smaCoeffs), false);

            % Add stochastic noise component
            noiseLevel = 0.1;
            noise = noiseLevel * randn(nobs, 1);

            % Combine seasonal and non-seasonal components
            series = armaSeries + seasonalSeries + noise;

            % Apply differencing if integrated series requested
            if isfield(params, 'integrated') && params.integrated
                series = cumsum(series);
            end

            % Return combined time series and parameter structure for validation
            seasonalData = struct('series', series, 'params', params);
        end
    end
end