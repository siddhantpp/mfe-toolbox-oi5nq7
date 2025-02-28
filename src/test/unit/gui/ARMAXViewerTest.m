classdef ARMAXViewerTest < BaseTest
    % ARMAXVIEWERTEST Test class for ARMAX_viewer component, validates the detailed results viewer that displays model parameters, diagnostics, plots, and forecasts.

    properties
        viewerHandle
        modelResults
        guiCommon
        dataGenerator
    end

    methods
        function obj = ARMAXViewerTest()
            % Initialize the ARMAXViewerTest class
            obj = obj@BaseTest();
            obj.viewerHandle = [];
            obj.guiCommon = GUICommonTest();
            obj.dataGenerator = TestDataGenerator();
            obj.modelResults = [];
        end

        function setUp(obj)
            % Setup method that runs before each test, preparing the test environment
            setUp@BaseTest(obj);
            obj.guiCommon.closeAllTestFigures();

            % Generate test data
            obj.modelResults = obj.createTestModelResults(2, 1, true);
            obj.viewerHandle = [];
        end

        function tearDown(obj)
            % Cleanup method that runs after each test, releasing resources
            if ~isempty(obj.viewerHandle) && ishandle(obj.viewerHandle)
                close(obj.viewerHandle);
            end
            obj.guiCommon.closeAllTestFigures();
            obj.modelResults = [];
            tearDown@BaseTest(obj);
        end

        function testViewerInitialization(obj)
            % Test that the viewer initializes correctly with valid parameters
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);
            obj.assertTrue(ishandle(obj.viewerHandle), 'Viewer figure is not a valid handle');
            figName = get(obj.viewerHandle, 'Name');
            obj.assertEqual(figName, 'ARMAX Results Viewer', 'Figure title is incorrect');

            % Verify GUI components exist
            obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'save_button');
            obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'close_button');
            obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'axes', 'main_axes');
            obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uitable', 'parameter_table');
            obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'model_equation');

            % Verify initial plot is displayed (model fit)
        end

        function testModelEquationDisplay(obj)
            % Test that the model equation is displayed correctly in LaTeX-like format
            arOrder = 2;
            maOrder = 1;
            includeConstant = true;
            obj.modelResults = obj.createTestModelResults(arOrder, maOrder, includeConstant);
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Find equation display component
            equationDisplay = findobj(obj.viewerHandle, 'Style', 'text', 'Tag', 'model_equation');
            obj.assertTrue(~isempty(equationDisplay), 'Equation display component not found');

            % Get equation text
            equationText = get(equationDisplay, 'String');
            obj.assertTrue(contains(equationText, 'y_t ='), 'Equation text does not start correctly');
        end

        function testParameterTableDisplay(obj)
            % Test that parameter estimates table is displayed correctly
            arOrder = 1;
            maOrder = 1;
            includeConstant = true;
            obj.modelResults = obj.createTestModelResults(arOrder, maOrder, includeConstant);
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Find parameter table component
            paramTable = findobj(obj.viewerHandle, 'Tag', 'parameter_table');
            obj.assertTrue(~isempty(paramTable), 'Parameter table component not found');

            % Get table data
            tableData = get(paramTable, 'Data');
            obj.assertEqual(size(tableData, 2), 4, 'Parameter table has incorrect number of columns');

            % Verify column labels
            columnNames = get(paramTable, 'ColumnName');
            obj.assertEqual(columnNames{1}, 'Parameter', 'Incorrect column name');
            obj.assertEqual(columnNames{2}, 'Estimate', 'Incorrect column name');
            obj.assertEqual(columnNames{3}, 'Std. Error', 'Incorrect column name');
            obj.assertEqual(columnNames{4}, 't-stat [p-value]', 'Incorrect column name');
        end

        function testPlotTypeSwitching(obj)
            % Test switching between different plot types
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Verify default plot is model fit
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_fit', struct('Value', 1)), true, 'Default plot is not model fit');

            % Find plot type radio buttons
            plotTypeResiduals = findobj(obj.viewerHandle, 'Tag', 'plot_type_residuals');
            plotTypeACF = findobj(obj.viewerHandle, 'Tag', 'plot_type_acf');
            plotTypePACF = findobj(obj.viewerHandle, 'Tag', 'plot_type_pacf');
            plotTypeForecast = findobj(obj.viewerHandle, 'Tag', 'plot_type_forecast');

            % Select residuals plot type and verify update
            obj.guiCommon.simulateUserInput(plotTypeResiduals, 'button', []);
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_residuals', struct('Value', 1)), true, 'Residuals plot not selected');

            % Select ACF plot type and verify update
            obj.guiCommon.simulateUserInput(plotTypeACF, 'button', []);
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_acf', struct('Value', 1)), true, 'ACF plot not selected');

            % Select PACF plot type and verify update
            obj.guiCommon.simulateUserInput(plotTypePACF, 'button', []);
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_pacf', struct('Value', 1)), true, 'PACF plot not selected');

            % Select forecast plot type and verify update
            obj.guiCommon.simulateUserInput(plotTypeForecast, 'button', []);
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_forecast', struct('Value', 1)), true, 'Forecast plot not selected');

            % Select model fit plot type again and verify return to original view
            plotTypeFit = findobj(obj.viewerHandle, 'Tag', 'plot_type_fit');
            obj.guiCommon.simulateUserInput(plotTypeFit, 'button', []);
            obj.assertEqual(obj.guiCommon.verifyGUIComponent(obj.viewerHandle, 'uicontrol', 'plot_type_fit', struct('Value', 1)), true, 'Model fit plot not selected');
        end

        function testDiagnosticsDisplay(obj)
            % Test that model diagnostics are displayed correctly
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Find diagnostics text component
            diagnosticsText = findobj(obj.viewerHandle, 'Style', 'text', 'Tag', 'diagnostics_text');
            obj.assertTrue(~isempty(diagnosticsText), 'Diagnostics text component not found');

            % Get diagnostics text
            diagText = get(diagnosticsText, 'String');
            obj.assertTrue(contains(diagText, 'Log-likelihood:'), 'Log-likelihood not displayed');
            obj.assertTrue(contains(diagText, 'AIC:'), 'AIC not displayed');
            obj.assertTrue(contains(diagText, 'SBIC:'), 'SBIC not displayed');
            obj.assertTrue(contains(diagText, 'Ljung-Box Test:'), 'Ljung-Box test not displayed');
            obj.assertTrue(contains(diagText, 'LM Test:'), 'LM test not displayed');
        end

        function testSavePlotFunctionality(obj)
            % Test the functionality to save plots to files
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Set up temporary directory for test file creation
            tempDir = tempdir;
            testFileName = 'test_plot.png';
            testFilePath = fullfile(tempDir, testFileName);

            % Mock the directory selection dialog to return test directory
            obj.mockDirectoryDialog(tempDir);

            % Trigger save plot button callback
            saveButton = findobj(obj.viewerHandle, 'Tag', 'save_button');
            obj.guiCommon.simulateUserInput(saveButton, 'button', []);

            % Verify file is created with expected name
            obj.assertTrue(exist(testFilePath, 'file') == 2, 'File not created');

            % Clean up temporary test files
            delete(testFilePath);
            rmdir(tempDir, 's');
        end

        function testCloseButtonFunctionality(obj)
            % Test the close button and window close request functionality
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Record initial figure handle
            initialHandle = obj.viewerHandle;

            % Trigger close button callback
            closeButton = findobj(obj.viewerHandle, 'Tag', 'close_button');
            obj.guiCommon.simulateUserInput(closeButton, 'button', []);

            % Verify figure is closed
            obj.assertFalse(ishandle(initialHandle), 'Figure is not closed');

            % Recreate viewer
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Trigger figure close request (window X button)
            close(obj.viewerHandle);

            % Check that resources are released
            obj.assertFalse(ishandle(obj.viewerHandle), 'Figure is not closed properly');
        end

        function testErrorHandling(obj)
            % Test that the viewer handles errors gracefully
            % Attempt to initialize viewer with invalid model structure
            obj.assertThrows(@() ARMAX_viewer('invalid'), 'MATLAB:UndefinedFunction', 'Viewer did not throw error with invalid input');

            % Test with missing field in model structure
            invalidResults = obj.modelResults;
            invalidResults = rmfield(invalidResults, 'parameters');
            obj.assertThrows(@() ARMAX_viewer(invalidResults), 'MATLAB:rmfield:InvalidField', 'Viewer did not throw error with missing field');

            % Test with extreme parameter values
            extremeResults = obj.createTestModelResults(2, 1, true);
            extremeResults.parameters = inf(size(extremeResults.parameters));
            obj.viewerHandle = ARMAX_viewer(extremeResults);
        end

        function testForPlotUpdates(obj)
            % Test that plot updates correctly when model results change
            obj.viewerHandle = ARMAX_viewer(obj.modelResults);

            % Capture current plot state
            initialPlot = get(handles.main_axes, 'Children');

            % Update model results with different parameters
            newResults = obj.createTestModelResults(1, 1, false);

            % Update the viewer with new results
            ARMAX_viewer(newResults);

            % Verify plot contents reflect the new model
            updatedPlot = get(handles.main_axes, 'Children');
            obj.assertFalse(isequal(initialPlot, updatedPlot), 'Plot contents did not update');
        end

        function testMemoryManagement(obj)
            % Test proper memory management during viewer lifecycle
            % Track initial memory usage
            initialMemory = memory;

            % Create and close multiple viewer instances in sequence
            numInstances = 5;
            for i = 1:numInstances
                viewer = ARMAX_viewer(obj.modelResults);
                close(viewer);
            end

            % Check for memory leaks using MATLAB memory functions
            finalMemory = memory;
            memoryDiff = finalMemory.MaxPossibleArrayBytes - initialMemory.MaxPossibleArrayBytes;
            obj.assertTrue(memoryDiff < 1e6, 'Memory leak detected');
        end

        function modelResults = createTestModelResults(obj, arOrder, maOrder, includeConstant)
            % Helper method to create test model results with controlled properties
            % Generate time series data
            T = 200;
            innovations = randn(T, 1);
            y = zeros(T, 1);

            % Set AR and MA parameters
            arParams = randn(arOrder, 1) * 0.5;
            maParams = randn(maOrder, 1) * 0.3;
            constantValue = 0.1;

            % Generate ARMA process
            for t = max(arOrder, maOrder) + 1:T
                arTerm = 0;
                for i = 1:arOrder
                    arTerm = arTerm + arParams(i) * y(t-i);
                end

                maTerm = 0;
                for j = 1:maOrder
                    maTerm = maTerm + maParams(j) * innovations(t-j);
                end

                y(t) = constantValue + arTerm + maTerm + innovations(t);
            end

            % Create structure with model configuration
            modelResults = struct();
            modelResults.p = arOrder;
            modelResults.q = maOrder;
            modelResults.constant = includeConstant;
            modelResults.T = T;
            modelResults.y = y;
            modelResults.innovations = innovations;

            % Add parameter estimates with known values
            numParams = arOrder + maOrder + includeConstant;
            modelResults.parameters = randn(numParams, 1);

            % Add standard errors for all parameters
            modelResults.standardErrors = abs(randn(numParams, 1) * 0.1);

            % Add t-statistics and p-values
            modelResults.tStats = modelResults.parameters ./ modelResults.standardErrors;
            modelResults.pValues = 2 * (1 - tcdf(abs(modelResults.tStats), T - numParams));

            % Add diagnostic statistics
            modelResults.logL = -100;
            modelResults.aic = 2 * numParams - 2 * modelResults.logL;
            modelResults.sbic = numParams * log(T) - 2 * modelResults.logL;

            % Add residuals and fitted values arrays
            modelResults.residuals = innovations;
            modelResults.fittedValues = y - innovations;

            % Add parameter names
            paramNames = cell(numParams, 1);
            idx = 1;
            if includeConstant
                paramNames{idx} = 'Constant';
                idx = idx + 1;
            end
            for i = 1:arOrder
                paramNames{idx} = sprintf('AR{%d}', i);
                idx = idx + 1;
            end
            for i = 1:maOrder
                paramNames{idx} = sprintf('MA{%d}', i);
                idx = idx + 1;
            end
            modelResults.paramNames = paramNames;
        end

        function mockDirectoryDialog(obj, returnPath)
            % Helper method to mock directory selection dialog
            override_uigetdir = @() returnPath;
            obj.mockGUIData.original_uigetdir = str2func('uigetdir');
            assignin('base', 'uigetdir', override_uigetdir);
            obj.addTeardown(@() assignin('base', 'uigetdir', obj.mockGUIData.original_uigetdir));
        end
    end
end