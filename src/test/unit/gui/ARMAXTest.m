classdef ARMAXTest < BaseTest
    % ARMAXTEST Test class for the ARMAX GUI component, which provides interactive time series modeling with ARMAX.
    % Tests focus on initialization, parameter validation, model estimation, visualization, and user interaction.

    properties
        armaFigure          % Handle to the ARMAX GUI figure
        testHandles         % Structure containing handles to GUI components
        testData            % Structure containing test data
        guiCommon           % GUICommonTest instance
        dataGenerator       % TestDataGenerator instance
    end

    methods
        function obj = ARMAXTest()
            % ARMAXTEST Initialize a new ARMAXTest instance with test configuration

            % Call the superclass constructor from BaseTest
            obj = obj@BaseTest();

            % Initialize armaFigure handle to empty
            obj.armaFigure = [];

            % Initialize testHandles structure
            obj.testHandles = struct();

            % Create a GUICommonTest instance for shared GUI testing utilities
            obj.guiCommon = GUICommonTest();

            % Create a TestDataGenerator instance for generating test data
            obj.dataGenerator = TestDataGenerator;

            % Initialize empty testData structure
            obj.testData = struct();
        end

        function setUp(obj)
            % SETUP Prepares the test environment before each test method runs

            % Call superclass setUp method
            setUp@BaseTest(obj);

            % Close any open figures from previous tests
            close all;

            % Generate test data using TestDataGenerator
            obj.testData = obj.generateTestData(500, false);

            % Initialize ARMAX GUI in test mode
            obj.armaFigure = ARMAX('test');

            % Store figure handle for later use
            handles = guidata(obj.armaFigure);
            obj.testHandles = handles;

            % Wait for rendering to complete with drawnow
            drawnow;
        end

        function tearDown(obj)
            % TEARDOWN Cleans up the test environment after each test method completes

            % Close ARMAX GUI figure if it exists
            if ~isempty(obj.armaFigure) && ishandle(obj.armaFigure)
                close(obj.armaFigure);
            end

            % Clear handles and data structures
            obj.armaFigure = [];
            obj.testHandles = struct();
            obj.testData = struct();

            % Call GUICommon.closeAllTestFigures to ensure cleanup
            obj.guiCommon.closeAllTestFigures();

            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end

        function testInitialization(obj)
            % TESTINITIALIZATION Tests that the ARMAX GUI initializes correctly with default settings

            % Verify ARMAX figure exists and is visible
            obj.assertTrue(ishandle(obj.armaFigure), 'ARMAX figure does not exist');
            obj.assertEqual(get(obj.armaFigure, 'Visible'), 'on', 'ARMAX figure is not visible');

            % Check figure title contains 'ARMAX Model Estimation'
            obj.assertTrue(contains(get(obj.armaFigure, 'Name'), 'ARMAX Model Estimation'), 'Incorrect figure title');

            % Verify that default AR order is 1
            obj.assertEqual(str2double(get(obj.testHandles.AROrderEdit, 'String')), 1, 'Incorrect default AR order');

            % Verify that default MA order is 0
            obj.assertEqual(str2double(get(obj.testHandles.MAOrderEdit, 'String')), 0, 'Incorrect default MA order');

            % Verify that constant term is included by default
            obj.assertTrue(get(obj.testHandles.ConstantCheckbox, 'Value'), 'Incorrect default constant setting');

            % Check that all UI components are properly created with expected properties
            obj.guiCommon.verifyGUIComponent(obj.armaFigure, 'uicontrol', 'LoadDataButton', struct('String', 'Load Data'));
            obj.guiCommon.verifyGUIComponent(obj.armaFigure, 'uicontrol', 'EstimateButton', struct('String', 'Estimate'));
            obj.guiCommon.verifyGUIComponent(obj.armaFigure, 'uicontrol', 'ViewResultsButton', struct('String', 'View Results'));
        end

        function testLoadData(obj)
            % TESTLOADDATA Tests the data loading functionality of the ARMAX GUI

            % Create temporary data file with known time series
            tempFile = [tempdir, 'test_timeseries.mat'];
            timeSeries = obj.testData.y; %#ok<*PROPLC>
            save(tempFile, 'timeSeries');

            % Mock the file selection dialog to return the temporary file
            obj.mockFileDialog('load', tempFile);

            % Trigger the LoadDataButton callback
            obj.guiCommon.simulateUserInput(obj.testHandles.LoadDataButton, 'button', []);

            % Verify data is loaded correctly into the GUI
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(isequal(loadedData.data, timeSeries), 'Data not loaded correctly');

            % Check that the time series plot is updated with the data
            % (Basic check: verify axes limits are reasonable)
            axes_handle = obj.testHandles.TimeSeriesAxes;
            ylim = get(axes_handle, 'YLim');
            obj.assertTrue(ylim(1) < min(timeSeries) && ylim(2) > max(timeSeries), 'Time series plot not updated');

            % Verify that the Estimate button is enabled after data loading
            obj.assertEqual(get(obj.testHandles.EstimateButton, 'Enable'), 'on', 'Estimate button not enabled');

            % Clean up temporary test file
            delete(tempFile);
        end

        function testParameterValidation(obj)
            % TESTPARAMETERVALIDATION Tests the validation of AR and MA order input parameters

            % Simulate user input of invalid AR order (negative value)
            obj.guiCommon.simulateUserInput(obj.testHandles.AROrderEdit, 'edit', '-1');
            obj.assertEqual(str2double(get(obj.testHandles.AROrderEdit, 'String')), 1, 'AR order not reverted after invalid input');

            % Simulate user input of invalid AR order (non-integer)
            obj.guiCommon.simulateUserInput(obj.testHandles.AROrderEdit, 'edit', '1.5');
            obj.assertEqual(str2double(get(obj.testHandles.AROrderEdit, 'String')), 1, 'AR order not reverted after invalid input');

            % Simulate user input of invalid MA order (negative value)
            obj.guiCommon.simulateUserInput(obj.testHandles.MAOrderEdit, 'edit', '-1');
            obj.assertEqual(str2double(get(obj.testHandles.MAOrderEdit, 'String')), 0, 'MA order not reverted after invalid input');

            % Simulate user input of valid orders and verify acceptance
            obj.guiCommon.simulateUserInput(obj.testHandles.AROrderEdit, 'edit', '2');
            obj.assertEqual(str2double(get(obj.testHandles.AROrderEdit, 'String')), 2, 'Valid AR order not accepted');
            obj.guiCommon.simulateUserInput(obj.testHandles.MAOrderEdit, 'edit', '1');
            obj.assertEqual(str2double(get(obj.testHandles.MAOrderEdit, 'String')), 1, 'Valid MA order not accepted');
        end

        function testModelEstimation(obj)
            % TESTMODELESTIMATION Tests the ARMAX model estimation with valid parameters and data

            % Set up known test data and load into GUI
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);

            % Configure ARMA parameters (AR=2, MA=1, constant=true)
            obj.guiCommon.simulateUserInput(obj.testHandles.AROrderEdit, 'edit', '2');
            obj.guiCommon.simulateUserInput(obj.testHandles.MAOrderEdit, 'edit', '1');
            obj.guiCommon.simulateUserInput(obj.testHandles.ConstantCheckbox, 'checkbox', 1);

            % Trigger the EstimateButton callback
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Verify estimation completes without errors
            % (Check for existence of results structure)
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(isstruct(loadedData.results), 'Estimation did not complete successfully');

            % Check that fitted values are displayed in time series plot
            % (Basic check: verify axes limits are reasonable)
            axes_handle = obj.testHandles.TimeSeriesAxes;
            ylim = get(axes_handle, 'YLim');
            obj.assertTrue(ylim(1) < min(timeSeries) && ylim(2) > max(timeSeries), 'Time series plot not updated with fitted values');

            % Verify that model results structure is created with correct fields
            obj.assertTrue(isfield(loadedData.results, 'parameters'), 'Model results missing parameters');
            obj.assertTrue(isfield(loadedData.results, 'residuals'), 'Model results missing residuals');

            % Confirm that ViewResults button is enabled after estimation
            obj.assertEqual(get(obj.testHandles.ViewResultsButton, 'Enable'), 'on', 'ViewResults button not enabled');

            % Verify diagnostic statistics are displayed in the status area
            statusText = get(obj.testHandles.StatusText, 'String');
            obj.assertTrue(~isempty(statusText), 'Diagnostic statistics not displayed');
        end

        function testDistributionSelection(obj)
            % TESTDISTRIBUTIONSELECTION Tests the error distribution selection functionality

            % Set up test data and load into GUI
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);

            % Verify default distribution is Normal
            obj.assertEqual(get(obj.testHandles.DistributionPopup, 'Value'), 1, 'Default distribution not Normal');

            % Change distribution to Student's t
            set(obj.testHandles.DistributionPopup, 'Value', 2);
            obj.assertEqual(get(obj.testHandles.DistributionPopup, 'Value'), 2, 'Distribution not changed to Student''s t');

            % Estimate model and verify t distribution is used
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            loadedData = guidata(obj.armaFigure);
            obj.assertEqual(loadedData.results.distribution, 't', 'Distribution not set to t after estimation');

            % Change distribution to GED
            set(obj.testHandles.DistributionPopup, 'Value', 3);
            obj.assertEqual(get(obj.testHandles.DistributionPopup, 'Value'), 3, 'Distribution not changed to GED');

            % Estimate model and verify GED distribution is used
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            loadedData = guidata(obj.armaFigure);
            obj.assertEqual(loadedData.results.distribution, 'ged', 'Distribution not set to GED after estimation');

            % Change distribution to Skewed t
            set(obj.testHandles.DistributionPopup, 'Value', 4);
            obj.assertEqual(get(obj.testHandles.DistributionPopup, 'Value'), 4, 'Distribution not changed to Skewed t');

            % Estimate model and verify Skewed t distribution is used
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            loadedData = guidata(obj.armaFigure);
            obj.assertEqual(loadedData.results.distribution, 'skewt', 'Distribution not set to Skewed t after estimation');
        end

        function testDiagnosticsDisplay(obj)
            % TESTDIAGNOSTICSDISPLAY Tests the display of model diagnostics after estimation

            % Set up test data and estimate model with known parameters
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Check ACF checkbox and verify ACF plot is displayed
            obj.guiCommon.simulateUserInput(obj.testHandles.ACFCheckbox, 'checkbox', 1);
            obj.assertTrue(ishandle(findobj(obj.testHandles.DiagnosticsAxes, 'Type', 'hggroup')), 'ACF plot not displayed');

            % Verify ACF confidence intervals are shown correctly
            obj.assertTrue(ishandle(findobj(obj.testHandles.DiagnosticsAxes, 'Type', 'line')), 'ACF confidence intervals not displayed');

            % Check PACF checkbox and verify PACF plot is displayed
            obj.guiCommon.simulateUserInput(obj.testHandles.PACFCheckbox, 'checkbox', 1);
            obj.assertTrue(ishandle(findobj(obj.testHandles.DiagnosticsAxes, 'Type', 'hggroup')), 'PACF plot not displayed');

            % Verify PACF confidence intervals are shown correctly
            obj.assertTrue(ishandle(findobj(obj.testHandles.DiagnosticsAxes, 'Type', 'line')), 'PACF confidence intervals not displayed');

            % Check Residuals checkbox and verify residuals plot is displayed
            obj.guiCommon.simulateUserInput(obj.testHandles.ResidualCheckbox, 'checkbox', 1);
            obj.assertTrue(ishandle(findobj(obj.testHandles.DiagnosticsAxes, 'Type', 'line')), 'Residuals plot not displayed');

            % Verify that diagnostic checkboxes are mutually exclusive
            obj.assertFalse(get(obj.testHandles.ACFCheckbox, 'Value') && get(obj.testHandles.PACFCheckbox, 'Value'), 'ACF and PACF checkboxes not mutually exclusive');
            obj.assertFalse(get(obj.testHandles.ACFCheckbox, 'Value') && get(obj.testHandles.ResidualCheckbox, 'Value'), 'ACF and Residuals checkboxes not mutually exclusive');
            obj.assertFalse(get(obj.testHandles.PACFCheckbox, 'Value') && get(obj.testHandles.ResidualCheckbox, 'Value'), 'PACF and Residuals checkboxes not mutually exclusive');
        end

        function testViewResultsButton(obj)
            % TESTVIEWRESULTSBUTTON Tests the View Results button and integration with ARMAX_viewer

            % Set up test data and estimate model
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Verify ViewResults button is enabled after estimation
            obj.assertEqual(get(obj.testHandles.ViewResultsButton, 'Enable'), 'on', 'ViewResults button not enabled');

            % Mock the ARMAX_viewer call to track invocation
            viewerCalled = false;
            resultsPassed = [];
            mockViewer = @(results) set([viewerCalled, resultsPassed], [true, results]);
            restoreViewer = onCleanup(@() assignin('base', 'ARMAX_viewer', @ARMAX_viewer));
            assignin('base', 'ARMAX_viewer', mockViewer);

            % Click the ViewResults button
            obj.guiCommon.simulateUserInput(obj.testHandles.ViewResultsButton, 'button', []);

            % Verify ARMAX_viewer is called with correct model results
            obj.assertTrue(viewerCalled, 'ARMAX_viewer not called');
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(isequal(resultsPassed, loadedData.results), 'ARMAX_viewer not called with correct results');

            % Verify ARMAX_viewer window is displayed with correct title
            % (Check for existence of a figure with 'ARMAX Results' in the title)
            viewerFigure = obj.guiCommon.findAllFiguresByType('ARMAX Results');
            obj.assertTrue(~isempty(viewerFigure), 'ARMAX_viewer window not displayed');

            % Close viewer window and verify control returns to main ARMAX window
            close(viewerFigure);
            obj.assertTrue(ishandle(obj.armaFigure), 'ARMAX window closed unexpectedly');
        end

        function testForecastGeneration(obj)
            % TESTFORECASTGENERATION Tests the forecast generation functionality

            % Set up test data and estimate model
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Set forecast horizon to 5 periods
            obj.guiCommon.simulateUserInput(obj.testHandles.ForecastHorizonEdit, 'edit', '5');

            % Trigger forecast generation
            % (Estimation automatically generates forecasts)

            % Verify forecasts are computed correctly
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(isfield(loadedData.results, 'forecasts'), 'Forecasts not computed');
            obj.assertEqual(length(loadedData.results.forecasts.values), 5, 'Incorrect forecast horizon');

            % Check that forecast values and confidence intervals are displayed in the plot
            % (Basic check: verify axes limits are reasonable)
            axes_handle = obj.testHandles.TimeSeriesAxes;
            ylim = get(axes_handle, 'YLim');
            obj.assertTrue(ylim(1) < min(loadedData.results.forecasts.values) && ylim(2) > max(loadedData.results.forecasts.values), 'Forecasts not displayed in plot');

            % Change forecast horizon and verify update in forecasts
            obj.guiCommon.simulateUserInput(obj.testHandles.ForecastHorizonEdit, 'edit', '10');
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(length(loadedData.results.forecasts.values) == 10, 'Forecasts not updated after changing horizon');
        end

        function testSaveResults(obj)
            % TESTSAVERESULTS Tests the functionality to save model results to a file

            % Set up test data and estimate model
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Create temporary path for saving results
            tempFile = [tempdir, 'test_results.mat'];

            % Mock the file save dialog to return the temporary path
            obj.mockFileDialog('save', tempFile);

            % Trigger the SaveButton callback
            obj.guiCommon.simulateUserInput(obj.testHandles.SaveButton, 'button', []);

            % Verify that results file is created with correct content
            obj.assertTrue(exist(tempFile, 'file') == 2, 'Results file not created');
            savedData = load(tempFile);
            obj.assertTrue(isfield(savedData, 'saveResults'), 'Results not saved to file');

            % Load saved results and verify they match the model results
            loadedData = guidata(obj.armaFigure);
            obj.assertTrue(isequal(savedData.saveResults, loadedData.results), 'Saved results do not match model results');

            % Clean up temporary files
            delete(tempFile);
        end

        function testErrorHandling(obj)
            % TESTERRORHANDLING Tests that the GUI handles error conditions gracefully

            % Attempt to estimate model without loading data
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            % Verify appropriate error message is displayed
            % (Check for existence of a dialog with 'No Data' in the title)
            errorFigure = obj.guiCommon.findAllFiguresByType('No Data');
            obj.assertTrue(~isempty(errorFigure), 'Error message not displayed');
            close(errorFigure);

            % Load invalid data and verify error handling
            tempFile = [tempdir, 'test_invalid_data.txt'];
            fid = fopen(tempFile, 'w');
            fprintf(fid, 'Invalid Data');
            fclose(fid);
            obj.mockFileDialog('load', tempFile);
            obj.guiCommon.simulateUserInput(obj.testHandles.LoadDataButton, 'button', []);
            errorFigure = obj.guiCommon.findAllFiguresByType('Data Error');
            obj.assertTrue(~isempty(errorFigure), 'Error message not displayed for invalid data');
            close(errorFigure);
            delete(tempFile);

            % Test with extreme parameter values and verify graceful handling
            % (Set AR order to a very large number)
            obj.guiCommon.simulateUserInput(obj.testHandles.AROrderEdit, 'edit', '1000');
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            errorFigure = obj.guiCommon.findAllFiguresByType('Estimation Error');
            obj.assertTrue(~isempty(errorFigure), 'Error message not displayed for extreme parameters');
            close(errorFigure);

            % Verify application remains stable after error conditions
            obj.assertTrue(ishandle(obj.armaFigure), 'ARMAX window closed unexpectedly after error');
        end

        function testCloseButton(obj)
            % TESTCLOSEBUTTON Tests the Close button functionality and confirmation dialog

            % Set up test data and estimate model
            timeSeries = obj.testData.y;
            handles = guidata(obj.armaFigure);
            handles.data = timeSeries;
            guidata(obj.armaFigure, handles);
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);

            % Mock the ARMAX_close_dialog to return 'Cancel'
            obj.mockFileDialog('close', 'Cancel');

            % Trigger the Close_button callback
            obj.guiCommon.simulateUserInput(obj.testHandles.Close_button, 'button', []);

            % Verify figure remains open when cancellation is chosen
            obj.assertTrue(ishandle(obj.armaFigure), 'ARMAX window closed unexpectedly after cancellation');

            % Mock the ARMAX_close_dialog to return 'Yes'
            obj.mockFileDialog('close', 'Yes');

            % Trigger the Close_button callback
            obj.guiCommon.simulateUserInput(obj.testHandles.Close_button, 'button', []);

            % Verify figure is closed when confirmation is chosen
            obj.assertFalse(ishandle(obj.armaFigure), 'ARMAX window not closed after confirmation');

            % Handle figure cleanup to prevent test failures
            obj.armaFigure = [];
        end

        function testPerformance(obj)
            % TESTPERFORMANCE Tests the performance characteristics of the ARMAX GUI

            % Generate large test dataset (1000+ observations)
            largeData = obj.generateTestData(1500, false);

            % Measure time to load and display data
            loadTime = obj.measureExecutionTime(@(x) obj.loadDataIntoGUI(x), largeData.y);
            obj.assertTrue(loadTime < 5, 'Loading data took too long');

            % Measure time to estimate model with various parameter configurations
            estimationTime = obj.measureExecutionTime(@(x) obj.estimateModelInGUI(x), largeData.y);
            obj.assertTrue(estimationTime < 10, 'Estimation took too long');

            % Verify responsive UI during long operations
            % (Check that waitbar is displayed during estimation)
            % (This is difficult to automate fully, but we can check for waitbar existence)
            obj.guiCommon.simulateUserInput(obj.testHandles.EstimateButton, 'button', []);
            waitbarHandle = findobj('Type', 'figure', 'Name', 'Model Estimation');
            obj.assertTrue(ishandle(waitbarHandle), 'Waitbar not displayed during estimation');
            close(waitbarHandle);

            % Verify memory usage remains within acceptable limits
            memoryInfo = obj.checkMemoryUsage(@(x) obj.estimateModelInGUI(x), largeData.y);
            obj.assertTrue(memoryInfo.memoryDifferenceMB < 200, 'Memory usage exceeded limit');
        end

        function testGUILayout(obj)
            % TESTGUILAYOUT Tests the visual layout and component organization of the ARMAX GUI

            % Verify relative positions of major components (plots, controls)
            % Check visibility and enabled state of all components
            % Verify component sizes and proportions are appropriate
            % Test tab order for logical interaction flow
            % Verify resizing behavior maintains usable interface

            % Define expected layout structure
            expectedLayout = struct();

            % Check figure properties
            expectedLayout.figure = struct('Resize', 'on');

            % Check component groups
            expectedLayout.components.modelConfig = {
                struct('Type', 'uicontrol', 'Tag', 'AROrderEdit'),
                struct('Type', 'uicontrol', 'Tag', 'MAOrderEdit'),
                struct('Type', 'uicontrol', 'Tag', 'ConstantCheckbox'),
                struct('Type', 'uicontrol', 'Tag', 'EstimateButton')
            };

            expectedLayout.components.diagnostics = {
                struct('Type', 'uicontrol', 'Tag', 'ACFCheckbox'),
                struct('Type', 'uicontrol', 'Tag', 'PACFCheckbox'),
                struct('Type', 'uicontrol', 'Tag', 'ResidualCheckbox'),
                struct('Type', 'uicontrol', 'Tag', 'StatusText')
            };

            expectedLayout.components.buttons = {
                struct('Type', 'uicontrol', 'Tag', 'ViewResultsButton'),
                struct('Type', 'uicontrol', 'Tag', 'SaveButton'),
                struct('Type', 'uicontrol', 'Tag', 'Close_button')
            };

            % Verify layout
            obj.guiCommon.verifyGUILayout(obj.armaFigure, expectedLayout);
        end

        function testARMAXHelp(obj)
            % TESTARMAXHELP Tests the Help functionality of the ARMAX GUI

            % Mock the help dialog display
            helpCalled = false;
            mockHelp = @() set(helpCalled, true);
            restoreHelp = onCleanup(@() assignin('base', 'help', @help));
            assignin('base', 'help', mockHelp);

            % Trigger the Help menu callback
            helpMenu = findobj(obj.armaFigure, 'Tag', 'Help_Callback');
            obj.guiCommon.simulateUserInput(helpMenu, 'button', []);

            % Verify help information is displayed with correct content
            obj.assertTrue(helpCalled, 'Help information not displayed');

            % Verify help window can be closed properly
            % (This is difficult to automate fully, but we can check that the help function was called)
        end

        function testARMAXAbout(obj)
            % TESTARMAXABOUT Tests the About dialog functionality of the ARMAX GUI

            % Mock the ARMAX_about dialog
            aboutCalled = false;
            mockAbout = @() set(aboutCalled, true);
            restoreAbout = onCleanup(@() assignin('base', 'ARMAX_about', @ARMAX_about));
            assignin('base', 'ARMAX_about', mockAbout);

            % Trigger the About menu callback
            aboutMenu = findobj(obj.armaFigure, 'Tag', 'About_Callback');
            obj.guiCommon.simulateUserInput(aboutMenu, 'button', []);

            % Verify About dialog is displayed with correct version information
            obj.assertTrue(aboutCalled, 'About dialog not displayed');

            % Verify About dialog can be closed properly
            % (This is difficult to automate fully, but we can check that the ARMAX_about function was called)
        end

        function testData = generateTestData(obj, numObservations, includeExogenous)
            % GENERATETESTDATA Helper method to generate standardized test data for ARMAX GUI testing
            %   numObservations - Number of observations in the time series
            %   includeExogenous - Logical flag to include exogenous variables

            % Use dataGenerator to create time series data with known properties
            testData = struct();
            testData.numObs = numObservations;
            testData.ar = [0.5, -0.2];
            testData.ma = [0.3, 0.1];
            testData.constant = 0.001;
            testData.distribution = 'normal';

            % Add exogenous variables if requested
            if includeExogenous
                testData.exogenousVars = true;
                testData.exogenousCoef = [0.5, -0.3];
                testData.numExoVars = 2;
            else
                testData.exogenousVars = false;
                testData.exogenousCoef = [];
                testData.numExoVars = 0;
            end

            % Add specific characteristics for testing diagnostics
            testData.testDiagnostics = true;

            % Generate time series data using TestDataGenerator function
            tsData = obj.dataGenerator('generateTimeSeriesData', testData);

            % Return structured test data for ARMAX testing
            testData = tsData;
        end

        function mockFileDialog(obj, dialogType, returnPath)
            % MOCKFILEDIALOG Helper method to mock file dialog operations for testing
            %   dialogType - 'load' or 'save'
            %   returnPath - Path to return from the dialog

            switch lower(dialogType)
                case 'load'
                    % Override uigetfile function
                    originalUigetfile = @uigetfile;
                    mockUigetfile = @(varargin) deal(fileparts(returnPath), returnPath);
                    assignin('base', 'uigetfile', mockUigetfile);
                    cleanupUigetfile = onCleanup(@() assignin('base', 'uigetfile', originalUigetfile));

                case 'save'
                    % Override uiputfile function
                    originalUiputfile = @uiputfile;
                    mockUiputfile = @(varargin) deal(fileparts(returnPath), returnPath);
                    assignin('base', 'uiputfile', mockUiputfile);
                    cleanupUiputfile = onCleanup(@() assignin('base', 'uiputfile', originalUiputfile));

                case 'close'
                    % Mock the ARMAX_close_dialog
                    originalCloseDialog = @ARMAX_close_dialog;
                    mockCloseDialog = @() returnPath;
                    assignin('base', 'ARMAX_close_dialog', mockCloseDialog);
                    cleanupCloseDialog = onCleanup(@() assignin('base', 'ARMAX_close_dialog', originalCloseDialog));

                otherwise
                    error('Invalid dialog type');
            end
        end

        function success = verifyModelEstimation(obj, handles, expectedParams)
            % VERIFYMODELESTIMATION Helper method to verify model estimation results
            %   handles - GUI handles structure
            %   expectedParams - Expected parameter values

            % Extract model results from handles structure
            results = handles.results;

            % Compare estimated parameters with expected values
            % (Implement specific checks based on model configuration)
            % (This is a placeholder - implement actual checks here)

            % Check standard errors are reasonable
            % (Implement checks to ensure standard errors are not NaN or Inf)

            % Verify diagnostic statistics are computed
            % (Check that Ljung-Box test and LM test results are available)

            % Return true if all validation checks pass
            success = true;
        end

        function loadDataIntoGUI(obj, data)
            % Helper function to load data into the GUI for performance testing
            handles = guidata(obj.armaFigure);
            handles.data = data;
            handles.hasData = true;
            guidata(obj.armaFigure, handles);
            UpdateDataPlot(handles);
        end

        function estimateModelInGUI(obj, data)
            % Helper function to estimate the model in the GUI for performance testing
            handles = guidata(obj.armaFigure);
            handles.data = data;
            handles.hasData = true;
            guidata(obj.armaFigure, handles);
            EstimateButton_Callback(obj.testHandles.EstimateButton, [], handles);
        end
    end
end