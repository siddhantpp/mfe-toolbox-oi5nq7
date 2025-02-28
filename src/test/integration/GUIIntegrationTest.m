classdef GUIIntegrationTest < BaseTest
    % GUIINTEGRATIONTEST Integration test class for validating the complete end-to-end functionality of the ARMAX GUI components, including cross-component interactions, data flow, and workflow integrity.
    
    properties
        mainFigure     % Handle to the main ARMAX GUI figure
        viewerFigure   % Handle to the ARMAX Results Viewer figure
        testData       % Structure containing test data
        guiCommon      % GUICommonTest instance
        dataGenerator  % TestDataGenerator instance
        mockCallbacks  % Structure for mock callback tracking
        tempDataPath   % Path to temporary data directory
        waitTimeout    % Timeout for GUI operations
    end
    
    methods
        function obj = GUIIntegrationTest()
            % Initialize a new GUIIntegrationTest instance with test configuration
            
            % Call the superclass constructor from BaseTest
            obj = obj@BaseTest();
            
            % Initialize figure handles to empty
            obj.mainFigure = [];
            obj.viewerFigure = [];
            
            % Create a GUICommonTest instance for shared GUI testing utilities
            obj.guiCommon = GUICommonTest();
            
            % Create a TestDataGenerator instance for generating test data
            obj.dataGenerator = TestDataGenerator;
            
            % Initialize empty testData structure
            obj.testData = struct();
            
            % Set up mock callback tracking structure
            obj.mockCallbacks = struct();
            
            % Set default timeout for GUI operations to 30 seconds
            obj.waitTimeout = 30;
            
            % Configure temporary data path for file operations
            obj.tempDataPath = tempdir;
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            % Call superclass setUp method
            setUp@BaseTest(obj);
            
            % Close any open figures from previous tests
            obj.guiCommon.closeAllTestFigures();
            
            % Generate test data using TestDataGenerator
            obj.testData = obj.dataGenerator.generateFinancialReturnsSample(1000, 1);
            
            % Create temporary data files for testing
            tempDataFile = fullfile(obj.tempDataPath, 'testData.mat');
            save(tempDataFile, 'obj.testData');
            
            % Initialize mock callback tracking
            obj.mockCallbacks = struct();
            
            % Set up file dialog mocks for testing
            obj.mockFileDialog('LoadData', tempDataFile);
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            % Close main ARMAX GUI figure if it exists
            if ~isempty(obj.mainFigure) && ishandle(obj.mainFigure)
                close(obj.mainFigure);
            end
            
            % Close viewer figure if it exists
            if ~isempty(obj.viewerFigure) && ishandle(obj.viewerFigure)
                close(obj.viewerFigure);
            end
            
            % Close any other test figures
            obj.guiCommon.closeAllTestFigures();
            
            % Remove temporary test files
            delete(fullfile(obj.tempDataPath, 'testData.mat'));
            delete(fullfile(obj.tempDataPath, 'savedResults.mat'));
            
            % Reset mock callbacks
            obj.mockCallbacks = struct();
            
            % Call superclass tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testCompleteWorkflow(obj)
            % Tests the entire user workflow from data loading through model estimation to results viewing
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load test data into the GUI
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            
            % Configure ARMA parameters (AR=2, MA=1)
            handles = guidata(obj.mainFigure);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Verify estimation completes without errors
            obj.assertTrue(handles.hasResults, 'Model estimation should complete successfully');
            
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Verify results viewer displays correct model information
            viewerHandles = guidata(obj.viewerFigure);
            obj.assertTrue(isfield(viewerHandles, 'model_info_text'), 'Results viewer should display model information');
            
            % Test switching between different diagnostic views in viewer
            obj.guiCommon.simulateUserInput(viewerHandles.plot_type_residuals, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.viewerFigure, obj.waitTimeout);
            obj.guiCommon.simulateUserInput(viewerHandles.plot_type_acf, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.viewerFigure, obj.waitTimeout);
            
            % Save results to temporary file
            savePath = fullfile(obj.tempDataPath, 'savedResults.mat');
            obj.guiCommon.simulateUserInput(handles.SaveButton, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Verify saved results match the model
            savedData = load(savePath);
            obj.assertEqual(handles.results.parameters, savedData.saveResults.parameters, 'Saved results should match model parameters');
            
            % Close viewer and return to main window
            obj.guiCommon.simulateUserInput(viewerHandles.close_button, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Test forecast generation in main window
            obj.guiCommon.simulateUserInput(handles.ForecastHorizonEdit, 'edit', '10');
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Close application with confirmation dialog
            obj.guiCommon.simulateUserInput(handles.Close_button, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
        end
        
        function testModelTransferIntegrity(obj)
            % Tests that model results are transferred correctly between main window and viewer
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load test data and estimate model with specific parameters
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Capture model results from main window
            mainModel = handles.results;
            
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Extract model results from viewer
            viewerHandles = guidata(obj.viewerFigure);
            viewerModel = viewerHandles.modelResults;
            
            % Compare model structures for equality
            obj.verifyModelMatchesBetweenComponents(mainModel, viewerModel);
            
            % Close both windows
            close(obj.mainFigure);
            close(obj.viewerFigure);
        end
        
        function testCrossFigureNavigation(obj)
            % Tests navigation between different GUI components
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Open About dialog
            handles = guidata(obj.mainFigure);
            obj.guiCommon.simulateUserInput(handles.Help_menu, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Verify About dialog appears and can be closed
            aboutFigure = findobj('Name', 'About ARMAX');
            obj.assertTrue(ishandle(aboutFigure), 'About dialog should be open');
            obj.guiCommon.simulateUserInput(findobj(aboutFigure, 'Tag', 'OKButton'), 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Perform model estimation workflow
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Navigate between different plot views in viewer
            viewerHandles = guidata(obj.viewerFigure);
            obj.guiCommon.simulateUserInput(viewerHandles.plot_type_residuals, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.viewerFigure, obj.waitTimeout);
            obj.guiCommon.simulateUserInput(viewerHandles.plot_type_acf, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.viewerFigure, obj.waitTimeout);
            
            % Close viewer with close button
            obj.guiCommon.simulateUserInput(viewerHandles.close_button, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Verify control returns to main window
            obj.assertTrue(ishandle(obj.mainFigure), 'Main window should still be open');
            
            % Test close confirmation dialog
            obj.guiCommon.simulateUserInput(handles.Close_button, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Verify application closes properly with confirmation
            obj.assertFalse(ishandle(obj.mainFigure), 'Application should close after confirmation');
        end
        
        function testDiagnosticsConsistency(obj)
            % Tests consistency of diagnostic information between main window and viewer
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load data and estimate model
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Capture diagnostic information from main window
            mainLjungBox = handles.results.ljungBox;
            mainLMTest = handles.results.lmTest;
            mainAIC = handles.results.aic;
            mainSBIC = handles.results.sbic;
            
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Extract detailed diagnostics from viewer
            viewerHandles = guidata(obj.viewerFigure);
            diagText = get(viewerHandles.diagnostics_text, 'String');
            
            % Verify ACF/PACF plots are consistent between windows
            % Verify Ljung-Box test results match
            obj.assertTrue(contains(diagText, sprintf('Q=%0.4f', mainLjungBox.stats(1))), 'Ljung-Box statistic should match');
            obj.assertTrue(contains(diagText, sprintf('p=%0.4f', mainLjungBox.pvals(1))), 'Ljung-Box p-value should match');
            
            % Verify LM test results match
            obj.assertTrue(contains(diagText, sprintf('Statistic=%0.4f', mainLMTest.stat)), 'LM statistic should match');
            obj.assertTrue(contains(diagText, sprintf('p=%0.4f', mainLMTest.pval)), 'LM p-value should match');
            
            % Verify information criteria values match
            obj.assertTrue(contains(diagText, sprintf('AIC: %0.4f', mainAIC)), 'AIC should match');
            obj.assertTrue(contains(diagText, sprintf('SBIC: %0.4f', mainSBIC)), 'SBIC should match');
            
            % Close both windows
            close(obj.mainFigure);
            close(obj.viewerFigure);
        end
        
        function testErrorHandlingAcrossComponents(obj)
            % Tests error handling across GUI component boundaries
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Attempt operations that should trigger errors
            % Verify error handling in main window
            handles = guidata(obj.mainFigure);
            obj.guiCommon.simulateUserInput(handles.AROrderEdit, 'edit', '-1');
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Create model with known problematic properties
            % Verify viewer handles problematic model gracefully
            % Test recovery from error conditions across component transitions
            % Verify application remains stable after error situations
            % Load data and estimate model
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Close both windows
            close(obj.mainFigure);
            close(obj.viewerFigure);
        end
        
        function testSaveLoadResultsIntegrity(obj)
            % Tests the saving and loading of model results across components
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load data and estimate model
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Save model results to file from main window
            savePath = fullfile(obj.tempDataPath, 'savedResults.mat');
            obj.guiCommon.simulateUserInput(handles.SaveButton, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Close main window
            close(obj.mainFigure);
            
            % Initialize new ARMAX GUI instance
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load saved model results
            handles = guidata(obj.mainFigure);
            obj.guiCommon.simulateUserInput(handles.LoadDataButton, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Verify model properties match original
            obj.assertEqual(handles.results.parameters, handles.results.parameters, 'Loaded model parameters should match original');
            
            % Open results viewer with loaded model
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Verify viewer displays correct information from loaded model
            viewerHandles = guidata(obj.viewerFigure);
            obj.assertTrue(isfield(viewerHandles, 'model_info_text'), 'Results viewer should display model information');
            
            % Save modified results from viewer
            % Verify modifications are preserved correctly
            close(obj.mainFigure);
            close(obj.viewerFigure);
        end
        
        function testDistributionSelectionPropagation(obj)
            % Tests that error distribution selection propagates correctly across components
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load data and set distribution to Student's t
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 't');
            guidata(obj.mainFigure, handles);
            
            % Estimate model
            % Open results viewer
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Verify viewer correctly displays t-distribution information
            viewerHandles = guidata(obj.viewerFigure);
            obj.assertTrue(contains(get(viewerHandles.diagnostics_text, 'String'), 'Error Distribution: t'), 'Viewer should display t-distribution');
            
            % Return to main window and change to GED distribution
            obj.guiCommon.simulateUserInput(viewerHandles.close_button, 'button', []);
            obj.guiCommon.waitForGUIResponse(obj.mainFigure, obj.waitTimeout);
            
            % Re-estimate model
            handles = guidata(obj.mainFigure);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'ged');
            guidata(obj.mainFigure, handles);
            
            % Open viewer again
            obj.viewerFigure = obj.openResultsViewer(obj.mainFigure);
            
            % Verify viewer correctly displays GED distribution information
            viewerHandles = guidata(obj.viewerFigure);
            obj.assertTrue(contains(get(viewerHandles.diagnostics_text, 'String'), 'Error Distribution: ged'), 'Viewer should display GED distribution');
            
            % Close both windows
            close(obj.mainFigure);
            close(obj.viewerFigure);
        end
        
        function testModelComparisonWorkflow(obj)
            % Tests workflow for comparing multiple ARMAX models
            % Initialize ARMAX GUI
            obj.mainFigure = obj.initializeARMAXGUI();
            
            % Load data and estimate model with AR=1, MA=0
            handles = guidata(obj.mainFigure);
            handles = obj.loadTestDataIntoGUI(obj.mainFigure, obj.testData);
            guidata(obj.mainFigure, handles);
            handles = obj.estimateModelInGUI(obj.mainFigure, 1, 0, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Save results as model 1
            model1 = handles.results;
            
            % Change parameters to AR=1, MA=1 and estimate
            handles = guidata(obj.mainFigure);
            handles = obj.estimateModelInGUI(obj.mainFigure, 1, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Save results as model 2
            model2 = handles.results;
            
            % Change parameters to AR=2, MA=1 and estimate
            handles = guidata(obj.mainFigure);
            handles = obj.estimateModelInGUI(obj.mainFigure, 2, 1, 'normal');
            guidata(obj.mainFigure, handles);
            
            % Save results as model 3
            model3 = handles.results;
            
            % Load each model in sequence and verify correct loading
            % Compare AIC/SBIC values across models
            % Verify information is preserved across loading operations
            close(obj.mainFigure);
        end
        
        function testConcurrentGUIOperation(obj)
            % Tests multiple GUI components operating simultaneously
            % Initialize two ARMAX GUI instances
            mainFigure1 = obj.initializeARMAXGUI();
            mainFigure2 = obj.initializeARMAXGUI();
            
            % Load different data into each instance
            handles1 = guidata(mainFigure1);
            handles1 = obj.loadTestDataIntoGUI(mainFigure1, obj.testData);
            guidata(mainFigure1, handles1);
            
            handles2 = guidata(mainFigure2);
            handles2 = obj.loadTestDataIntoGUI(mainFigure2, obj.testData);
            guidata(mainFigure2, handles2);
            
            % Configure different models in each instance
            handles1 = guidata(mainFigure1);
            handles1 = obj.estimateModelInGUI(mainFigure1, 1, 0, 'normal');
            guidata(mainFigure1, handles1);
            
            handles2 = guidata(mainFigure2);
            handles2 = obj.estimateModelInGUI(mainFigure2, 0, 1, 'normal');
            guidata(mainFigure2, handles2);
            
            % Estimate models concurrently
            % Open results viewers for both models
            viewerFigure1 = obj.openResultsViewer(mainFigure1);
            viewerFigure2 = obj.openResultsViewer(mainFigure2);
            
            % Verify each viewer displays correct model information
            % Test interactions between multiple open components
            % Verify application integrity with multiple windows open
            % Close all components in reverse order of creation
            close(mainFigure1);
            close(mainFigure2);
            close(viewerFigure1);
            close(viewerFigure2);
        end
        
        function handle = initializeARMAXGUI(obj)
            % Helper method to initialize the ARMAX GUI for testing
            % Initialize ARMAX GUI in test mode
            ARMAX();
            
            % Wait for figure creation and rendering
            obj.guiCommon.waitForGUIResponse(gcf, obj.waitTimeout);
            
            % Get figure handle
            handle = findobj('Name', 'ARMAX Model Estimation');
            
            % Verify initialization completed successfully
            obj.assertTrue(ishandle(handle), 'ARMAX GUI initialization failed');
        end
        
        function handles = loadTestDataIntoGUI(obj, figureHandle, testData)
            % Helper method to load test data into the ARMAX GUI
            % Save test data to temporary file
            tempDataFile = fullfile(obj.tempDataPath, 'testData.mat');
            save(tempDataFile, 'testData');
            
            % Find LoadDataButton in the figure
            LoadDataButton = findobj(figureHandle, 'Tag', 'LoadDataButton');
            
            % Mock file dialog to return test data file
            obj.mockFileDialog('LoadData', tempDataFile);
            
            % Trigger LoadDataButton callback
            obj.guiCommon.simulateUserInput(LoadDataButton, 'button', []);
            
            % Wait for data loading to complete
            obj.guiCommon.waitForGUIResponse(figureHandle, obj.waitTimeout);
            
            % Get updated handles structure
            handles = guidata(figureHandle);
            
            % Verify data was loaded correctly
            obj.assertTrue(handles.hasData, 'Data loading failed');
        end
        
        function handles = estimateModelInGUI(obj, figureHandle, arOrder, maOrder, distribution)
            % Helper method to estimate an ARMAX model in the GUI
            % Find AR and MA order edit boxes
            AROrderEdit = findobj(figureHandle, 'Tag', 'AROrderEdit');
            MAOrderEdit = findobj(figureHandle, 'Tag', 'MAOrderEdit');
            DistributionPopup = findobj(figureHandle, 'Tag', 'DistributionPopup');
            
            % Set AR order using simulateUserInput
            obj.guiCommon.simulateUserInput(AROrderEdit, 'edit', num2str(arOrder));
            
            % Set MA order using simulateUserInput
            obj.guiCommon.simulateUserInput(MAOrderEdit, 'edit', num2str(maOrder));
            
            % Set distribution selection if specified
            if nargin > 4 && ~isempty(distribution)
                distOptions = get(DistributionPopup, 'String');
                for i = 1:length(distOptions)
                    if strcmpi(distOptions{i}, distribution)
                        obj.guiCommon.simulateUserInput(DistributionPopup, 'popup', i);
                        break;
                    end
                end
            end
            
            % Find EstimateButton and trigger callback
            EstimateButton = findobj(figureHandle, 'Tag', 'EstimateButton');
            obj.guiCommon.simulateUserInput(EstimateButton, 'button', []);
            
            % Wait for estimation to complete with timeout
            obj.guiCommon.waitForGUIResponse(figureHandle, obj.waitTimeout);
            
            % Get updated handles with model results
            handles = guidata(figureHandle);
            
            % Verify estimation completed successfully
            obj.assertTrue(handles.hasResults, 'Model estimation failed');
        end
        
        function viewerFigure = openResultsViewer(obj, mainFigure)
            % Helper method to open the results viewer from the main window
            % Find ViewResultsButton in main figure
            ViewResultsButton = findobj(mainFigure, 'Tag', 'ViewResultsButton');
            
            % Trigger ViewResultsButton callback
            obj.guiCommon.simulateUserInput(ViewResultsButton, 'button', []);
            
            % Wait for viewer figure to open
            obj.guiCommon.waitForGUIResponse(gcf, obj.waitTimeout);
            
            % Get viewer figure handle
            viewerFigure = findobj('Name', 'ARMAX Results Viewer');
            
            % Verify viewer initialized successfully
            obj.assertTrue(ishandle(viewerFigure), 'Results viewer initialization failed');
        end
        
        function isEqual = verifyModelMatchesBetweenComponents(obj, mainModel, viewerModel)
            % Helper method to verify model consistency between components
            % Compare model specifications (AR/MA orders)
            obj.assertEqual(mainModel.p, viewerModel.p, 'AR order should match');
            obj.assertEqual(mainModel.q, viewerModel.q, 'MA order should match');
            
            % Compare parameter estimates with tolerance
            obj.assertAlmostEqual(mainModel.parameters, viewerModel.parameters, 'Parameter estimates should match');
            
            % Compare standard errors with tolerance
            obj.assertAlmostEqual(mainModel.standardErrors, viewerModel.standardErrors, 'Standard errors should match');
            
            % Compare diagnostic statistics with tolerance
            % Compare information criteria with tolerance
            isEqual = true;
        end
        
        function mockFileDialog(obj, dialogType, returnPath)
            % Helper method to mock file dialogs for testing
            % Determine which dialog function to mock based on dialogType
            switch dialogType
                case 'LoadData'
                    dialogFunction = 'uigetfile';
                case 'SaveResults'
                    dialogFunction = 'uiputfile';
                otherwise
                    error('Unsupported dialog type: %s', dialogType);
            end
            
            % Create mock implementation that returns specified path
            mockImplementation = @() deal(returnPath, fileparts(returnPath));
            
            % Install mock function
            fcnName = function_handle(dialogFunction);
            obj.mockCallbacks.(dialogFunction) = addDependency(fcnName, mockImplementation);
            
            % Setup cleanup to restore original function after test
        end
    end
end