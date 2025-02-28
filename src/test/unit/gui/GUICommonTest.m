classdef GUICommonTest < BaseTest
    % GUICOMMONTEST Common utility class for GUI component testing in the MFE Toolbox.
    % Provides shared functionality and helper methods for testing all GUI components.
    
    properties
        dataGenerator    % TestDataGenerator instance
        mockGUIData      % Structure for mock GUI data
        defaultTimeout   % Default timeout for GUI operations
        captureScreenshots % Flag for capturing screenshots
    end
    
    methods
        function obj = GUICommonTest()
            % Initialize a new GUICommonTest instance with default settings
            obj = obj@BaseTest();
            obj.dataGenerator = @TestDataGenerator;
            obj.defaultTimeout = 10; % Default timeout in seconds
            obj.captureScreenshots = false;
            obj.mockGUIData = struct();
        end
        
        function setUp(obj)
            % Prepares the test environment before each test method runs
            setUp@BaseTest(obj);
            % Set fixed seed for reproducibility
            rng(123, 'twister');
            % Close any open figures from previous tests
            close all;
            % Reset mockGUIData
            obj.mockGUIData = struct();
        end
        
        function tearDown(obj)
            % Cleans up the test environment after each test method completes
            obj.closeAllTestFigures();
            obj.mockGUIData = struct();
            tearDown@BaseTest(obj);
        end
        
        function result = verifyGUIComponent(obj, figureHandle, componentType, componentTag, expectedProperties)
            % Verifies that a GUI component exists and has expected properties
            %
            % INPUTS:
            %   figureHandle - Handle to the figure containing the component
            %   componentType - Type of the component (e.g., 'uicontrol', 'axes')
            %   componentTag - Tag of the component to find
            %   expectedProperties - Structure containing expected property values
            %
            % OUTPUTS:
            %   result - True if component exists and matches expectations
            
            % Validate input parameters
            parametercheck(figureHandle, 'figureHandle');
            parametercheck(componentType, 'componentType');
            parametercheck(componentTag, 'componentTag');
            
            % Initialize result
            result = false;
            
            % Find component in the figure
            component = findobj(figureHandle, 'Type', componentType, 'Tag', componentTag);
            
            % Verify component exists
            if isempty(component)
                error('Component of type "%s" with tag "%s" not found in figure.', componentType, componentTag);
            end
            
            % If no expected properties provided, just verify existence
            if nargin < 5 || isempty(expectedProperties)
                result = true;
                return;
            end
            
            % Check each expected property
            propFields = fieldnames(expectedProperties);
            for i = 1:length(propFields)
                propName = propFields{i};
                expectedValue = expectedProperties.(propName);
                
                % Get actual property value
                try
                    actualValue = get(component, propName);
                catch ME
                    error('Failed to get property "%s" from component: %s', propName, ME.message);
                end
                
                % Compare property values
                if ~isequal(actualValue, expectedValue)
                    error('Property "%s" mismatch. Expected: %s, Actual: %s', ...
                        propName, mat2str(expectedValue), mat2str(actualValue));
                end
            end
            
            % All checks passed
            result = true;
        end
        
        function simulateUserInput(obj, componentHandle, inputType, inputValue)
            % Simulates user input on a GUI component
            %
            % INPUTS:
            %   componentHandle - Handle to the GUI component
            %   inputType - Type of input ('edit', 'checkbox', 'popup', 'button')
            %   inputValue - Value to set (depends on input type)
            
            % Validate component handle
            if ~ishandle(componentHandle)
                error('Invalid component handle provided.');
            end
            
            % Get component type
            componentType = get(componentHandle, 'Type');
            
            % Process based on input type
            switch lower(inputType)
                case 'edit'
                    % For edit boxes, set the 'String' property
                    set(componentHandle, 'String', inputValue);
                    % Trigger callback if available
                    callback = get(componentHandle, 'Callback');
                    if ~isempty(callback)
                        if iscell(callback)
                            feval(callback{1}, componentHandle, [], callback{2:end});
                        else
                            feval(callback, componentHandle, []);
                        end
                    end
                    
                case 'checkbox'
                    % For checkboxes, set the 'Value' property
                    set(componentHandle, 'Value', inputValue);
                    % Trigger callback if available
                    callback = get(componentHandle, 'Callback');
                    if ~isempty(callback)
                        if iscell(callback)
                            feval(callback{1}, componentHandle, [], callback{2:end});
                        else
                            feval(callback, componentHandle, []);
                        end
                    end
                    
                case 'popup'
                    % For popup menus, set the 'Value' property
                    set(componentHandle, 'Value', inputValue);
                    % Trigger callback if available
                    callback = get(componentHandle, 'Callback');
                    if ~isempty(callback)
                        if iscell(callback)
                            feval(callback{1}, componentHandle, [], callback{2:end});
                        else
                            feval(callback, componentHandle, []);
                        end
                    end
                    
                case 'button'
                    % For buttons, trigger the callback
                    callback = get(componentHandle, 'Callback');
                    if ~isempty(callback)
                        if iscell(callback)
                            feval(callback{1}, componentHandle, [], callback{2:end});
                        else
                            feval(callback, componentHandle, []);
                        end
                    end
                    
                otherwise
                    error('Unsupported input type: %s', inputType);
            end
        end
        
        function mockResults = createMockModelResults(obj, arOrder, maOrder, includeConstant)
            % Creates mock model results structure for GUI testing
            %
            % INPUTS:
            %   arOrder - AR order (number of AR parameters)
            %   maOrder - MA order (number of MA parameters)
            %   includeConstant - Whether to include a constant term
            %
            % OUTPUTS:
            %   mockResults - Mock model results structure
            
            % Default values
            if nargin < 4
                includeConstant = true;
            end
            
            % Generate sample data
            sampleSize = 1000;
            
            % Create parameters for time series data
            tsParams = struct();
            tsParams.ar = 0.7 * rand(arOrder, 1) - 0.35; % Random AR parameters between -0.35 and 0.35
            tsParams.ma = 0.5 * rand(maOrder, 1) - 0.25; % Random MA parameters between -0.25 and 0.25
            if includeConstant
                tsParams.constant = 0.01 * randn(); % Small random constant
            end
            tsParams.numObs = sampleSize;
            tsParams.distribution = 'normal';
            
            % Generate time series data using TestDataGenerator function
            tsData = obj.dataGenerator('generateTimeSeriesData', tsParams);
            
            % Extract data, parameters and residuals
            data = tsData.y;
            residuals = tsData.innovations;
            arParams = tsData.trueParameters.ar;
            maParams = tsData.trueParameters.ma;
            if includeConstant
                constantValue = tsData.trueParameters.constant;
            else
                constantValue = 0;
            end
            
            % Number of parameters
            numParams = arOrder + maOrder + includeConstant;
            
            % Create model structure
            mockResults = struct();
            
            % Set parameters (use the ones we generated with)
            if includeConstant
                mockResults.parameters = [constantValue; arParams; maParams];
            else
                mockResults.parameters = [arParams; maParams];
            end
            
            % Create standard errors, t-stats, and p-values
            mockResults.stderrors = abs(randn(numParams, 1) * 0.05);
            mockResults.tstat = mockResults.parameters ./ mockResults.stderrors;
            mockResults.pvals = 2 * (1 - tcdf(abs(mockResults.tstat), sampleSize - numParams));
            
            % Parameter names
            paramNames = cell(numParams, 1);
            paramIdx = 1;
            
            if includeConstant
                paramNames{paramIdx} = 'Constant';
                paramIdx = paramIdx + 1;
            end
            
            for i = 1:arOrder
                paramNames{paramIdx} = sprintf('AR{%d}', i);
                paramIdx = paramIdx + 1;
            end
            
            for i = 1:maOrder
                paramNames{paramIdx} = sprintf('MA{%d}', i);
                paramIdx = paramIdx + 1;
            end
            
            mockResults.paramNames = paramNames;
            
            % Add model information
            mockResults.p = arOrder;
            mockResults.q = maOrder;
            mockResults.constant = includeConstant;
            
            % Add data and fitted values
            mockResults.T = sampleSize;
            mockResults.data = data;
            mockResults.residuals = residuals;
            mockResults.fitted = data - residuals;
            
            % Add diagnostic statistics
            mockResults.R2 = 1 - var(residuals) / var(data);
            mockResults.adjR2 = 1 - (1 - mockResults.R2) * (sampleSize - 1) / (sampleSize - numParams - 1);
            
            % Calculate log-likelihood assuming normal errors
            sigma2 = mean(residuals.^2);
            mockResults.logL = -0.5 * sampleSize * (log(2*pi) + log(sigma2) + 1);
            
            % Calculate AIC and SBIC
            mockResults.AIC = -2 * mockResults.logL / sampleSize + 2 * numParams / sampleSize;
            mockResults.SBIC = -2 * mockResults.logL / sampleSize + numParams * log(sampleSize) / sampleSize;
            
            % Generate 10-period ahead forecasts
            forecastHorizon = 10;
            mockResults.forecasts = zeros(forecastHorizon, 1);
            mockResults.forecastErrors = randn(forecastHorizon, 1) * sqrt(sigma2);
            mockResults.forecastVariances = repmat(sigma2, forecastHorizon, 1);
            
            % Use actual data and estimated parameters to compute forecasts
            for h = 1:forecastHorizon
                forecast = constantValue * includeConstant;
                
                % Add AR component
                for i = 1:min(h-1, arOrder)
                    if h-i > 0
                        forecast = forecast + arParams(i) * mockResults.forecasts(h-i);
                    end
                end
                for i = min(h-1, arOrder)+1:arOrder
                    idx = sampleSize - (i - (h-1));
                    if idx > 0
                        forecast = forecast + arParams(i) * data(idx);
                    end
                end
                
                % Add MA component
                for i = 1:min(h-1, maOrder)
                    if h-i > 0
                        forecast = forecast + maParams(i) * mockResults.forecastErrors(h-i);
                    end
                end
                for i = min(h-1, maOrder)+1:maOrder
                    idx = sampleSize - (i - (h-1));
                    if idx > 0
                        forecast = forecast + maParams(i) * residuals(idx);
                    end
                end
                
                mockResults.forecasts(h) = forecast;
            end
        end
        
        function success = waitForGUIResponse(obj, figureHandle, timeout)
            % Waits for a GUI to process an action with timeout
            %
            % INPUTS:
            %   figureHandle - Handle to the figure to wait for
            %   timeout - Timeout in seconds (optional, default is obj.defaultTimeout)
            %
            % OUTPUTS:
            %   success - True if operation completed before timeout
            
            % Set default timeout if not provided
            if nargin < 3 || isempty(timeout)
                timeout = obj.defaultTimeout;
            end
            
            % Validate figure handle
            if ~ishandle(figureHandle)
                error('Invalid figure handle provided.');
            end
            
            % Start timer
            startTime = tic;
            
            % Process event queue to allow GUI updates
            while toc(startTime) < timeout
                % Process MATLAB event queue
                drawnow;
                
                % Check if operation has completed (can be customized based on specific criteria)
                % Here, we're just checking if the figure is still valid
                if ~ishandle(figureHandle)
                    success = true;
                    return;
                end
                
                % Add a short pause to prevent CPU hogging
                pause(0.1);
            end
            
            % If we reach here, timeout occurred
            success = false;
            
            % Optionally capture screenshot on timeout for debugging
            if obj.captureScreenshots
                if ishandle(figureHandle)
                    screenshotFile = sprintf('gui_timeout_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
                    obj.captureGUIScreenshot(figureHandle, screenshotFile);
                    warning('GUI operation timed out. Screenshot saved to: %s', screenshotFile);
                else
                    warning('GUI operation timed out. Figure no longer exists.');
                end
            else
                warning('GUI operation timed out.');
            end
        end
        
        function figHandles = findAllFiguresByType(obj, namePattern)
            % Finds all open figures of a specific type by name pattern
            %
            % INPUTS:
            %   namePattern - Regular expression pattern to match figure names
            %
            % OUTPUTS:
            %   figHandles - Array of figure handles matching the pattern
            
            % Get all figure handles
            allFigs = findobj('Type', 'figure');
            
            % If no figures open, return empty array
            if isempty(allFigs)
                figHandles = [];
                return;
            end
            
            % Filter figures based on name pattern
            matchIdx = [];
            for i = 1:length(allFigs)
                figName = get(allFigs(i), 'Name');
                if ~isempty(regexp(figName, namePattern, 'once'))
                    matchIdx = [matchIdx, i]; %#ok<AGROW>
                end
            end
            
            % Return matching figure handles
            if isempty(matchIdx)
                figHandles = [];
            else
                figHandles = allFigs(matchIdx);
            end
        end
        
        function closeAllTestFigures(obj)
            % Closes all test-related figures
            
            % Define patterns for test figures
            testPatterns = {'Test', 'Mock', 'ARMAX', 'Figure\s\d+', 'GUI'};
            
            % Find and close figures matching each pattern
            for i = 1:length(testPatterns)
                pattern = testPatterns{i};
                figHandles = obj.findAllFiguresByType(pattern);
                
                % Close each figure
                for j = 1:length(figHandles)
                    if ishandle(figHandles(j))
                        delete(figHandles(j));
                    end
                end
            end
            
            % Additional cleanup: close any remaining figures created during tests
            remainingFigs = findobj('Type', 'figure');
            if ~isempty(remainingFigs)
                % If created within the last minute, close it (assuming it's from our test)
                currentTime = now;
                for i = 1:length(remainingFigs)
                    try
                        % This may not work on all MATLAB versions
                        createTime = get(remainingFigs(i), 'CreateTime');
                        if (currentTime - createTime) * 24 * 60 < 1 % Less than 1 minute old
                            delete(remainingFigs(i));
                        end
                    catch
                        % If CreateTime property doesn't exist, use a more conservative approach
                        % Just try to close figures without specific names
                        if isempty(get(remainingFigs(i), 'Name'))
                            delete(remainingFigs(i));
                        end
                    end
                end
            end
        end
        
        function filePath = captureGUIScreenshot(obj, figureHandle, fileName)
            % Captures a screenshot of a figure for test documentation
            %
            % INPUTS:
            %   figureHandle - Handle to the figure to capture
            %   fileName - Name for the saved file (without extension)
            %
            % OUTPUTS:
            %   filePath - Path to saved screenshot file
            
            % Check if screenshot capture is enabled
            if ~obj.captureScreenshots
                filePath = '';
                return;
            end
            
            % Validate figure handle
            if ~ishandle(figureHandle)
                error('Invalid figure handle provided.');
            end
            
            % Ensure figure is visible and refreshed
            set(figureHandle, 'Visible', 'on');
            drawnow;
            
            % Define directory for screenshots
            screenshotDir = 'test_screenshots';
            if ~exist(screenshotDir, 'dir')
                mkdir(screenshotDir);
            end
            
            % Add timestamp to filename to avoid overwrites
            if nargin < 3 || isempty(fileName)
                fileName = sprintf('gui_%s', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            % Ensure filename has .png extension
            if ~endsWith(fileName, '.png')
                fileName = [fileName, '.png'];
            end
            
            % Define full file path
            filePath = fullfile(screenshotDir, fileName);
            
            % Capture and save the screenshot
            frame = getframe(figureHandle);
            imwrite(frame.cdata, filePath);
        end
        
        function result = verifyGUILayout(obj, figureHandle, expectedLayout)
            % Verifies the overall layout of a GUI figure
            %
            % INPUTS:
            %   figureHandle - Handle to the figure to verify
            %   expectedLayout - Structure containing expected layout properties
            %
            % OUTPUTS:
            %   result - True if layout matches expectations
            
            % Validate figure handle
            if ~ishandle(figureHandle)
                error('Invalid figure handle provided.');
            end
            
            % Initialize result
            result = true;
            
            % Check figure properties if specified
            if isfield(expectedLayout, 'figure')
                figProps = expectedLayout.figure;
                figPropFields = fieldnames(figProps);
                
                for i = 1:length(figPropFields)
                    propName = figPropFields{i};
                    expectedValue = figProps.(propName);
                    
                    % Get actual property value
                    try
                        actualValue = get(figureHandle, propName);
                    catch ME
                        error('Failed to get figure property "%s": %s', propName, ME.message);
                    end
                    
                    % Compare property values
                    if ~isequal(actualValue, expectedValue)
                        error('Figure property "%s" mismatch. Expected: %s, Actual: %s', ...
                            propName, mat2str(expectedValue), mat2str(actualValue));
                        result = false;
                        return;
                    end
                end
            end
            
            % Check component groups if specified
            if isfield(expectedLayout, 'components')
                compGroups = fieldnames(expectedLayout.components);
                
                for i = 1:length(compGroups)
                    groupName = compGroups{i};
                    groupProps = expectedLayout.components.(groupName);
                    
                    % Check each component in the group
                    for j = 1:length(groupProps)
                        compProps = groupProps{j};
                        
                        % Extract component type and tag
                        if ~isfield(compProps, 'Type') || ~isfield(compProps, 'Tag')
                            error('Component specifications must include Type and Tag fields.');
                        end
                        
                        compType = compProps.Type;
                        compTag = compProps.Tag;
                        
                        % Remove Type and Tag from properties to check
                        propsToCheck = compProps;
                        propsToCheck = rmfield(propsToCheck, 'Type');
                        propsToCheck = rmfield(propsToCheck, 'Tag');
                        
                        % Verify component
                        try
                            compResult = obj.verifyGUIComponent(figureHandle, compType, compTag, propsToCheck);
                            if ~compResult
                                result = false;
                                return;
                            end
                        catch ME
                            error('Component verification failed: %s', ME.message);
                            result = false;
                            return;
                        end
                    end
                end
            end
            
            % Check relative positioning if specified
            if isfield(expectedLayout, 'positioning')
                posRules = expectedLayout.positioning;
                
                for i = 1:length(posRules)
                    rule = posRules{i};
                    
                    % Get component handles
                    comp1 = findobj(figureHandle, 'Type', rule.comp1.Type, 'Tag', rule.comp1.Tag);
                    comp2 = findobj(figureHandle, 'Type', rule.comp2.Type, 'Tag', rule.comp2.Tag);
                    
                    if isempty(comp1) || isempty(comp2)
                        error('Cannot find components for positioning rule %d.', i);
                    end
                    
                    % Get positions
                    pos1 = get(comp1, 'Position');
                    pos2 = get(comp2, 'Position');
                    
                    % Check relation
                    switch lower(rule.relation)
                        case 'above'
                            if pos1(2) <= pos2(2)
                                error('Component %s should be above %s but is not.', ...
                                    rule.comp1.Tag, rule.comp2.Tag);
                                result = false;
                                return;
                            end
                        case 'below'
                            if pos1(2) >= pos2(2)
                                error('Component %s should be below %s but is not.', ...
                                    rule.comp1.Tag, rule.comp2.Tag);
                                result = false;
                                return;
                            end
                        case 'leftof'
                            if pos1(1) >= pos2(1)
                                error('Component %s should be left of %s but is not.', ...
                                    rule.comp1.Tag, rule.comp2.Tag);
                                result = false;
                                return;
                            end
                        case 'rightof'
                            if pos1(1) <= pos2(1)
                                error('Component %s should be right of %s but is not.', ...
                                    rule.comp1.Tag, rule.comp2.Tag);
                                result = false;
                                return;
                            end
                        otherwise
                            error('Unknown positioning relation: %s', rule.relation);
                    end
                end
            end
        end
        
        function componentTree = getComponentTree(obj, figureHandle)
            % Gets a hierarchical representation of GUI components
            %
            % INPUTS:
            %   figureHandle - Handle to the figure
            %
            % OUTPUTS:
            %   componentTree - Hierarchical structure of GUI components
            
            % Validate figure handle
            if ~ishandle(figureHandle)
                error('Invalid figure handle provided.');
            end
            
            % Create root node for figure
            componentTree = struct();
            componentTree.handle = figureHandle;
            componentTree.type = 'figure';
            componentTree.tag = get(figureHandle, 'Tag');
            componentTree.name = get(figureHandle, 'Name');
            componentTree.position = get(figureHandle, 'Position');
            componentTree.children = {};
            
            % Get all immediate children of the figure
            childHandles = findobj(figureHandle, 'Parent', figureHandle);
            
            % Process each child
            for i = 1:length(childHandles)
                child = obj.processComponent(childHandles(i));
                componentTree.children{end+1} = child;
            end
        end
        
        function componentInfo = processComponent(obj, componentHandle)
            % Helper function to process a component and its children recursively
            
            % Create component info structure
            componentInfo = struct();
            componentInfo.handle = componentHandle;
            componentInfo.type = get(componentHandle, 'Type');
            
            % Get common properties (will silently fail for properties that don't exist)
            try componentInfo.tag = get(componentHandle, 'Tag'); catch, componentInfo.tag = ''; end
            try componentInfo.string = get(componentHandle, 'String'); catch, componentInfo.string = ''; end
            try componentInfo.position = get(componentHandle, 'Position'); catch, componentInfo.position = []; end
            try componentInfo.visible = get(componentHandle, 'Visible'); catch, componentInfo.visible = ''; end
            try componentInfo.enabled = get(componentHandle, 'Enable'); catch, componentInfo.enabled = ''; end
            
            % Special handling for specific component types
            switch componentInfo.type
                case 'uicontrol'
                    try componentInfo.style = get(componentHandle, 'Style'); catch, componentInfo.style = ''; end
                    try componentInfo.value = get(componentHandle, 'Value'); catch, componentInfo.value = []; end
                case 'axes'
                    try componentInfo.title = get(get(componentHandle, 'Title'), 'String'); catch, componentInfo.title = ''; end
                    try componentInfo.xlabel = get(get(componentHandle, 'XLabel'), 'String'); catch, componentInfo.xlabel = ''; end
                    try componentInfo.ylabel = get(get(componentHandle, 'YLabel'), 'String'); catch, componentInfo.ylabel = ''; end
            end
            
            % Get children of this component
            childHandles = findobj(componentHandle, 'Parent', componentHandle);
            componentInfo.children = {};
            
            % Process each child recursively
            for i = 1:length(childHandles)
                child = obj.processComponent(childHandles(i));
                componentInfo.children{end+1} = child;
            end
        end
    end
end