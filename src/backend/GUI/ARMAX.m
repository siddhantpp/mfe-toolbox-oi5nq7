function varargout = ARMAX(varargin)
% ARMAX Implements the main Graphical User Interface for ARMAX modeling
%
% The ARMAX GUI provides an interactive environment for time series analysis
% using AutoRegressive Moving Average models with eXogenous inputs (ARMAX).
% It offers a comprehensive set of tools for model specification, estimation,
% diagnostics, and forecasting with an intuitive visual interface.
%
% USAGE:
%   ARMAX
%   h = ARMAX
%   ARMAX(data)
%
% INPUTS:
%   data - [OPTIONAL] Time series data to analyze (T x 1 vector)
%
% OUTPUTS:
%   h    - [OPTIONAL] Handle to the ARMAX figure
%
% See also ARMAX_VIEWER, ARMAXFILTER, ARMAFOR, SACF, SPACF, LJUNGBOX, LMTEST1
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ARMAX_OpeningFcn, ...
                   'gui_OutputFcn',  @ARMAX_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before ARMAX is made visible.
function ARMAX_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ARMAX (see VARARGIN)

% Choose default command line output for ARMAX
handles.output = hObject;

% Initialize empty data variables in handles structure
handles.data = [];
handles.hasData = false;
handles.hasResults = false;
handles.hasForecast = false;
handles.unsavedChanges = false;

% Set default model parameters
handles.modelParams.ar_order = 1;
handles.modelParams.ma_order = 0;
handles.modelParams.constant = true;
handles.modelParams.distribution = 'normal';
handles.modelParams.forecast_horizon = 10;

% Initialize time series axes
axes(handles.TimeSeriesAxes);
cla;
title('Time Series Data');
xlabel('Time');
ylabel('Value');
grid on;

% Initialize diagnostics axes
axes(handles.DiagnosticsAxes);
cla;
title('Diagnostics');
grid on;

% Set status text
set(handles.StatusText, 'String', 'Ready. Load data to begin analysis.');

% Initialize UI element states
set(handles.ViewResultsButton, 'Enable', 'off');
set(handles.EstimateButton, 'Enable', 'off');

% Set default values in UI elements
set(handles.AROrderEdit, 'String', num2str(handles.modelParams.ar_order));
set(handles.MAOrderEdit, 'String', num2str(handles.modelParams.ma_order));
set(handles.ConstantCheckbox, 'Value', handles.modelParams.constant);
set(handles.ForecastHorizonEdit, 'String', num2str(handles.modelParams.forecast_horizon));

% Set distribution popup default
distOptions = get(handles.DistributionPopup, 'String');
if strcmp(handles.modelParams.distribution, 'normal')
    set(handles.DistributionPopup, 'Value', 1);
else
    % Find the index of the selected distribution
    for i = 1:length(distOptions)
        if strcmpi(distOptions{i}, handles.modelParams.distribution)
            set(handles.DistributionPopup, 'Value', i);
            break;
        end
    end
end

% If input data was provided, load it
if nargin > 3 && ~isempty(varargin) && isnumeric(varargin{1})
    % Validate input data
    inputData = varargin{1};
    try
        inputData = datacheck(inputData, 'input data');
        inputData = columncheck(inputData, 'input data');
        
        % Store the data
        handles.data = inputData;
        handles.hasData = true;
        
        % Update plot
        handles = UpdateDataPlot(handles);
        
        % Enable estimation button
        set(handles.EstimateButton, 'Enable', 'on');
        
        % Update status
        set(handles.StatusText, 'String', sprintf('Loaded %d observations.', length(inputData)));
    catch ME
        % Handle error when loading input data
        errordlg(['Error loading input data: ' ME.message], 'Data Error');
    end
end

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = ARMAX_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in LoadDataButton.
function LoadDataButton_Callback(hObject, eventdata, handles)
% hObject    handle to LoadDataButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Open file dialog to load data
[filename, pathname] = uigetfile({'*.mat', 'MAT-files (*.mat)'; ...
                                 '*.txt;*.csv', 'Text Files (*.txt, *.csv)'; ...
                                 '*.*', 'All Files (*.*)'}, ...
                                 'Load Time Series Data');
if isequal(filename, 0) || isequal(pathname, 0)
    % User cancelled
    return;
end

% Attempt to load the file
try
    fullpath = fullfile(pathname, filename);
    [~, ~, ext] = fileparts(filename);
    
    if strcmpi(ext, '.mat')
        % Load MAT file
        loadedData = load(fullpath);
        
        % Extract variable from MAT file
        if isstruct(loadedData)
            % If it's a structure, try to find the first numeric field
            fields = fieldnames(loadedData);
            foundData = false;
            for i = 1:length(fields)
                if isnumeric(loadedData.(fields{i}))
                    loadedData = loadedData.(fields{i});
                    foundData = true;
                    break;
                end
            end
            
            if ~foundData
                error('Could not find numeric data in the MAT file.');
            end
        end
    else
        % Load text file
        loadedData = dlmread(fullpath);
    end
    
    % Validate data
    loadedData = datacheck(loadedData, 'data');
    
    % Ensure column vector
    loadedData = columncheck(loadedData, 'data');
    
    % Store in handles structure
    handles.data = loadedData;
    handles.hasData = true;
    handles.hasResults = false;
    handles.hasForecast = false;
    
    % Update time series plot
    handles = UpdateDataPlot(handles);
    
    % Enable estimation button
    set(handles.EstimateButton, 'Enable', 'on');
    
    % Update status
    set(handles.StatusText, 'String', sprintf('Loaded %d observations from %s.', length(loadedData), filename));
    
    % Update handles structure
    guidata(hObject, handles);
catch ME
    % Display error dialog
    errordlg(['Error loading data: ' ME.message], 'Data Error');
end

% --- Executes on button press in EstimateButton.
function EstimateButton_Callback(hObject, eventdata, handles)
% hObject    handle to EstimateButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if data is loaded
if ~handles.hasData
    errordlg('Please load data before estimating the model.', 'No Data');
    return;
end

% Get model parameters from UI
ar_order = str2double(get(handles.AROrderEdit, 'String'));
ma_order = str2double(get(handles.MAOrderEdit, 'String'));
constant = get(handles.ConstantCheckbox, 'Value');
forecast_horizon = str2double(get(handles.ForecastHorizonEdit, 'String'));

% Get distribution type
dist_idx = get(handles.DistributionPopup, 'Value');
dist_options = get(handles.DistributionPopup, 'String');
distribution = lower(dist_options{dist_idx});

% Validate parameters
try
    % Create options structure for armaxfilter
    options = struct('constant', constant, ...
                     'p', ar_order, ...
                     'q', ma_order, ...
                     'distribution', distribution);
    
    % Show waitbar during estimation
    wb = waitbar(0, 'Estimating ARMAX model...', 'Name', 'Model Estimation');
    
    % Set up waitbar update function
    updateFcn = @(x, optimValues, state) waitbar_update(x, optimValues, state, wb);
    options.optimopts = optimset('Display', 'iter', 'OutputFcn', updateFcn);
    
    % Call armaxfilter to estimate the model
    results = armaxfilter(handles.data, [], options);
    
    % Close waitbar
    if ishandle(wb)
        close(wb);
    end
    
    % Store results in handles
    handles.results = results;
    handles.hasResults = true;
    handles.hasForecast = false;
    handles.unsavedChanges = true;
    
    % Generate forecasts if requested
    if forecast_horizon > 0
        handles = GenerateForecasts(handles);
    end
    
    % Update plots
    handles = UpdateDataPlot(handles);
    
    % Reset diagnostic plot selection
    set(handles.ACFCheckbox, 'Value', 0);
    set(handles.PACFCheckbox, 'Value', 0);
    set(handles.ResidualCheckbox, 'Value', 1);
    handles = UpdateDiagnosticsPlot(handles);
    
    % Enable results button
    set(handles.ViewResultsButton, 'Enable', 'on');
    
    % Display model diagnostics in status area
    DisplayModelDiagnostics(handles);
    
    % Update handles structure
    guidata(hObject, handles);
catch ME
    % Close waitbar if it exists
    if exist('wb', 'var') && ishandle(wb)
        close(wb);
    end
    
    % Display error dialog
    errordlg(['Error estimating model: ' ME.message], 'Estimation Error');
end

% Waitbar update function
function stop = waitbar_update(x, optimValues, state, wb)
stop = false;
if ishandle(wb)
    switch state
        case 'init'
            waitbar(0, wb, 'Initializing estimation...');
        case 'iter'
            % Update progress (approximate since we don't know total iterations)
            waitbar(min(0.9, optimValues.iteration/100), wb, ...
                    sprintf('Iteration %d: fval = %.4g', optimValues.iteration, optimValues.fval));
        case 'done'
            waitbar(1, wb, 'Finalizing model estimation...');
    end
end

% --- Executes on edit of AROrderEdit.
function AROrderEdit_Callback(hObject, eventdata, handles)
% hObject    handle to AROrderEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get AR order from edit box
ar_order = str2double(get(hObject, 'String'));

% Validate AR order
if isnan(ar_order) || ar_order < 0 || floor(ar_order) ~= ar_order
    % Invalid input, reset to previous value
    set(hObject, 'String', num2str(handles.modelParams.ar_order));
    warndlg('AR order must be a non-negative integer.', 'Invalid Input');
else
    % Store new AR order
    handles.modelParams.ar_order = ar_order;
    guidata(hObject, handles);
end

% --- Executes on edit of MAOrderEdit.
function MAOrderEdit_Callback(hObject, eventdata, handles)
% hObject    handle to MAOrderEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get MA order from edit box
ma_order = str2double(get(hObject, 'String'));

% Validate MA order
if isnan(ma_order) || ma_order < 0 || floor(ma_order) ~= ma_order
    % Invalid input, reset to previous value
    set(hObject, 'String', num2str(handles.modelParams.ma_order));
    warndlg('MA order must be a non-negative integer.', 'Invalid Input');
else
    % Store new MA order
    handles.modelParams.ma_order = ma_order;
    guidata(hObject, handles);
end

% --- Executes on button press in ConstantCheckbox.
function ConstantCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to ConstantCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Store constant setting
handles.modelParams.constant = get(hObject, 'Value');
guidata(hObject, handles);

% --- Executes on selection change in DistributionPopup.
function DistributionPopup_Callback(hObject, eventdata, handles)
% hObject    handle to DistributionPopup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get selected distribution
dist_idx = get(hObject, 'Value');
dist_options = get(hObject, 'String');
handles.modelParams.distribution = lower(dist_options{dist_idx});

% Update handles structure
guidata(hObject, handles);

% --- Executes on edit of ForecastHorizonEdit.
function ForecastHorizonEdit_Callback(hObject, eventdata, handles)
% hObject    handle to ForecastHorizonEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get forecast horizon from edit box
forecast_horizon = str2double(get(hObject, 'String'));

% Validate forecast horizon
if isnan(forecast_horizon) || forecast_horizon < 1 || floor(forecast_horizon) ~= forecast_horizon
    % Invalid input, reset to previous value
    set(hObject, 'String', num2str(handles.modelParams.forecast_horizon));
    warndlg('Forecast horizon must be a positive integer.', 'Invalid Input');
else
    % Store new forecast horizon
    handles.modelParams.forecast_horizon = forecast_horizon;
    
    % If model has been estimated, update forecasts
    if handles.hasResults
        handles = GenerateForecasts(handles);
        handles = UpdateDataPlot(handles);
    end
    
    % Update handles structure
    guidata(hObject, handles);
end

% --- Executes on button press in ACFCheckbox.
function ACFCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to ACFCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% If checked, uncheck other diagnostic options
if get(hObject, 'Value')
    set(handles.PACFCheckbox, 'Value', 0);
    set(handles.ResidualCheckbox, 'Value', 0);
    
    % Update diagnostics plot
    handles = UpdateDiagnosticsPlot(handles);
    guidata(hObject, handles);
end

% --- Executes on button press in PACFCheckbox.
function PACFCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to PACFCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% If checked, uncheck other diagnostic options
if get(hObject, 'Value')
    set(handles.ACFCheckbox, 'Value', 0);
    set(handles.ResidualCheckbox, 'Value', 0);
    
    % Update diagnostics plot
    handles = UpdateDiagnosticsPlot(handles);
    guidata(hObject, handles);
end

% --- Executes on button press in ResidualCheckbox.
function ResidualCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to ResidualCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% If checked, uncheck other diagnostic options
if get(hObject, 'Value')
    set(handles.ACFCheckbox, 'Value', 0);
    set(handles.PACFCheckbox, 'Value', 0);
    
    % Update diagnostics plot
    handles = UpdateDiagnosticsPlot(handles);
    guidata(hObject, handles);
end

% --- Executes on button press in ViewResultsButton.
function ViewResultsButton_Callback(hObject, eventdata, handles)
% hObject    handle to ViewResultsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if results exist
if ~handles.hasResults
    errordlg('No model results available. Please estimate the model first.', 'No Results');
    return;
end

% Launch the results viewer with the model results
ARMAX_viewer(handles.results);

% --- Executes on button press in SaveButton.
function SaveButton_Callback(hObject, eventdata, handles)
% hObject    handle to SaveButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if results exist
if ~handles.hasResults
    errordlg('No model results available. Please estimate the model first.', 'No Results');
    return;
end

% Open save dialog
[filename, pathname] = uiputfile({'*.mat', 'MAT-files (*.mat)'}, 'Save Model Results');
if isequal(filename, 0) || isequal(pathname, 0)
    % User cancelled
    return;
end

% Create results structure for saving
saveResults = handles.results;

% Add forecasts if available
if handles.hasForecast
    saveResults.forecasts = handles.forecasts;
end

% Save to file
try
    save(fullfile(pathname, filename), 'saveResults');
    handles.unsavedChanges = false;
    guidata(hObject, handles);
    set(handles.StatusText, 'String', sprintf('Results saved to %s', filename));
catch ME
    errordlg(['Error saving results: ' ME.message], 'Save Error');
end

% --- Executes on button press in Close_button.
function Close_button_Callback(hObject, eventdata, handles)
% hObject    handle to Close_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Call Exit_Callback to handle application closure
Exit_Callback(hObject, eventdata, handles);

% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Call Exit_Callback to handle application closure
Exit_Callback(hObject, eventdata, handles);

% --- Menu callback for Help option
function Help_Callback(hObject, eventdata, handles)
% hObject    handle to Help menu item (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Display help information
msgbox({
    'ARMAX Model Estimation Tool', '', ...
    'This tool allows you to estimate ARMAX (AutoRegressive Moving Average with eXogenous inputs) models.', '', ...
    'Quick Guide:', ...
    '1. Load your time series data using the "Load Data" button', ...
    '2. Set the AR order (p) and MA order (q) for your model', ...
    '3. Select the error distribution type', ...
    '4. Click "Estimate" to fit the model', ...
    '5. View diagnostics in the bottom panel', ...
    '6. Click "View Results" for detailed model information', ...
    '7. Save your results using the "Save" button', '', ...
    'For more information, see the MFE Toolbox documentation.'
    }, 'Help', 'help');

% --- Menu callback for About option
function About_Callback(hObject, eventdata, handles)
% hObject    handle to About menu item (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Launch About dialog
ARMAX_about();

% --- Menu callback for Exit option
function Exit_Callback(hObject, eventdata, handles)
% hObject    handle to Exit menu item (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if there are unsaved results
if handles.hasResults && handles.unsavedChanges
    % Show confirmation dialog
    choice = ARMAX_close_dialog();
    
    % Process user choice
    if strcmp(choice, 'Yes')
        % Save results first
        SaveButton_Callback(handles.SaveButton, eventdata, handles);
        delete(handles.figure1);
    elseif strcmp(choice, 'No')
        % Close without saving
        delete(handles.figure1);
    end
    % If user cancels, do nothing
else
    % No results to save, close directly
    delete(handles.figure1);
end

% --- Helper function to generate forecasts
function handles = GenerateForecasts(handles)
% Validate that model has been estimated
if ~handles.hasResults
    return;
end

% Get forecast horizon
forecast_horizon = handles.modelParams.forecast_horizon;

% Extract model parameters from results
results = handles.results;
p = results.p;
q = results.q;
constant = results.constant;
parameters = results.parameters;
data = results.y;

% Set up options for armafor
error_dist = results.distribution;  % Use the same error distribution as the model

% Create distribution parameters structure if needed
dist_params = struct();
if strcmp(error_dist, 't')
    % For t distribution, last parameter is degrees of freedom
    numParams = length(parameters);
    dist_params.nu = parameters(numParams);
elseif strcmp(error_dist, 'ged')
    % For GED, last parameter is shape parameter
    numParams = length(parameters);
    dist_params.nu = parameters(numParams);
elseif strcmp(error_dist, 'skewt')
    % For skewed t, last two parameters are dof and lambda
    numParams = length(parameters);
    dist_params.nu = parameters(numParams-1);
    dist_params.lambda = parameters(numParams);
end

% Generate forecasts using armafor
try
    [forecasts, variances] = armafor(parameters, data, p, q, constant, [], ...
                                   forecast_horizon, [], 'exact', 1000, ...
                                   error_dist, dist_params);
    
    % Store forecast results
    handles.forecasts.values = forecasts;
    handles.forecasts.variances = variances;
    handles.forecasts.horizon = forecast_horizon;
    handles.hasForecast = true;
    handles.unsavedChanges = true;
catch ME
    % Handle error in forecast generation
    warning('Error generating forecasts: %s', ME.message);
    handles.hasForecast = false;
end

% --- Helper function to update the time series plot
function handles = UpdateDataPlot(handles)
% Get axes handle
ax = handles.TimeSeriesAxes;

% Clear current plot
axes(ax);
cla;

% Check if data is available
if ~handles.hasData
    title('Time Series Data');
    xlabel('Time');
    ylabel('Value');
    grid on;
    return;
end

% Plot the original data
T = length(handles.data);
t = 1:T;
plot(ax, t, handles.data, 'b-', 'LineWidth', 1);
hold(ax, 'on');

% Plot fitted values if model has been estimated
if handles.hasResults
    % Extract fitted values (original - residuals)
    fitted = handles.data - handles.results.residuals;
    plot(ax, t, fitted, 'r--', 'LineWidth', 1);
    
    % Plot forecasts if available
    if handles.hasForecast
        forecast_horizon = handles.forecasts.horizon;
        forecasts = handles.forecasts.values;
        std_forecasts = sqrt(handles.forecasts.variances);
        
        % Create time index for forecasts
        t_forecast = (T+1):(T+forecast_horizon);
        
        % Plot forecasts
        plot(ax, t_forecast, forecasts, 'r-', 'LineWidth', 1);
        
        % Plot confidence intervals (95%)
        upper_ci = forecasts + 1.96 * std_forecasts;
        lower_ci = forecasts - 1.96 * std_forecasts;
        
        % Plot confidence bounds
        plot(ax, t_forecast, upper_ci, 'r:', 'LineWidth', 1);
        plot(ax, t_forecast, lower_ci, 'r:', 'LineWidth', 1);
        
        % Fill confidence interval area
        fill([t_forecast, fliplr(t_forecast)], ...
             [upper_ci', fliplr(lower_ci')], ...
             'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    end
    
    % Add legend
    if handles.hasForecast
        legend(ax, 'Data', 'Fitted', 'Forecast', 'Location', 'Best');
    else
        legend(ax, 'Data', 'Fitted', 'Location', 'Best');
    end
else
    % Just data, no model yet
    legend(ax, 'Data', 'Location', 'Best');
end

% Add labels and grid
title(ax, 'Time Series Data');
xlabel(ax, 'Time');
ylabel(ax, 'Value');
grid(ax, 'on');
hold(ax, 'off');

% --- Helper function to update the diagnostics plot
function handles = UpdateDiagnosticsPlot(handles)
% Get axes handle
ax = handles.DiagnosticsAxes;

% Clear current plot
axes(ax);
cla;

% Check what should be plotted
if get(handles.ACFCheckbox, 'Value')
    % Plot ACF if model has been estimated
    if handles.hasResults
        % Compute ACF of residuals
        maxLag = min(20, floor(length(handles.results.residuals)/4));
        [acf, ~, acf_ci] = sacf(handles.results.residuals, maxLag);
        
        % Plot ACF
        bar(ax, 1:maxLag, acf, 'b');
        hold(ax, 'on');
        
        % Add confidence bounds
        plot(ax, [1 maxLag], [acf_ci(1,1) acf_ci(1,1)], 'r--');
        plot(ax, [1 maxLag], [acf_ci(1,2) acf_ci(1,2)], 'r--');
        
        % Add labels
        title(ax, 'Autocorrelation Function (ACF) of Residuals');
        xlabel(ax, 'Lag');
        ylabel(ax, 'Correlation');
        grid(ax, 'on');
        hold(ax, 'off');
    else
        % No model estimated yet
        text(0.5, 0.5, 'Estimate a model to view ACF', ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        title(ax, 'Autocorrelation Function (ACF)');
    end
    
elseif get(handles.PACFCheckbox, 'Value')
    % Plot PACF if model has been estimated
    if handles.hasResults
        % Compute PACF of residuals
        maxLag = min(20, floor(length(handles.results.residuals)/4));
        [pacf, ~, pacf_ci] = spacf(handles.results.residuals, maxLag);
        
        % Plot PACF
        bar(ax, 1:maxLag, pacf, 'b');
        hold(ax, 'on');
        
        % Add confidence bounds
        plot(ax, [1 maxLag], [pacf_ci(1,1) pacf_ci(1,1)], 'r--');
        plot(ax, [1 maxLag], [pacf_ci(1,2) pacf_ci(1,2)], 'r--');
        
        % Add labels
        title(ax, 'Partial Autocorrelation Function (PACF) of Residuals');
        xlabel(ax, 'Lag');
        ylabel(ax, 'Partial Correlation');
        grid(ax, 'on');
        hold(ax, 'off');
    else
        % No model estimated yet
        text(0.5, 0.5, 'Estimate a model to view PACF', ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        title(ax, 'Partial Autocorrelation Function (PACF)');
    end
    
elseif get(handles.ResidualCheckbox, 'Value')
    % Plot residuals if model has been estimated
    if handles.hasResults
        % Get residuals
        residuals = handles.results.residuals;
        t = 1:length(residuals);
        
        % Plot residuals
        plot(ax, t, residuals, 'b-');
        hold(ax, 'on');
        
        % Add horizontal line at zero
        plot(ax, [1 length(residuals)], [0 0], 'k--');
        
        % Add horizontal lines at +/- 2 standard deviations
        std_resid = std(residuals);
        plot(ax, [1 length(residuals)], [2*std_resid 2*std_resid], 'r--');
        plot(ax, [1 length(residuals)], [-2*std_resid -2*std_resid], 'r--');
        
        % Add labels
        title(ax, 'Model Residuals');
        xlabel(ax, 'Time');
        ylabel(ax, 'Residual');
        grid(ax, 'on');
        hold(ax, 'off');
    else
        % No model estimated yet
        text(0.5, 0.5, 'Estimate a model to view residuals', ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        title(ax, 'Model Residuals');
    end
else
    % Nothing selected, show empty plot
    title(ax, 'Diagnostics');
    text(0.5, 0.5, 'Select a diagnostic plot', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    grid(ax, 'on');
end

% --- Helper function to display model diagnostics in the status area
function DisplayModelDiagnostics(handles)
% Check if model has been estimated
if ~handles.hasResults
    return;
end

% Extract results
results = handles.results;
p = results.p;
q = results.q;
constant = results.constant;
logL = results.logL;
aic = results.aic;
sbic = results.sbic;

% Format model specification
modelSpec = sprintf('ARMAX(%d,%d)', p, q);
if constant
    modelSpec = [modelSpec ' with constant'];
end

% Get Ljung-Box test results
lb_results = results.ljungBox;
lb_stats = lb_results.stats;
lb_pvals = lb_results.pvals;
lb_sig = lb_results.isRejected5pct;

% Get LM test results
lm_results = results.lmTest;
lm_stat = lm_results.stat;
lm_pval = lm_results.pval;
lm_sig = lm_results.sig(2);  % 5% significance level

% Format diagnostic message
diagMsg = sprintf('Model: %s, Error Distribution: %s\n', modelSpec, results.distribution);
diagMsg = [diagMsg sprintf('Log-likelihood: %.4f, AIC: %.4f, SBIC: %.4f\n', logL, aic, sbic)];
diagMsg = [diagMsg sprintf('Ljung-Box Q(%d): %.4f [p=%.4f] %s\n', ...
                         lb_results.lags(1), lb_stats(1), lb_pvals(1), ...
                         iif(lb_sig(1), 'SIGNIFICANT', 'not significant'))];
diagMsg = [diagMsg sprintf('LM Test: %.4f [p=%.4f] %s', ...
                         lm_stat, lm_pval, ...
                         iif(lm_sig, 'SIGNIFICANT', 'not significant'))];

% Update status text
set(handles.StatusText, 'String', diagMsg);

% Inline if function
function result = iif(condition, trueVal, falseVal)
if condition
    result = trueVal;
else
    result = falseVal;
end