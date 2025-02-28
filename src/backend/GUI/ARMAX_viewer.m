function varargout = ARMAX_viewer(varargin)
% ARMAX_VIEWER Detailed results viewer for ARMAX model estimation
%
% This function provides a comprehensive graphical interface for viewing and
% analyzing ARMAX (AutoRegressive Moving Average with eXogenous inputs) model
% estimation results including parameter estimates, diagnostics, residuals,
% autocorrelation functions, and forecasts.
%
% USAGE:
%   ARMAX_viewer()
%   ARMAX_viewer(results)
%   h = ARMAX_viewer(...)
%
% INPUTS:
%   results - [OPTIONAL] Structure containing ARMAX model estimation results
%             from armaxfilter function. If not provided, an empty viewer is shown.
%
% OUTPUTS:
%   h - [OPTIONAL] Handle to the created figure window
%
% See also: ARMAX, ARMAXFILTER, ARMAFOR, SACF, SPACF, LJUNGBOX, LMTEST1
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Initialize the GUI (handles empty creation)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ARMAX_viewer_OpeningFcn, ...
                   'gui_OutputFcn',  @ARMAX_viewer_OutputFcn, ...
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

function ARMAX_viewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function runs when ARMAX_viewer is first invoked
% Initialize the GUI and set default values

% Choose default command line output for ARMAX_viewer
handles.output = hObject;

% Center the figure on the screen
center_figure(hObject);

% Initialize plot selection radio buttons
handles.currentPlotType = 'fit';  % Default plot type

% Initialize model results to empty
handles.modelResults = [];
handles.hasResults = false;
handles.hasForecast = false;
handles.forecastHorizon = 10;  % Default forecast horizon
handles.forecastResults = [];
handles.hasUnsavedChanges = false;

% Set default parameters table data (empty)
set(handles.parameter_table, 'Data', cell(0, 3), 'ColumnName', {'Parameter', 'Estimate', 'Std. Error'});

% Process input arguments if provided
if nargin > 3
    % Check if model results were provided
    if ~isempty(varargin) && isstruct(varargin{1})
        % Load the provided model results
        handles = loadModelResults(handles, varargin{1});
        handles.hasResults = true;
    end
end

% Set the default plot type to 'fit'
set(handles.plot_type_fit, 'Value', 1);
set(handles.plot_type_residuals, 'Value', 0);
set(handles.plot_type_acf, 'Value', 0);
set(handles.plot_type_pacf, 'Value', 0);
set(handles.plot_type_forecast, 'Value', 0);

% Update the displayed plot
if handles.hasResults
    handles = updatePlots(handles);
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ARMAX_viewer wait for user response
% uiwait(handles.figure1);

function varargout = ARMAX_viewer_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;

function plotTypeChanged_Callback(hObject, eventdata, handles)
% Callback function executed when plot type selection changes

% Determine which radio button was selected
if get(handles.plot_type_fit, 'Value') == 1
    handles.currentPlotType = 'fit';
elseif get(handles.plot_type_residuals, 'Value') == 1
    handles.currentPlotType = 'residuals';
elseif get(handles.plot_type_acf, 'Value') == 1
    handles.currentPlotType = 'acf';
elseif get(handles.plot_type_pacf, 'Value') == 1
    handles.currentPlotType = 'pacf';
elseif get(handles.plot_type_forecast, 'Value') == 1
    handles.currentPlotType = 'forecast';
    
    % Generate forecasts if they don't exist yet
    if ~handles.hasForecast && handles.hasResults
        handles.forecastResults = generateForecasts(handles, handles.forecastHorizon);
        handles.hasForecast = true;
    end
end

% Update the plot
handles = updatePlots(handles);

% Update handles structure
guidata(hObject, handles);

function savePlot_Callback(hObject, eventdata, handles)
% Callback for Save button - saves the current plot to a file

% If there is no data loaded, do nothing
if ~handles.hasResults
    return;
end

% Get the directory to save the plot
savedir = uigetdir(pwd, 'Select directory to save plot');
if savedir == 0  % User cancelled
    return;
end

% Generate default filename based on plot type
switch handles.currentPlotType
    case 'fit'
        filename = 'ARMAX_model_fit.png';
    case 'residuals'
        filename = 'ARMAX_residuals.png';
    case 'acf'
        filename = 'ARMAX_acf.png';
    case 'pacf'
        filename = 'ARMAX_pacf.png';
    case 'forecast'
        filename = 'ARMAX_forecast.png';
    otherwise
        filename = 'ARMAX_plot.png';
end

% Full path to save file
filepath = fullfile(savedir, filename);

% Save current plot
try
    % Get handle to main plot axes
    axesHandle = handles.main_axes;
    
    % Create temporary figure with just the plot
    tempFig = figure('Visible', 'off');
    newAxes = copyobj(axesHandle, tempFig);
    set(newAxes, 'Position', get(groot, 'DefaultAxesPosition'));
    
    % Save the figure
    saveas(tempFig, filepath);
    
    % Close temporary figure
    close(tempFig);
    
    % Display confirmation message
    msgbox(['Plot saved to ' filepath], 'Save Successful');
    
    % Reset unsaved changes flag
    handles.hasUnsavedChanges = false;
    guidata(hObject, handles);
catch
    % Display error message if save fails
    errordlg(['Failed to save plot to ' filepath], 'Save Error');
end

function closeButton_Callback(hObject, eventdata, handles)
% Callback for Close button - closes the viewer

% Check if there are unsaved changes
if handles.hasUnsavedChanges
    % Show confirmation dialog
    choice = ARMAX_close_dialog();
    if strcmp(choice, 'No')
        % User cancelled closing
        return;
    end
end

% Close the figure
delete(handles.figure1);

function figure1_CloseRequestFcn(hObject, eventdata, handles)
% Callback when user attempts to close window (X button)
% Routes to the Close button callback to handle confirmation

% Call the close button callback
closeButton_Callback(handles.close_button, eventdata, handles);

function handles = loadModelResults(handles, results)
% Loads ARMAX model results into the viewer and updates the GUI components

% Validate results structure
if ~isstruct(results) || ~isfield(results, 'parameters') || ~isfield(results, 'standardErrors')
    errordlg('Invalid model results structure', 'Data Error');
    return;
end

% Store the results in the handles structure
handles.modelResults = results;
handles.hasResults = true;
handles.hasUnsavedChanges = true;

% Get model information
p = results.p;
q = results.q;
constant = results.constant;
r = results.r;
T = results.T;

% Set model information in the GUI
if isfield(handles, 'model_info_text')
    modelInfo = sprintf('Model: ARMAX(%d,%d,%d)\nSample size: %d\n', p, q, r, T);
    set(handles.model_info_text, 'String', modelInfo);
end

% Display the model equation in a nicely formatted way
displayModelEquation(handles, results);

% Set up parameter table
paramNames = results.paramNames;
paramValues = results.parameters;
stdErrors = results.standardErrors;
tStats = results.tStats;
pValues = results.pValues;

% Create table data
numParams = length(paramNames);
tableData = cell(numParams, 4);
for i = 1:numParams
    tableData{i,1} = paramNames{i};
    tableData{i,2} = sprintf('%.4f', paramValues(i));
    tableData{i,3} = sprintf('%.4f', stdErrors(i));
    tableData{i,4} = sprintf('%.4f [%.3f]', tStats(i), pValues(i));
end

% Update parameter table
set(handles.parameter_table, 'Data', tableData);
set(handles.parameter_table, 'ColumnName', {'Parameter', 'Estimate', 'Std. Error', 't-stat [p-value]'});

% Display diagnostic information
displayDiagnostics(handles, results);

% Update plot to show model fit
handles.currentPlotType = 'fit';
set(handles.plot_type_fit, 'Value', 1);
handles = updatePlots(handles);

% Generate forecasts if forecast horizon is specified
if isfield(handles, 'forecastHorizon') && handles.forecastHorizon > 0
    handles.forecastResults = generateForecasts(handles, handles.forecastHorizon);
    handles.hasForecast = true;
end

function handles = updatePlots(handles)
% Updates the displayed plot based on the current plot type selection

% If no results are loaded, display empty plot with message
if ~handles.hasResults
    cla(handles.main_axes);
    text(0.5, 0.5, 'No model results loaded', 'Parent', handles.main_axes, ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    return;
end

% Clear current plot
cla(handles.main_axes);

% Get model results
results = handles.modelResults;
y = results.y;
residuals = results.residuals;
T = results.T;

% Create appropriate plot based on the selected type
switch handles.currentPlotType
    case 'fit'
        % Plot original data and fitted values
        % Calculate fitted values (original minus residuals)
        fitted = y - residuals;
        
        % Plot both series
        plot(handles.main_axes, 1:T, y, 'b', 'LineWidth', 1);
        hold(handles.main_axes, 'on');
        plot(handles.main_axes, 1:T, fitted, 'r--', 'LineWidth', 1);
        hold(handles.main_axes, 'off');
        
        % Add labels and legend
        xlabel(handles.main_axes, 'Time');
        ylabel(handles.main_axes, 'Value');
        title(handles.main_axes, 'Model Fit');
        legend(handles.main_axes, 'Original Data', 'Fitted Values', 'Location', 'Best');
        
    case 'residuals'
        % Plot residuals
        plot(handles.main_axes, 1:T, residuals, 'b', 'LineWidth', 1);
        hold(handles.main_axes, 'on');
        
        % Add horizontal lines at 0 and +/- 2*std(residuals)
        res_std = std(residuals);
        plot(handles.main_axes, [1 T], [0 0], 'k--');
        plot(handles.main_axes, [1 T], [2*res_std 2*res_std], 'r--');
        plot(handles.main_axes, [1 T], [-2*res_std -2*res_std], 'r--');
        hold(handles.main_axes, 'off');
        
        % Add labels
        xlabel(handles.main_axes, 'Time');
        ylabel(handles.main_axes, 'Residual');
        title(handles.main_axes, 'Model Residuals');
        
    case 'acf'
        % Compute and plot autocorrelation function
        maxLag = min(20, floor(T/4));
        [acf, ~, acf_ci] = sacf(residuals, maxLag);
        
        % Create bar plot for ACF
        bar(handles.main_axes, 1:maxLag, acf, 'b');
        hold(handles.main_axes, 'on');
        
        % Add confidence interval lines
        plot(handles.main_axes, [1 maxLag], [acf_ci(1,2) acf_ci(1,2)], 'r--');
        plot(handles.main_axes, [1 maxLag], [acf_ci(1,1) acf_ci(1,1)], 'r--');
        hold(handles.main_axes, 'off');
        
        % Add labels
        xlabel(handles.main_axes, 'Lag');
        ylabel(handles.main_axes, 'Autocorrelation');
        title(handles.main_axes, 'Residual Autocorrelation Function (ACF)');
        
    case 'pacf'
        % Compute and plot partial autocorrelation function
        maxLag = min(20, floor(T/4));
        [pacf, ~, pacf_ci] = spacf(residuals, maxLag);
        
        % Create bar plot for PACF
        bar(handles.main_axes, 1:maxLag, pacf, 'b');
        hold(handles.main_axes, 'on');
        
        % Add confidence interval lines
        plot(handles.main_axes, [1 maxLag], [pacf_ci(1,2) pacf_ci(1,2)], 'r--');
        plot(handles.main_axes, [1 maxLag], [pacf_ci(1,1) pacf_ci(1,1)], 'r--');
        hold(handles.main_axes, 'off');
        
        % Add labels
        xlabel(handles.main_axes, 'Lag');
        ylabel(handles.main_axes, 'Partial Autocorrelation');
        title(handles.main_axes, 'Residual Partial Autocorrelation Function (PACF)');
        
    case 'forecast'
        % Plot forecasts if available
        if ~handles.hasForecast
            handles.forecastResults = generateForecasts(handles, handles.forecastHorizon);
            handles.hasForecast = true;
        end
        
        % Get forecast data
        forecasts = handles.forecastResults.forecasts;
        forecastVar = handles.forecastResults.variances;
        horizon = length(forecasts);
        
        % Calculate confidence intervals (95%)
        z_value = 1.96; % ~95% confidence
        forecast_ci_upper = forecasts + z_value * sqrt(forecastVar);
        forecast_ci_lower = forecasts - z_value * sqrt(forecastVar);
        
        % Plot original data
        plot(handles.main_axes, 1:T, y, 'b', 'LineWidth', 1);
        hold(handles.main_axes, 'on');
        
        % Plot forecasts with confidence intervals
        t_forecast = T+1:T+horizon;
        plot(handles.main_axes, t_forecast, forecasts, 'r', 'LineWidth', 1);
        plot(handles.main_axes, t_forecast, forecast_ci_upper, 'r--');
        plot(handles.main_axes, t_forecast, forecast_ci_lower, 'r--');
        
        % Fill confidence interval area
        fill([t_forecast, fliplr(t_forecast)], ...
             [forecast_ci_upper', fliplr(forecast_ci_lower')], ...
             'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        
        hold(handles.main_axes, 'off');
        
        % Add labels and legend
        xlabel(handles.main_axes, 'Time');
        ylabel(handles.main_axes, 'Value');
        title(handles.main_axes, 'Forecast');
        legend(handles.main_axes, 'Historical Data', 'Forecast', '95% Confidence Interval', ...
               'Location', 'Best');
end

% Add grid for better readability
grid(handles.main_axes, 'on');

% Flag that there are unsaved changes
handles.hasUnsavedChanges = true;

function displayModelEquation(handles, model)
% Formats and displays the ARMAX model equation in LaTeX-like format

% Get model parameters
p = model.p;
q = model.q;
r = model.r;
constant = model.constant;
params = model.parameters;
stderrs = model.standardErrors;

% Start building the equation
equation = 'y_t = ';

% Track the current parameter index
param_idx = 1;

% Add constant if present
if constant
    const_value = params(param_idx);
    const_stderr = stderrs(param_idx);
    
    if const_value >= 0
        equation = [equation sprintf('%.4f', const_value)];
    else
        equation = [equation sprintf('- %.4f', abs(const_value))];
    end
    equation = [equation sprintf(' (%.4f)', const_stderr)];
    
    param_idx = param_idx + 1;
else
    equation = [equation '0'];
end

% Add AR terms
for i = 1:p
    ar_value = params(param_idx);
    ar_stderr = stderrs(param_idx);
    
    if ar_value >= 0
        equation = [equation sprintf(' + %.4f', ar_value)];
    else
        equation = [equation sprintf(' - %.4f', abs(ar_value))];
    end
    
    equation = [equation sprintf(' (%.4f)', ar_stderr)];
    equation = [equation sprintf(' y_{t-%d}', i)];
    
    param_idx = param_idx + 1;
end

% Add MA terms
for i = 1:q
    ma_value = params(param_idx);
    ma_stderr = stderrs(param_idx);
    
    if ma_value >= 0
        equation = [equation sprintf(' + %.4f', ma_value)];
    else
        equation = [equation sprintf(' - %.4f', abs(ma_value))];
    end
    
    equation = [equation sprintf(' (%.4f)', ma_stderr)];
    equation = [equation sprintf(' \\epsilon_{t-%d}', i)];
    
    param_idx = param_idx + 1;
end

% Add exogenous variable terms
for i = 1:r
    x_value = params(param_idx);
    x_stderr = stderrs(param_idx);
    
    if x_value >= 0
        equation = [equation sprintf(' + %.4f', x_value)];
    else
        equation = [equation sprintf(' - %.4f', abs(x_value))];
    end
    
    equation = [equation sprintf(' (%.4f)', x_stderr)];
    equation = [equation sprintf(' x_%d_{t}', i)];
    
    param_idx = param_idx + 1;
end

% Add the current error term
equation = [equation ' + \\epsilon_t'];

% Set the equation text
set(handles.model_equation, 'String', equation);

function displayDiagnostics(handles, results)
% Displays diagnostic test results and model fit statistics

% Get log-likelihood and information criteria
logL = results.logL;
aic = results.aic;
sbic = results.sbic;

% Get Ljung-Box test results
ljungBox = results.ljungBox;
lags = ljungBox.lags;
lb_pvals = ljungBox.pvals;

% Get LM test results
lmTest = results.lmTest;
lm_pval = lmTest.pval;

% Format the diagnostic text
diagText = '';

% Add log-likelihood and information criteria
diagText = sprintf('Log-likelihood: %.4f\n', logL);
diagText = [diagText sprintf('AIC: %.4f\n', aic)];
diagText = [diagText sprintf('SBIC: %.4f\n\n', sbic)];

% Add Ljung-Box test results
diagText = [diagText 'Ljung-Box Test:\n'];
for i = 1:length(lags)
    diagText = [diagText sprintf('  Lag %d: Q=%.4f (p=%.4f)\n', ...
                lags(i), ljungBox.stats(i), lb_pvals(i))];
end
diagText = [diagText '\n'];

% Add LM test results
diagText = [diagText sprintf('LM Test: Statistic=%.4f (p=%.4f)\n\n', ...
            lmTest.stat, lm_pval)];

% Add distribution information
diagText = [diagText sprintf('Error Distribution: %s\n', results.distribution)];
if strcmp(results.distribution, 't')
    % Add degrees of freedom for t-distribution
    numParams = length(results.parameters);
    dof_idx = numParams;  % Last parameter for t-distribution is DoF
    diagText = [diagText sprintf('  Degrees of Freedom: %.4f\n', ...
                results.parameters(dof_idx))];
elseif strcmp(results.distribution, 'ged')
    % Add shape parameter for GED
    numParams = length(results.parameters);
    shape_idx = numParams;  % Last parameter for GED is shape
    diagText = [diagText sprintf('  Shape Parameter: %.4f\n', ...
                results.parameters(shape_idx))];
elseif strcmp(results.distribution, 'skewt')
    % Add parameters for skewed t
    numParams = length(results.parameters);
    dof_idx = numParams - 1;  % Second-to-last parameter is DoF
    skew_idx = numParams;     % Last parameter is skewness
    diagText = [diagText sprintf('  Degrees of Freedom: %.4f\n', ...
                results.parameters(dof_idx))];
    diagText = [diagText sprintf('  Skewness: %.4f\n', ...
                results.parameters(skew_idx))];
end

% Set the diagnostic text
set(handles.diagnostics_text, 'String', diagText);

function forecastResults = generateForecasts(handles, horizon)
% Generates forecasts from the estimated ARMAX model for visualization

% Get model results
results = handles.modelResults;

% Extract model parameters
p = results.p;
q = results.q;
constant = results.constant;
r = results.r;
params = results.parameters;
y = results.y;
x = results.x;

% Extract AR and MA parameters
if constant
    const_val = params(1);
    ar_params = params(1+1:1+p);
    ma_params = params(1+p+1:1+p+q);
    if r > 0
        x_params = params(1+p+q+1:1+p+q+r);
    else
        x_params = [];
    end
else
    const_val = 0;
    ar_params = params(1:p);
    ma_params = params(p+1:p+q);
    if r > 0
        x_params = params(p+q+1:p+q+r);
    else
        x_params = [];
    end
end

% Set up forecast options
method = 'exact';  % Use exact forecasting method
shocks = [];       % No known future shocks
nsim = 1000;       % Number of simulations if using simulation method
error_dist = results.distribution;  % Use the same error distribution as the model

% Additional parameters for error distribution
dist_params = struct();
if strcmp(error_dist, 't')
    numParams = length(params);
    dist_params.nu = params(numParams);  % Last parameter is DoF
elseif strcmp(error_dist, 'ged')
    numParams = length(params);
    dist_params.nu = params(numParams);  % Last parameter is shape
elseif strcmp(error_dist, 'skewt')
    numParams = length(params);
    dist_params.nu = params(numParams-1);  % Second-to-last is DoF
    dist_params.lambda = params(numParams);  % Last parameter is skewness
end

% Generate forecasts
[forecasts, variances] = armafor(params, y, p, q, constant, x, horizon, ...
                               shocks, method, nsim, error_dist, dist_params);

% Return forecast results
forecastResults = struct();
forecastResults.forecasts = forecasts;
forecastResults.variances = variances;
forecastResults.horizon = horizon;

function center_figure(figure_handle)
% Centers the figure on the screen for optimal visibility

% Get screen size
screen_size = get(0, 'ScreenSize');
screen_width = screen_size(3);
screen_height = screen_size(4);

% Get figure dimensions
figure_position = get(figure_handle, 'Position');
figure_width = figure_position(3);
figure_height = figure_position(4);

% Calculate center position
center_x = (screen_width - figure_width) / 2;
center_y = (screen_height - figure_height) / 2;

% Set the figure position
set(figure_handle, 'Position', [center_x, center_y, figure_width, figure_height]);

function handles = initGUIComponents(hFigure)
% Initializes GUI components programmatically when loading from a figure file
% This allows operating without directly loading the .fig file, breaking dependency

% Create handles structure
handles = guihandles(hFigure);

% Find the main axes for plotting
handles.main_axes = findobj(hFigure, 'Type', 'axes', 'Tag', 'main_axes');
if isempty(handles.main_axes)
    % Create main axes if not found
    handles.main_axes = axes('Parent', hFigure, 'Position', [0.1, 0.35, 0.8, 0.55], ...
                             'Tag', 'main_axes');
end

% Find or create the model info panel
handles.model_info_panel = findobj(hFigure, 'Type', 'uipanel', 'Tag', 'model_info_panel');
if isempty(handles.model_info_panel)
    handles.model_info_panel = uipanel('Parent', hFigure, ...
                                      'Title', 'Model Information', ...
                                      'Position', [0.05, 0.05, 0.45, 0.25], ...
                                      'Tag', 'model_info_panel');
end

% Find or create the model equation text
handles.model_equation = findobj(hFigure, 'Type', 'uicontrol', 'Style', 'text', ...
                                'Tag', 'model_equation');
if isempty(handles.model_equation)
    handles.model_equation = uicontrol('Parent', handles.model_info_panel, ...
                                      'Style', 'text', ...
                                      'Units', 'normalized', ...
                                      'Position', [0.05, 0.6, 0.9, 0.35], ...
                                      'HorizontalAlignment', 'left', ...
                                      'FontSize', 9, ...
                                      'Tag', 'model_equation');
end

% Find or create the model info text
handles.model_info_text = findobj(hFigure, 'Type', 'uicontrol', 'Style', 'text', ...
                                 'Tag', 'model_info_text');
if isempty(handles.model_info_text)
    handles.model_info_text = uicontrol('Parent', handles.model_info_panel, ...
                                       'Style', 'text', ...
                                       'Units', 'normalized', ...
                                       'Position', [0.05, 0.2, 0.4, 0.35], ...
                                       'HorizontalAlignment', 'left', ...
                                       'FontSize', 9, ...
                                       'Tag', 'model_info_text');
end

% Find or create the diagnostics text
handles.diagnostics_text = findobj(hFigure, 'Type', 'uicontrol', 'Style', 'text', ...
                                  'Tag', 'diagnostics_text');
if isempty(handles.diagnostics_text)
    handles.diagnostics_text = uicontrol('Parent', handles.model_info_panel, ...
                                        'Style', 'text', ...
                                        'Units', 'normalized', ...
                                        'Position', [0.5, 0.05, 0.45, 0.5], ...
                                        'HorizontalAlignment', 'left', ...
                                        'FontSize', 8, ...
                                        'Tag', 'diagnostics_text');
end

% Find or create parameter table
handles.parameter_table = findobj(hFigure, 'Type', 'uitable', 'Tag', 'parameter_table');
if isempty(handles.parameter_table)
    handles.parameter_table = uitable('Parent', hFigure, ...
                                     'Position', [0.55, 0.05, 0.4, 0.25], ...
                                     'ColumnName', {'Parameter', 'Estimate', 'Std. Error', 't-stat [p-value]'}, ...
                                     'ColumnWidth', {100, 80, 80, 120}, ...
                                     'Tag', 'parameter_table');
end

% Find or create plot type radio buttons
handles.plot_type_panel = findobj(hFigure, 'Type', 'uipanel', 'Tag', 'plot_type_panel');
if isempty(handles.plot_type_panel)
    handles.plot_type_panel = uipanel('Parent', hFigure, ...
                                     'Title', 'Plot Type', ...
                                     'Position', [0.05, 0.92, 0.9, 0.06], ...
                                     'Tag', 'plot_type_panel');
                                 
    % Create radio buttons for plot types
    handles.plot_type_fit = uicontrol('Parent', handles.plot_type_panel, ...
                                     'Style', 'radiobutton', ...
                                     'String', 'Model Fit', ...
                                     'Units', 'normalized', ...
                                     'Position', [0.05, 0.2, 0.15, 0.6], ...
                                     'Value', 1, ...
                                     'Tag', 'plot_type_fit', ...
                                     'Callback', @(hObject,eventdata)ARMAX_viewer('plotTypeChanged_Callback',hObject,eventdata,guidata(hObject)));
                                 
    handles.plot_type_residuals = uicontrol('Parent', handles.plot_type_panel, ...
                                          'Style', 'radiobutton', ...
                                          'String', 'Residuals', ...
                                          'Units', 'normalized', ...
                                          'Position', [0.25, 0.2, 0.15, 0.6], ...
                                          'Value', 0, ...
                                          'Tag', 'plot_type_residuals', ...
                                          'Callback', @(hObject,eventdata)ARMAX_viewer('plotTypeChanged_Callback',hObject,eventdata,guidata(hObject)));
                                      
    handles.plot_type_acf = uicontrol('Parent', handles.plot_type_panel, ...
                                     'Style', 'radiobutton', ...
                                     'String', 'ACF', ...
                                     'Units', 'normalized', ...
                                     'Position', [0.45, 0.2, 0.15, 0.6], ...
                                     'Value', 0, ...
                                     'Tag', 'plot_type_acf', ...
                                     'Callback', @(hObject,eventdata)ARMAX_viewer('plotTypeChanged_Callback',hObject,eventdata,guidata(hObject)));
                                 
    handles.plot_type_pacf = uicontrol('Parent', handles.plot_type_panel, ...
                                      'Style', 'radiobutton', ...
                                      'String', 'PACF', ...
                                      'Units', 'normalized', ...
                                      'Position', [0.65, 0.2, 0.15, 0.6], ...
                                      'Value', 0, ...
                                      'Tag', 'plot_type_pacf', ...
                                      'Callback', @(hObject,eventdata)ARMAX_viewer('plotTypeChanged_Callback',hObject,eventdata,guidata(hObject)));
                                  
    handles.plot_type_forecast = uicontrol('Parent', handles.plot_type_panel, ...
                                         'Style', 'radiobutton', ...
                                         'String', 'Forecast', ...
                                         'Units', 'normalized', ...
                                         'Position', [0.85, 0.2, 0.15, 0.6], ...
                                         'Value', 0, ...
                                         'Tag', 'plot_type_forecast', ...
                                         'Callback', @(hObject,eventdata)ARMAX_viewer('plotTypeChanged_Callback',hObject,eventdata,guidata(hObject)));
end

% Find or create save and close buttons
handles.save_button = findobj(hFigure, 'Type', 'uicontrol', 'Style', 'pushbutton', ...
                             'Tag', 'save_button');
if isempty(handles.save_button)
    handles.save_button = uicontrol('Parent', hFigure, ...
                                   'Style', 'pushbutton', ...
                                   'String', 'Save Plot', ...
                                   'Position', [580, 10, 100, 30], ...
                                   'Tag', 'save_button', ...
                                   'Callback', @(hObject,eventdata)ARMAX_viewer('savePlot_Callback',hObject,eventdata,guidata(hObject)));
end

handles.close_button = findobj(hFigure, 'Type', 'uicontrol', 'Style', 'pushbutton', ...
                              'Tag', 'close_button');
if isempty(handles.close_button)
    handles.close_button = uicontrol('Parent', hFigure, ...
                                    'Style', 'pushbutton', ...
                                    'String', 'Close', ...
                                    'Position', [690, 10, 100, 30], ...
                                    'Tag', 'close_button', ...
                                    'Callback', @(hObject,eventdata)ARMAX_viewer('closeButton_Callback',hObject,eventdata,guidata(hObject)));
end

% Initialize empty model results
handles.modelResults = [];
handles.hasResults = false;
handles.hasForecast = false;
handles.forecastHorizon = 10;
handles.hasUnsavedChanges = false;
handles.currentPlotType = 'fit';

% Set close request function
set(hFigure, 'CloseRequestFcn', @(hObject,eventdata)ARMAX_viewer('figure1_CloseRequestFcn',hObject,eventdata,guidata(hObject)));