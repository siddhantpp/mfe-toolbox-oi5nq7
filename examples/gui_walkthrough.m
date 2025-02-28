function gui_walkthrough()
% GUI_WALKTHROUGH A comprehensive walkthrough of the ARMAX GUI functionality
%
% This script guides users through the ARMAX (AutoRegressive Moving Average with
% eXogenous inputs) modeling interface of the MFE Toolbox. It demonstrates how to:
%   1. Load financial time series data
%   2. Configure model parameters
%   3. Estimate ARMAX models
%   4. Interpret diagnostic results
%   5. Generate and visualize forecasts
%
% The walkthrough provides a step-by-step introduction to time series modeling
% through an interactive graphical interface.
%
% See also: ARMAX, armaxfilter, ARMAX_viewer

% Display welcome message
display_section_header('WELCOME TO THE ARMAX GUI WALKTHROUGH');
disp('This walkthrough will guide you through using the ARMAX GUI for time series modeling.');
disp('You will learn how to load data, configure models, interpret results, and generate forecasts.');
disp('Follow the instructions and press Enter when prompted to continue to the next step.');
wait_for_user('Press Enter to begin the walkthrough...');

% Ensure MFE Toolbox is on the path
if ~exist('ARMAX', 'file')
    % Try to add the toolbox to the path
    try
        addpath('../src/backend/GUI');
        addpath('../src/backend/timeseries');
        addpath('../src/backend/utility');
        if ~exist('ARMAX', 'file')
            error('MFE Toolbox not found on the MATLAB path. Please add the toolbox to your path before running this walkthrough.');
        end
    catch
        error('MFE Toolbox not found on the MATLAB path. Please add the toolbox to your path before running this walkthrough.');
    end
end

% Load example data
display_section_header('STEP 1: LOADING DATA');
disp('First, we''ll load some example financial time series data to analyze.');
disp('For this walkthrough, we''ll use stock return data from the included example dataset.');

% Prepare example data for visualization
data = prepare_example_data();

% Plot the data before loading into the GUI
figure;
plot(data.dates, data.returns);
title(['Example Financial Returns: ' data.asset_name]);
xlabel('Date');
ylabel('Returns');
grid on;

disp(' ');
disp('Above is a plot of the time series data we''ll be working with.');
disp(['This shows daily returns for ' data.asset_name ' over approximately ' ...
      num2str(round(length(data.returns)/252)) ' years.']);
disp('We will now launch the ARMAX GUI to model this series.');
wait_for_user('Press Enter to launch the ARMAX GUI...');

% Launch the ARMAX GUI with the data preloaded
h_armax = ARMAX(data.returns);

% Provide instructions for using the GUI
display_section_header('STEP 2: EXPLORING THE ARMAX GUI');
disp('The ARMAX GUI is now open with our data preloaded. Let''s examine its components:');
disp(' ');
disp('1. The left panel contains model configuration options:');
disp('   - AR Order: The number of autoregressive terms');
disp('   - MA Order: The number of moving average terms');
disp('   - Distribution: The error distribution assumption');
disp('   - Constant: Whether to include a constant term');
disp(' ');
disp('2. The main area shows plots of:');
disp('   - Time series data and fitted values');
disp('   - Model diagnostics (ACF, PACF, residuals)');
disp(' ');
disp('3. The bottom area provides model statistics and buttons for:');
disp('   - Viewing detailed results');
disp('   - Saving the model');
disp('   - Closing the application');
wait_for_user('Press Enter to continue to the next step...');

% Guide user to load data
display_section_header('STEP 3: LOADING DATA INTO THE GUI');
disp('For this walkthrough, we''ve already loaded the data automatically.');
disp('However, in normal usage, you would load data using the "Load Data" button:');
disp(' ');
disp('1. Click the "Load Data" button in the GUI');
disp('2. Navigate to a MAT file containing time series data');
disp('3. Select the file and click "Open"');
disp(' ');
disp('The ARMAX GUI supports loading:');
disp('- MAT files containing numeric arrays');
disp('- Text/CSV files with numeric data');
disp(' ');
disp('The loaded data should be a column vector of observations with the most recent at the end.');
wait_for_user('Press Enter to continue to the next step...');

% Configure model parameters
display_section_header('STEP 4: CONFIGURING MODEL PARAMETERS');
disp('Now let''s configure the ARMA model parameters:');
disp(' ');
disp('The AR (AutoRegressive) order determines how many lagged values of the series to include.');
disp('The MA (Moving Average) order determines how many lagged error terms to include.');
disp(' ');
disp('For financial returns, common models include:');
disp('- AR(1): Returns may exhibit some persistence (momentum)');
disp('- MA(1): Returns may have short-term reversals');
disp('- ARMA(1,1): Combination of both effects');
disp(' ');
disp('In the GUI:');
disp('1. Set AR Order to 1 (enter 1 in the AR Order field)');
disp('2. Set MA Order to 1 (enter 1 in the MA Order field)');
disp('3. Ensure "Include Constant" is checked (usually appropriate for returns)');
disp('4. From the Distribution dropdown, select "t" (financial returns often have fat tails)');
disp(' ');
disp('The error distribution options are:');
disp('- normal: Standard normal distribution (symmetric, light tails)');
disp('- t: Student''s t distribution (symmetric, heavy tails)');
disp('- ged: Generalized Error Distribution (flexible tail thickness)');
disp('- skewt: Hansen''s skewed t (asymmetric, heavy tails)');
wait_for_user('After configuring the model, press Enter to continue...');

% Estimate the model
display_section_header('STEP 5: ESTIMATING THE MODEL');
disp('Now we''re ready to estimate the ARMA(1,1) model:');
disp(' ');
disp('1. Click the "Estimate" button in the GUI');
disp('2. The GUI will display a progress bar while estimating the model');
disp('3. Once complete, the time series plot will update with fitted values');
disp('4. The diagnostic plot will show residuals by default');
disp(' ');
disp('NOTE: Estimation may take a few seconds, especially for larger datasets');
disp('or more complex models. The exact time depends on your computer''s speed.');
disp(' ');
disp('Behind the scenes, the GUI is calling the armaxfilter function, which:');
disp('1. Optimizes model parameters using maximum likelihood estimation');
disp('2. Computes standard errors for parameter estimates');
disp('3. Calculates various diagnostic statistics');
disp('4. Generates residuals for diagnostic checking');
wait_for_user('After the model has been estimated, press Enter to continue...');

% Examine diagnostics
display_section_header('STEP 6: EXAMINING MODEL DIAGNOSTICS');
disp('After estimation, we should examine the model diagnostics to assess fit:');
disp(' ');
disp('1. The status area at the bottom shows key statistics:');
disp('   - Log-likelihood: Higher values indicate better fit');
disp('   - AIC/SBIC: Lower values indicate better fit, balancing goodness of fit and complexity');
disp('   - Ljung-Box and LM tests: Test for remaining autocorrelation in residuals');
disp(' ');
disp('2. To examine different diagnostics, select checkboxes on the left:');
disp('   - Residuals: Shows model residuals, looking for white noise (no pattern)');
disp('   - ACF: Autocorrelation Function of residuals (should show no significant spikes)');
disp('   - PACF: Partial Autocorrelation Function (also should show no pattern)');
disp(' ');
disp('Try selecting each diagnostic option and observe the results:');
disp('1. First, check the "ACF" checkbox');
disp('2. Next, check the "PACF" checkbox');
disp('3. Finally, return to "Residuals" checkbox');

% Explain how to interpret these diagnostics
explain_model_diagnostics();
wait_for_user('After examining the diagnostics, press Enter to continue...');

% View detailed results
display_section_header('STEP 7: VIEWING DETAILED RESULTS');
disp('For a more detailed view of the results, click the "View Results" button.');
disp('This will open a new window showing:');
disp(' ');
disp('1. The model equation with estimated parameters');
disp('2. A table of parameter estimates with standard errors and t-statistics');
disp('3. Comprehensive diagnostic information');
disp('4. Various plots accessible through radio buttons at the top');
disp(' ');
disp('In the Results Viewer:');
disp('1. Examine the parameter estimates and their significance');
disp('   - Parameters with t-stats > 1.96 (or p-values < 0.05) are statistically significant');
disp('   - For AR(1) and MA(1), look at the AR(1) and MA(1) coefficients');
disp('   - The constant term represents the mean of the process when significant');
disp(' ');
disp('2. Try the different plot types (Model Fit, Residuals, ACF, PACF, Forecast)');
disp('3. You can save any plot using the "Save Plot" button');
wait_for_user('After exploring the Results Viewer, close it and press Enter to continue...');

% Try different model specifications
display_section_header('STEP 8: TRYING DIFFERENT MODEL SPECIFICATIONS');
disp('Now that you understand the interface, you might want to try different models:');
disp(' ');
disp('1. Return to the main ARMAX window');
disp('2. Try different AR and MA orders, for example:');
disp('   - AR(2), MA(0): Set AR Order=2, MA Order=0');
disp('   - AR(0), MA(1): Set AR Order=0, MA Order=1');
disp('   - AR(2), MA(2): Set AR Order=2, MA Order=2');
disp(' ');
disp('3. Re-estimate the model and compare AIC/SBIC values');
disp('   - Lower AIC/SBIC suggests a better model');
disp('   - But beware of overfitting with too many parameters!');
disp(' ');
disp('4. Try different error distributions:');
disp('   - For financial data, "t" often works well due to fat tails');
disp('   - "skewt" may be better if returns show asymmetry');
disp('   - The estimated degrees of freedom parameter will appear in results');
wait_for_user('After trying different model specifications, press Enter to continue...');

% Generating forecasts
display_section_header('STEP 9: GENERATING FORECASTS');
disp('ARMAX models are often used for forecasting future values:');
disp(' ');
disp('1. In the View Results window, select the "Forecast" plot type');
disp('2. This shows out-of-sample forecasts with confidence intervals');
disp('3. The default forecast horizon is 10 periods');
disp('4. You can modify the forecast horizon in the main ARMAX window');
disp(' ');
disp('To adjust the forecast horizon:');
disp('1. In the main ARMAX window, locate the "Forecast Horizon" field');
disp('2. Enter a different number (e.g., 20 for a longer horizon)');
disp('3. Re-estimate the model to update the forecasts');
disp(' ');
disp('Note: As the forecast horizon increases, the confidence intervals widen,');
disp('reflecting increasing uncertainty about future values.');
disp(' ');
disp('For financial returns, forecasts often revert to the mean (constant) quickly,');
disp('which is typical for near-random-walk processes with limited predictability.');
wait_for_user('After exploring forecasts, press Enter to continue...');

% Saving results
display_section_header('STEP 10: SAVING RESULTS');
disp('If you want to save your model results for later use:');
disp(' ');
disp('1. Click the "Save" button in the main ARMAX window');
disp('2. Choose a location and filename to save the results as a .mat file');
disp('3. The saved file will contain the full model specification and results');
disp(' ');
disp('To load saved results in the future:');
disp('1. Use the MATLAB load command to load the .mat file');
disp('2. The saved structure contains parameter estimates, diagnostics, and more');
disp('3. You can use armaxfilter directly with the saved parameters for further analysis');
disp(' ');
disp('The saved structure includes fields like:');
disp('- parameters: Estimated coefficient values');
disp('- standardErrors: Standard errors for the parameters');
disp('- residuals: Model residuals for further analysis');
disp('- logL, aic, sbic: Goodness-of-fit measures');
disp('- ljungBox, lmTest: Diagnostic test results');
wait_for_user('Press Enter to continue to the final step...');

% Closing the application
display_section_header('STEP 11: CLOSING THE APPLICATION');
disp('To close the ARMAX GUI:');
disp(' ');
disp('1. Click the "Close" button at the bottom of the window');
disp('2. If you have unsaved changes, you''ll be prompted to save them');
disp('3. The GUI and any open result viewers will close');
disp(' ');
disp('You can also use the File menu and select "Exit", or click the window''s close button (X).');
wait_for_user('When you''re ready to close the GUI, press Enter...');

% Wrap up the walkthrough
display_section_header('CONGRATULATIONS!');
disp('You have completed the ARMAX GUI walkthrough!');
disp(' ');
disp('You now know how to:');
disp('1. Load time series data into the ARMAX GUI');
disp('2. Configure and estimate ARMA/ARMAX models');
disp('3. Interpret model diagnostics and assess fit');
disp('4. View detailed results and parameter estimates');
disp('5. Generate forecasts and save results');
disp(' ');
disp('For more advanced usage, consider exploring:');
disp('- The command-line interface (armaxfilter function)');
disp('- Volatility modeling (GARCH, EGARCH, etc.)');
disp('- Multivariate time series analysis');
disp('- High-frequency data analysis');
disp(' ');
disp('Thank you for using the MFE Toolbox!');
end

function display_section_header(header_text)
% DISPLAY_SECTION_HEADER Displays formatted section headers
%
% INPUTS:
%   header_text - Text to display as header

% Print separator line
disp('=================================================================');
% Print header in uppercase
disp(['  ' upper(header_text)]);
% Print separator line
disp('=================================================================');
% Add blank line for readability
disp(' ');
end

function wait_for_user(prompt_text)
% WAIT_FOR_USER Pauses execution and waits for user acknowledgment before continuing
%
% INPUTS:
%   prompt_text - Text to display as prompt

disp(' ');
input([prompt_text ' ']);
clc;
end

function data = prepare_example_data()
% PREPARE_EXAMPLE_DATA Loads and prepares example financial data for the walkthrough
%
% OUTPUTS:
%   data - Structure with prepared time series

% Try to load example data from various possible locations
try
    load('examples/data/example_financial_data.mat', 'returns', 'dates', 'asset_names');
catch
    try
        load('example_financial_data.mat', 'returns', 'dates', 'asset_names');
    catch
        try
            load('../examples/data/example_financial_data.mat', 'returns', 'dates', 'asset_names');
        catch
            % If all attempts fail, create synthetic data for demonstration
            disp('Example data not found. Creating synthetic data for demonstration.');
            n = 1000;
            dates = (today-n+1:today)';
            returns = 0.0005 + 0.2*randn(n, 1) + 0.05*sin(2*pi*(1:n)'/252);
            asset_names = {'SyntheticAsset'};
        end
    end
end

% Select the first asset's returns for simplicity
asset_idx = 1;
asset_returns = returns(:, asset_idx);

% Format dates for display
if isdatetime(dates)
    formatted_dates = datestr(dates, 'yyyy-mm-dd');
else
    formatted_dates = datestr(dates, 'yyyy-mm-dd');
end

% Create data structure
data = struct();
data.returns = asset_returns;
data.dates = dates;
data.formatted_dates = formatted_dates;
data.asset_name = asset_names{asset_idx};
end

function explain_model_diagnostics()
% EXPLAIN_MODEL_DIAGNOSTICS Helper function to explain model diagnostic plots and statistics
%
% This function provides detailed explanations of how to interpret the various
% diagnostic tools available in the ARMAX GUI.

disp(' ');
disp('INTERPRETING MODEL DIAGNOSTICS:');
disp(' ');
disp('1. ACF (Autocorrelation Function):');
disp('   - Shows correlations between the series and its lags');
disp('   - In residuals, significant spikes indicate remaining patterns');
disp('   - Dashed red lines show significance bounds (typically ±1.96/√T)');
disp('   - For a good model, ~95% of spikes should be within bounds');
disp(' ');
disp('2. PACF (Partial Autocorrelation Function):');
disp('   - Shows correlations after removing effects of intermediate lags');
disp('   - Helps identify AR order: in AR(p) process, cuts off after lag p');
disp('   - In residuals, should show no significant pattern');
disp(' ');
disp('3. Residual Plot:');
disp('   - Should resemble white noise (random scatter around zero)');
disp('   - Look for: constant variance, no obvious patterns, no outliers');
disp('   - Dashed red lines show ±2 standard deviations');
disp(' ');
disp('4. Ljung-Box Test:');
disp('   - Tests null hypothesis of no autocorrelation up to specified lag');
disp('   - Low p-values (< 0.05) suggest remaining autocorrelation (bad)');
disp('   - "not significant" result is good (no evidence of remaining patterns)');
disp(' ');
disp('5. LM Test:');
disp('   - Tests for remaining ARCH effects in residuals');
disp('   - Significant result suggests need for GARCH modeling');
disp('   - "not significant" result is good for ARMA adequacy');
disp(' ');
disp('6. Information Criteria (AIC/SBIC):');
disp('   - Lower values indicate better models');
disp('   - AIC tends to select more complex models');
disp('   - SBIC penalizes complexity more heavily');
disp('   - Use to compare alternative specifications');
end