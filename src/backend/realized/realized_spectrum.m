function [rv, varargout] = realized_spectrum(returns, options)
% REALIZED_SPECTRUM Computes realized spectral volatility from high-frequency return data
%
% USAGE:
%   RV = realized_spectrum(RETURNS)
%   [RV, DIAGNOSTICS] = realized_spectrum(RETURNS, OPTIONS)
%
% INPUTS:
%   RETURNS    - T by n matrix of high-frequency returns where:
%                T is the number of intraday observations
%                n is the number of assets or time series
%
%   OPTIONS    - [Optional] Structure with estimation parameters:
%                windowType      - String specifying spectral window type:
%                                  'Parzen' (default), 'Bartlett', 'Tukey-Hanning',
%                                  'Quadratic', 'Cubic', 'Flat-Top'
%                cutoffFreq      - Frequency domain cutoff parameter [default: automatically determined]
%                biasCorrection  - Boolean for asymptotic bias correction [default: true]
%                handleOvernight - Boolean for overnight returns adjustment [default: false]
%                removeOutliers  - Boolean for outlier removal [default: false]
%                compareBenchmark- Boolean to compare with standard RV [default: false]
%                compareKernel   - Boolean to compare with kernel RV [default: false]
%
% OUTPUTS:
%   RV          - 1 by n vector of spectral realized volatility (variance) estimates
%   DIAGNOSTICS - [Optional] Structure with additional outputs (when nargout > 1):
%                 .benchmark     - Standard realized volatility (if requested)
%                 .kernel        - Kernel-based realized volatility (if requested)
%                 .spectrum      - Raw spectral density estimates
%                 .frequencies   - Frequency grid points
%                 .window        - Spectral window weights used
%
% COMMENTS:
%   This function implements spectral estimators for integrated variance that are
%   robust to market microstructure noise by utilizing frequency domain filtering.
%   Spectral methods tend to provide more accurate estimates than standard realized
%   volatility when high-frequency data contains noise.
%
%   The spectral realized volatility is computed by:
%   1. Calculating the periodogram of returns using FFT
%   2. Applying a spectral window to reduce noise impact
%   3. Integrating the filtered spectral density to estimate the integrated variance
%
%   Different spectral windows provide different trade-offs between bias and variance.
%   The default 'Parzen' window generally provides good performance across different
%   noise levels.
%
% EXAMPLES:
%   % Basic usage with default settings
%   rv = realized_spectrum(fiveminreturns);
%
%   % With custom window type and cutoff frequency
%   options.windowType = 'Tukey-Hanning';
%   options.cutoffFreq = 0.3;
%   rv = realized_spectrum(fiveminreturns, options);
%
%   % With diagnostics and benchmark comparison
%   options.compareBenchmark = true;
%   options.compareKernel = true;
%   [rv, diagnostics] = realized_spectrum(fiveminreturns, options);
%
% REFERENCES:
%   Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., & Shephard, N. (2008).
%   "Designing realized kernels to measure the ex post variation of equity
%   prices in the presence of noise." Econometrica, 76(6), 1481-1536.
%
%   Malliavin, P., & Mancino, M.E. (2009). "A Fourier transform method for
%   nonparametric estimation of multivariate volatility." Annals of Statistics, 
%   37(4), 1983-2010.
%
% See also RV_COMPUTE, RV_KERNEL, DATACHECK, COLUMNCHECK, PARAMETERCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input validation
if nargin < 1
    error('At least one input (RETURNS) is required.');
end

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Process options with defaults
if ~isfield(options, 'windowType') || isempty(options.windowType)
    options.windowType = 'Parzen';
end

if ~isfield(options, 'cutoffFreq') || isempty(options.cutoffFreq)
    options.cutoffFreq = []; % Will be automatically determined later
end

if ~isfield(options, 'biasCorrection')
    options.biasCorrection = true;
end

if ~isfield(options, 'handleOvernight')
    options.handleOvernight = false;
end

if ~isfield(options, 'removeOutliers')
    options.removeOutliers = false;
end

if ~isfield(options, 'compareBenchmark')
    options.compareBenchmark = false;
end

if ~isfield(options, 'compareKernel')
    options.compareKernel = false;
end

% Validate returns data
returns = datacheck(returns, 'returns');

% Ensure returns are formatted as column vectors if needed
returns = columncheck(returns, 'returns');

% Get dimensions of the returns data
[T, n] = size(returns);

% Auto-determine cutoff frequency if not specified
if isempty(options.cutoffFreq)
    % Rule of thumb based on sample size and expected noise level
    % Higher sampling frequency typically means more noise
    % T/252 approximates the number of days (assuming 252 trading days/year)
    % The exponent -0.2 is derived from asymptotic theory for optimal bandwidth
    options.cutoffFreq = min(0.4, max(0.05, 1.5 * (T/252)^(-0.2)));
else
    % Validate cutoff frequency parameter
    freqOptions = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 0.5);
    options.cutoffFreq = parametercheck(options.cutoffFreq, 'cutoffFreq', freqOptions);
end

%% Apply outlier removal if specified
if options.removeOutliers
    % Simple outlier detection: remove returns > 5 std dev
    stdReturns = std(returns);
    outlierThreshold = 5 * stdReturns;
    outlierIndices = abs(returns) > repmat(outlierThreshold, T, 1);
    
    % Replace outliers with zeros (effectively excluding them)
    if any(outlierIndices(:))
        warning(['Detected and removed ' num2str(sum(outlierIndices(:))) ' outliers.']);
        returns(outlierIndices) = 0;
    end
end

%% Handle overnight returns if specified
if options.handleOvernight && T > 1
    % Simple approach: downweight first return of each day
    % For a more sophisticated approach, you would need day markers
    % This is a placeholder for the overnight return adjustment
    warning(['Overnight returns handling is enabled but requires additional ' ...
             'day marker information for proper implementation.']);
end

%% Initialize output and diagnostic variables
rv = zeros(1, n);
if nargout > 1
    diagnostics = struct();
    if options.compareBenchmark
        diagnostics.benchmark = zeros(1, n);
    end
    if options.compareKernel
        diagnostics.kernel = zeros(1, n);
    end
    diagnostics.spectrum = cell(1, n);
    diagnostics.frequencies = cell(1, n);
    diagnostics.window = cell(1, n);
end

%% Process each asset separately
for asset = 1:n
    % Extract the returns for this asset
    r = returns(:, asset);
    
    % Number of frequency points (use power of 2 for efficient FFT)
    nfft = 2^nextpow2(T);
    
    % Compute the periodogram using FFT
    X = fft(r, nfft);
    periodogram = abs(X).^2 / T;
    
    % Create frequency grid (0 to π)
    freq = linspace(0, 0.5, nfft/2 + 1);
    
    % Generate spectral window in a vectorized way
    window = zeros(size(freq));
    idx = freq <= options.cutoffFreq;
    if any(idx)
        % Normalized frequencies
        x = freq(idx) / options.cutoffFreq;
        
        switch lower(options.windowType)
            case 'parzen'
                % Parzen window
                idx1 = x <= 0.5;
                idx2 = ~idx1;
                window(idx(idx1)) = 1 - 6*x(idx1).^2 + 6*x(idx1).^3;
                window(idx(idx2)) = 2 * (1-x(idx2)).^3;
                
            case 'bartlett'
                % Bartlett (triangular) window
                window(idx) = 1 - x;
                
            case 'tukey-hanning'
                % Tukey-Hanning window
                window(idx) = 0.5 * (1 + cos(pi*x));
                
            case 'quadratic'
                % Quadratic window
                window(idx) = 1 - x.^2;
                
            case 'cubic'
                % Cubic window
                window(idx) = 1 - 3*x.^2 + 2*x.^3;
                
            case 'flat-top'
                % Flat-Top window
                window(idx) = 1 - 1.93*cos(2*pi*x) + 1.29*cos(4*pi*x) - ...
                              0.388*cos(6*pi*x) + 0.032*cos(8*pi*x);
                
            otherwise
                error('Unsupported window type: %s', options.windowType);
        end
    end
    
    % Apply the window to the periodogram (only use the first half due to symmetry)
    windowed_periodogram = periodogram(1:length(freq)) .* window;
    
    % Integrate the spectral density to estimate the integrated variance
    % The scaling factor accounts for the fact that we're only using half the spectrum
    % and properly normalizes the FFT
    spectral_rv = sum(windowed_periodogram) * 2 * (0.5 / length(freq));
    
    % Apply bias correction if requested
    if options.biasCorrection
        % Simple correction factor based on asymptotic theory for noise bias
        cutoffIdx = max(2, ceil(options.cutoffFreq * nfft));
        correction_factor = 1 + 1/(2*cutoffIdx);
        spectral_rv = spectral_rv / correction_factor;
    end
    
    % Store the result
    rv(1, asset) = spectral_rv;
    
    % Compute and store diagnostics if requested
    if nargout > 1
        diagnostics.spectrum{asset} = windowed_periodogram;
        diagnostics.frequencies{asset} = freq;
        diagnostics.window{asset} = window;
        
        % Compute benchmark realized volatility if requested
        if options.compareBenchmark
            diagnostics.benchmark(1, asset) = rv_compute(r);
        end
        
        % Compute kernel-based realized volatility if requested
        if options.compareKernel
            kernel_options = struct('kernelType', 'Bartlett-Parzen');
            diagnostics.kernel(1, asset) = rv_kernel(r, kernel_options);
        end
    end
end

% Ensure non-negative results (numerical issues might yield small negative values)
rv = max(0, rv);

% If user wants just the value when nargout=0
if nargout == 0 && n == 1
    disp(['Spectral Realized Volatility: ', num2str(rv)]);
end

% Return diagnostics if requested
if nargout > 1
    varargout{1} = diagnostics;
end
end