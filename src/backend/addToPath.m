function success = addToPath(savePath, addOptionalDirs)
% ADDTOPATH  Adds all necessary directories of the MFE Toolbox to the MATLAB path
%
% USAGE:
%   success = addToPath()
%   success = addToPath(savePath)
%   success = addToPath(savePath, addOptionalDirs)
%
% INPUTS:
%   savePath       - [optional] Logical indicating whether to save the path permanently (Default: false)
%   addOptionalDirs - [optional] Logical indicating whether to add optional directories (Default: true)
%
% OUTPUTS:
%   success        - Logical indicating if the path was successfully configured
%
% COMMENTS:
%   This function adds all necessary directories for the MFE Toolbox to the MATLAB path
%   and handles platform-specific configurations. It also adds optional directories
%   if they exist and are requested. The function can save the path permanently if specified.
%
%   Mandatory directories:
%   - bootstrap
%   - crosssection
%   - distributions
%   - GUI
%   - multivariate
%   - tests
%   - timeseries
%   - univariate
%   - utility
%   - realized
%   - mex_source
%   - dlls (contains platform-specific MEX binaries)
%
%   Optional directories:
%   - duplication (contains work-alike functions)
%
% EXAMPLES:
%   % Add all directories to the path without saving
%   addToPath();
%
%   % Add all directories to the path and save the path permanently
%   addToPath(true);
%
%   % Add only mandatory directories to the path
%   addToPath(false, false);
%
% See also: ADDPATH, SAVEPATH

% Set default parameters if not provided
if nargin < 1 || isempty(savePath)
    savePath = false;
end

if nargin < 2 || isempty(addOptionalDirs)
    addOptionalDirs = true;
end

% Initialize success flag
success = false;

try
    % Get the directory of the current file (mfeRoot)
    mfeRoot = fileparts(mfilename('fullpath'));
    
    % Platform check
    if ispc()
        % Windows platform: add MEX DLLs
        dllDir = fullfile(mfeRoot, 'dlls');
        if exist(dllDir, 'dir')
            addpath(dllDir);
        else
            warning('MFE:MissingDirectory', 'DLLs directory not found: %s', dllDir);
        end
    else
        % For Unix platforms, skip Windows-specific DLLs
        % But we will still add MEX binaries for Unix when we add core directories
    end
    
    % Add core directories
    coreDirs = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                'multivariate', 'tests', 'timeseries', 'univariate', ...
                'utility', 'realized', 'mex_source'};
    
    % Add dlls directory for Unix platforms here, after the platform check
    if ~ispc()
        coreDirs{end+1} = 'dlls';
    end
    
    for i = 1:length(coreDirs)
        dirPath = fullfile(mfeRoot, coreDirs{i});
        if exist(dirPath, 'dir')
            addpath(dirPath);
        else
            warning('MFE:MissingDirectory', 'Core directory "%s" not found.', dirPath);
        end
    end
    
    % Optional directories check
    if addOptionalDirs
        optionalDir = fullfile(mfeRoot, 'duplication');
        if exist(optionalDir, 'dir')
            addpath(optionalDir);
            disp('Added optional directory with work-alike functions.');
        end
    end
    
    % Save path if requested
    if savePath
        saveResult = savepath();
        if saveResult ~= 0
            warning('MFE:PathNotSaved', 'Failed to save the path permanently. You may need administrator privileges.');
        else
            disp('MATLAB path saved permanently.');
        end
    elseif nargin < 1
        % Ask user if they want to save the path
        savePathInput = input('Do you want to save the path permanently? (Y/N) [N]: ', 's');
        if ~isempty(savePathInput) && upper(savePathInput(1)) == 'Y'
            saveResult = savepath();
            if saveResult ~= 0
                warning('MFE:PathNotSaved', 'Failed to save the path permanently. You may need administrator privileges.');
            else
                disp('MATLAB path saved permanently.');
            end
        end
    end
    
    % Set success flag
    success = true;
    
catch err
    warning('MFE:ConfigError', 'Error configuring MFE Toolbox path: %s', err.message);
    success = false;
end
end