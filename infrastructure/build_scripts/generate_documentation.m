function success = generate_documentation(outputDir, overwriteExisting)
% GENERATE_DOCUMENTATION Automatically generates comprehensive documentation for the MFE Toolbox
%
% This function extracts function help text from MATLAB source files, creates
% structured Markdown documentation files, and produces a consistent set of
% documentation pages according to standardized templates.
%
% USAGE:
%   success = generate_documentation()
%   success = generate_documentation(outputDir)
%   success = generate_documentation(outputDir, overwriteExisting)
%
% INPUTS:
%   outputDir         - [optional] Directory where documentation will be generated
%                       (default: 'docs/')
%   overwriteExisting - [optional] Logical flag indicating whether to overwrite
%                       existing documentation files (default: false)
%
% OUTPUTS:
%   success           - Logical value indicating if documentation was successfully generated
%
% COMMENTS:
%   This is part of the MFE Toolbox build system that automates the creation
%   of standardized documentation across all components.
%
% See also: parametercheck

% Define global constants
global TOOLBOX_VERSION TOOLBOX_ROOT DOC_DIRECTORIES DEFAULT_TEMPLATE_PATH DEFAULT_OUTPUT_DIR

% Set defaults if parameters not provided
if nargin < 1 || isempty(outputDir)
    DEFAULT_OUTPUT_DIR = 'docs/';
    outputDir = DEFAULT_OUTPUT_DIR;
else
    % Validate outputDir parameter
    if ~ischar(outputDir)
        error('outputDir must be a string representing a valid directory path.');
    end
end

if nargin < 2 || isempty(overwriteExisting)
    overwriteExisting = false;
else
    % Validate overwriteExisting parameter using parametercheck
    options.isscalar = true;
    overwriteExisting = parametercheck(overwriteExisting, 'overwriteExisting', options);
end

% Determine toolbox root directory based on relative path
scriptPath = mfilename('fullpath');
[scriptDir, ~, ~] = fileparts(scriptPath);
TOOLBOX_ROOT = fullfile(scriptDir, '..', '..');

% Define mandatory directories for documentation
DOC_DIRECTORIES = {'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                  'multivariate', 'tests', 'timeseries', 'univariate', ...
                  'utility', 'realized'};

% Set default template path
DEFAULT_TEMPLATE_PATH = fullfile(TOOLBOX_ROOT, 'infrastructure', 'templates', 'documentation_template.md');

% Extract toolbox version information from Contents.m
contentsPath = fullfile(TOOLBOX_ROOT, 'src', 'backend', 'Contents.m');
if ~exist(contentsPath, 'file')
    error('Could not find Contents.m at expected location: %s', contentsPath);
end

contentsInfo = help('Contents');
versionMatch = regexp(contentsInfo, 'Version\s+(\d+\.\d+)\s+\((\d+-\w+-\d+)\)', 'tokens', 'once');
if ~isempty(versionMatch)
    TOOLBOX_VERSION = versionMatch{1};
    releaseDate = versionMatch{2};
else
    % Default version if not found
    TOOLBOX_VERSION = '4.0';
    releaseDate = '28-Oct-2009';
end

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created documentation directory: %s\n', outputDir);
else
    % Check if existing documentation should be overwritten
    if ~overwriteExisting
        error(['Documentation directory already exists. ', ...
              'Use overwriteExisting=true to overwrite existing files.']);
    else
        fprintf('Warning: Overwriting existing documentation in %s\n', outputDir);
    end
end

% Statistics counters
stats = struct();
stats.totalComponents = length(DOC_DIRECTORIES);
stats.totalFunctions = 0;
stats.generatedFiles = 0;

% Generate main README.md with overview of the toolbox
readmePath = fullfile(outputDir, 'README.md');
fprintf('Generating main README.md...\n');
fh = fopen(readmePath, 'w');
fprintf(fh, '# MFE Toolbox v%s\n\n', TOOLBOX_VERSION);
fprintf(fh, '*MATLAB Financial Econometrics Toolbox - Release Date: %s*\n\n', releaseDate);
fprintf(fh, '## Overview\n\n');
fprintf(fh, 'The MFE Toolbox is a sophisticated MATLAB-based software suite ');
fprintf(fh, 'designed to provide comprehensive tools for financial time series ');
fprintf(fh, 'modeling, econometric analysis, and risk assessment.\n\n');
fprintf(fh, '## Components\n\n');

% Extract component descriptions from Contents.m
fid = fopen(contentsPath, 'r');
if fid ~= -1
    content = textscan(fid, '%s', 'delimiter', '\n');
    content = content{1};
    fclose(fid);
    
    % Find component sections in Contents.m
    inComponentSection = false;
    currentComponent = '';
    
    for i = 1:length(content)
        line = content{i};
        
        % Check if line contains a component section header
        componentMatch = regexp(line, '^% ([A-Za-z\s]+)$', 'tokens', 'once');
        if ~isempty(componentMatch) && ~contains(line, 'MFETOOLBOX')
            inComponentSection = true;
            currentComponent = componentMatch{1};
            fprintf(fh, '### %s\n\n', currentComponent);
            continue;
        end
        
        % Check if line contains a function description
        if inComponentSection && startsWith(line, '%   ')
            functionMatch = regexp(line, '%   ([a-zA-Z0-9_]+)\s*-\s*(.+)$', 'tokens', 'once');
            if ~isempty(functionMatch)
                funcName = functionMatch{1};
                funcDesc = functionMatch{2};
                fprintf(fh, '- **%s**: %s\n', funcName, funcDesc);
            end
        elseif inComponentSection && ~startsWith(line, '%')
            inComponentSection = false;
        end
    end
end
fprintf(fh, '\n## Documentation\n\n');
fprintf(fh, 'For detailed documentation on each component, please refer to the appropriate section:\n\n');

% Add links to component documentation
for i = 1:length(DOC_DIRECTORIES)
    compName = DOC_DIRECTORIES{i};
    fprintf(fh, '- [%s](%s/README.md)\n', capitalize(compName), compName);
end

fprintf(fh, '\n## Installation\n\n');
fprintf(fh, 'See the [Installation Guide](installation.md) for setup instructions.\n\n');
fprintf(fh, '## API Reference\n\n');
fprintf(fh, 'For complete API documentation, see the [API Reference](api_reference.md).\n');
fclose(fh);
stats.generatedFiles = stats.generatedFiles + 1;

% Generate component-specific documentation files for each subdirectory
componentInfos = cell(length(DOC_DIRECTORIES), 1);
for i = 1:length(DOC_DIRECTORIES)
    compName = DOC_DIRECTORIES{i};
    compPath = fullfile(TOOLBOX_ROOT, 'src', 'backend', compName);
    compOutputDir = fullfile(outputDir, compName);
    
    % Create component directory if it doesn't exist
    if ~exist(compOutputDir, 'dir')
        mkdir(compOutputDir);
    end
    
    fprintf('Generating documentation for %s component...\n', compName);
    componentInfos{i} = generate_component_docs(compPath, compOutputDir, DEFAULT_TEMPLATE_PATH);
    
    stats.totalFunctions = stats.totalFunctions + componentInfos{i}.functionCount;
    stats.generatedFiles = stats.generatedFiles + componentInfos{i}.generatedFiles;
end

% Generate API reference documentation
fprintf('Generating API reference...\n');
apiRefPath = fullfile(outputDir, 'api_reference.md');
apiSuccess = generate_api_reference(apiRefPath, componentInfos);
if apiSuccess
    stats.generatedFiles = stats.generatedFiles + 1;
end

% Generate installation and getting started guides
fprintf('Generating installation guide...\n');
installPath = fullfile(outputDir, 'installation.md');
fh = fopen(installPath, 'w');
fprintf(fh, '# Installation Guide\n\n');
fprintf(fh, '## Requirements\n\n');
fprintf(fh, '- MATLAB (compatible with version 4.0 or later)\n');
fprintf(fh, '- Statistics Toolbox (recommended)\n');
fprintf(fh, '- Optimization Toolbox (recommended)\n\n');
fprintf(fh, '## Installation Steps\n\n');
fprintf(fh, '1. Download the MFE Toolbox package (MFEToolbox.zip)\n');
fprintf(fh, '2. Extract the ZIP file to a directory of your choice\n');
fprintf(fh, '3. Start MATLAB\n');
fprintf(fh, '4. Navigate to the MFE Toolbox directory\n');
fprintf(fh, '5. Run the following command to add the toolbox to your MATLAB path:\n\n');
fprintf(fh, '```matlab\naddToPath\n```\n\n');
fprintf(fh, '6. To permanently add the toolbox to your MATLAB path, use:\n\n');
fprintf(fh, '```matlab\naddToPath(true)\n```\n\n');
fprintf(fh, '## Verification\n\n');
fprintf(fh, 'To verify that the toolbox is correctly installed, run:\n\n');
fprintf(fh, '```matlab\nhelp mfetoolbox\n```\n\n');
fprintf(fh, 'This should display the contents and version information for the MFE Toolbox.\n');
fclose(fh);
stats.generatedFiles = stats.generatedFiles + 1;

% Generate getting started guide
fprintf('Generating getting started guide...\n');
gettingStartedPath = fullfile(outputDir, 'getting_started.md');
fh = fopen(gettingStartedPath, 'w');
fprintf(fh, '# Getting Started with MFE Toolbox\n\n');
fprintf(fh, '## Basic Usage\n\n');
fprintf(fh, 'This guide will help you get started with the MFE Toolbox for MATLAB.\n\n');
fprintf(fh, '## Examples\n\n');
fprintf(fh, '### Time Series Analysis\n\n');
fprintf(fh, '```matlab\n');
fprintf(fh, '%% Load sample data\ndata = randn(100, 1);\n\n');
fprintf(fh, '%% Estimate an AR(1) model\n[parameters, stderrors, logL] = armaxfilter(data, 1, 0);\n\n');
fprintf(fh, '%% Generate forecasts\nforecasts = armafor(data, parameters, 1, 0, 10);\n\n');
fprintf(fh, '%% Plot the results\nplot([data; forecasts]);\nhold on;\nplot(length(data):length(data)+9, forecasts, ''r'', ''LineWidth'', 2);\nlegend(''Original Data'', ''Forecasts'');\n');
fprintf(fh, '```\n\n');
fprintf(fh, '### Volatility Modeling\n\n');
fprintf(fh, '```matlab\n');
fprintf(fh, '%% Load sample return data\nreturns = randn(1000, 1);\n\n');
fprintf(fh, '%% Estimate a GARCH(1,1) model\n[parameters, logL, ht] = tarchfit(returns, 1, 1, 0);\n\n');
fprintf(fh, '%% Plot the conditional volatility\nplot(sqrt(ht));\ntitle(''Estimated Conditional Volatility'');\n');
fprintf(fh, '```\n\n');
fprintf(fh, '## Common Tasks\n\n');
fprintf(fh, '- [Time Series Analysis](timeseries/README.md)\n');
fprintf(fh, '- [Volatility Modeling](univariate/README.md)\n');
fprintf(fh, '- [Distribution Analysis](distributions/README.md)\n');
fprintf(fh, '- [Statistical Testing](tests/README.md)\n\n');
fprintf(fh, '## Next Steps\n\n');
fprintf(fh, 'Explore the [API Reference](api_reference.md) for detailed information on all available functions.\n');
fclose(fh);
stats.generatedFiles = stats.generatedFiles + 1;

% Copy images and resources to documentation directory
fprintf('Copying documentation resources...\n');
resourcesSuccess = copy_documentation_resources(TOOLBOX_ROOT, outputDir);

% Generate table of contents and index pages
fprintf('Generating table of contents...\n');
tocPath = generate_table_of_contents(outputDir, stats);
if ~isempty(tocPath)
    stats.generatedFiles = stats.generatedFiles + 1;
end

% Display completion message with statistics
fprintf('\n=== Documentation Generation Complete ===\n');
fprintf('Total components processed: %d\n', stats.totalComponents);
fprintf('Total functions documented: %d\n', stats.totalFunctions);
fprintf('Total files generated: %d\n', stats.generatedFiles);
fprintf('Documentation available at: %s\n', outputDir);

% Return success status
success = true;
end

function results = generate_component_docs(componentPath, outputDir, templatePath)
% GENERATE_COMPONENT_DOCS Generates documentation for a specific toolbox component directory
%
% This function generates comprehensive documentation for all MATLAB functions
% within a specified component directory, using a standardized template format.
%
% USAGE:
%   results = generate_component_docs(componentPath, outputDir, templatePath)
%
% INPUTS:
%   componentPath - Path to the component directory
%   outputDir     - Directory where documentation will be generated
%   templatePath  - Path to the documentation template file
%
% OUTPUTS:
%   results       - Structure containing results and statistics:
%                   .componentName - Name of the component
%                   .functionCount - Number of functions documented
%                   .functionInfos - Cell array of function information structures
%                   .generatedFiles - Number of files generated

% Validate component path using parametercheck
if ~exist('componentPath', 'var') || isempty(componentPath)
    error('Component path must be specified.');
end

if ~exist('outputDir', 'var') || isempty(outputDir)
    error('Output directory must be specified.');
end

if ~exist('templatePath', 'var') || isempty(templatePath)
    error('Template path must be specified.');
end

% Ensure output directory exists
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get component name from path
[~, componentName, ~] = fileparts(componentPath);

% Load documentation template
if ~exist(templatePath, 'file')
    error('Template file not found: %s', templatePath);
end

fid = fopen(templatePath, 'r');
if fid == -1
    error('Could not open template file: %s', templatePath);
end
template = textscan(fid, '%s', 'delimiter', '\n');
template = strjoin(template{1}, '\n');
fclose(fid);

% Get list of MATLAB (.m) files in component directory
mFiles = dir(fullfile(componentPath, '*.m'));

% Initialize results structure
results = struct();
results.componentName = componentName;
results.functionCount = 0;
results.functionInfos = cell(length(mFiles), 1);
results.generatedFiles = 0;

% Process each MATLAB file
fprintf('  Processing %d files in %s component...\n', length(mFiles), componentName);
for i = 1:length(mFiles)
    filePath = fullfile(componentPath, mFiles(i).name);
    functionInfo = extract_function_info(filePath);
    
    if ~isempty(functionInfo)
        results.functionCount = results.functionCount + 1;
        results.functionInfos{results.functionCount} = functionInfo;
    end
end

% Generate component README file
readmePath = fullfile(outputDir, 'README.md');
fh = fopen(readmePath, 'w');
fprintf(fh, '# %s Component\n\n', capitalize(componentName));

% Find component description from Contents.m
global TOOLBOX_VERSION;
contentsPath = fullfile(fileparts(fileparts(componentPath)), 'Contents.m');
componentDesc = '';
if exist(contentsPath, 'file')
    contentsText = fileread(contentsPath);
    % Extract component description section
    componentPattern = sprintf('%% %s[^\n]*', capitalize(componentName));
    matches = regexp(contentsText, componentPattern, 'match');
    if ~isempty(matches)
        componentDesc = strrep(matches{1}, '%', '');
        componentDesc = strtrim(componentDesc);
    end
end

if isempty(componentDesc)
    componentDesc = [capitalize(componentName), ' Component'];
end

fprintf(fh, '*MFE Toolbox v%s*\n\n', TOOLBOX_VERSION);
fprintf(fh, '## Overview\n\n');
fprintf(fh, '%s\n\n', componentDesc);
fprintf(fh, '## Functions\n\n');

% List all functions with brief descriptions
for i = 1:results.functionCount
    func = results.functionInfos{i};
    % Get brief description (first line of help text)
    briefDesc = '';
    if ~isempty(func.helpText)
        helpLines = strsplit(func.helpText, '\n');
        if length(helpLines) > 0
            briefDesc = strtrim(helpLines{1});
            briefDesc = regexprep(briefDesc, '^[A-Z0-9_]+ ', ''); % Remove function name prefix
        end
    end
    fprintf(fh, '- [%s](%s.md) - %s\n', func.name, func.name, briefDesc);
end

fprintf(fh, '\n## Examples\n\n');
fprintf(fh, 'See individual function documentation for specific examples.\n\n');
fprintf(fh, '## Related Components\n\n');

% Find related components by checking function dependencies
global DOC_DIRECTORIES;
relatedComponents = {};
for i = 1:results.functionCount
    if isfield(results.functionInfos{i}, 'dependencies')
        deps = results.functionInfos{i}.dependencies;
        for j = 1:length(deps)
            for k = 1:length(DOC_DIRECTORIES)
                if strcmpi(deps{j}, DOC_DIRECTORIES{k}) && ~strcmpi(deps{j}, componentName)
                    relatedComponents{end+1} = deps{j};
                end
            end
        end
    end
end

relatedComponents = unique(relatedComponents);
if ~isempty(relatedComponents)
    for i = 1:length(relatedComponents)
        fprintf(fh, '- [%s](../%s/README.md)\n', capitalize(relatedComponents{i}), relatedComponents{i});
    end
else
    fprintf(fh, 'None\n');
end

fclose(fh);
results.generatedFiles = results.generatedFiles + 1;

% Generate individual function documentation files
for i = 1:results.functionCount
    func = results.functionInfos{i};
    funcPath = fullfile(outputDir, [func.name, '.md']);
    fprintf('  Generating documentation for %s...\n', func.name);
    
    % Format the content using the template
    content = struct();
    content.COMPONENT_NAME = capitalize(componentName);
    content.VERSION = TOOLBOX_VERSION;
    content.RELEASE_DATE = 'October 28, 2009'; % From Contents.m
    content.COMPONENT_DESCRIPTION = componentDesc;
    content.FUNCTION_NAME = func.name;
    
    % Get function description
    content.FUNCTION_DESCRIPTION = '';
    if isfield(func.sections, 'description')
        content.FUNCTION_DESCRIPTION = func.sections.description;
    elseif ~isempty(func.helpText)
        helpLines = strsplit(func.helpText, '\n');
        if length(helpLines) > 0
            content.FUNCTION_DESCRIPTION = strtrim(helpLines{1});
            content.FUNCTION_DESCRIPTION = regexprep(content.FUNCTION_DESCRIPTION, '^[A-Z0-9_]+ ', '');
        end
    end
    
    % Get function signature
    content.FUNCTION_SIGNATURE = func.signature;
    
    % Format parameter descriptions
    content.PARAMETER_DESCRIPTIONS = '';
    if isfield(func.sections, 'inputs')
        parameterLines = strsplit(func.sections.inputs, '\n');
        for j = 1:length(parameterLines)
            line = strtrim(parameterLines{j});
            if ~isempty(line)
                content.PARAMETER_DESCRIPTIONS = [content.PARAMETER_DESCRIPTIONS, '- ', line, '\n'];
            end
        end
    end
    
    % Format return value descriptions
    content.RETURN_DESCRIPTIONS = '';
    if isfield(func.sections, 'outputs')
        returnLines = strsplit(func.sections.outputs, '\n');
        for j = 1:length(returnLines)
            line = strtrim(returnLines{j});
            if ~isempty(line)
                content.RETURN_DESCRIPTIONS = [content.RETURN_DESCRIPTIONS, '- ', line, '\n'];
            end
        end
    end
    
    % Format examples
    content.EXAMPLE_CODE = '';
    if isfield(func.sections, 'examples')
        content.EXAMPLE_CODE = func.sections.examples;
    end
    
    % Component examples
    content.COMPONENT_EXAMPLES = 'See function examples above.';
    
    % Mathematical background (if available)
    content.FORMULA_NAME = '';
    content.LATEX_FORMULA = '';
    content.VARIABLE_DEFINITIONS = '';
    
    % References
    content.REFERENCES = '';
    if isfield(func.sections, 'references')
        referencesLines = strsplit(func.sections.references, '\n');
        for j = 1:length(referencesLines)
            line = strtrim(referencesLines{j});
            if ~isempty(line)
                content.REFERENCES = [content.REFERENCES, '- ', line, '\n'];
            end
        end
    end
    
    % Related components
    content.RELATED_COMPONENTS = '';
    if isfield(func, 'dependencies')
        for j = 1:length(func.dependencies)
            dep = func.dependencies{j};
            % Check if dependency is a component
            isComponent = false;
            for k = 1:length(DOC_DIRECTORIES)
                if strcmpi(dep, DOC_DIRECTORIES{k})
                    isComponent = true;
                    break;
                end
            end
            
            if isComponent
                content.RELATED_COMPONENTS = [content.RELATED_COMPONENTS, ...
                    sprintf('- [%s](../%s/README.md)\n', capitalize(dep), dep)];
            end
        end
    end
    
    % Format content as Markdown
    formattedContent = format_markdown(content, template);
    
    % Write to file
    fh = fopen(funcPath, 'w');
    fprintf(fh, '%s', formattedContent);
    fclose(fh);
    results.generatedFiles = results.generatedFiles + 1;
end

end

function functionInfo = extract_function_info(filePath)
% EXTRACT_FUNCTION_INFO Extracts function information and help text from MATLAB source file
%
% This function extracts information about a MATLAB function from its source file,
% including the function name, signature, help text, and structured help sections.
%
% USAGE:
%   functionInfo = extract_function_info(filePath)
%
% INPUTS:
%   filePath     - Path to the MATLAB source file
%
% OUTPUTS:
%   functionInfo - Structure containing function information:
%                  .name - Function name
%                  .signature - Function signature
%                  .helpText - Raw help text
%                  .sections - Structure with parsed help sections
%                  .dependencies - Cell array of detected dependencies

if ~exist(filePath, 'file')
    error('File not found: %s', filePath);
end

% Initialize function info structure
functionInfo = struct();
functionInfo.name = '';
functionInfo.signature = '';
functionInfo.helpText = '';
functionInfo.sections = struct();
functionInfo.dependencies = {};

% Get the function name from file name
[~, functionName, ~] = fileparts(filePath);
functionInfo.name = functionName;

% Open source file for reading
fid = fopen(filePath, 'r');
if fid == -1
    error('Could not open file: %s', filePath);
end

% Read the first line to get the function signature
firstLine = fgetl(fid);
fclose(fid);

% Extract function signature using regular expressions
signaturePattern = 'function\s+(?:(\[?[\w\s,]+\]?)\s*=\s*)?(\w+)\s*\(([\w\s,]*)\)';
tokens = regexp(firstLine, signaturePattern, 'tokens', 'once');
if ~isempty(tokens)
    outputs = tokens{1};
    funcName = tokens{2};
    inputs = tokens{3};
    
    if ~strcmp(funcName, functionName)
        % If file name doesn't match function name, use the function name instead
        functionInfo.name = funcName;
    end
    
    % Format the signature
    if isempty(outputs)
        functionInfo.signature = sprintf('%s(%s)', functionName, inputs);
    else
        functionInfo.signature = sprintf('[%s] = %s(%s)', outputs, functionName, inputs);
    end
else
    % Default signature if pattern not found
    functionInfo.signature = sprintf('%s()', functionName);
end

% Extract help text using MATLAB's help function
helpText = help(filePath);
functionInfo.helpText = helpText;

% Parse help text into sections
functionInfo.sections = parse_help_text(helpText);

% Extract dependencies by checking function calls in the file
content = fileread(filePath);
% Look for function calls (simple regex - could be improved for more complex cases)
funcCallPattern = '(\w+)\s*\(';
funcCalls = regexp(content, funcCallPattern, 'tokens');
funcCalls = [funcCalls{:}];
funcCalls = unique(funcCalls);

% Filter out known MATLAB built-in functions
matlabBuiltins = {'if', 'for', 'while', 'switch', 'case', 'otherwise', ...
                  'min', 'max', 'sum', 'mean', 'std', 'var', 'cov', 'corr', ...
                  'error', 'warning', 'disp', 'fprintf', 'sprintf', ...
                  'regexp', 'strfind', 'strrep', 'strtrim', 'strcmp', 'strcmpi', ...
                  'isempty', 'isnan', 'isinf', 'isfinite', 'isfloat', 'isinteger', ...
                  'length', 'size', 'numel', 'zeros', 'ones', 'eye', 'rand', 'randn', ...
                  'any', 'all', 'find', 'sort', 'unique'};

dependencies = {};
for i = 1:length(funcCalls)
    if ~ismember(funcCalls{i}, matlabBuiltins) && ~strcmp(funcCalls{i}, functionName)
        dependencies{end+1} = funcCalls{i};
    end
end
functionInfo.dependencies = unique(dependencies);

end

function sections = parse_help_text(helpText)
% PARSE_HELP_TEXT Parses MATLAB help text into structured sections
%
% This function parses standard MATLAB help text format and extracts
% structured sections like description, syntax, inputs, outputs, examples, etc.
%
% USAGE:
%   sections = parse_help_text(helpText)
%
% INPUTS:
%   helpText - Raw help text extracted from a MATLAB function
%
% OUTPUTS:
%   sections - Structure with parsed help text sections:
%              .description - Overall function description
%              .syntax - Usage syntax examples
%              .inputs - Input parameter descriptions
%              .outputs - Output parameter descriptions
%              .examples - Usage examples
%              .references - References to papers or documentation
%              .seeAlso - Related functions
%              .comments - Additional comments/notes

% Initialize sections structure
sections = struct();

% If help text is empty, return empty structure
if isempty(helpText)
    return;
end

% Split help text into lines
helpLines = strsplit(helpText, '\n');

% Identify section headers
sectionPatterns = {
    'USAGE'       , 'syntax';
    'INPUTS'      , 'inputs';
    'INPUT'       , 'inputs';
    'OUTPUTS'     , 'outputs';
    'OUTPUT'      , 'outputs';
    'PARAMETER'   , 'inputs';
    'PARAMETERS'  , 'inputs';
    'RETURNS'     , 'outputs';
    'RETURN'      , 'outputs';
    'EXAMPLES'    , 'examples';
    'EXAMPLE'     , 'examples';
    'REFERENCES'  , 'references';
    'REFERENCE'   , 'references';
    'SEE ALSO'    , 'seeAlso';
    'COMMENTS'    , 'comments';
    'COMMENT'     , 'comments';
    'NOTES'       , 'comments';
    'NOTE'        , 'comments';
};

% Extract description from first paragraph
descriptionLines = {};
lineIndex = 1;
while lineIndex <= length(helpLines)
    line = strtrim(helpLines{lineIndex});
    if isempty(line) || (lineIndex > 1 && any(cellfun(@(p) contains(upper(line), p), sectionPatterns(:,1))))
        break;
    end
    
    % Remove function name from first line if present
    if lineIndex == 1
        line = regexprep(line, '^[A-Z0-9_]+ ', '');
    end
    
    descriptionLines{end+1} = line;
    lineIndex = lineIndex + 1;
end

if ~isempty(descriptionLines)
    sections.description = strjoin(descriptionLines, '\n');
end

% Process remaining sections
currentSection = '';
sectionContent = {};

while lineIndex <= length(helpLines)
    line = strtrim(helpLines{lineIndex});
    lineIndex = lineIndex + 1;
    
    % Check if line starts a new section
    isNewSection = false;
    for i = 1:size(sectionPatterns, 1)
        if contains(upper(line), sectionPatterns{i,1})
            % Save previous section if any
            if ~isempty(currentSection) && ~isempty(sectionContent)
                sections.(currentSection) = strjoin(sectionContent, '\n');
            end
            
            % Start new section
            currentSection = sectionPatterns{i,2};
            sectionContent = {};
            isNewSection = true;
            break;
        end
    end
    
    if isNewSection
        continue;
    end
    
    % Add line to current section
    if ~isempty(currentSection) && ~isempty(line)
        sectionContent{end+1} = line;
    end
end

% Save final section if any
if ~isempty(currentSection) && ~isempty(sectionContent)
    sections.(currentSection) = strjoin(sectionContent, '\n');
end

end

function success = generate_api_reference(outputDir, componentInfos)
% GENERATE_API_REFERENCE Generates comprehensive API reference documentation
%
% This function generates a comprehensive API reference document that organizes
% all functions by category and provides an alphabetical index.
%
% USAGE:
%   success = generate_api_reference(outputDir, componentInfos)
%
% INPUTS:
%   outputDir     - Directory where API reference will be generated
%   componentInfos - Cell array of component information structures
%
% OUTPUTS:
%   success       - Logical value indicating if API reference was successfully generated

% Validate parameters
if ~exist('outputDir', 'var') || isempty(outputDir)
    error('Output directory must be specified.');
end

if ~exist('componentInfos', 'var') || isempty(componentInfos)
    error('Component information must be provided.');
end

% Build API reference file path
apiRefPath = fullfile(outputDir, 'api_reference.md');

% Open API reference file for writing
fh = fopen(apiRefPath, 'w');
if fh == -1
    error('Could not create API reference file: %s', apiRefPath);
end

% Write API reference header
fprintf(fh, '# MFE Toolbox API Reference\n\n');
global TOOLBOX_VERSION;
fprintf(fh, '*MFE Toolbox v%s - October 28, 2009*\n\n', TOOLBOX_VERSION);
fprintf(fh, 'This document provides comprehensive reference for all functions in the MFE Toolbox.\n\n');

% Create alphabetical index
fprintf(fh, '## Alphabetical Index\n\n');

% Collect all function information
allFunctions = {};
for i = 1:length(componentInfos)
    if ~isempty(componentInfos{i})
        componentName = componentInfos{i}.componentName;
        for j = 1:componentInfos{i}.functionCount
            funcInfo = componentInfos{i}.functionInfos{j};
            allFunctions{end+1} = struct('name', funcInfo.name, 'component', componentName);
        end
    end
end

% Sort functions alphabetically
functionNames = cellfun(@(f) f.name, allFunctions, 'UniformOutput', false);
[~, sortIdx] = sort(lower(functionNames));
allFunctions = allFunctions(sortIdx);

% Group functions by first letter for the index
currentLetter = '';
for i = 1:length(allFunctions)
    func = allFunctions{i};
    firstLetter = upper(func.name(1));
    
    if ~strcmp(firstLetter, currentLetter)
        currentLetter = firstLetter;
        fprintf(fh, '\n### %s\n\n', currentLetter);
    end
    
    fprintf(fh, '- [%s](%s/%s.md) - %s component\n', ...
        func.name, func.component, func.name, capitalize(func.component));
end

% Create component-based organization
fprintf(fh, '\n## Components\n\n');
for i = 1:length(componentInfos)
    if ~isempty(componentInfos{i})
        componentName = componentInfos{i}.componentName;
        fprintf(fh, '### %s\n\n', capitalize(componentName));
        
        % Add link to component documentation
        fprintf(fh, '[%s Documentation](%s/README.md)\n\n', capitalize(componentName), componentName);
        
        % List functions in this component
        fprintf(fh, '| Function | Description |\n');
        fprintf(fh, '|----------|--------------|\n');
        
        for j = 1:componentInfos{i}.functionCount
            funcInfo = componentInfos{i}.functionInfos{j};
            
            % Get brief description
            briefDesc = '';
            if ~isempty(funcInfo.helpText)
                helpLines = strsplit(funcInfo.helpText, '\n');
                if length(helpLines) > 0
                    briefDesc = strtrim(helpLines{1});
                    briefDesc = regexprep(briefDesc, '^[A-Z0-9_]+ ', '');
                end
            end
            
            fprintf(fh, '| [%s](%s/%s.md) | %s |\n', ...
                funcInfo.name, componentName, funcInfo.name, briefDesc);
        end
        
        fprintf(fh, '\n');
    end
end

% Add common parameter types reference
fprintf(fh, '## Common Parameter Types\n\n');
fprintf(fh, 'The following parameter types are commonly used throughout the MFE Toolbox:\n\n');

fprintf(fh, '| Parameter Type | Description |\n');
fprintf(fh, '|---------------|--------------|\n');
fprintf(fh, '| T×1 vector    | Time series data with T observations |\n');
fprintf(fh, '| T×K matrix    | Multiple time series or predictor variables |\n');
fprintf(fh, '| scalar integer | Order parameter (p, q, m, etc.) |\n');
fprintf(fh, '| parameter vector | Vector of model parameters |\n');
fprintf(fh, '| options structure | Structure containing estimation options |\n');

fclose(fh);
success = true;
end

function tocPath = generate_table_of_contents(outputDir, documentationInfo)
% GENERATE_TABLE_OF_CONTENTS Generates table of contents for documentation
%
% This function generates a hierarchical table of contents in Markdown format
% to facilitate navigation through the documentation.
%
% USAGE:
%   tocPath = generate_table_of_contents(outputDir, documentationInfo)
%
% INPUTS:
%   outputDir        - Directory where documentation is generated
%   documentationInfo - Structure containing information about generated documentation
%
% OUTPUTS:
%   tocPath          - Path to the generated table of contents file

% Validate parameters
if ~exist('outputDir', 'var') || isempty(outputDir)
    error('Output directory must be specified.');
end

if ~exist('documentationInfo', 'var')
    error('Documentation information must be provided.');
end

% Build table of contents file path
tocPath = fullfile(outputDir, 'TABLE_OF_CONTENTS.md');

% Open table of contents file for writing
fh = fopen(tocPath, 'w');
if fh == -1
    error('Could not create table of contents file: %s', tocPath);
    tocPath = '';
    return;
end

% Write table of contents header
fprintf(fh, '# MFE Toolbox Documentation\n\n');
global TOOLBOX_VERSION;
fprintf(fh, '*MFE Toolbox v%s - October 28, 2009*\n\n', TOOLBOX_VERSION);
fprintf(fh, 'This documentation provides comprehensive information about the MFE Toolbox,\n');
fprintf(fh, 'a MATLAB Financial Econometrics Toolbox for time series analysis, volatility modeling,\n');
fprintf(fh, 'and statistical testing.\n\n');

% Main sections
fprintf(fh, '## Main Documentation\n\n');
fprintf(fh, '- [README](README.md) - Overview and introduction\n');
fprintf(fh, '- [Installation Guide](installation.md) - Setup instructions\n');
fprintf(fh, '- [Getting Started](getting_started.md) - Basic usage examples\n');
fprintf(fh, '- [API Reference](api_reference.md) - Complete function reference\n\n');

% Component documentation
fprintf(fh, '## Components\n\n');
global DOC_DIRECTORIES;
for i = 1:length(DOC_DIRECTORIES)
    componentName = DOC_DIRECTORIES{i};
    fprintf(fh, '- [%s](%s/README.md)\n', capitalize(componentName), componentName);
    
    % Get list of function documentation files in this component
    funcFiles = dir(fullfile(outputDir, componentName, '*.md'));
    if length(funcFiles) > 0
        fprintf(fh, '  - Functions:\n');
        
        % List functions (excluding README.md)
        for j = 1:length(funcFiles)
            if ~strcmpi(funcFiles(j).name, 'README.md')
                [funcName, ~] = strtok(funcFiles(j).name, '.');
                fprintf(fh, '    - [%s](%s/%s)\n', funcName, componentName, funcFiles(j).name);
            end
        end
    end
end

fprintf(fh, '\n## Additional Resources\n\n');
fprintf(fh, '- [Mathematical Background](mathematical_background.md)\n');
fprintf(fh, '- [References](references.md)\n');
fprintf(fh, '- [Change Log](changelog.md)\n');

fclose(fh);
end

function success = copy_documentation_resources(sourceDir, outputDir)
% COPY_DOCUMENTATION_RESOURCES Copies images and other resources to documentation directory
%
% This function copies resources like images, diagrams, and other files
% needed for documentation to the output directory.
%
% USAGE:
%   success = copy_documentation_resources(sourceDir, outputDir)
%
% INPUTS:
%   sourceDir  - Source directory containing resource files
%   outputDir  - Directory where documentation is generated
%
% OUTPUTS:
%   success    - Logical value indicating if resources were successfully copied

% Validate parameters
if ~exist('sourceDir', 'var') || isempty(sourceDir)
    error('Source directory must be specified.');
end

if ~exist('outputDir', 'var') || isempty(outputDir)
    error('Output directory must be specified.');
end

% Create images directory in output if it doesn't exist
imagesDir = fullfile(outputDir, 'images');
if ~exist(imagesDir, 'dir')
    mkdir(imagesDir);
end

% Source directories for resources
resourceDirs = {
    fullfile(sourceDir, 'docs', 'images'),
    fullfile(sourceDir, 'infrastructure', 'resources'),
    fullfile(sourceDir, 'media')
};

% Track success
success = true;
fileCount = 0;

% Copy resources from each potential directory
for i = 1:length(resourceDirs)
    resourceDir = resourceDirs{i};
    if exist(resourceDir, 'dir')
        % Copy image files (png, jpg, gif, svg)
        imgFiles = [dir(fullfile(resourceDir, '*.png')); 
                   dir(fullfile(resourceDir, '*.jpg')); 
                   dir(fullfile(resourceDir, '*.gif')); 
                   dir(fullfile(resourceDir, '*.svg'))];
                   
        for j = 1:length(imgFiles)
            srcPath = fullfile(resourceDir, imgFiles(j).name);
            destPath = fullfile(imagesDir, imgFiles(j).name);
            try
                copyfile(srcPath, destPath);
                fileCount = fileCount + 1;
            catch
                fprintf('Warning: Could not copy resource file: %s\n', srcPath);
                success = false;
            end
        end
    end
end

% If no resources found, create placeholder README in images dir
if fileCount == 0
    readmePath = fullfile(imagesDir, 'README.md');
    fh = fopen(readmePath, 'w');
    fprintf(fh, '# Documentation Images\n\n');
    fprintf(fh, 'This directory contains images used in the MFE Toolbox documentation.\n');
    fclose(fh);
end

fprintf('Copied %d resource files to documentation directory.\n', fileCount);

% Create additional documentation files if they don't exist
additionalFiles = {
    'mathematical_background.md', 
    'references.md', 
    'changelog.md'
};

for i = 1:length(additionalFiles)
    filePath = fullfile(outputDir, additionalFiles{i});
    if ~exist(filePath, 'file')
        fh = fopen(filePath, 'w');
        
        % Format filename for title
        [~, title, ~] = fileparts(additionalFiles{i});
        title = strrep(title, '_', ' ');
        title = capitalize(title);
        
        fprintf(fh, '# %s\n\n', title);
        
        % Add appropriate content based on file type
        switch lower(additionalFiles{i})
            case 'mathematical_background.md'
                fprintf(fh, '## Statistical Distributions\n\n');
                fprintf(fh, '### Generalized Error Distribution (GED)\n\n');
                fprintf(fh, 'The probability density function (PDF) of the GED is given by:\n\n');
                fprintf(fh, '$$f(x|\\mu,\\sigma,\\nu) = \\frac{\\nu}{2\\lambda\\sigma\\Gamma(1/\\nu)}e^{-(|x-\\mu|/\\lambda\\sigma)^\\nu}$$\n\n');
                fprintf(fh, 'Where:\n');
                fprintf(fh, '- $\\mu$ is the location parameter\n');
                fprintf(fh, '- $\\sigma$ is the scale parameter\n');
                fprintf(fh, '- $\\nu$ is the shape parameter\n');
                fprintf(fh, '- $\\lambda = \\sqrt{\\frac{2^{-2/\\nu}\\Gamma(1/\\nu)}{\\Gamma(3/\\nu)}}$\n');
                
                fprintf(fh, '\n### Hansen''s Skewed T Distribution\n\n');
                fprintf(fh, 'The PDF is given by:\n\n');
                fprintf(fh, '$$f(z|\\eta,\\lambda) = \\begin{cases}\n');
                fprintf(fh, 'bc(1 + \\frac{1}{\\eta-2}(\\frac{bz+a}{1-\\lambda})^2)^{-(\\eta+1)/2} & \\text{if } z < -a/b \\\\\n');
                fprintf(fh, 'bc(1 + \\frac{1}{\\eta-2}(\\frac{bz+a}{1+\\lambda})^2)^{-(\\eta+1)/2} & \\text{if } z \\geq -a/b\n');
                fprintf(fh, '\\end{cases}$$\n\n');
                fprintf(fh, 'Where:\n');
                fprintf(fh, '- $\\eta$ is the degrees of freedom parameter, $2 < \\eta < \\infty$\n');
                fprintf(fh, '- $\\lambda$ is the skewness parameter, $-1 < \\lambda < 1$\n');
                
                fprintf(fh, '\n## Time Series Models\n\n');
                fprintf(fh, '### ARMA Models\n\n');
                fprintf(fh, 'The ARMA(p,q) model is defined as:\n\n');
                fprintf(fh, '$$y_t = c + \\sum_{i=1}^{p} \\phi_i y_{t-i} + \\sum_{j=1}^{q} \\theta_j \\epsilon_{t-j} + \\epsilon_t$$\n\n');
                fprintf(fh, 'Where:\n');
                fprintf(fh, '- $y_t$ is the time series value at time t\n');
                fprintf(fh, '- $c$ is a constant term\n');
                fprintf(fh, '- $\\phi_i$ are the autoregressive coefficients\n');
                fprintf(fh, '- $\\theta_j$ are the moving average coefficients\n');
                fprintf(fh, '- $\\epsilon_t$ is white noise\n');
                
                fprintf(fh, '\n### GARCH Models\n\n');
                fprintf(fh, 'The GARCH(p,q) model is defined as:\n\n');
                fprintf(fh, '$$\\sigma_t^2 = \\omega + \\sum_{i=1}^{p} \\alpha_i \\epsilon_{t-i}^2 + \\sum_{j=1}^{q} \\beta_j \\sigma_{t-j}^2$$\n\n');
                fprintf(fh, 'Where:\n');
                fprintf(fh, '- $\\sigma_t^2$ is the conditional variance at time t\n');
                fprintf(fh, '- $\\omega$ is a constant term (must be positive)\n');
                fprintf(fh, '- $\\alpha_i$ are the ARCH coefficients\n');
                fprintf(fh, '- $\\beta_j$ are the GARCH coefficients\n');
                
            case 'references.md'
                fprintf(fh, '## Academic References\n\n');
                fprintf(fh, '### Time Series Analysis\n\n');
                fprintf(fh, '1. Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control. Holden-Day.\n\n');
                fprintf(fh, '2. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.\n\n');
                fprintf(fh, '### Volatility Models\n\n');
                fprintf(fh, '1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.\n\n');
                fprintf(fh, '2. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007.\n\n');
                fprintf(fh, '3. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347-370.\n\n');
                fprintf(fh, '### Statistical Distributions\n\n');
                fprintf(fh, '1. Hansen, B. E. (1994). Autoregressive conditional density estimation. International Economic Review, 35(3), 705-730.\n\n');
                fprintf(fh, '2. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347-370.\n\n');
                
            case 'changelog.md'
                fprintf(fh, '## Version History\n\n');
                fprintf(fh, '### Version 4.0 (October 28, 2009)\n\n');
                fprintf(fh, '- Initial version with comprehensive functionality\n');
                fprintf(fh, '- Includes Bootstrap, Cross-sectional Analysis, Distribution Functions\n');
                fprintf(fh, '- Full suite of Time Series and Volatility models\n');
                fprintf(fh, '- Implemented GUI for ARMAX modeling\n');
                fprintf(fh, '- MEX optimization for high-performance computation\n\n');
        end
        
        fclose(fh);
    end
end

end

function formattedContent = format_markdown(content, template)
% FORMAT_MARKDOWN Formats content as Markdown with proper syntax
%
% This function takes structured content and a template, replacing placeholders
% in the template with the appropriate content to create a well-formatted
% Markdown document.
%
% USAGE:
%   formattedContent = format_markdown(content, template)
%
% INPUTS:
%   content    - Structure containing content fields to insert
%   template   - Template string with placeholders
%
% OUTPUTS:
%   formattedContent - Formatted Markdown content

% Make a copy of the template
formattedContent = template;

% Get field names from content structure
fieldNames = fieldnames(content);

% Replace placeholders with content
for i = 1:length(fieldNames)
    fieldName = fieldNames{i};
    fieldValue = content.(fieldName);
    
    % Create placeholder pattern {{FIELD_NAME}}
    placeholder = ['{{', fieldName, '}}'];
    
    % Replace placeholder with content
    formattedContent = strrep(formattedContent, placeholder, fieldValue);
end

% Handle any remaining placeholders
remainingPlaceholders = regexp(formattedContent, '{{[A-Z_]+}}', 'match');
for i = 1:length(remainingPlaceholders)
    formattedContent = strrep(formattedContent, remainingPlaceholders{i}, '');
end

end

function capitalized = capitalize(str)
% CAPITALIZE Capitalizes the first letter of a string
%
% USAGE:
%   capitalized = capitalize(str)
%
% INPUTS:
%   str         - Input string
%
% OUTPUTS:
%   capitalized - String with first letter capitalized

if isempty(str)
    capitalized = str;
    return;
end

capitalized = [upper(str(1)), str(2:end)];
end