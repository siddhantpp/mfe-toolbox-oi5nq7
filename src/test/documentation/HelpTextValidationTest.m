classdef HelpTextValidationTest < BaseTest
    % HELPTEXTVALIDATIONTEST Validates help text documentation across MFE Toolbox
    %
    % This test class ensures that all MATLAB functions in the MFE Toolbox have
    % proper help text documentation that follows a standardized format and
    % contains all required elements, including parameter and return value
    % descriptions and usage examples.
    %
    % The validation checks for:
    %   1. Existence of help text
    %   2. Compliance with standard format
    %   3. Complete parameter documentation
    %   4. Complete return value documentation
    %   5. Presence of usage examples
    %   6. Validity of "See Also" references
    %
    % USAGE:
    %   test = HelpTextValidationTest();
    %   results = test.runAllTests();
    %
    % EXAMPLES:
    %   % Run all test methods
    %   test = HelpTextValidationTest();
    %   results = test.runAllTests();
    %
    %   % Run specific test method
    %   test = HelpTextValidationTest();
    %   test.testHelpTextFormat();
    %
    % See also: BaseTest, TestReporter
    
    properties
        toolboxDirectories  % Cell array of directories to scan
        validationResults   % Struct to store validation results
        excludeFiles        % Cell array of files to exclude from validation
        reporter            % TestReporter instance
        standardFormat      % Struct defining required help text format
    end
    
    methods
        function obj = HelpTextValidationTest()
            % Initialize a new instance of the HelpTextValidationTest class
            %
            % OUTPUTS:
            %   obj - Initialized HelpTextValidationTest instance
            
            % Call parent constructor
            obj = obj@BaseTest('Help Text Validation');
            
            % Initialize properties
            obj.toolboxDirectories = {};
            obj.validationResults = struct();
            obj.excludeFiles = {};
            obj.reporter = TestReporter('Help Text Validation Report');
            
            % Define standard format requirements
            obj.standardFormat = struct(...
                'requiredSections', {'Description', 'Syntax', 'Inputs', 'Outputs'}, ...
                'optionalSections', {'Examples', 'See Also', 'Comments', 'Notes'}, ...
                'parameterFormat', 'name - description', ...
                'returnFormat', 'name - description' ...
            );
        end
        
        function setUp(obj)
            % Prepares the test environment by configuring paths and initializing resources
            %
            % OUTPUTS:
            %   void - No return value
            
            % Call parent setUp method
            setUp@BaseTest(obj);
            
            % Initialize toolbox directories to scan
            obj.toolboxDirectories = {
                'bootstrap', 'crosssection', 'distributions', 'GUI', ...
                'multivariate', 'tests', 'timeseries', 'univariate', ...
                'utility', 'realized'
            };
            
            % Initialize excluded files (private functions, etc.)
            obj.excludeFiles = {
                'Contents.m', 'addToPath.m', 'buildZipFile.m'
            };
            
            % Initialize reporter
            obj.reporter = TestReporter('Help Text Validation Report');
            obj.reporter.setVerboseOutput(obj.verbose);
            
            % Initialize validation results
            obj.validationResults = struct(...
                'totalFiles', 0, ...
                'passedFiles', 0, ...
                'failedFiles', 0, ...
                'fileResults', struct() ...
            );
            
            % Define standard format expectations
            obj.standardFormat = struct(...
                'requiredSections', {'Description', 'Inputs', 'Outputs'}, ...
                'optionalSections', {'Examples', 'See Also', 'Comments', 'Notes'}, ...
                'parameterFormat', '^\s*(\w+)\s*-\s*(.+)$', ...  % Regex for parameter docs
                'returnFormat', '^\s*(\w+)\s*-\s*(.+)$' ...      % Regex for return docs
            );
        end
        
        function tearDown(obj)
            % Cleans up after tests by resetting the environment
            %
            % OUTPUTS:
            %   void - No return value
            
            % Generate summary report
            if obj.verbose
                fprintf('\n===== Help Text Validation Summary =====\n');
                fprintf('Total files checked: %d\n', obj.validationResults.totalFiles);
                fprintf('Passed: %d (%.1f%%)\n', obj.validationResults.passedFiles, ...
                    (obj.validationResults.passedFiles / max(1, obj.validationResults.totalFiles)) * 100);
                fprintf('Failed: %d (%.1f%%)\n', obj.validationResults.failedFiles, ...
                    (obj.validationResults.failedFiles / max(1, obj.validationResults.totalFiles)) * 100);
                fprintf('======================================\n\n');
            end
            
            % Record validation statistics to the reporter
            obj.reporter.addTestResult('HelpTextValidation', 'Documentation', ...
                obj.validationResults.failedFiles == 0, ...
                struct('message', sprintf('%d of %d files validated successfully', ...
                obj.validationResults.passedFiles, obj.validationResults.totalFiles)));
            
            % Clear temporary variables
            obj.validationResults = struct();
            
            % Call parent tearDown method
            tearDown@BaseTest(obj);
        end
        
        function testHelpTextExists(obj)
            % Tests that every function file has help text documentation
            %
            % OUTPUTS:
            %   void - No return value
            
            % Initialize counters
            filesChecked = 0;
            filesWithHelp = 0;
            filesWithoutHelp = 0;
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract help text using MATLAB's help function
                    helpText = help(filePath);
                    
                    % Check if help text exists and is not empty
                    hasHelpText = ~isempty(helpText) && ~strcmp(helpText, ' ');
                    
                    if hasHelpText
                        filesWithHelp = filesWithHelp + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'Help text exists'));
                    else
                        filesWithoutHelp = filesWithoutHelp + 1;
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', 'Missing help text'));
                        obj.reporter.logError(sprintf('Missing help text in file: %s', filePath));
                    end
                    
                    % Assert that help text exists
                    obj.assertTrue(hasHelpText, sprintf('Missing help text in file: %s', filePath));
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = filesWithHelp;
            obj.validationResults.failedFiles = filesWithoutHelp;
            
            % Final assertion
            obj.assertEqual(filesChecked, filesWithHelp, ...
                sprintf('%d of %d files are missing help text', filesWithoutHelp, filesChecked));
        end
        
        function testHelpTextFormat(obj)
            % Tests that help text follows the standard format for MFE Toolbox
            %
            % OUTPUTS:
            %   void - No return value
            
            % Initialize counters
            filesChecked = 0;
            validFormatFiles = 0;
            invalidFormatFiles = 0;
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract help text using MATLAB's help function
                    helpText = help(filePath);
                    
                    % Validate help text format
                    result = obj.validateHelpTextFormat(helpText, fileName);
                    
                    if result.isValid
                        validFormatFiles = validFormatFiles + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'Help text format is valid'));
                    else
                        invalidFormatFiles = invalidFormatFiles + 1;
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', result.message, 'issues', result.issues));
                        obj.reporter.logError(sprintf('Invalid help text format in file: %s - %s', filePath, result.message));
                    end
                    
                    % Assert that format is valid
                    obj.assertTrue(result.isValid, sprintf('Invalid help text format in file: %s - %s', filePath, result.message));
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = validFormatFiles;
            obj.validationResults.failedFiles = invalidFormatFiles;
            
            % Final assertion
            obj.assertEqual(filesChecked, validFormatFiles, ...
                sprintf('%d of %d files have invalid help text format', invalidFormatFiles, filesChecked));
        end
        
        function testParameterDocumentation(obj)
            % Tests that function parameters are properly documented in help text
            %
            % OUTPUTS:
            %   void - No return value
            
            % Initialize counters
            filesChecked = 0;
            validParamDocs = 0;
            invalidParamDocs = 0;
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract function signature
                    signature = obj.extractFunctionSignature(filePath);
                    
                    % Skip if not a function file or couldn't parse signature
                    if isempty(signature) || ~isfield(signature, 'inputs')
                        continue;
                    end
                    
                    % Extract help text and parameter documentation
                    helpText = help(filePath);
                    paramDocs = obj.extractParameterDocs(helpText);
                    
                    % Check if all parameters are documented
                    allParamsDocumented = true;
                    missingParams = {};
                    
                    % Skip validation for functions with no input parameters
                    if isempty(signature.inputs)
                        validParamDocs = validParamDocs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'No parameters to document'));
                        continue;
                    end
                    
                    % Check each parameter
                    for k = 1:length(signature.inputs)
                        paramName = signature.inputs{k};
                        if ~isfield(paramDocs, paramName)
                            allParamsDocumented = false;
                            missingParams{end+1} = paramName;
                        end
                    end
                    
                    if allParamsDocumented
                        validParamDocs = validParamDocs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'All parameters documented'));
                    else
                        invalidParamDocs = invalidParamDocs + 1;
                        message = sprintf('Missing documentation for parameters: %s', strjoin(missingParams, ', '));
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', message, 'missingParams', {missingParams}));
                        obj.reporter.logError(sprintf('Parameter documentation issues in file: %s - %s', filePath, message));
                    end
                    
                    % Assert that all parameters are documented
                    obj.assertTrue(allParamsDocumented, sprintf('Missing parameter documentation in file: %s', filePath));
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = validParamDocs;
            obj.validationResults.failedFiles = invalidParamDocs;
            
            % Final assertion
            obj.assertEqual(filesChecked, validParamDocs, ...
                sprintf('%d of %d files have incomplete parameter documentation', invalidParamDocs, filesChecked));
        end
        
        function testReturnValueDocumentation(obj)
            % Tests that function return values are properly documented in help text
            %
            % OUTPUTS:
            %   void - No return value
            
            % Initialize counters
            filesChecked = 0;
            validReturnDocs = 0;
            invalidReturnDocs = 0;
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract function signature
                    signature = obj.extractFunctionSignature(filePath);
                    
                    % Skip if not a function file or couldn't parse signature
                    if isempty(signature) || ~isfield(signature, 'outputs')
                        continue;
                    end
                    
                    % Extract help text and return value documentation
                    helpText = help(filePath);
                    returnDocs = obj.extractReturnDocs(helpText);
                    
                    % Skip validation for functions with no output parameters
                    if isempty(signature.outputs)
                        validReturnDocs = validReturnDocs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'No return values to document'));
                        continue;
                    end
                    
                    % Check if all return values are documented
                    allReturnsDocumented = true;
                    missingReturns = {};
                    
                    % Check each return value
                    for k = 1:length(signature.outputs)
                        returnName = signature.outputs{k};
                        if ~isfield(returnDocs, returnName)
                            allReturnsDocumented = false;
                            missingReturns{end+1} = returnName;
                        end
                    end
                    
                    if allReturnsDocumented
                        validReturnDocs = validReturnDocs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'All return values documented'));
                    else
                        invalidReturnDocs = invalidReturnDocs + 1;
                        message = sprintf('Missing documentation for return values: %s', strjoin(missingReturns, ', '));
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', message, 'missingReturns', {missingReturns}));
                        obj.reporter.logError(sprintf('Return value documentation issues in file: %s - %s', filePath, message));
                    end
                    
                    % Assert that all return values are documented
                    obj.assertTrue(allReturnsDocumented, sprintf('Missing return value documentation in file: %s', filePath));
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = validReturnDocs;
            obj.validationResults.failedFiles = invalidReturnDocs;
            
            % Final assertion
            obj.assertEqual(filesChecked, validReturnDocs, ...
                sprintf('%d of %d files have incomplete return value documentation', invalidReturnDocs, filesChecked));
        end
        
        function testExamplesExist(obj)
            % Tests that help text includes usage examples for relevant functions
            %
            % OUTPUTS:
            %   void - No return value
            
            % Initialize counters
            filesChecked = 0;
            filesWithExamples = 0;
            filesWithoutExamples = 0;
            
            % Scan all toolbox directories (excluding utility functions)
            relevantDirs = setdiff(obj.toolboxDirectories, {'utility'});
            
            for i = 1:length(relevantDirs)
                dirPath = relevantDirs{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract help text and check for Examples section
                    helpText = help(filePath);
                    sections = obj.extractHelpSections(helpText);
                    
                    if isfield(sections, 'Examples') && ~isempty(sections.Examples)
                        filesWithExamples = filesWithExamples + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'Examples section exists'));
                    else
                        filesWithoutExamples = filesWithoutExamples + 1;
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', 'Missing Examples section'));
                        obj.reporter.logMessage(sprintf('Missing examples in file: %s', filePath), 'warning');
                    end
                    
                    % Assert that examples exist (warning only)
                    % Note: Not making this a hard failure as some functions may not need examples
                    try
                        obj.assertTrue(isfield(sections, 'Examples') && ~isempty(sections.Examples), ...
                            sprintf('Missing examples in file: %s', filePath));
                    catch
                        % Convert to warning instead of failure
                        warning('Missing examples in file: %s', filePath);
                    end
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = filesWithExamples;
            obj.validationResults.warningFiles = filesWithoutExamples;
            
            % Report statistics but don't make this a hard failure
            if obj.verbose
                fprintf('Examples validation: %d of %d files have usage examples (%.1f%%)\n', ...
                    filesWithExamples, filesChecked, (filesWithExamples / filesChecked) * 100);
            end
        end
        
        function testSeeAlsoReferences(obj)
            % Tests that 'See Also' references in help text point to existing functions
            %
            % OUTPUTS:
            %   void - No return value
            
            % Get list of all available function names in the toolbox
            allFunctions = obj.getToolboxFunctionList();
            
            % Initialize counters
            filesChecked = 0;
            validRefs = 0;
            invalidRefs = 0;
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    filePath = fullfile(dirPath, fileName);
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    filesChecked = filesChecked + 1;
                    
                    % Extract help text and See Also references
                    helpText = help(filePath);
                    references = obj.extractSeeAlsoReferences(helpText);
                    
                    % Skip if no references
                    if isempty(references)
                        validRefs = validRefs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'No See Also references to validate'));
                        continue;
                    end
                    
                    % Check if all references are valid
                    allRefsValid = true;
                    invalidRefList = {};
                    
                    for k = 1:length(references)
                        refName = references{k};
                        % Check if reference exists in the toolbox
                        if ~any(strcmp(refName, allFunctions))
                            allRefsValid = false;
                            invalidRefList{end+1} = refName;
                        end
                    end
                    
                    if allRefsValid
                        validRefs = validRefs + 1;
                        obj.logValidationResult(filePath, struct('isValid', true, 'message', 'All See Also references are valid'));
                    else
                        invalidRefs = invalidRefs + 1;
                        message = sprintf('Invalid See Also references: %s', strjoin(invalidRefList, ', '));
                        obj.logValidationResult(filePath, struct('isValid', false, 'message', message, 'invalidRefs', {invalidRefList}));
                        obj.reporter.logError(sprintf('Invalid See Also references in file: %s - %s', filePath, message));
                    end
                    
                    % Assert that all references are valid
                    obj.assertTrue(allRefsValid, sprintf('Invalid See Also references in file: %s', filePath));
                end
            end
            
            % Update validation results
            obj.validationResults.totalFiles = filesChecked;
            obj.validationResults.passedFiles = validRefs;
            obj.validationResults.failedFiles = invalidRefs;
            
            % Final assertion
            obj.assertEqual(filesChecked, validRefs, ...
                sprintf('%d of %d files have invalid See Also references', invalidRefs, filesChecked));
        end
        
        function result = validateHelpTextFormat(obj, helpText, fileName)
            % Validates that help text follows the standard format
            %
            % INPUTS:
            %   helpText - Help text string to validate
            %   fileName - Name of the file being validated
            %
            % OUTPUTS:
            %   result - Validation results with isValid flag and list of issues
            
            % Initialize result
            result = struct('isValid', true, 'message', 'Help text format is valid', 'issues', {{}});
            
            % Check if help text exists
            if isempty(helpText)
                result.isValid = false;
                result.message = 'Help text is empty';
                result.issues{end+1} = 'Empty help text';
                return;
            end
            
            % Check if help text starts with function name
            fileNameNoExt = regexprep(fileName, '\.m$', '');
            if ~contains(upper(helpText(1:min(length(helpText), 100))), upper(fileNameNoExt))
                result.isValid = false;
                result.issues{end+1} = 'Help text does not start with function name';
            end
            
            % Parse help text into sections
            sections = obj.extractHelpSections(helpText);
            
            % Check for required sections
            for i = 1:length(obj.standardFormat.requiredSections)
                requiredSection = obj.standardFormat.requiredSections{i};
                if ~isfield(sections, requiredSection) || isempty(sections.(requiredSection))
                    result.isValid = false;
                    result.issues{end+1} = sprintf('Missing required section: %s', requiredSection);
                end
            end
            
            % If any issues found, update message
            if ~result.isValid
                result.message = sprintf('Help text format issues: %s', strjoin(result.issues, ', '));
            end
        end
        
        function signature = extractFunctionSignature(obj, filePath)
            % Extracts function signature from an M-file
            %
            % INPUTS:
            %   filePath - Path to the M-file
            %
            % OUTPUTS:
            %   signature - Function signature details including inputs and outputs
            
            % Initialize empty signature
            signature = struct('name', '', 'inputs', {{}}, 'outputs', {{}});
            
            % Read the first few lines of the file to find function declaration
            try
                fid = fopen(filePath, 'r');
                if fid == -1
                    return;
                end
                
                % Read first 20 lines (should be enough to find function declaration)
                lines = cell(20, 1);
                for i = 1:20
                    line = fgetl(fid);
                    if ~ischar(line)
                        break;
                    end
                    lines{i} = line;
                end
                fclose(fid);
                
                % Find function declaration line
                funcLine = '';
                for i = 1:length(lines)
                    if ~isempty(lines{i}) && contains(lines{i}, 'function')
                        funcLine = lines{i};
                        break;
                    end
                end
                
                % Return empty if no function declaration found
                if isempty(funcLine)
                    return;
                end
                
                % Parse function signature using regex
                % Matches patterns like: function [out1, out2] = funcName(in1, in2, in3)
                pattern = 'function\s+(?:\[([^\]]*)\]\s*=\s*|\s*([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*)?([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?';
                matches = regexp(funcLine, pattern, 'tokens', 'once');
                
                if isempty(matches)
                    return;
                end
                
                % Extract outputs
                outputs = matches{1};
                if isempty(outputs) && ~isempty(matches{2})
                    outputs = matches{2}; % Single output without brackets
                end
                
                % Extract function name and inputs
                functionName = matches{3};
                inputs = matches{4};
                
                % Parse output parameters
                if ~isempty(outputs)
                    outputParams = strtrim(strsplit(outputs, ','));
                    signature.outputs = outputParams;
                end
                
                % Parse input parameters
                if ~isempty(inputs)
                    inputParams = strtrim(strsplit(inputs, ','));
                    signature.inputs = inputParams;
                end
                
                % Set function name
                signature.name = functionName;
                
            catch e
                % Handle errors during parsing
                warning('Error parsing function signature for %s: %s', filePath, e.message);
                signature = struct('name', '', 'inputs', {{}}, 'outputs', {{}});
            end
        end
        
        function sections = extractHelpSections(obj, helpText)
            % Extracts individual sections from help text
            %
            % INPUTS:
            %   helpText - Help text string
            %
            % OUTPUTS:
            %   sections - Help text sections (Description, Syntax, Inputs, Outputs, Examples, See Also)
            
            % Initialize sections struct
            sections = struct();
            
            % Split help text into lines
            lines = strsplit(helpText, '\n');
            
            % Define known section headers
            knownSections = [obj.standardFormat.requiredSections, obj.standardFormat.optionalSections];
            currentSection = 'Description'; % Default section
            sections.Description = '';
            
            % Process each line
            for i = 1:length(lines)
                line = lines{i};
                
                % Check if line is a section header
                isSectionHeader = false;
                newSection = '';
                
                % Try to match section headers with various formats
                % 1. SECTION: or SECTION -
                for j = 1:length(knownSections)
                    pattern = sprintf('\\b%s[\\s]*[:|-]', upper(knownSections{j}));
                    if ~isempty(regexp(line, pattern, 'once'))
                        newSection = knownSections{j};
                        isSectionHeader = true;
                        break;
                    end
                end
                
                % 2. SECTION
                if ~isSectionHeader
                    for j = 1:length(knownSections)
                        pattern = sprintf('^\\s*%s\\s*$', upper(knownSections{j}));
                        if ~isempty(regexp(line, pattern, 'once'))
                            newSection = knownSections{j};
                            isSectionHeader = true;
                            break;
                        end
                    end
                end
                
                % Start new section if section header found
                if isSectionHeader
                    currentSection = newSection;
                    if ~isfield(sections, currentSection)
                        sections.(currentSection) = '';
                    end
                    continue;
                end
                
                % Append line to current section
                if isfield(sections, currentSection)
                    if ~isempty(sections.(currentSection))
                        sections.(currentSection) = [sections.(currentSection), '\n', line];
                    else
                        sections.(currentSection) = line;
                    end
                else
                    sections.(currentSection) = line;
                end
            end
            
            % Trim whitespace from all sections
            sectionNames = fieldnames(sections);
            for i = 1:length(sectionNames)
                sections.(sectionNames{i}) = strtrim(sections.(sectionNames{i}));
            end
        end
        
        function paramDocs = extractParameterDocs(obj, helpText)
            % Extracts parameter documentation from help text
            %
            % INPUTS:
            %   helpText - Help text string
            %
            % OUTPUTS:
            %   paramDocs - Parameter documentation with names and descriptions
            
            % Initialize empty struct
            paramDocs = struct();
            
            % Extract sections from help text
            sections = obj.extractHelpSections(helpText);
            
            % Return empty if no Inputs section
            if ~isfield(sections, 'Inputs') || isempty(sections.Inputs)
                return;
            end
            
            % Split Inputs section into lines
            inputLines = strsplit(sections.Inputs, '\n');
            
            % Parse parameter documentation using regex
            for i = 1:length(inputLines)
                line = strtrim(inputLines{i});
                if isempty(line)
                    continue;
                end
                
                % Try different parameter documentation patterns
                % Pattern 1: paramName - description
                matches = regexp(line, obj.standardFormat.parameterFormat, 'tokens', 'once');
                if ~isempty(matches)
                    paramName = matches{1};
                    paramDesc = matches{2};
                    paramDocs.(paramName) = paramDesc;
                    continue;
                end
                
                % Pattern 2: paramName: description
                matches = regexp(line, '^\s*(\w+)\s*:\s*(.+)$', 'tokens', 'once');
                if ~isempty(matches)
                    paramName = matches{1};
                    paramDesc = matches{2};
                    paramDocs.(paramName) = paramDesc;
                    continue;
                end
                
                % Pattern 3: paramName = description
                matches = regexp(line, '^\s*(\w+)\s*=\s*(.+)$', 'tokens', 'once');
                if ~isempty(matches)
                    paramName = matches{1};
                    paramDesc = matches{2};
                    paramDocs.(paramName) = paramDesc;
                end
            end
        end
        
        function returnDocs = extractReturnDocs(obj, helpText)
            % Extracts return value documentation from help text
            %
            % INPUTS:
            %   helpText - Help text string
            %
            % OUTPUTS:
            %   returnDocs - Return value documentation with names and descriptions
            
            % Initialize empty struct
            returnDocs = struct();
            
            % Extract sections from help text
            sections = obj.extractHelpSections(helpText);
            
            % Return empty if no Outputs section
            if ~isfield(sections, 'Outputs') || isempty(sections.Outputs)
                return;
            end
            
            % Split Outputs section into lines
            outputLines = strsplit(sections.Outputs, '\n');
            
            % Parse return value documentation using regex
            for i = 1:length(outputLines)
                line = strtrim(outputLines{i});
                if isempty(line)
                    continue;
                end
                
                % Try different return documentation patterns
                % Pattern 1: returnName - description
                matches = regexp(line, obj.standardFormat.returnFormat, 'tokens', 'once');
                if ~isempty(matches)
                    returnName = matches{1};
                    returnDesc = matches{2};
                    returnDocs.(returnName) = returnDesc;
                    continue;
                end
                
                % Pattern 2: returnName: description
                matches = regexp(line, '^\s*(\w+)\s*:\s*(.+)$', 'tokens', 'once');
                if ~isempty(matches)
                    returnName = matches{1};
                    returnDesc = matches{2};
                    returnDocs.(returnName) = returnDesc;
                    continue;
                end
                
                % Pattern 3: returnName = description
                matches = regexp(line, '^\s*(\w+)\s*=\s*(.+)$', 'tokens', 'once');
                if ~isempty(matches)
                    returnName = matches{1};
                    returnDesc = matches{2};
                    returnDocs.(returnName) = returnDesc;
                end
            end
        end
        
        function references = extractSeeAlsoReferences(obj, helpText)
            % Extracts 'See Also' references from help text
            %
            % INPUTS:
            %   helpText - Help text string
            %
            % OUTPUTS:
            %   references - List of referenced function names
            
            % Initialize empty cell array
            references = {};
            
            % Extract sections from help text
            sections = obj.extractHelpSections(helpText);
            
            % Return empty if no See Also section
            if ~isfield(sections, 'See Also') || isempty(sections.('See Also'))
                return;
            end
            
            % Parse See Also section
            seeAlsoText = sections.('See Also');
            
            % Remove common formatting elements
            seeAlsoText = regexprep(seeAlsoText, '[,;:]', ' '); % Replace separators with spaces
            
            % Split by spaces and filter out empty strings
            words = strsplit(seeAlsoText);
            words = words(~cellfun(@isempty, words));
            
            % Further clean up each reference
            for i = 1:length(words)
                word = strtrim(words{i});
                if ~isempty(word)
                    % Remove any non-alphanumeric characters except underscore
                    word = regexprep(word, '[^a-zA-Z0-9_]', '');
                    if ~isempty(word)
                        references{end+1} = word;
                    end
                end
            end
        end
        
        function functionList = getToolboxFunctionList(obj)
            % Builds a list of all function names in the toolbox
            %
            % OUTPUTS:
            %   functionList - List of all function names in the toolbox
            
            % Initialize empty cell array
            functionList = {};
            
            % Scan all toolbox directories
            for i = 1:length(obj.toolboxDirectories)
                dirPath = obj.toolboxDirectories{i};
                
                % Get all .m files in directory
                files = dir(fullfile(dirPath, '*.m'));
                
                for j = 1:length(files)
                    fileName = files(j).name;
                    
                    % Skip excluded files
                    if any(strcmp(fileName, obj.excludeFiles))
                        continue;
                    end
                    
                    % Extract function name (remove .m extension)
                    functionName = regexprep(fileName, '\.m$', '');
                    functionList{end+1} = functionName;
                end
            end
        end
        
        function logValidationResult(obj, fileName, result)
            % Logs the results of help text validation for a file
            %
            % INPUTS:
            %   fileName - Name of the file being validated
            %   result - Validation results with isValid flag and list of issues
            %
            % OUTPUTS:
            %   void - No return value
            
            % Update file results
            fileId = sprintf('file%d', length(fieldnames(obj.validationResults.fileResults)) + 1);
            obj.validationResults.fileResults.(fileId) = struct(...
                'fileName', fileName, ...
                'isValid', result.isValid, ...
                'message', result.message ...
            );
            
            % If there are additional fields in result, add them to the file result
            extraFields = setdiff(fieldnames(result), {'isValid', 'message'});
            for i = 1:length(extraFields)
                field = extraFields{i};
                obj.validationResults.fileResults.(fileId).(field) = result.(field);
            end
            
            % Log message based on validation result
            if result.isValid
                if obj.verbose
                    obj.reporter.logMessage(sprintf('[PASS] %s: %s', fileName, result.message));
                end
            else
                obj.reporter.logError(sprintf('[FAIL] %s: %s', fileName, result.message));
            end
        end
    end
end