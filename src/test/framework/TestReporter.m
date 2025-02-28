classdef TestReporter
    % TESTREPORTER A class responsible for collecting, formatting, and generating
    % comprehensive reports of test execution results in the MFE Toolbox testing framework.
    %
    % This class supports multiple output formats, detailed failure analysis, and
    % statistical summaries of test execution.
    %
    % Properties:
    %   testResults          - Struct storing individual test results
    %   reportFormats        - Cell array of output formats ('text', 'html', 'xml')
    %   reportTitle          - String title used in generated reports
    %   reportOutputPath     - String path where reports will be saved
    %   verboseOutput        - Logical flag for immediate console output of results
    %   includePerformanceData - Logical flag for including performance metrics
    %   statistics           - Struct with aggregated test metrics
    %   lastGeneratedReports - Struct tracking paths to most recent reports
    %
    % Methods:
    %   TestReporter             - Constructor to initialize a new TestReporter instance
    %   addTestResult            - Records the result of a single test execution
    %   addTestResults           - Records multiple test results from a test suite or runner
    %   generateReport           - Generates reports in all configured formats
    %   generateTextReport       - Generates a plain text report
    %   generateHTMLReport       - Generates an HTML report with styling
    %   getTestStatistics        - Returns statistical information about test execution
    %   displaySummary           - Displays a summary of test results to the console
    %   getFailedTests           - Returns a list of all failed tests with details
    %   setVerboseOutput         - Sets the verbose output flag
    %   setReportFormats         - Sets the output formats for test reports
    %   setReportOutputPath      - Sets the output directory for generated reports
    %   setReportTitle           - Sets the title used in generated reports
    %   setIncludePerformanceData - Sets whether to include performance metrics
    %   getLastGeneratedReports  - Returns paths to recently generated reports
    %   clearResults             - Clears all recorded test results and statistics
    %   formatTimestamp          - Formats a timestamp for report display
    %   formatDuration           - Formats a duration for human-readable display
    
    properties
        testResults          % Struct storing test outcomes
        reportFormats        % Cell array of output formats
        reportTitle          % String title for reports
        reportOutputPath     % String path for output files
        verboseOutput        % Logical flag for immediate output
        includePerformanceData % Logical flag for performance metrics
        statistics           % Struct with test metrics
        lastGeneratedReports % Struct with recent report paths
    end
    
    methods
        function obj = TestReporter(reportTitle)
            % TESTREPORTER Initializes a new TestReporter instance with default settings
            %
            % USAGE:
            %   reporter = TestReporter()
            %   reporter = TestReporter(reportTitle)
            %
            % INPUT:
            %   reportTitle - (optional) String title for reports
            %
            % OUTPUT:
            %   obj - TestReporter instance
            
            % Initialize empty test results structure
            obj.testResults = struct();
            
            % Set default report formats
            obj.reportFormats = {'text'};
            
            % Set report title
            if nargin > 0 && ~isempty(reportTitle)
                if ~ischar(reportTitle)
                    error('reportTitle must be a string');
                end
                obj.reportTitle = reportTitle;
            else
                obj.reportTitle = 'MFE Toolbox Test Report';
            end
            
            % Set default output path to current directory
            obj.reportOutputPath = pwd;
            
            % Default settings
            obj.verboseOutput = false;
            obj.includePerformanceData = true;
            
            % Initialize statistics and report tracking
            obj.statistics = struct();
            obj.lastGeneratedReports = struct();
        end
        
        function addTestResult(obj, testName, testCategory, passed, details)
            % ADDTESTRESULT Records the result of a single test execution
            %
            % USAGE:
            %   obj.addTestResult(testName, testCategory, passed, details)
            %
            % INPUTS:
            %   testName     - String name of the test
            %   testCategory - String category of the test
            %   passed       - Logical flag indicating test success/failure
            %   details      - (optional) Structure with detailed test information
            %
            % OUTPUTS:
            %   void - No return value
            
            % Validate input parameters
            if ~ischar(testName)
                error('testName must be a string');
            end
            
            if ~ischar(testCategory)
                error('testCategory must be a string');
            end
            
            if ~islogical(passed) || numel(passed) ~= 1
                error('passed must be a logical scalar value');
            end
            
            % Create a new result entry
            resultIndex = length(fieldnames(obj.testResults)) + 1;
            resultId = sprintf('test%d', resultIndex);
            
            % Store basic test information
            obj.testResults.(resultId).name = testName;
            obj.testResults.(resultId).category = testCategory;
            obj.testResults.(resultId).passed = passed;
            obj.testResults.(resultId).timestamp = now;
            
            % Store detailed information if provided
            if nargin > 4 && ~isempty(details)
                obj.testResults.(resultId).details = details;
            end
            
            % Update statistics
            if ~isfield(obj.statistics, testCategory)
                obj.statistics.(testCategory).total = 0;
                obj.statistics.(testCategory).passed = 0;
                obj.statistics.(testCategory).failed = 0;
            end
            
            obj.statistics.(testCategory).total = obj.statistics.(testCategory).total + 1;
            if passed
                obj.statistics.(testCategory).passed = obj.statistics.(testCategory).passed + 1;
            else
                obj.statistics.(testCategory).failed = obj.statistics.(testCategory).failed + 1;
            end
            
            % Display immediate output if verbose mode is enabled
            if obj.verboseOutput
                if passed
                    fprintf('[PASS] %s: %s\n', testCategory, testName);
                else
                    fprintf('[FAIL] %s: %s\n', testCategory, testName);
                    if nargin > 4 && isfield(details, 'message')
                        fprintf('       Error: %s\n', details.message);
                    end
                end
            end
        end
        
        function addTestResults(obj, results, category)
            % ADDTESTRESULTS Records multiple test results from a test suite or runner
            %
            % USAGE:
            %   obj.addTestResults(results, category)
            %
            % INPUTS:
            %   results  - Structure with test results or cell array of result structures
            %   category - (optional) String category for all tests
            %
            % OUTPUTS:
            %   void - No return value
            
            % Validate inputs
            if ~isstruct(results) && ~iscell(results)
                error('Results must be a structure or cell array of structures');
            end
            
            if nargin < 3
                category = 'General';
            elseif ~ischar(category)
                error('category must be a string');
            end
            
            % Process results based on format
            if isstruct(results)
                % Extract test results from structure
                if isfield(results, 'tests')
                    % Handle test runner results format
                    tests = results.tests;
                    for i = 1:length(tests)
                        testName = tests{i}.name;
                        passed = tests{i}.passed;
                        details = tests{i};
                        obj.addTestResult(testName, category, passed, details);
                    end
                elseif isfield(results, 'results')
                    % Handle test suite results format
                    for i = 1:length(results.results)
                        testName = results.results{i}.name;
                        passed = results.results{i}.passed;
                        details = results.results{i};
                        obj.addTestResult(testName, category, passed, details);
                    end
                else
                    % Handle individual test result
                    if isfield(results, 'name') && isfield(results, 'passed')
                        testName = results.name;
                        passed = results.passed;
                        obj.addTestResult(testName, category, passed, results);
                    else
                        error('Invalid test result structure format');
                    end
                end
            elseif iscell(results)
                % Process each result in the cell array
                for i = 1:length(results)
                    if isstruct(results{i}) && isfield(results{i}, 'name') && isfield(results{i}, 'passed')
                        testName = results{i}.name;
                        passed = results{i}.passed;
                        obj.addTestResult(testName, category, passed, results{i});
                    else
                        error('Invalid test result structure at index %d', i);
                    end
                end
            end
        end
        
        function reports = generateReport(obj)
            % GENERATEREPORT Generates test reports in all configured formats
            %
            % USAGE:
            %   reports = obj.generateReport()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   reports - Structure with report generation status and file paths
            
            % Ensure we have test results to report
            if isempty(fieldnames(obj.testResults))
                error('No test results available. Run tests before generating reports.');
            end
            
            % Calculate final statistics
            stats = obj.getTestStatistics();
            
            % Initialize result structure
            reports = struct();
            obj.lastGeneratedReports = struct();
            
            % Generate reports in all configured formats
            for i = 1:length(obj.reportFormats)
                format = obj.reportFormats{i};
                
                % Generate appropriate filename based on format
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                filename = sprintf('mfe_test_report_%s.%s', timestamp, format);
                outputPath = fullfile(obj.reportOutputPath, filename);
                
                % Generate report in specified format
                switch lower(format)
                    case 'text'
                        reportPath = obj.generateTextReport(outputPath);
                        reports.text = reportPath;
                        obj.lastGeneratedReports.text = reportPath;
                    case 'html'
                        reportPath = obj.generateHTMLReport(outputPath);
                        reports.html = reportPath;
                        obj.lastGeneratedReports.html = reportPath;
                    case 'xml'
                        % XML format could be implemented in the future
                        warning('XML report format is not yet implemented');
                        reports.xml = '';
                        obj.lastGeneratedReports.xml = '';
                    otherwise
                        warning('Unsupported report format: %s', format);
                end
            end
        end
        
        function reportPath = generateTextReport(obj, outputPath)
            % GENERATETEXTREPORT Generates a plain text report of test results
            %
            % USAGE:
            %   reportPath = obj.generateTextReport(outputPath)
            %
            % INPUTS:
            %   outputPath - String path where the report should be saved
            %
            % OUTPUTS:
            %   reportPath - String path to the generated report file
            
            % Validate input
            if ~ischar(outputPath)
                error('outputPath must be a string');
            end
            
            % Open file for writing
            fileId = fopen(outputPath, 'w');
            if fileId == -1
                error('Failed to open file for writing: %s', outputPath);
            end
            
            % Get statistics
            stats = obj.getTestStatistics();
            
            % Write report header
            fprintf(fileId, '==========================================================\n');
            fprintf(fileId, '%s\n', obj.reportTitle);
            fprintf(fileId, 'Generated: %s\n', obj.formatTimestamp(now));
            fprintf(fileId, '==========================================================\n\n');
            
            % Write summary section
            fprintf(fileId, 'SUMMARY\n');
            fprintf(fileId, '----------\n');
            fprintf(fileId, 'Total Tests: %d\n', stats.totalTests);
            fprintf(fileId, 'Passed: %d (%.1f%%)\n', stats.totalPassed, stats.passRate);
            fprintf(fileId, 'Failed: %d (%.1f%%)\n', stats.totalFailed, 100 - stats.passRate);
            fprintf(fileId, '\n');
            
            % Write category breakdown if multiple categories exist
            categories = fieldnames(obj.statistics);
            if length(categories) > 1
                fprintf(fileId, 'CATEGORY BREAKDOWN\n');
                fprintf(fileId, '----------\n');
                for i = 1:length(categories)
                    cat = categories{i};
                    catStats = obj.statistics.(cat);
                    catPassRate = (catStats.passed / catStats.total) * 100;
                    fprintf(fileId, '%s: %d tests, %d passed (%.1f%%), %d failed\n', ...
                        cat, catStats.total, catStats.passed, catPassRate, catStats.failed);
                end
                fprintf(fileId, '\n');
            end
            
            % Write detailed results section
            fprintf(fileId, 'DETAILED RESULTS\n');
            fprintf(fileId, '----------\n');
            
            % Group results by category
            resultFields = fieldnames(obj.testResults);
            for i = 1:length(categories)
                cat = categories{i};
                fprintf(fileId, 'Category: %s\n', cat);
                
                % Find all tests in this category
                categoryTests = 0;
                for j = 1:length(resultFields)
                    result = obj.testResults.(resultFields{j});
                    if strcmp(result.category, cat)
                        categoryTests = categoryTests + 1;
                        status = 'PASS';
                        if ~result.passed
                            status = 'FAIL';
                        end
                        fprintf(fileId, '[%s] %s\n', status, result.name);
                    end
                end
                
                if categoryTests == 0
                    fprintf(fileId, '(No tests in this category)\n');
                end
                fprintf(fileId, '\n');
            end
            
            % Write failure details section if there are failures
            if stats.totalFailed > 0
                fprintf(fileId, 'FAILURE DETAILS\n');
                fprintf(fileId, '----------\n');
                
                for j = 1:length(resultFields)
                    result = obj.testResults.(resultFields{j});
                    if ~result.passed
                        fprintf(fileId, 'Test: %s (Category: %s)\n', result.name, result.category);
                        if isfield(result, 'details')
                            if isfield(result.details, 'message')
                                fprintf(fileId, 'Error: %s\n', result.details.message);
                            end
                            if isfield(result.details, 'stack')
                                fprintf(fileId, 'Stack Trace:\n');
                                for k = 1:length(result.details.stack)
                                    frame = result.details.stack(k);
                                    fprintf(fileId, '  File: %s, Line: %d, Function: %s\n', ...
                                        frame.file, frame.line, frame.name);
                                end
                            end
                            if isfield(result.details, 'expected') && isfield(result.details, 'actual')
                                fprintf(fileId, 'Expected: %s\n', mat2str(result.details.expected));
                                fprintf(fileId, 'Actual: %s\n', mat2str(result.details.actual));
                            end
                        end
                        fprintf(fileId, '\n');
                    end
                end
            end
            
            % Write performance metrics section if enabled
            if obj.includePerformanceData && isfield(stats, 'performance')
                fprintf(fileId, 'PERFORMANCE METRICS\n');
                fprintf(fileId, '----------\n');
                fprintf(fileId, 'Total Execution Time: %s\n', obj.formatDuration(stats.performance.totalTime));
                fprintf(fileId, 'Average Test Time: %s\n', obj.formatDuration(stats.performance.averageTime));
                fprintf(fileId, 'Slowest Test: %s (%s)\n', stats.performance.slowestTest, ...
                    obj.formatDuration(stats.performance.slowestTime));
                fprintf(fileId, 'Fastest Test: %s (%s)\n', stats.performance.fastestTest, ...
                    obj.formatDuration(stats.performance.fastestTime));
                fprintf(fileId, '\n');
            end
            
            % Close the file
            fclose(fileId);
            
            % Return the path to the generated file
            reportPath = outputPath;
        end
        
        function reportPath = generateHTMLReport(obj, outputPath)
            % GENERATEHTMLREPORT Generates an HTML format report of test results with styling
            %
            % USAGE:
            %   reportPath = obj.generateHTMLReport(outputPath)
            %
            % INPUTS:
            %   outputPath - String path where the report should be saved
            %
            % OUTPUTS:
            %   reportPath - String path to the generated report file
            
            % Validate input
            if ~ischar(outputPath)
                error('outputPath must be a string');
            end
            
            % Open file for writing
            fileId = fopen(outputPath, 'w');
            if fileId == -1
                error('Failed to open file for writing: %s', outputPath);
            end
            
            % Get statistics
            stats = obj.getTestStatistics();
            
            % Write HTML header with styling
            fprintf(fileId, '<!DOCTYPE html>\n');
            fprintf(fileId, '<html lang="en">\n');
            fprintf(fileId, '<head>\n');
            fprintf(fileId, '  <meta charset="UTF-8">\n');
            fprintf(fileId, '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n');
            fprintf(fileId, '  <title>%s</title>\n', obj.reportTitle);
            fprintf(fileId, '  <style>\n');
            fprintf(fileId, '    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n');
            fprintf(fileId, '    h1 { color: #333366; }\n');
            fprintf(fileId, '    h2 { color: #333366; margin-top: 30px; border-bottom: 1px solid #cccccc; }\n');
            fprintf(fileId, '    .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }\n');
            fprintf(fileId, '    .pass { color: green; }\n');
            fprintf(fileId, '    .fail { color: red; }\n');
            fprintf(fileId, '    table { border-collapse: collapse; width: 100%%; }\n');
            fprintf(fileId, '    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }\n');
            fprintf(fileId, '    th { background-color: #f2f2f2; }\n');
            fprintf(fileId, '    tr:hover { background-color: #f5f5f5; }\n');
            fprintf(fileId, '    .details-container { margin-left: 20px; border-left: 3px solid #ddd; padding-left: 10px; }\n');
            fprintf(fileId, '    .toggle-button { cursor: pointer; color: blue; text-decoration: underline; }\n');
            fprintf(fileId, '    .hidden { display: none; }\n');
            fprintf(fileId, '  </style>\n');
            fprintf(fileId, '  <script>\n');
            fprintf(fileId, '    function toggleDetails(id) {\n');
            fprintf(fileId, '      var details = document.getElementById(id);\n');
            fprintf(fileId, '      if (details.classList.contains("hidden")) {\n');
            fprintf(fileId, '        details.classList.remove("hidden");\n');
            fprintf(fileId, '      } else {\n');
            fprintf(fileId, '        details.classList.add("hidden");\n');
            fprintf(fileId, '      }\n');
            fprintf(fileId, '    }\n');
            fprintf(fileId, '  </script>\n');
            fprintf(fileId, '</head>\n');
            fprintf(fileId, '<body>\n');
            
            % Report header
            fprintf(fileId, '  <h1>%s</h1>\n', obj.reportTitle);
            fprintf(fileId, '  <p>Generated: %s</p>\n', obj.formatTimestamp(now));
            
            % Summary section
            fprintf(fileId, '  <h2>Summary</h2>\n');
            fprintf(fileId, '  <div class="summary">\n');
            fprintf(fileId, '    <p>Total Tests: <strong>%d</strong></p>\n', stats.totalTests);
            fprintf(fileId, '    <p>Passed: <strong class="pass">%d</strong> (%.1f%%)</p>\n', stats.totalPassed, stats.passRate);
            fprintf(fileId, '    <p>Failed: <strong class="fail">%d</strong> (%.1f%%)</p>\n', stats.totalFailed, (100 - stats.passRate));
            fprintf(fileId, '  </div>\n');
            
            % Category breakdown if multiple categories exist
            categories = fieldnames(obj.statistics);
            if length(categories) > 1
                fprintf(fileId, '  <h2>Category Breakdown</h2>\n');
                fprintf(fileId, '  <table>\n');
                fprintf(fileId, '    <tr><th>Category</th><th>Total Tests</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr>\n');
                
                for i = 1:length(categories)
                    cat = categories{i};
                    catStats = obj.statistics.(cat);
                    catPassRate = (catStats.passed / catStats.total) * 100;
                    fprintf(fileId, '    <tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%.1f%%</td></tr>\n', ...
                        cat, catStats.total, catStats.passed, catStats.failed, catPassRate);
                end
                
                fprintf(fileId, '  </table>\n');
            end
            
            % Detailed results table
            fprintf(fileId, '  <h2>Detailed Results</h2>\n');
            
            % Group by category
            for i = 1:length(categories)
                cat = categories{i};
                fprintf(fileId, '  <h3>%s</h3>\n', cat);
                fprintf(fileId, '  <table>\n');
                fprintf(fileId, '    <tr><th>Status</th><th>Test Name</th><th>Actions</th></tr>\n');
                
                % Find all tests in this category
                categoryTests = 0;
                resultFields = fieldnames(obj.testResults);
                for j = 1:length(resultFields)
                    result = obj.testResults.(resultFields{j});
                    if strcmp(result.category, cat)
                        categoryTests = categoryTests + 1;
                        if result.passed
                            status = '<span class="pass">PASS</span>';
                            actions = '';
                        else
                            status = '<span class="fail">FAIL</span>';
                            detailsId = sprintf('details-%s', resultFields{j});
                            actions = sprintf('<span class="toggle-button" onclick="toggleDetails(''%s'')">Show Details</span>', detailsId);
                        end
                        
                        fprintf(fileId, '    <tr><td>%s</td><td>%s</td><td>%s</td></tr>\n', ...
                            status, result.name, actions);
                        
                        % If test failed, add hidden details section
                        if ~result.passed
                            fprintf(fileId, '    <tr><td colspan="3">\n');
                            fprintf(fileId, '      <div id="%s" class="details-container hidden">\n', detailsId);
                            
                            if isfield(result, 'details')
                                if isfield(result.details, 'message')
                                    fprintf(fileId, '        <p><strong>Error:</strong> %s</p>\n', result.details.message);
                                end
                                
                                if isfield(result.details, 'stack')
                                    fprintf(fileId, '        <p><strong>Stack Trace:</strong></p>\n');
                                    fprintf(fileId, '        <ul>\n');
                                    for k = 1:length(result.details.stack)
                                        frame = result.details.stack(k);
                                        fprintf(fileId, '          <li>File: %s, Line: %d, Function: %s</li>\n', ...
                                            frame.file, frame.line, frame.name);
                                    end
                                    fprintf(fileId, '        </ul>\n');
                                end
                                
                                if isfield(result.details, 'expected') && isfield(result.details, 'actual')
                                    fprintf(fileId, '        <p><strong>Expected:</strong> %s</p>\n', ...
                                        mat2str(result.details.expected));
                                    fprintf(fileId, '        <p><strong>Actual:</strong> %s</p>\n', ...
                                        mat2str(result.details.actual));
                                end
                            else
                                fprintf(fileId, '        <p>No detailed information available</p>\n');
                            end
                            
                            fprintf(fileId, '      </div>\n');
                            fprintf(fileId, '    </td></tr>\n');
                        end
                    end
                end
                
                if categoryTests == 0
                    fprintf(fileId, '    <tr><td colspan="3">(No tests in this category)</td></tr>\n');
                end
                
                fprintf(fileId, '  </table>\n');
            end
            
            % Performance metrics section if enabled
            if obj.includePerformanceData && isfield(stats, 'performance')
                fprintf(fileId, '  <h2>Performance Metrics</h2>\n');
                fprintf(fileId, '  <table>\n');
                fprintf(fileId, '    <tr><th>Metric</th><th>Value</th></tr>\n');
                fprintf(fileId, '    <tr><td>Total Execution Time</td><td>%s</td></tr>\n', ...
                    obj.formatDuration(stats.performance.totalTime));
                fprintf(fileId, '    <tr><td>Average Test Time</td><td>%s</td></tr>\n', ...
                    obj.formatDuration(stats.performance.averageTime));
                fprintf(fileId, '    <tr><td>Slowest Test</td><td>%s (%s)</td></tr>\n', ...
                    stats.performance.slowestTest, obj.formatDuration(stats.performance.slowestTime));
                fprintf(fileId, '    <tr><td>Fastest Test</td><td>%s (%s)</td></tr>\n', ...
                    stats.performance.fastestTest, obj.formatDuration(stats.performance.fastestTime));
                fprintf(fileId, '  </table>\n');
            end
            
            % HTML footer
            fprintf(fileId, '</body>\n');
            fprintf(fileId, '</html>\n');
            
            % Close the file
            fclose(fileId);
            
            % Return the path to the generated file
            reportPath = outputPath;
        end
        
        function stats = getTestStatistics(obj)
            % GETTESTSTATISTICS Returns statistical information about test execution results
            %
            % USAGE:
            %   stats = obj.getTestStatistics()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   stats - Structure with test statistics including counts and pass rates
            
            % Initialize statistics structure
            stats = struct();
            
            % Calculate total tests and results
            stats.totalTests = 0;
            stats.totalPassed = 0;
            stats.totalFailed = 0;
            
            % Get category statistics
            categories = fieldnames(obj.statistics);
            for i = 1:length(categories)
                cat = categories{i};
                catStats = obj.statistics.(cat);
                stats.totalTests = stats.totalTests + catStats.total;
                stats.totalPassed = stats.totalPassed + catStats.passed;
                stats.totalFailed = stats.totalFailed + catStats.failed;
            end
            
            % Calculate pass rate
            if stats.totalTests > 0
                stats.passRate = (stats.totalPassed / stats.totalTests) * 100;
            else
                stats.passRate = 0;
            end
            
            % Include per-category statistics
            stats.categories = obj.statistics;
            
            % Calculate performance metrics if available
            if obj.includePerformanceData
                resultFields = fieldnames(obj.testResults);
                totalTime = 0;
                slowestTime = 0;
                fastestTime = Inf;
                slowestTest = '';
                fastestTest = '';
                
                for i = 1:length(resultFields)
                    result = obj.testResults.(resultFields{i});
                    
                    % Check if execution time is available
                    if isfield(result, 'details') && isfield(result.details, 'executionTime')
                        execTime = result.details.executionTime;
                        totalTime = totalTime + execTime;
                        
                        % Update slowest test
                        if execTime > slowestTime
                            slowestTime = execTime;
                            slowestTest = result.name;
                        end
                        
                        % Update fastest test
                        if execTime < fastestTime
                            fastestTime = execTime;
                            fastestTest = result.name;
                        end
                    end
                end
                
                % Only include performance data if we have execution times
                if totalTime > 0
                    stats.performance = struct();
                    stats.performance.totalTime = totalTime;
                    stats.performance.averageTime = totalTime / length(resultFields);
                    stats.performance.slowestTest = slowestTest;
                    stats.performance.slowestTime = slowestTime;
                    stats.performance.fastestTest = fastestTest;
                    stats.performance.fastestTime = fastestTime;
                end
            end
        end
        
        function displaySummary(obj)
            % DISPLAYSUMMARY Displays a summary of test results to the console
            %
            % USAGE:
            %   obj.displaySummary()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   void - No return value
            
            % Get statistics
            stats = obj.getTestStatistics();
            
            % Display summary header
            fprintf('==========================================================\n');
            fprintf('TEST EXECUTION SUMMARY\n');
            fprintf('Timestamp: %s\n', obj.formatTimestamp(now));
            fprintf('==========================================================\n\n');
            
            % Display overall results
            fprintf('Total Tests: %d\n', stats.totalTests);
            fprintf('Passed: %d (%.1f%%)\n', stats.totalPassed, stats.passRate);
            fprintf('Failed: %d (%.1f%%)\n', stats.totalFailed, 100 - stats.passRate);
            fprintf('\n');
            
            % Display category breakdown if multiple categories exist
            categories = fieldnames(obj.statistics);
            if length(categories) > 1
                fprintf('CATEGORY BREAKDOWN:\n');
                for i = 1:length(categories)
                    cat = categories{i};
                    catStats = obj.statistics.(cat);
                    catPassRate = (catStats.passed / catStats.total) * 100;
                    fprintf('%s: %d tests, %d passed (%.1f%%), %d failed\n', ...
                        cat, catStats.total, catStats.passed, catPassRate, catStats.failed);
                end
                fprintf('\n');
            end
            
            % Display failed tests if any
            if stats.totalFailed > 0
                fprintf('FAILED TESTS:\n');
                resultFields = fieldnames(obj.testResults);
                for i = 1:length(resultFields)
                    result = obj.testResults.(resultFields{i});
                    if ~result.passed
                        fprintf('- %s (%s)\n', result.name, result.category);
                        if isfield(result, 'details') && isfield(result.details, 'message')
                            fprintf('  Error: %s\n', result.details.message);
                        end
                    end
                end
                fprintf('\n');
            end
            
            % Display performance summary if available
            if isfield(stats, 'performance')
                fprintf('PERFORMANCE SUMMARY:\n');
                fprintf('Total Execution Time: %s\n', obj.formatDuration(stats.performance.totalTime));
                fprintf('Average Test Time: %s\n', obj.formatDuration(stats.performance.averageTime));
                fprintf('\n');
            end
            
            fprintf('==========================================================\n');
        end
        
        function failedTests = getFailedTests(obj)
            % GETFAILEDTESTS Returns a list of all failed tests with detailed information
            %
            % USAGE:
            %   failedTests = obj.getFailedTests()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   failedTests - Structure with failed test details grouped by category
            
            % Initialize result structure
            failedTests = struct();
            
            % Get all test results
            resultFields = fieldnames(obj.testResults);
            
            % No tests have been run
            if isempty(resultFields)
                return;
            end
            
            % Track categories with failures
            categoriesWithFailures = {};
            
            % Find all failed tests and group by category
            for i = 1:length(resultFields)
                result = obj.testResults.(resultFields{i});
                if ~result.passed
                    category = result.category;
                    
                    % Add category to tracking list if it's new
                    if ~isfield(failedTests, category)
                        failedTests.(category) = struct();
                        categoriesWithFailures{end+1} = category;
                    end
                    
                    % Add test to the appropriate category
                    testId = sprintf('test%d', length(fieldnames(failedTests.(category))) + 1);
                    failedTests.(category).(testId) = struct();
                    failedTests.(category).(testId).name = result.name;
                    failedTests.(category).(testId).timestamp = result.timestamp;
                    
                    % Add details if available
                    if isfield(result, 'details')
                        failedTests.(category).(testId).details = result.details;
                    end
                end
            end
            
            % Add list of categories with failures for easier processing
            failedTests.categoriesWithFailures = categoriesWithFailures;
        end
        
        function obj = setVerboseOutput(obj, verboseFlag)
            % SETVERBOSEOUTPUT Sets the verbose output flag for immediate result display
            %
            % USAGE:
            %   obj = obj.setVerboseOutput(verboseFlag)
            %
            % INPUTS:
            %   verboseFlag - Logical flag indicating whether to display immediate results
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            % Validate input
            if ~islogical(verboseFlag) || numel(verboseFlag) ~= 1
                error('verboseFlag must be a logical scalar value');
            end
            
            obj.verboseOutput = verboseFlag;
        end
        
        function obj = setReportFormats(obj, formats)
            % SETREPORTFORMATS Sets the output formats for test reports
            %
            % USAGE:
            %   obj = obj.setReportFormats(formats)
            %
            % INPUTS:
            %   formats - Cell array of strings with formats ('text', 'html', 'xml')
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            % Validate input
            if ~iscell(formats)
                error('formats must be a cell array of strings');
            end
            
            % Validate each format
            validFormats = {'text', 'html', 'xml'};
            for i = 1:length(formats)
                if ~ischar(formats{i})
                    error('Each format must be a string');
                end
                
                format = lower(formats{i});
                if ~any(strcmp(format, validFormats))
                    error('Unsupported format: %s. Supported formats are: text, html, xml', formats{i});
                end
            end
            
            obj.reportFormats = formats;
        end
        
        function obj = setReportOutputPath(obj, outputPath)
            % SETREPORTOUTPUTPATH Sets the output directory for generated reports
            %
            % USAGE:
            %   obj = obj.setReportOutputPath(outputPath)
            %
            % INPUTS:
            %   outputPath - String path to directory for report output
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            % Validate input
            if ~ischar(outputPath)
                error('outputPath must be a string');
            end
            
            % Check if directory exists
            if ~exist(outputPath, 'dir')
                error('Output directory does not exist: %s', outputPath);
            end
            
            obj.reportOutputPath = outputPath;
        end
        
        function obj = setReportTitle(obj, title)
            % SETREPORTTITLE Sets the title used in generated reports
            %
            % USAGE:
            %   obj = obj.setReportTitle(title)
            %
            % INPUTS:
            %   title - String title for reports
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            % Validate input
            if ~ischar(title) || isempty(title)
                error('title must be a non-empty string');
            end
            
            obj.reportTitle = title;
        end
        
        function obj = setIncludePerformanceData(obj, includeFlag)
            % SETINCLUDEPERFORMANCEDATA Sets whether performance metrics should be included in reports
            %
            % USAGE:
            %   obj = obj.setIncludePerformanceData(includeFlag)
            %
            % INPUTS:
            %   includeFlag - Logical flag indicating whether to include performance data
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            % Validate input
            if ~islogical(includeFlag) || numel(includeFlag) ~= 1
                error('includeFlag must be a logical scalar value');
            end
            
            obj.includePerformanceData = includeFlag;
        end
        
        function reports = getLastGeneratedReports(obj)
            % GETLASTGENERATEDREPORTS Returns paths to reports generated in the last run
            %
            % USAGE:
            %   reports = obj.getLastGeneratedReports()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   reports - Structure with report file paths by format
            
            if isempty(fieldnames(obj.lastGeneratedReports))
                warning('No reports have been generated yet.');
            end
            
            reports = obj.lastGeneratedReports;
        end
        
        function obj = clearResults(obj)
            % CLEARRESULTS Clears all recorded test results and statistics
            %
            % USAGE:
            %   obj = obj.clearResults()
            %
            % INPUTS:
            %   None
            %
            % OUTPUTS:
            %   obj - TestReporter instance (for method chaining)
            
            obj.testResults = struct();
            obj.statistics = struct();
            obj.lastGeneratedReports = struct();
        end
        
        function formattedTime = formatTimestamp(obj, timestamp)
            % FORMATTIMESTAMP Formats a timestamp for report display
            %
            % USAGE:
            %   formattedTime = obj.formatTimestamp()
            %   formattedTime = obj.formatTimestamp(timestamp)
            %
            % INPUTS:
            %   timestamp - (optional) MATLAB datenum timestamp
            %
            % OUTPUTS:
            %   formattedTime - Formatted timestamp string
            
            if nargin < 2 || isempty(timestamp)
                timestamp = now;
            end
            
            formattedTime = datestr(timestamp, 'yyyy-mm-dd HH:MM:SS');
        end
        
        function formattedDuration = formatDuration(obj, seconds)
            % FORMATDURATION Formats a duration in seconds for human-readable display
            %
            % USAGE:
            %   formattedDuration = obj.formatDuration(seconds)
            %
            % INPUTS:
            %   seconds - Duration in seconds
            %
            % OUTPUTS:
            %   formattedDuration - Formatted duration string
            
            % Validate input
            if ~isnumeric(seconds) || numel(seconds) ~= 1
                error('seconds must be a numeric scalar value');
            end
            
            % Handle very small durations
            if seconds < 0.001
                formattedDuration = sprintf('%.3f ms', seconds * 1000);
                return;
            end
            
            % Convert to hours, minutes, seconds
            hours = floor(seconds / 3600);
            minutes = floor((seconds - hours * 3600) / 60);
            secs = seconds - hours * 3600 - minutes * 60;
            
            % Format the string based on duration length
            if hours > 0
                formattedDuration = sprintf('%dh %dm %.2fs', hours, minutes, secs);
            elseif minutes > 0
                formattedDuration = sprintf('%dm %.2fs', minutes, secs);
            else
                formattedDuration = sprintf('%.3fs', secs);
            end
        end
    end
end