% Script to convert building coordinates from a csv into a mesh for MATLAB
% and save them

% Define the input and output directories
inputDir = '../data/cityscapes/';  % Replace with your input directory
outputDir = '../data/matlab_meshes/'; % Replace with your output directory

% Check if the output directory exists; if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);  % Create the output directory if it doesn't exist
    fprintf('Created output directory: %s\n', outputDir);
end


% Get a list of all CSV files in the input directory
csvFiles = dir(fullfile(inputDir, '*.csv'));

% Loop through each CSV file
for i = 1:length(csvFiles)
    % Get the full path of the current CSV file
    csvFilePath = fullfile(inputDir, csvFiles(i).name);
    
    % Read the CSV file (skip the header)
    data = csvread(csvFilePath, 1, 0);  % Skip the header row
    
    % Initialize a cell array to store the transformed data
    buildingCoords = cell(1,size(data, 1));  % One cell for each row in the CSV file
    buildingHeights = cell(1,size(data, 1));
    
    % Loop over each row in the CSV file
    for j = 1:size(data, 1)
        % Extract x1, y1, x2, y2 from the current row
        x1 = data(j, 1);
        y1 = data(j, 2);
        x2 = data(j, 3);
        y2 = data(j, 4);
        height = data(j,5);
        
        % Rearrange the values into the desired 4x2 format
        transformed_row = [x1, y1; x2, y1; x2, y2; x1, y2];
        height_row = [0 height];
        
        % Store the transformed row into the cell array
        buildingCoords{1,j} = transformed_row;
        buildingHeights{1,j} = height_row;
    end
    
    % Create the output file name by replacing the .csv extension with .mat
    [~, fileName, ~] = fileparts(csvFiles(i).name);  % Get the file name without extension
    matFileName = fullfile(outputDir, [fileName, '.mat']);  % Set the .mat file path
    
    % Save the cell array to a .mat file
    save(matFileName, 'buildingCoords','buildingHeights');


end

fprintf("Completed converting all csv files into meshes\n")