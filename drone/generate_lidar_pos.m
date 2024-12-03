% Directory containing the cityscape files
cityscapeDir = '../data/matlab_meshes/'; % Adjust the directory path if necessary

% Directory containing the initial positions CSV files
positionsDir = '../data/drone_positions/'; % Adjust the directory path if necessary

% Directory to save the point cloud data
outputDir = '../data/pointclouds/'; % Adjust the output directory if necessary

% Ensure the output directory exists
if ~exist(outputDir, 'dir')
    mkdir(outputDir); % Create the directory if it doesn't exist
end

% Get a list of all .mat files in the cityscape directory
cityscapeFiles = dir(fullfile(cityscapeDir, 'city_*.mat'));

% Loop through each .mat cityscape file
for fileIdx = 1:length(cityscapeFiles)
    % Construct full cityscape file path
    cityscapeFile = fullfile(cityscapeDir, cityscapeFiles(fileIdx).name);

    % Load the cityscape data
    load(cityscapeFile);
    

    % Load the initial positions CSV (same name as cityscape, but in positionsDir)
    positionsFile = fullfile(positionsDir, [cityscapeFiles(fileIdx).name(1:end-4), '.csv']);
    
    if exist(positionsFile, 'file')
        % Read the initial positions (x, y, z) from the CSV
        initialPositions = readmatrix(positionsFile);

        % Create a directory for this cityscape in the output folder
        cityOutputDir = fullfile(outputDir, cityscapeFiles(fileIdx).name(1:end-4));
        if ~exist(cityOutputDir, 'dir')
            mkdir(cityOutputDir); % Create the directory for this cityscape
        end

        % Loop over each position in the CSV file
        for posIdx = 1:size(initialPositions, 1)
            % Get the initial position (x, y, z) where z is fixed to 1
            initialPosition = [initialPositions(posIdx, 1), initialPositions(posIdx, 2), 1];  % Set z = 1

          % Create the UAV scenario
            scene = uavScenario(UpdateRate=2, ReferenceLocation=[75 -46 0]);
        
            % Add a ground plane
            color.Gray = 0.651*ones(1,3);
            color.Green = [0.3922 0.8314 0.0745];
            color.Red = [1 0 0];
            addMesh(scene, "polygon", {[0 0; 250 0; 250 250; 0 250], [-4 0]}, color.Gray);
        
            % Add extruded meshes for each building with varying heights from 10-30
            for i = 1:size(buildingCoords, 2)
                addMesh(scene, "polygon", {buildingCoords{i}(1:4,:), [0 10]}, color.Green);
            end

            % Re-create the UAV platform for each position
            plat = uavPlatform("UAV", scene, ReferenceFrame="NED", ...
                InitialPosition=initialPosition, InitialOrientation=eul2quat([0 0 0]));

            % Set up platform mesh and apply rotation
            updateMesh(plat, "quadrotor", {10}, color.Red, [0 0 0], eul2quat([0 0 pi]));

            % Set up the lidar model
            lidarmodel = uavLidarPointCloudGenerator(AzimuthResolution=0.3324099, ...
                ElevationLimits=[-20 20], ElevationResolution=1.25, ...
                MaxRange=90, UpdateRate=2, HasOrganizedOutput=true);

            % Set up a new lidar sensor for this UAV platform
            lidar = uavSensor("Lidar", plat, lidarmodel, MountingLocation=[0, 0, -1]);

            % Initialize pointCloud object
            pt = pointCloud(nan(1, 1, 3));

            % Setup and update the scene
            setup(scene);
            updateSensors(scene);

            % Read lidar data
            [isupdated, lidarSampleTime, pt] = read(lidar);

            % Flatten the point cloud data (assuming pt.Location is a 3D array)
            flat_pointCloud = reshape(pt.Location, [], 3);

            % Construct the output file name (just using the position index)
            outputFileName = fullfile(cityOutputDir, sprintf('pointcloud_%d.csv', posIdx));
            outputFileNameMAT = fullfile(cityOutputDir, sprintf('pointcloud_%d.mat', posIdx));

            % Write the point cloud data to a CSV file
            writematrix(flat_pointCloud, outputFileName);
            save(outputFileNameMAT, 'pt');
        end
    else
        warning('Positions file %s not found for cityscape %s.', positionsFile, cityscapeFiles(fileIdx).name);
    end
end
