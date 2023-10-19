% Define the video file name and its path
videoFile = 'C:\Users\user-pc\Desktop\18May2023\1.avi';

% Read the video file
videoObj = VideoReader(videoFile);

% Define the start and end frames
startFrame = 1;  % Starting frame number
endFrame = videoObj.NumFrames; % Ending frame number

% Initialize the previous frame variable
prevFrame = [];

% Create a figure for displaying the matched features
figure;

% Initialize the first camera pose
R_total = [eye(3)];
t_total = [0; 0; 0];

R_totals{1} = eye(3);
t_totals{1} = [0; 0; 0];

R_prev = eye(3);
t_prev = [0; 0; 0];

% Iterate through frames
for frameNum = startFrame:endFrame
    disp(frameNum);

    % Read the current frame
    frame = read(videoObj, frameNum);
    
    % Perform operations on the frame (e.g., display or process it)
    %imshow(frame); % Display the frame
    
    % Detect SURF features between the current frame and the previous frame
    if ~isempty(prevFrame)
        % Convert frames to grayscale
        grayCurrentFrame = rgb2gray(frame);
        grayPrevFrame = rgb2gray(prevFrame);
        
        % Detect SURF features in both frames
        pointsCurrentFrame = detectSURFFeatures(grayCurrentFrame,  ...
            "MetricThreshold", 250, "NumOctaves", 4, "NumScaleLevels", 6);
        pointsPrevFrame = detectSURFFeatures(grayPrevFrame,  ...
            "MetricThreshold", 250, "NumOctaves", 4, "NumScaleLevels", 6);
        
        % Extract feature descriptors
        [featuresCurrentFrame, validPointsCurrentFrame] = ...
            extractFeatures(grayCurrentFrame, pointsCurrentFrame);
        [featuresPrevFrame, validPointsPrevFrame] = ...
            extractFeatures(grayPrevFrame, pointsPrevFrame);
        
        % Match features between frames
        indexPairs = matchFeatures(featuresCurrentFrame, featuresPrevFrame);
        matchedPointsCurrentFrame = validPointsCurrentFrame(indexPairs(:, 1));
        matchedPointsPrevFrame = validPointsPrevFrame(indexPairs(:, 2));
        
        % Remove outliers using RANSAC
        [tform, inlierPointsCurrentFrame, inlierPointsPrevFrame] = ...
            estimateGeometricTransform(matchedPointsCurrentFrame, ...
            matchedPointsPrevFrame, 'affine', 'MaxNumTrials', 500);
        
        % Visualize the matched features without outliers
        %showMatchedFeatures(grayCurrentFrame, grayPrevFrame, ...
        %    inlierPointsCurrentFrame, inlierPointsPrevFrame, 'montage');
       % title('Matched SURF Features (Outliers Removed)');
        
        % Calculate the essential matrix
        [E, inliers] = estimateEssentialMatrix(inlierPointsCurrentFrame, ...
            inlierPointsPrevFrame, cameraParams);

        % Extract the inlier points using logical indexing
        inlierPointsCurrentFrame = inlierPointsCurrentFrame(inliers);   
        inlierPointsPrevFrame = inlierPointsPrevFrame(inliers);

        % Decompose the essential matrix to obtain camera poses
        relativePose = estrelpose(E, cameraParams.Intrinsics, ...
            inlierPointsCurrentFrame, inlierPointsPrevFrame);
        
        % Compute the rotation matrix and translation vector
        R = relativePose(1).R;
        t = relativePose(1).Translation;

        relativePose_prev = rigidtform3d(R_prev,t_prev);
        
        camProjection1 = cameraProjection(cameraParams.Intrinsics,relativePose_prev);
        camProjection2 = cameraProjection(cameraParams.Intrinsics,relativePose(1));

        %worldPoints = triangulate(matchedPointsPrevFrame,matchedPointsCurrentFrame,camProjection1,camProjection2);
        %disp("3D World Points Relative to Prev Frame");
        %disp(worldPoints);

        % Update the camera pose
        R_total = R_total * R;
        t_total = R_total * t.' + t_total;

        R_totals{frameNum} = R_total;
        t_totals{frameNum} = t_total;

        %disp("Camera Pose relative to World Frame")
        %disp(R_total)
        %disp(t_total)
        
        % Print the rotation matrix and translation vector
        %disp("Rotation Matrix (R):");
        %disp(R);
        %disp("Translation Vector (t):");
        %disp(t);
    end
    
    % Store the current frame as the previous frame for the next iteration
    prevFrame = frame;
 
    % Wait for a key press to display the next frame
    % Comment out this line if you want the frames to be displayed continuously
    pause;
end
