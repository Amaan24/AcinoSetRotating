%Define project, camera and frames
project = '25Apr2023';
camera = 1; %Camera 1 or 2
%frame1 = 10490;
%frames = [12000];
%frames = [12225];
frames = [12200, 12225, 12250, 12275];
%frames = [12200, 12225, 12250, 12275, 14150, 14175, 14200, 14250];
static_frame = 10490;

%Load detected checkerboard corners and encoder values
synced_data = py.open(strcat('C:\Users\user-pc\Desktop\AcinoSetRotating\data\', project, '\synced_data.pkl'),'rb');
synced_data = py.pickle.load(synced_data);

load(strcat('C:\Users\user-pc\Desktop\AcinoSetRotating\data\', project, '\checkerboard_corners.mat'))

%World Points
%Get initial position of checkerboard point using triangulation
%Change to bundle adjustment?
points1 = cell2mat(cornerPoints(static_frame,1));
points2 = cell2mat(cornerPoints(static_frame,2));
worldPoints = triangulate(points1,points2,stereoParams);

if (camera == 1)
    angles = count_to_rad(double(py.array.array('d',py.numpy.nditer(synced_data{'enc1tick'}))));

    % Camera 1 initial pose
    R_i = [1, 0, 0; 0, 1, 0; 0, 0, 1];
    t_i = [0; 0; 0];

    intrinsics = stereoParams.CameraParameters1.Intrinsics;

    cornerPoints = cornerPoints(:, 1);

elseif (camera == 2)
    angles = count_to_rad(double(py.array.array('d',py.numpy.nditer(synced_data{'enc2tick'}))));

    % Camera 2 initial pose
    R_i = stereoParams.PoseCamera2.R;
    t_i = stereoParams.PoseCamera2.Translation;

    intrinsics = stereoParams.CameraParameters2.Intrinsics;

    cornerPoints = cornerPoints(:, 2);

else
    disp("Invalid camera. Exiting script.")
    quit();
end

%Solve optimisation
RCWInitial = [0.1 0.1];
XInitial = [RCWInitial];

b_angles = pi/10 .* ones([1,2]); %Radians
 
ub = [b_angles];
lb = -1 .* ub;

options = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', 1000);

X_solution = lsqnonlin(@(X) residual(X, R_i, t_i, angles, intrinsics, ...
    cornerPoints, worldPoints, frames), XInitial, lb, ub, options)

vidIn = VideoReader('C:\Users\user-pc\Desktop\AcinoSetRotating\data\25Apr2023\1.avi');

%Display results
%Camera at t0 to motor frame
R_CM = rot_x(X_solution(1)) * rot_z(X_solution(2));

%Motor frame at ti to camera frame at ti
R_MC = R_CM.';

%Homogenous coordinates
R_i_homog = [R_i t_i; [0, 0, 0, 1]]; %World to camera
R_CM_homog = [R_CM [0;0;0]; [0, 0, 0, 1]]; %Camera to motor
R_MC_homog = [R_MC [0;0;0]; [0, 0, 0, 1]];

for i = 1:length(frames)
   pic = read(vidIn, frames(i));
    imshow(pic);
    title("Frame " + frames(i))
    axis on
    hold on;

    %Optimised
    R_MM_homog = [rot_y(angles(frames(i))).' [0; 0; 0]; [0, 0, 0, 1]]; 

    R_homog = R_MC_homog * R_MM_homog * R_CM_homog * R_i_homog;

    R = R_homog(1:3, 1:3);
    t = R_homog(1:3, 4);
    Rt = rigidtform3d(R, t);

    projectedPoints = world2img(worldPoints, Rt, intrinsics);
    
    corners = cell2mat(cornerPoints(frames(i),1));

    plot(corners(:,1), corners(:,2), 'b*')
    plot(projectedPoints(:,1), projectedPoints(:,2), 'go');

    %Not optimised
    R = rot_y(angles(frames(i))).' * R_i;
    t = rot_y(angles(frames(i))).' * t_i;
    Rt = rigidtform3d(R,t);
    projectedPoints = world2img(worldPoints, Rt, intrinsics);
        
    plot(projectedPoints(:,1), projectedPoints(:,2), 'r+');
    legend('Detected Corner Locations', 'Optimised Corner Locations', ...
        'Unoptimised Corner Locations', 'AutoUpdate','off')

    pause
    key = get(gcf,'CurrentKey');
    if(strcmp (key , 'q'))
        break;
    end
end

close all;



 

%Function Definitions
%Cost Function
%Minimises the difference between the measured checkerboard points and the reprojected points.
%Points in the world frame are converted to the camera frame at time t:
%Rotate from world frame to camera frame at t0.(R initial)
%Rotate from camera frame at t0 to motor frame. (R_CM) (This is what we are
%solving for
%Rotate about motor y axis by angle measured by encoder (with 1 deg noise)
%(R_MM)
%Rotate from motor frame to camera frame at t1. (R_MC)
%Then, points are reprojected into the image frame and the residual is calculated.

function r = residual(X, Ri, ti, enc_angles, intrinsics, cornerPoints, worldPoints, frames)
    r = zeros(20, 2, length(frames));
    for i = 1:length(frames)
        angles = X(1:2);
    
        %Camera at t0 to motor frame
        R_CM = rot_x(angles(1)) * rot_z(angles(2));
    
        %Motor frame at ti to camera frame at ti
        R_MC = R_CM.';
         
        %Homogenous coordinates
        R_i_homog = [Ri ti; [0, 0, 0, 1]]; %World to camera
        R_CM_homog = [R_CM [0;0;0]; [0, 0, 0, 1]]; %Camera to motor
        R_MM_homog = [rot_y(enc_angles(frames(i))).' [0; 0; 0]; [0, 0, 0, 1]]; 
        R_MC_homog = [R_MC [0;0;0]; [0, 0, 0, 1]];
    
        R_homog = R_MC_homog * R_MM_homog * R_CM_homog * R_i_homog;

        R = R_homog(1:3, 1:3);
        t = R_homog(1:3, 4);
    
        Rt = rigidtform3d(R,t);
    
        projectedPoints = world2img(worldPoints,Rt,intrinsics);
        differences = projectedPoints - cell2mat(cornerPoints(frames(i)));
     
        for j = 1:20 %Number of checkerboard corners
            for k = 1:2 %Dimensionality of measurements
                r(i,j,k) = differences(j,k);
            end
        end
    end
    r = reshape(r, 1, []);
end

%Rotation about x axis
function r = rot_x(theta)
    r = [1,     0,          0;
         0, cos(theta), -sin(theta);
         0, sin(theta), cos(theta)]; 
end

%Rotation about y axis
function r = rot_y(theta)
    r = [cos(theta), 0, -sin(theta);
            0,       1,     0;
         sin(theta), 0,     cos(theta)]; 
end

%Rotation about z axis
function r = rot_z(theta)
    r = [cos(theta), -sin(theta), 0;
         sin(theta), cos(theta),  0;
             0,           0,      1]; 
end

%Encoder ticks to angles
function ang = count_to_rad(enc_count)
    ang = enc_count * 2 * pi / 102000;
end
