%Define project, camera and frames
project = '25Apr2023';
camera = 1; %Camera 1 or 2
frame1 = 10490;
frame2 = 12300;
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


%Reproject Using only initial R and motor rototation
R_new = rot_y(angles(frame2)).' * R_i;
t_new = rot_y(angles(frame2)).' * t_i;

Rt = rigidtform3d(R_new,t_new);
projectedPoints = world2img(worldPoints, Rt, intrinsics);

vidIn = VideoReader('C:\Users\user-pc\Desktop\AcinoSetRotating\data\25Apr2023\1.avi');
pic = read(vidIn, frame2);
imshow(pic);
axis on
hold on;

for i = 1:20
    plot(projectedPoints(i,1), projectedPoints(i,2), 'r+');
end

pause;

%Solve optimisation
RCWInitial = zeros(1,3);

XInitial = RCWInitial;
lb = [-pi, -pi, -pi]; 
ub = [pi, pi, pi];
options = optimoptions('lsqnonlin', 'Display', 'iter');

X_solution = lsqnonlin(@(X) residual(X, R_i, t_i, angles, intrinsics, ...
    cornerPoints, worldPoints, frame2), XInitial, lb, ub, options)

%Reproject Using Optimised transformations
pic = read(vidIn, frame2);
imshow(pic);
axis on
hold on;

R_i = [R_i t_i; [0, 0, 0, 1]];

RCW = rot_y(X_solution(2)) * rot_x(X_solution(1)) * rot_z(X_solution(3));

R_CM = [RCW [0; 0; 0]; [0, 0, 0, 1]];
R_MM = [rot_y(angles(frame2)).' [0; 0; 0]; [0, 0, 0, 1]];
R_MC = [RCW.' [0; 0; 0]; [0, 0, 0, 1]];

R_new = R_MC * R_MM * R_CM * R_i;

R = R_new(1:3, 1:3);
t = R_new(1:3, 4);

Rt = rigidtform3d(R, t);

projectedPoints = world2img(worldPoints, Rt, intrinsics);

for i = 1:20
    plot(projectedPoints(i,1), projectedPoints(i,2), 'r+');
end




 

%Function Definitions
%Cost Function
%Minimises the difference between the measured checkerboard points and the reprojected points.
%Points in the world frame are converted to the camera frame at time t:
%Rotate from world frame to camera frame at t0.
%Rotate from camera frame at t0 to motor frame.
%Rotate about motor y axis by angle measured by encoder (with 1 deg noise)
%Rotate from motor frame to camera frame at t1.

%Then, points are reprojected into the image frame and the residual is calculated.

function r = residual(X, Ri, ti, enc_angles, intrinsics, cornerPoints, worldPoints, frame)
    r = zeros(20, 2);

    angles = X(1:3);
 
    RCW = rot_y(angles(2)) * rot_x(angles(1)) * rot_z(angles(3));

    R_i = [Ri ti; [0, 0, 0, 1]];
    R_CM = [RCW [0; 0; 0]; [0, 0, 0, 1]];
    R_MM = [rot_y(enc_angles(frame)).' [0; 0; 0]; [0, 0, 0, 1]]; 
    R_MC = [RCW.' [0; 0; 0]; [0, 0, 0, 1]];

    R_new = R_MC * R_MM * R_CM * R_i;

    R = R_new(1:3, 1:3);
    t = R_new(1:3, 4);

    Rt = rigidtform3d(R,t);

    projectedPoints = world2img(worldPoints,Rt,intrinsics);
    differences = projectedPoints - cell2mat(cornerPoints(frame));
 
    for j = 1:20 %Number of checkerboard corners
        for k = 1:2 %Dimensionality of measurements
            r(j,k) = differences(j,k);
        end
    end
    r = reshape(r, 1, 40);
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
