clear
clc

vidIn = VideoReader('C:\Users\user-pc\Desktop\25Apr2023\1.avi');
cornerPoints1 = {};
for ii = 1:vidIn.NumFrames
  pic = read(vidIn, ii);
  [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(pic);

  if ~isequal(size(imagePoints), [20 2])
      %J = insertMarker(pic,imagePoints,'o','Color','red','Size',5);
      %imshow(J)
      imagePoints = zeros([20 2]);
      %w = waitforbuttonpress
  end

  cornerPoints1 = [cornerPoints1; imagePoints];
  
  
  %X = sprintf('1- %d', ii);
  disp(ii)
end

vidIn = VideoReader('C:\Users\user-pc\Desktop\25Apr2023\2.avi');
cornerPoints2 = {};
for ii = 1:vidIn.NumFrames
  pic = read(vidIn, ii);
  [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(pic);

  if ~isequal(size(imagePoints), [20 2])      
      imagePoints = zeros([20 2]);
      %J = insertMarker(pic,imagePoints,'o','Color','red','Size',5);
      %imshow(J)
      %w = waitforbuttonpress
  end

  cornerPoints2 = [cornerPoints2; imagePoints];
  %X = sprintf('2- %d', ii);
  disp(ii)
end
   
cornerPoints = [cornerPoints1 cornerPoints2];

save("C:\Users\user-pc\Desktop\25Apr2023\checkerboard_corners", "cornerPoints")