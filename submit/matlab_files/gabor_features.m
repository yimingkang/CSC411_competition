% Extracts gabor features from each image in the passed in images vector
% Parameters:
%   column downsampling:
%   row downsampling:
%   # of scales:
%   # of orientations:
%   # of rows in 2-D Gabor Filter:
%   # of columns in 2-D Gabor Filter:

function features = gabor_features(images)

n = size(images,3);
gaborArray = gaborFilterBank(5,8,9,9);
features = zeros(n,2560);

for i = 1:n
    featureVector = gaborFeatures(images(:,:,i),gaborArray,4,4);
    features(i,:) = featureVector;
end
