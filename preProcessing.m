%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wright State University                                Machine Learning %
% Written By: Alec Petrack                 Instructor: Dr. Tanvi Banerjee %
% Modified: 11/17/2019                                                     %
%                                                                         %
% Code Description: Final project code for machine learning.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

% % Load Data
[X,featureLabels] = xlsread('heart.xlsx'); % read data

% % pre-process data
temp1 = zscore(X(:,1)); % zscore age
temp2 = double(encode(X(:,3))); % one hot encode chest pain
temp3 = zscore(X(:,4)); % zscore trestbps
temp4 = zscore(X(:,5)); % zscore chol
temp5 = encode(X(:,7)); % one hot encode thal
temp6 = zscore(X(:,8)); % zscore thalach
temp7 = zscore(X(:,10)); % zscore oldpeak
temp8 = zscore(X(:,11)); % zscore slope
temp9 = zscore(X(:,12)); % zscore ca
temp10 = encode(X(:,13)); % zscore thal

scaledData = [temp1,X(:,2),temp2,temp3,temp4,X(:,6),temp5,temp6,X(:,9),...
    temp7,temp8,temp9,temp10,X(:,end)];

xlswrite('processedData.xlsx',scaledData)


% encodes string or char target data into one hot/integer encoded labels
function [oneHotEncoded,integerEncoded] = encode(data)
categories = repmat(unique(data),1,size(data,1))';  % get classification categories
labels = repmat(data,1,size(categories,2));    % repeat labels for one hot encoding
labels = labels;  % turn cell into char matrix
oneHotEncoded = labels==categories; % one hot encode output Y
[integerEncoded,~] = find(oneHotEncoded'); % convert into integer encoding
end