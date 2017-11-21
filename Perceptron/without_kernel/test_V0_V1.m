%clear;
clc;
data = load('/Users/zhanghao/Documents/MATLAB/ML/HW2/hw1data.mat');
X = data.X;
Y = data.Y;
%split the data to training and testing
training_X = X(1:6000,:);
testing_X = X(6001:10000,:);
training_Y = Y(1:6000,:);
testing_Y = Y(6001:10000,:);
training_tmp_X = training_X;
training_tmp_Y = training_Y;
%generate the w using perceptron
w = zeros(9, size(X, 2));
for i = 0:8
    %training_tmp_X = training_tmp_X(training_tmp_Y >= i, :);
    %training_tmp_Y = training_tmp_Y(training_tmp_Y >= i, 1);
    training_new_Y = ones(size(training_tmp_Y,1), 1);
    training_new_Y(training_tmp_Y == i, 1) = -1;
    new_data = [training_tmp_X training_new_Y];
    w(i+1, :) = generatePerceptronV0(new_data);
    %w(i+1, :) = generatePerceptronV1(new_data);
end
%test the classifier
testing_results = zeros(size(testing_Y,1), 1);
for i =1:size(testing_X,1)
    j = 1;
    while j<10 
        if dot(w(j, :), testing_X(i, :)) < 0
            break;
        end
        j = j+1;
    end
    testing_results(i,1) = j-1;
end
%calculate the accurancy
V1 = sum(testing_results == testing_Y)/size(testing_Y,1);