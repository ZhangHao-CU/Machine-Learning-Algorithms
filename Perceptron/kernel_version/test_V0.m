clear;
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
a = zeros(9, size(training_X, 1));
for i = 0:8
    %training_tmp_X = training_tmp_X(training_tmp_Y >= i, :);
    %training_tmp_Y = training_tmp_Y(training_tmp_Y >= i, 1);
    training_new_Y = ones(size(training_tmp_Y,1), 1);
    training_new_Y(training_tmp_Y == i, 1) = -1;
    new_data = [training_tmp_X training_new_Y];
    %a(i+1, :) = generateKernelPerceptronV0(new_data);
    a(i+1, :) = generateKernelPerceptronV1(new_data);
end
%test the classifier
testing_results = zeros(size(testing_Y,1), 1);
for i =1:size(testing_X,1)
    j = 1;
    while j<10 
        tmp_sum = 0;
        training_new_Y = ones(size(training_tmp_Y,1), 1);
        training_new_Y(training_tmp_Y == j-1, 1) = -1;
        for k = find(a(j,:) == 1)
            tmp_sum = tmp_sum+training_new_Y(k, 1)*Kernel(training_tmp_X(k, :),testing_X(i,:));
        end
        if tmp_sum < 0
            break;
        end
        j = j+1;
    end
    testing_results(i,1) = j-1;
end
%calculate the accurancy
V1 = sum(testing_results == testing_Y)/size(testing_Y,1);