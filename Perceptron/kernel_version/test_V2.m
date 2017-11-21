clear;
clc;
data = load('/Users/zhanghao/Documents/MATLAB/ML/HW2/hw1data.mat');
X = data.X;
Y = data.Y;
%split the data to training and testing
training_X = X(1:9000,:);
testing_X = X(9001:10000,:);
training_Y = Y(1:9000,:);
testing_Y = Y(9001:10000,:);
training_tmp_X = training_X;
training_tmp_Y = training_Y;
%generate the w using perceptron
a = cell(9,1);
c = cell(9,1);
for i = 0:8
    %training_tmp_X = training_tmp_X(training_tmp_Y >= i, :);
    %training_tmp_Y = training_tmp_Y(training_tmp_Y >= i, 1);
    training_new_Y = ones(size(training_tmp_Y,1), 1);
    training_new_Y(training_tmp_Y == i, 1) = -1;
    new_data = [training_tmp_X training_new_Y];
    %w(i+1, :) = generatePerceptronV0(new_data);
    %w(i+1, :) = generatePerceptronV1(new_data);
    [a{i+1,1}, c{i+1,1}] = generateKernelPerceptronV2(new_data);
end
%test the classifier
testing_results = zeros(size(testing_Y,1), 1);
for i =1:size(testing_X,1)
    j = 1;
    while j<10 
        c_sum = 0;
        training_new_Y = ones(size(training_tmp_Y,1), 1);
        training_new_Y(training_tmp_Y == j-1, 1) = -1;
        for k = 1:size(c{j,1}, 1)
            w_sum = 0;
            for q = find(a{j,1}(k,:))
                w_sum = w_sum+training_new_Y(q, 1)*Kernel(training_tmp_X(q, :),testing_X(i,:));
            end
            c_sum = c_sum + c{j,1}(k,1)*sign(w_sum);
        end
        if c_sum < 0
            break;
        end
        j = j+1;
    end
    testing_results(i,1) = j-1;
end
%calculate the accurancy
V2 = sum(testing_results == testing_Y)/size(testing_Y,1);