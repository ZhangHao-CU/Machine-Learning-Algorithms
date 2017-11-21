function [a,c] = generateKernelPerceptronV2(data)
    N = size(data, 1);
    X = data(:, 1:end-1);
    %X = [X zeros(N, 1)];
    Y = data(:, end);
    T = N + 1;
    a(1,:) = zeros(1, size(X, 1));
    %w(1,:) = zeros(1, size(X, 2));
    c(1,1) = 0;
    k = 1;
    for t = 1:T
        i = mod(t, N) + 1;
        tmp_sum = 0;
        for j = find(a(k,:) == 1)
            tmp_sum = tmp_sum+Y(j, 1)*Kernel(X(j, :),X(i,:));
        end
        if Y(i, 1)*tmp_sum <= 0
            a(k+1,:) = a(k,:);
            a(k+1,i) = a(k+1,i) + 1;
            c(k+1,1) = 1;
            k = k+1;
        else
            c(k,1) = c(k,1) + 1;
        end
    end
end