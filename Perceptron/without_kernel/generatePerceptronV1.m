function w = generatePerceptronV1(data)
    N = size(data, 1);
    X = data(:, 1:end-1);
    %X = [X zeros(N, 1)];
    Y = data(:, end);
    T = N + 1;
    w = zeros(1, size(X, 2));
    each_p = zeros(1, N);
    for t = 1:T
        for j = 1:N
            each_p(1, j) = Y(j, 1)*(X(j, :)*w')';
        end
        [~, i] = min(each_p);
        if (each_p(1, i)<=0)
            w = w + Y(i, 1)*X(i, :);
        else 
            return;
        end
    end
end