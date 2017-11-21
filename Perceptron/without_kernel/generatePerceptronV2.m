function [w,c] = generatePerceptronV2(data)
    N = size(data, 1);
    X = data(:, 1:end-1);
    %X = [X zeros(N, 1)];
    Y = data(:, end);
    T = N + 1;
    w(1,:) = zeros(1, size(X, 2));
    c(1,1) = 0;
    k = 1;
    for t = 1:T
        i = mod(t, N) + 1;
        if (Y(i, 1)*dot(w(k,:), X(i, :))<=0)
            w(k+1,:) = w(k,:) + Y(i, 1)*X(i, :);
            c(k+1,1) = 1;
            k = k+1;
        else
            c(k,1) = c(k,1) + 1;
        end
    end
end