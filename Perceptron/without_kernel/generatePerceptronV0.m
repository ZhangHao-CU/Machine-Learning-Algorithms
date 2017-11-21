function w = generatePerceptronV0(data)
    N = size(data, 1);
    X = data(:, 1:end-1);
    %X = [X zeros(N, 1)];
    Y = data(:, end);
    T = N + 1;
    w = zeros(1, size(X, 2));
    for t = 1:T
        i = mod(t, N) + 1;
        if (Y(i, 1)*dot(w, X(i, :))<=0)
            w = w + Y(i, 1)*X(i, :);
        end
    end
end