function a = generateKernelPerceptronV0(data)
    N = size(data, 1);
    X = data(:, 1:end-1);
    %X = [X zeros(N, 1)];
    Y = data(:, end);
    T = N + 1;
    a = zeros(1, size(X, 1));
    for t = 1:T
        i = mod(t, N) + 1;
        tmp_sum = 0;
        for j = find(a == 1)
            tmp_sum = tmp_sum+Y(j, 1)*Kernel(X(j, :),X(i,:));
        end
        if Y(i, 1)*tmp_sum <= 0
            a(1,i) = a(1,i) + 1;
        end
    end
end