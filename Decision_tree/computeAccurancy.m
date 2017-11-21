function acc = computeAccurancy(t, X, Y)
    testing_results = zeros(size(Y,1),1);
    for i = 1:size(X,1)
        index_node = 1;
        entry_X = X(i,:);
        while((t.children(index_node,1) ~= 0) && (t.children(index_node,2) ~= 0))
            if(entry_X(1,t.feature(index_node,1)) <= t.cutpoint(index_node,1))
                index_node = t.children(index_node,1);
            else
                index_node = t.children(index_node,2);
            end
        end
        [~,find_result] = max(t.class_p(index_node,:));
        testing_results(i,1) = find_result-1;
    end
    acc = sum(testing_results == Y)/size(Y,1);
end