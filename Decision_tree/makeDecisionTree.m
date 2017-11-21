function t = makeDecisionTree(X,Y,maxDepth,t)
%initiate the parameters
%t.nodeNum = nodesNum;
%t.splitTimes = splitTimes;
clear layer_table;
layer_table = cell(maxDepth,1);
layer_table{1,1} = 1;
clear content_table;
content_table = cell(size(layer_table{1,1},2),1);
content_table{1,1} = [X Y];
for depth = 1:maxDepth
    if size(layer_table{depth,1}) == 0
        break;
    else
    first_index = layer_table{depth,1}(1,1)-1;
    for i = layer_table{depth,1}
        for j = 1:10
            t.class_p(i,j) = size(find(content_table{i-first_index,1}(:,end) == j-1),1)/size(content_table{i-first_index,1}(:,end),1);
        end
        
        [t.feature(i,1), t.cutpoint(i,1)] = selectOptimalFandT(content_table{i-first_index,1}(:,1:end-1),content_table{i-first_index,1}(:,end));
        if t.feature(i,1) ~= 0 && depth ~= maxDepth
            t.nodeNum = t.nodeNum + 1;
            t.children(i, 1) = t.nodeNum;
            t.nodeNum = t.nodeNum + 1;
            t.children(i, 2) = t.nodeNum;
        else 
            t.children(i, 1) = 0;
            t.children(i, 2) = 0;
        end
    end
    %update the content_table
    next_size = t.nodeNum - layer_table{depth,1}(1,end);
    clear next_content_table;
    next_content_table = cell(next_size,1);
    j = 0;
    for i = layer_table{depth,1}
        if t.feature(i,1) ~= 0 && depth ~= maxDepth
            index_l = find(content_table{i-first_index,1}(:,t.feature(i,1)) <= t.cutpoint(i,1));
            index_h = find(content_table{i-first_index,1}(:,t.feature(i,1)) > t.cutpoint(i,1));
            j = j+1;
            next_content_table{j,1} = content_table{i-first_index,1}(index_l',:);
            j = j+1;
            next_content_table{j,1} = content_table{i-first_index,1}(index_h',:);
        end
    end
    clear content_table;
    content_table = next_content_table;
    %update the layer table
    layer_table{depth+1,1} = (layer_table{depth,1}(1,end)+1 : t.nodeNum);
    end
end