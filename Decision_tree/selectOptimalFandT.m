function [optimal_f,optimal_c] = selectOptimalFandT(X,Y)
optimal_f = 0;
optimal_c = NaN;
min_E = 0;
clear temp_table;
temp_table = tabulate(Y);
for i = 1:size(temp_table,1)
    temp_p = temp_table(i,3)/100;
    min_E = min_E - temp_p*log2(temp_p);%- (temp_table(i,3)/100)^2;
end
for feature = 1:size(X,2)
    clear sort_table;
    sort_table = sort(unique(X(:,feature)));
    for cutpoint_i = 1:size(sort_table,1)
        cutpoint = sort_table(cutpoint_i,1);
        index_l = find(X(:,feature) <= cutpoint);
        index_h = find(X(:,feature) > cutpoint);
        new_Y_l = Y(index_l',:);
        new_Y_h = Y(index_h',:);
        E_l = 0;
        E_h = 0;
        clear temp_table;
        temp_table = tabulate(new_Y_l);
        for i = 1:size(temp_table,1)
            temp_p = temp_table(i,3)/100;
            E_l = E_l - temp_p*log2(temp_p);%- (temp_table(i,3)/100)^2;
        end
        clear temp_table;
        temp_table = tabulate(new_Y_h);
        for i = 1:size(temp_table,1)
            temp_p = temp_table(i,3)/100;
            E_h = E_h - temp_p*log2(temp_p);%- (temp_table(i,3)/100)^2;
        end
        E_c = (size(new_Y_l,1)/size(Y,1))*E_l+(size(new_Y_h,1)/size(Y,1))*E_h;
        if E_c < min_E
            min_E = E_c;
            optimal_f = feature;
            optimal_c = cutpoint;
        end
    end
end
end