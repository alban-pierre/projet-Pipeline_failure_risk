function [err, auc_curves] = auc_error(prediction, y)

    % Function that computes the auc error of an algorithm

    % Dimensions :
    % N : Number of examples
    % D : Dimension of prediction

    % Input :
    % prediction : (N*D) : A prediction made by an algorithm
    % y          : (N*D) : The true values

    % Output :
    % err        : (1*D) : The auc computed
    % auc_curves : {1*D} : The auc curves

    assert(all(size(prediction) == size(y)), 'Error : To compute the auc error the prediction and the true values must have the same size');
    
    D = size(y,2);
    auc_curves = {};
    err = zeros(1,D);
    
    for j=1:D
        [s, is] = sort(prediction(:,j), 'descend');
        truesorted = y(is,j);
        n = size(truesorted,1);
        cs = cumsum(truesorted,1);
        ics = (1:n)' - cs;
        curve = zeros(1,ics(end,1)+1);
        for i=1:n
            curve(1,ics(i,1)+1) = cs(i,1);
        end
        curve = curve / max(max(curve,[],2),1);
        auc_curves{j} = curve;
        err(1,j) = mean(curve);
    end
    
end
