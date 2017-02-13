function [mean_err, sorted_err] = abs_error(prediction, y)

    % Function that computes the absolute error of an algorithm

    % Dimensions :
    % N : Number of examples
    % D : Dimension of prediction

    % Input :
    % prediction : (N*D) : A prediction made by an algorithm
    % y          : (N*D) : The true values

    % Output :
    % mean_err   : (1*D) : The mean absolute error between prediction and y
    % sorted_err : (N*D) : All the absolute errors, sorted from the smallest to the highest

    assert(all(size(prediction) == size(y)), 'Error : To compute the auc error the prediction and the true values must have the same size');
    
    D = size(y,2);

    err = abs(prediction - y);
    mean_err = mean(err, 1);
    sorted_err = zeros(size(err));
    
    for i=1:D
        sorted_err(:,i) = sort(err(:,i));
    end

end
