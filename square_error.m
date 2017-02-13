function [mean_err, sorted_err] = square_error(prediction, y)

    % Function that computes the square error of an algorithm

    % Dimensions :
    % N : Number of examples
    % D : Dimension of prediction

    % Input :
    % prediction : (N*D) : A prediction made by an algorithm
    % y          : (N*D) : The true values

    % Output :
    % mean_err   : (1*D) : The mean square error between prediction and y
    % sorted_err : (N*D) : All the square errors, sorted from the smallest to the highest

    assert(all(size(prediction) == size(y)), 'Error : To compute the auc error the prediction and the true values must have the same size');
    
    D = size(y,2);

    err = (prediction - y).^2;
    mean_err = mean(err, 1);
    sorted_err = zeros(size(err));
    
    for i=1:D
        sorted_err(:,i) = sort(err(:,i));
    end

end
