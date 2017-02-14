function [prediction_test, err_train, sorted_err_train] = prediction(algo, trainx, trainy, testx)

    % Function that makes only one prediction, and return its error

    % Dimensions :
    % Ntr : Number of training examples
    % Nte : Number of testing examples
    % Dx  : Dimension of examples
    % Dy  : Dimension of prediction
    
    % Input :
    % algo   : .(str)   : All parameters that defines an algorithm
    % trainx : (Ntr*Dx) : Training set
    % trainy : (Ntr*Dy) : Training output
    % testx  : (Nte*Dx) : Testing set

    % Output :
    % prediction_test  : (Nte*Dx) : Testing output prediction
    % err_train        : (1*Dy)   : The training mean error between the prediction and trainy
    % sorted_err_train : (Ntr*Dy) : All the training errors, sorted from the smallest to the highest
    
    % Choices of kernels
    if (algo.kernel == 0)
        kernel = @(x1, x2) (x1*x2');
    elseif (algo.kernel == 1)
        kernel = @(x1, x2) gaussian_kernel(x1, x2, algo.kernel_hyp);
    elseif (algo.kernel == 2)
        kernel = @(x1, x2) laplacian_kernel(x1, x2, algo.kernel_hyp);
    end
        
    % Choices of regression algorithms
    if (algo.regression == 0)
        prediction_train = randi(2,n-testsize,2)-1;
        prediction_test = randi(2,testsize,2)-1;
    elseif (algo.regression == 1)
        [prediction_train, prediction_test] = ridge_regression(trainx, trainy, testx, algo.regr_hyp);
    elseif (algo.regression == 2)
        [prediction_train, prediction_test] = kernel_ridge_regression(kernel, trainx, trainy, testx, algo.regr_hyp);
    end
    
    % Choices of errors
    if (algo.error == 0)
        [err_train, sorted_err_train] = auc_error(prediction_train, trainy);
    elseif (algo.error == 1)
        [err_train, sorted_err_train] = abs_error(prediction_train, trainy);
    elseif (algo.error == 2)
        [err_train, sorted_err_train] = square_error(prediction_train, trainy);
    end

end
