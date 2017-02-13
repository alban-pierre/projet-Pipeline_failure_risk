function [prediction_train, prediction_test] = kernel_ridge_regression(kernel, trainx, trainy, testx, hyp1)

    % Prediction based on kernel ridge regression

    % Dimensions :
    % Ntr : Number of training examples
    % Nte : Number of testing examples
    % Dx  : Dimension of examples
    % Dy  : Dimension of prediction
    
    % Input :
    % kernel : @(N1*D, N2*D) -> (N1*N2) : A kernel
    % trainx : (Ntr*Dx)                 : Training set
    % trainy : (Ntr*Dy)                 : Training output
    % testx  : (Nte*Dx)                 : Testing set
    % hyp1   : (1*1)                    : Hyperparameter lambda, set to 1 by default

    % Output :
    % prediction_train : (Ntr*Dy) : The training prediction
    % prediction_test  : (Nte*Dy) : The testing prediction
    
    assert(nargin >= 4, 'Error : kernel_ridge_regression requires at least 4 arguments : kernel, trainx, trainy and testx');
    assert(size(trainx,1) == size(trainy,1), 'trainx and trainy must contain the same number of examples');
    assert(size(trainx,2) == size(testx,2), 'trainx and testx examples must have the same dimension');

    if (nargin < 5)
        lambda = 1;
    else
        lambda = hyp1;
    end

    [Ntr, Dx] = size(trainx);

    ktr = kernel(trainx, trainx);
    assert(all(size(ktr) == [Ntr,Ntr]), 'Error : the kernel has dimension problems, it must be a function @(N1*D, N2*D) -> (N1*N2)');
    alpha = (ktr + lambda*Ntr*eye(Ntr))^(-1)*trainy;

    kte = kernel(trainx, testx);
    assert(all(size(kte) == [Ntr,Nte]), 'Error : the kernel has dimension problems, it must be a function @(N1*D, N2*D) -> (N1*N2)');
    
    prediction_train = ktr'*alpha;
    prediction_test = kte'*alpha;

end
