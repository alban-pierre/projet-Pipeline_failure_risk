function [prediction_train, prediction_test] = ridge_regression(trainx, trainy, testx, hyp1)

    % Prediction based on ridge regression

    % Dimensions :
    % Ntr : Number of training examples
    % Nte : Number of testing examples
    % Dx  : Dimension of examples
    % Dy  : Dimension of prediction
    
    % Input :
    % trainx : (Ntr*Dx) : Training set
    % trainy : (Ntr*Dy) : Training output
    % testx  : (Nte*Dx) : Testing set
    % hyp1   : (1*1)    : Hyperparameter lambda, set to 1 by default

    % Output :
    % prediction_train : (Ntr*Dy) : The training prediction
    % prediction_test  : (Nte*Dy) : The testing prediction
    
    assert(nargin >= 3, 'Error : ridge_regression requires at least 3 arguments : trainx, trainy and testx');
    assert(size(trainx,1) == size(trainy,1), 'trainx and trainy must contain the same number of examples');
    assert(size(trainx,2) == size(testx,2), 'trainx and testx examples must have the same dimension');
    
    
    if (nargin < 4)
        lambda = 1;
    else
        lambda = hyp1;
    end

    [Ntr, Dx] = size(trainx);

    if (Ntr > Dx)
        wrr = (trainx'*trainx + lambda*Ntr*eye(Dx))^(-1)*(trainx')*trainy;
    else
        wrr = trainx'*(trainx*(trainx') + lambda*Ntr*eye(Ntr))^(-1)*trainy;
    end
    
    prediction_train = trainx*wrr;
    prediction_test = testx*wrr;

end
