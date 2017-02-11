function [prediction_train, prediction_test] = prediction1(trainx, trainy, testx)

    % Prediction based on ...

    lambda = 0.1;

    [n, d] = size(trainx);
    
    wrr = (trainx'*trainx + lambda*n*eye(d))^(-1)*(trainx')*trainy;

    prediction_train = trainx*wrr;
    prediction_test = testx*wrr;

end
