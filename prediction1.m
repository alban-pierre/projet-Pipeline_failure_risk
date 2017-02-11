function [prediction_train, prediction_test] = prediction1(trainx, trainy, testx)

    % Prediction based on ...

%    prediction_train = randi(2,n-testsize,2)-1;
%    prediction_test = randi(2,testsize,2)-1;

	sigma = 1;
	lambda = 0.1;

	if (0)
	d = sqdist(trainx', trainx');
    k = exp(-d/(2*sigma^2));
	alpha = ((k+lambda*eye(size(trainx,1)))^-1)*trainy;
	d = sqdist(trainx', testx');
    d = exp(-d/(2*sigma^2));
	pred = (alpha'*d)';
	end

	[n, d] = size(trainx);
	
	wrr = (trainx'*trainx + lambda*n*eye(d))^(-1)*(trainx')*trainy;

	prediction_train = trainx*wrr;
	prediction_test = testx*wrr;
end
