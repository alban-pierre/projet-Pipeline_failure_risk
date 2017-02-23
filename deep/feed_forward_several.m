function [NN, resl, resbl] = feed_forward_several(NN, datax)

    % Function that computes the result of the neural network for one example

    % Dimensions :
    % N  : Number of examples in the batch
    % Dx : Dimension of examples
    % Dy : Dimension of output
    % Dc : Dimension of the before last layer
    
    % Input :
    % NN    : (structure) : The neural network, containing coefficients, some parameters, etc
    % datax : (N*Dx)      : Training set
    
    % Output :
    % NN    : (structure) : The neural network, containing coefficients, some parameters, etc
    % resl  : (N*Dy)      : The output of the neural network
    % resbl : (N*Dc)      : The output of the before last layer of the neural network

    sigmoid = @(x) (1./(1+exp(-x)));

    a{1} = datax';
    for i=1:NN.nbr_layers-1
        a{i+1} = sigmoid(NN.w{i}*a{i} + NN.b{i});
    end
    resl = a{NN.nbr_layers}';
    resbl = a{NN.nbr_layers-1}';
    
end
