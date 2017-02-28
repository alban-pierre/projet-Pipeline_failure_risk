function [NN, a] = feed_forward_several(NN, datax)

    % Function that computes the result of the neural network for one example

    % Dimensions :
    % N  : Number of examples in the batch
    % Dx : Dimension of examples
    % Dy : Dimension of output
    % Dc : Dimension of the before last layer
    
    % Input :
    % NN    : (structure) : The neural network, containing coefficients, some parameters, etc
    % datax : (N*Dx)      : Training set

    % Optionnal input :
    % dropout_ind : {1*NbrLayers}(Da*1) : The indexes to keep if we use dropout
    
    % Output :
    % NN    : (structure) : The neural network, containing coefficients, some parameters, etc
    % resl  : (N*Dy)      : The output of the neural network
    % resbl : (N*Dc)      : The output of the before last layer of the neural network

    sigmoid = @(x) (1./(1+exp(-x)));

    a{1} = datax';

    if (NN.activation_function == 1) % sigmoid
        for i=1:NN.nbr_layers-1
            a{i+1} = sigmoid(NN.w{i}*a{i} + NN.b{i});
        end
    else % ReLU
        for i=1:NN.nbr_layers-2
            a{i+1} = max(NN.w{i}*a{i} + NN.b{i}, 0);
        end
        i = NN.nbr_layers-1;
        a{i+1} = sigmoid(NN.w{i}*a{i} + NN.b{i});
    end
end
