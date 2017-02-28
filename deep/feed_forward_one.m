function NN = feed_forward_one(NN, datax)

    % Function that computes the result of the neural network for one example

    % Dimensions :
    % D : Dimension of examples
    
    % Input :
    % NN    : (structure) : The neural network, containing coefficients, some parameters, etc
    % datax : (1*D)       : The example that we have to compute the output on
    
    % Output :
    % NN : (structure) : The neural network, containing coefficients, some parameters, etc

    sigmoid = @(x) (1./(1+exp(-x)));
    
    NN.a{1} = datax(1,:)';

    if (NN.activation_function == 1) % sigmoid
        for i=1:NN.nbr_layers-1
            NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
            NN.a{i+1} = sigmoid(NN.z{i});
        end
    else % ReLU
        for i=1:NN.nbr_layers-2
            NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
            NN.a{i+1} = max(NN.z{i},0);
        end
        i = NN.nbr_layers-1;
        NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
        NN.a{i+1} = sigmoid(NN.z{i});
    end
    
end
