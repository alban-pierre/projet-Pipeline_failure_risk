function NN = create_a_NN(algo)

    % Function that creates a neural network, defines its parameters

    % Input :
    % algo   : (structure) : All parameters that defines an algorithm

    % Output :
    % NN : (structure) : The neural network, containing coefficients, some parameters, etc
    
    for i=1:size(algo.deep.sizes,2)-1
        NN.w{i} = randn(algo.deep.sizes(1,1+i), algo.deep.sizes(1,i))/1;
        NN.b{i} = randn(algo.deep.sizes(1,1+i),1)/1;
    end

    NN.sizes = algo.deep.sizes;

    NN.nbr_layers = size(NN.sizes,2);
    
    NN.dropout = algo.deep.dropout;

    NN.activation_function = algo.deep.activation_function;

end
