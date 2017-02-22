function NN = feed_forward_one(NN, datax)

    sigmoid = @(x) (1./(1+exp(-x)));

    %dsigmoid = @(x) (sigmoid(x).*(1-sigmoid(x)));

    %res = zeros(size(datax,1),NN.sizes(1,end));
    
    NN.a{1} = datax(1,:)';
    
    for i=1:NN.nbr_layers-1
        NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
        NN.a{i+1} = sigmoid(NN.z{i});
    end
    
end
