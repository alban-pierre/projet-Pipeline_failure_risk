function [NN, res] = feed_forward_several(NN, datax)

    sigmoid = @(x) (1./(1+exp(-x)));

    %dsigmoid = @(x) (sigmoid(x).*(1-sigmoid(x)));

    res = zeros(size(datax,1),NN.sizes(1,end));
    
    for j=1:size(datax,1)
        NN.a{1} = datax(j,:)';
    
        for i=1:NN.nbr_layers-1
            NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
            NN.a{i+1} = sigmoid(NN.z{i});
        end

        res(j,:) = NN.a{NN.nbr_layers};
    end
    
end
