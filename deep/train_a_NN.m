function NN = train_a_NN(NN, algo, trainx, trainy, learn_rate)

    % Function that train the neural network, every sample is used almost once

    % Dimensions :
    % N  : Number of training examples
    % Dx : Dimension of examples
    % Dy : Dimension of prediction

    % Input :
    % NN         : (structure) : The neural network, containing coefficients, some parameters, etc
    % algo       : (structure) : All parameters that defines an algorithm
    % trainx     : (N*Dx)      : Training set
    % trainy     : (N*Dy)      : Training output
    % learn_rate : (1*1)       : The learning rate of our neural network

    % Output :
    % NN : (structure) : The neural network, containing coefficients, some parameters, etc

    batchsize = algo.deep.batchsize;

    if (algo.deep.uniformbatch == 1)
        % We take samples uniformly in the data
        for j=batchsize:batchsize:size(trainx,1)
            NN = update_a_NN(NN, algo, trainx(j-batchsize+1:j,:), trainy(j-batchsize+1:j,:), learn_rate);
        end
    elseif (algo.deep.uniformbatch == 0)
        % We take samples so that each output as the same probability
        u = unique(trainy, 'rows');
        g = sqdist(trainy', u') == 0;
        ny = size(g,2);
        sg = sum(g,1);
        bg = randi(ny, 1, batchsize);
        sbg = sum(bg == (1:ny)', 2);
        r = zeros(batchsize, 1);
        ri = 0;
        for i=1:ny
            li{i} = 1:size(trainy,1);
            li{i} = li{i}(1,g(:,i) == 1);
            %r(ri+1:ri+sbg(i,1),1) = li{i}(1,randi(sg(1,i), 1, sbg(i,1)))';
            ri = ri+sbg(i,1);
        end

        for j=1:size(trainx,1)/batchsize
            r = zeros(batchsize, 1);
            ri = 0;
            for i=1:ny
                r(ri+1:ri+sbg(i,1),1) = li{i}(1,randi(sg(1,i), 1, sbg(i,1)))';
                ri = ri+sbg(i,1);
            end
            r = r(randperm(batchsize),:);
            NN = update_a_NN(NN, algo, trainx(r,:), trainy(r,:), learn_rate);
        end
    else
        [N, Dy] = size(trainy);
        minority = any(trainy ~= zeros(1,Dy),2);
        for j=1:(N/batchsize)
            keep = rand(N,1) - minority < 1/algo.deep.uniformbatch;
            r = ((1:N)(1,keep))';
            if (size(r,1) < batchsize)
                printf(2, 'Warning : smaller batchsize taken due to non uniform sampling');
                r = r(randperm(size(r,1))',1);
            else
                r = r((randperm(size(r,1))(1, 1:batchsize))',1);
            end
            NN = update_a_NN(NN, algo, trainx(r,:), trainy(r,:), learn_rate);
        end
    end
    
end
