function NN = train_a_NN(NN, algo, datax, datay, learn_rate)


    batchsize = algo.deep.batchsize;

    if (algo.deep.uniformbatch)
		% We take samples uniformly in the data
        for j=batchsize:batchsize:size(datax,1)
            NN = update_a_NN(NN, algo, datax(j-batchsize+1:j,:), datay(j-batchsize+1:j,:), learn_rate);
        end
    else
		% We take samples so that each output as the same probability
        u = unique(datay, 'rows');
        g = sqdist(datay', u') == 0;
        ny = size(g,2);
        sg = sum(g,1);
        bg = randi(ny, 1, batchsize);
        sbg = sum(bg == (1:ny)', 2);
        r = zeros(batchsize, 1);
        ri = 0;
        for i=1:ny
            li{i} = 1:size(datay,1);
            li{i} = li{i}(1,g(:,i) == 1);
            %r(ri+1:ri+sbg(i,1),1) = li{i}(1,randi(sg(1,i), 1, sbg(i,1)))';
            ri = ri+sbg(i,1);
        end

        for j=1:size(datax,1)/batchsize
            r = zeros(batchsize, 1);
            ri = 0;
            for i=1:ny
                r(ri+1:ri+sbg(i,1),1) = li{i}(1,randi(sg(1,i), 1, sbg(i,1)))';
                ri = ri+sbg(i,1);
            end
            r = r(randperm(batchsize),:);
            %datay(r,:)
            %fprintf(1, '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n');
            NN = update_a_NN(NN, algo, datax(r,:), datay(r,:), learn_rate);
        end
    end
    
end
