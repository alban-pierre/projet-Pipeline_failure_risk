function NN = update_a_NN(NN, y, learn_rate)

    err = NN.a{NN.nbr_layers} - y;

    sigmoid = @(x) (1./(1+exp(-x)));

    dsigmoid = @(x) (sigmoid(x).*(1-sigmoid(x)));

    err = abs(NN.a{NN.nbr_layers}' - y)*[0.6;0.4];
	dCda = (NN.a{NN.nbr_layers}' - y).*[0.6,0.4];
	NN.dCdai = dCda;
	%dadb{NN.nbr_layers-1} = dsigmoid(NN.z{NN.nbr_layers-1});
    %a = NN.a{NN.nbr_layers}.*(1-NN.a{NN.nbr_layers});
    %assert(all(dadb{NN.nbr_layers-1} == a));

    %dadw = NN.a{NN.nbr_layers-1}'.*dadb{NN.nbr_layers-1};
    %dada = NN.w{NN.nbr_layers-1}.*dadb{NN.nbr_layers-1};

    

    % Computes derivatives layer by layer
    for i=NN.nbr_layers-1:-1:1
        dadb{i} = dsigmoid(NN.z{i});
        a = NN.a{i+1}.*(1-NN.a{i+1});
        assert(all(dadb{i} == a));
        
        %dadw{i} = NN.a{i}'.*dadb{i};
        dada{i} = NN.w{i}.*dadb{i};
    end

    % Computes derivatives of C
    for i=NN.nbr_layers-1:-1:1
		%size(dCda)
		%size(dadb{i})
		assert(size(dCda, 2) == size(dadb{i},1));
        dCdb{i} = dCda'.*dadb{i};
        dCdw{i} = (dCda'.*dadb{i})*(NN.a{i}');
        if (i>1)
            dCda = dCda*dada{i};
        end
    end

    % Update parameters
    for i=NN.nbr_layers-1:-1:1
        NN.b{i} = NN.b{i} - learn_rate*dCdb{i};
        NN.w{i} = NN.w{i} - learn_rate*dCdw{i};
		%size(NN.w{i})
		%size(dCdw{i})
		%size(NN.b{i})
		%size(dCdb{i})
	end
	NN.dCdw = dCdw;
	NN.dCdb = dCdb;
	NN.dada = dada;
	NN.dadb = dadb;
	NN.dCda = dCda;
end
