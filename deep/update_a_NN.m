function NN = update_a_NN(NN, algo, datax, datay, learn_rate)

    %err = NN.a{NN.nbr_layers} - y;

    sigmoid = @(x) (1./(1+exp(-x)));

    %dsigmoid = @(x) (sigmoid(x).*(1-sigmoid(x)));

    %err = abs(NN.a{NN.nbr_layers}' - y)*[0.6;0.4];
    for i=NN.nbr_layers-1:-1:1
        dCdb{i} = zeros(size(NN.b{i}));
        dCdw{i} = zeros(size(NN.w{i}));
    end

    
    for j=1:size(datax,1)
    
        NN = feed_forward_one(NN, datax(j,:));
        
        if (algo.deep.costfunction == 1)
            dCda = 2*(NN.a{NN.nbr_layers}' - datay(j,:)).*[0.6,0.4];
            for i=NN.nbr_layers-1:-1:1
                dadb{i} = NN.a{i+1}.*(1-NN.a{i+1});
                dada{i} = NN.w{i}.*dadb{i};
            end
            
            % Computes derivatives of C
            for i=NN.nbr_layers-1:-1:1
                assert(size(dCda, 2) == size(dadb{i},1));
                dCdb{i} = dCdb{i} + dCda'.*dadb{i};
                dCdw{i} = dCdw{i} + (dCda'.*dadb{i})*(NN.a{i}');
                if (i>1)
                    dCda = dCda*dada{i};
                end
            end
            
        else
            %dCda = (datay(j,:).*(1./(NN.a{NN.nbr_layers})') + (1-datay(j,:)).*(-1./(1-NN.a{NN.nbr_layers})')).*[0.6,0.4];
            for i=NN.nbr_layers-2:-1:1
                dadb{i} = NN.a{i+1}.*(1-NN.a{i+1});
                dada{i} = NN.w{i}.*dadb{i};
            end

            d = datay(j,:)' - NN.a{NN.nbr_layers};
            dCdb{NN.nbr_layers-1} = dCdb{NN.nbr_layers-1} + d;
            dCdw{NN.nbr_layers-1} = dCdw{NN.nbr_layers-1} + d*(NN.a{NN.nbr_layers-1}');
            dCda = d'*NN.w{NN.nbr_layers-1};
            
            % Computes derivatives of C
            for i=NN.nbr_layers-2:-1:1
                assert(size(dCda, 2) == size(dadb{i},1));
                dCdb{i} = dCdb{i} + dCda'.*dadb{i};
                dCdw{i} = dCdw{i} + (dCda'.*dadb{i})*(NN.a{i}');
                if (i>1)
                    dCda = dCda*dada{i};
                end
            end
 
        end
        %NN.dCdai = dCda;
        %dadb{NN.nbr_layers-1} = dsigmoid(NN.z{NN.nbr_layers-1});
        %a = NN.a{NN.nbr_layers}.*(1-NN.a{NN.nbr_layers});
        %assert(all(dadb{NN.nbr_layers-1} == a));
        
        %dadw = NN.a{NN.nbr_layers-1}'.*dadb{NN.nbr_layers-1};
        %dada = NN.w{NN.nbr_layers-1}.*dadb{NN.nbr_layers-1};

        

        % Computes derivatives layer by layer
%        for i=NN.nbr_layers-1:-1:1
            %dadb{i} = dsigmoid(NN.z{i});
            %a = NN.a{i+1}.*(1-NN.a{i+1});
            %assert(all(dadb{i} == a));
%            dadb{i} = NN.a{i+1}.*(1-NN.a{i+1});
            
            %dadw{i} = NN.a{i}'.*dadb{i};
%            dada{i} = NN.w{i}.*dadb{i};
%        end

        % Computes derivatives of C
%        for i=NN.nbr_layers-1:-1:1
            %size(dCda)
            %size(dadb{i})
%            assert(size(dCda, 2) == size(dadb{i},1));
%            dCdb{i} = dCdb{i} + dCda'.*dadb{i};
%            dCdw{i} = dCdw{i} + (dCda'.*dadb{i})*(NN.a{i}');
%            if (i>1)
%                dCda = dCda*dada{i};
%            end
%        end
    end

    for i=NN.nbr_layers-1:-1:1
        dCdb{i} = dCdb{i}./size(datax,1);
        dCdw{i} = dCdw{i}./size(datax,1);
    end
    
    % Update parameters
    for i=NN.nbr_layers-1:-1:1
        NN.b{i} = NN.b{i} + learn_rate*dCdb{i};
        NN.w{i} = NN.w{i} + learn_rate*dCdw{i};

        %dCdb
        %dCdw
        %size(NN.w{i})
        %size(dCdw{i})
        %size(NN.b{i})
        %size(dCdb{i})
    end
    %NN.dCdw = dCdw;
    %NN.dCdb = dCdb;
    %NN.dada = dada;
    %NN.dadb = dadb;
    %NN.dCda = dCda;
end
