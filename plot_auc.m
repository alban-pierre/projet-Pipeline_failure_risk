function plot_auc(auc, nfig, titles)

    % Plots the train and test auc of 2014 and 2015

    % Dimensions :
    % N : Number of auc curves
    
    % Input :
    % auc    : ((1 or 2)*N) : Train (and test) auc curves
    % nfig   : ((1 or 2)*1) : The id of figures, set it to -1 to have a new figure
    % titles : {strings}    : Titles of plots
    
    if (nargin < 2)
        nfig = -1;
    end
    
    N = size(auc,2);
    
    colors = rand(max(N,6), 3);
    colors = colors - repmat(max(0,sum(colors,2)-2)/2,1,3);
    colors(1,:) = [0,0,0];
    colors(2,:) = [1,0,0];
    colors(3,:) = [0,0,1];
    colors(4,:) = [0,1,0];
    colors(5,:) = [1,0,1];
    colors(6,:) = [0,1,1];


    if (nfig < 0)
        figure;
    else
        figure(nfig(1,1));
    end
    hold on;

    for i=1:N
        if size(auc{1,i} > 0)
            c = auc{1,i};
            plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
        end
    end
    if (nargin < 3)
        title('Training AUC');
    else
        title(titles{1});
    end
    xlabel('False positive');
    ylabel('True positive');

    
    if (size(auc,1) > 1)
        
        if (nfig < 0)
            figure;
        else
            figure(nfig(end,1));
        end
        hold on;
        
        for i=1:N
            if size(auc{2,i} > 0)
                c = auc{2,i};
                plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
            end
        end
        if (nargin < 3)
            title('Testing AUC');
        else
            title(titles{2});
        end
        xlabel('False positive');
        ylabel('True positive');
    end

end
