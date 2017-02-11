function plot_auc(auc14, auc15)

    % Plots the train and test auc of 2014 and 2015

    n = max(size(auc14,2), size(auc15,2));
    
    colors = rand(max(n,6), 3);
    colors = colors - repmat(max(0,sum(colors,2)-2)/2,1,3);
    colors(1,:) = [0,0,0];
    colors(2,:) = [1,0,0];
    colors(3,:) = [0,0,1];
    colors(4,:) = [0,1,0];
    colors(5,:) = [1,0,1];
    colors(6,:) = [0,1,1];


    figure; hold on;
    for i=1:size(auc14,2)
        if size(auc14{1,i} > 0)
            c = auc14{1,i};
            plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
        end
    end
    title('Training AUC for 2014');
    xlabel('False positive');
    ylabel('True positive');

    
    if (size(auc14,1) > 1)        
        figure; hold on;
        for i=1:size(auc14,2)
            if size(auc14{2,i} > 0)
                c = auc14{2,i};
                plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
            end
        end
        title('Testing AUC for 2014');
        xlabel('False positive');
        ylabel('True positive');
    end

    figure; hold on;
    for i=1:size(auc15,2)
        if size(auc15{1,i} > 0)
            c = auc15{1,i};
            plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
        end
    end
    title('Training AUC for 2015');
    xlabel('False positive');
    ylabel('True positive');

    if (size(auc15,1) > 1)        
        figure; hold on;
        for i=1:size(auc15,2)
            if size(auc15{2,i} > 0)
                c = auc15{2,i};
                plot((0:size(c,2)-1)/(size(c,2)-1), c, '-', 'color', colors(i,:));
            end
        end
        title('Testing AUC for 2015');
        xlabel('False positive');
        ylabel('True positive');
    end
    
end
