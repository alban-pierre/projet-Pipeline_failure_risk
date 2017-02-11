function [score, auc14, auc15] = compute_auc(prediction, y)

    % Function that computes the score of an algorithm

    % 2014
    [s, is] = sort(prediction(:,1), 'descend');
    truesorted = y(is,1);
    n = size(truesorted,1);
    cs = cumsum(truesorted(:,1),1);
    ics = (1:n)' - cumsum(truesorted(:,1),1);
    curve = zeros(1,ics(end,1)+1);
    for i=1:n
        curve(1,ics(i,1)+1) = cs(i,1);
    end
    curve = curve / max(max(curve,[],2),1);
    auc14 = curve;
    score(1,1) = mean(curve);
    
    %2015
    [s, is] = sort(prediction(:,2), 'descend');
    truesorted = y(is,2);
    n = size(truesorted,1);
    cs = cumsum(truesorted(:,1),1);
    ics = (1:n)' - cumsum(truesorted(:,1),1);
    curve = zeros(1,ics(end,1)+1);
    for i=1:n
        curve(1,ics(i,1)+1) = cs(i,1);
    end
    curve = curve / max(max(curve,[],2),1);
    auc15 = curve;
    score(2,1) = mean(curve);

    score = [0.6,0.4]*score;
    
end
