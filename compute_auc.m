function [score, auc14, auc15] = compute_auc(prediction, y)

    % Function that computes the score of an algorithm

    % 2014
    [s, is] = sort(prediction(:,1));
    truesorted = y(is,1);
    n = size(truesorted,1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    
    for i=1:n
        curve(1,i-sum(truesorted(1:i,:),1)) = sum(truesorted(1:i,:),1);
    end
    curve = curve / max(curve,[],2);
    auc14 = curve;
    score(1,1) = mean(curve);
	
    % 2015
    [s, is] = sort(prediction(:,2));
    truesorted = y(is,2);
    n = size(truesorted,1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    
    for i=1:n
        curve(1,i-sum(truesorted(1:i,:),1)) = sum(truesorted(1:i,:),1);
    end
    curve = curve / max(curve,[],2);
    auc15 = curve;
    score(2,1) = mean(curve);

    score = [0.6,0.4]*score;
end
