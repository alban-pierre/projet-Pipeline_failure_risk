function [score, auc14, auc15] = compute_auc(prediction, y)

    % Function that computes the score of an algorithm

    % 2014
	tt = time();
    [s, is] = sort(prediction(:,1));
    truesorted = y(is,1);
    n = size(truesorted,1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    
    for i=1:n
        curve(1,i-sum(truesorted(1:i,:),1)) = sum(truesorted(1:i,:),1);
    end
    curve = curve / max(max(curve,[],2),1);
    auc14 = curve;
    score(1,1) = mean(curve);
	time() - tt
    
    % 2015
	tt = time();
    [s, is] = sort(prediction(:,2));
    truesorted = y(is,2);
    n = size(truesorted,1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    
    for i=1:n
        curve(1,i-sum(truesorted(1:i,:),1)) = sum(truesorted(1:i,:),1);
    end
    curve = curve / max(max(curve,[],2),1);
    auc15 = curve;
    score(2,1) = mean(curve);

    score = [0.6,0.4]*score;
	time() - tt

    % 2014
	tt = time();
    [s, is] = sort(prediction(:,1));
    truesorted = y(is,1);
    n = size(truesorted,1);
    cs = cumsum(truesorted(:,1),1);
    ics = (1:size(cs,1))' - cumsum(truesorted(:,1),1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    for i=1:n
        curve(1,ics(i,1)) = cs(i,1);
    end
    curve = curve / max(max(curve,[],2),1);
	abs(sum(auc14 - curve))
    auc14 = curve;
    score(1,1) = mean(curve);
	time() - tt

	%2015
	tt = time();
    [s, is] = sort(prediction(:,2));
    truesorted = y(is,2);
    n = size(truesorted,1);
    cs = cumsum(truesorted(:,1),1);
    ics = (1:size(cs,1))' - cumsum(truesorted(:,1),1);
    s = n-sum(truesorted, 1);
    curve = zeros(1,s);
    for i=1:n
        curve(1,ics(i,1)) = cs(i,1);
    end
    curve = curve / max(max(curve,[],2),1);
    abs(sum(auc15 - curve))
	auc15 = curve;
	score(2,1) = mean(curve);
	score = [0.6,0.4]*score;
	time() - tt
	
end
