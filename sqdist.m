function d = sqdist(a, b)

    % SQDIST - computes squared Euclidean distance matrix
    %          computes a rectangular matrix of pairwise distances
    % between points in A (given in columns) and points in B
    % NB: very fast implementation taken from Roland Bunschoten

    % Dimensions :
    % D  : Dimension of observed points
    % Na : Number of points of a
    % Nb : Number of points of b

    % Input :
    % a : (D*Na) : Points a
    % b : (D*Nb) : Points b

    % Output :
    % d : (Na*Nb) : Square Euclidean distance matrix
    
    
    aa = sum(a.*a,1);
    bb = sum(b.*b,1);
    ab = a'*b; 
    d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

end
