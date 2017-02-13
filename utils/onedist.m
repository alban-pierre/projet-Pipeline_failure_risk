function dist = onedist(a, b)

    % onedist - computes the norm one distance matrix
    %          computes a rectangular matrix of pairwise distances
    % between points in A (given in columns) and points in B
    
    % Dimensions :
    % D  : Dimension of observed points
    % Na : Number of points of a
    % Nb : Number of points of b

    % Input :
    % a : (D*Na) : Points a
    % b : (D*Nb) : Points b

    % Output :
    % dist : (Na*Nb) : Norm one distance matrix

    Na = size(a,2);
    Nb = size(b,2);
    dist = zeros(Na, Nb);
    D = size(b,1);

    for i = 1:D
        dist = dist + abs(repmat(a(i,:), Nb, 1)' - repmat(b(i,:), Na, 1));
    end
    
end
