function create_submit_file(prediction_test)

    % Fonction that creates the file to submit on the data challenge site

    ma = max(prediction_test,[],1);
    mi = min(prediction_test,[],1);

    pred = (prediction_test-repmat(mi,9713,1)) ./ (repmat(ma-mi,9713,1));

    FID = fopen('../../submit/submit.csv', 'w+');

    fprintf(FID, 'Id;2014;2015\n');

    for i=1:9713
        fprintf(FID, '%d;', i+19427);
        fprintf(FID, '%.3f;%.3f\n', pred(i,1), pred(i,2));
    end
        
    fclose(FID);
    
end
