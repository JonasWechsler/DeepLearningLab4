function Assignment4
    main();
end

function book_data = load(book_fname)
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c');
    fclose(fid);
end

function architecture = default_architecture()
    architecture = struct('m', 100, 'eta', 0.1, 'seq_length', 25);
end

function model = default_model(m, K)
    sig = 0.01;
    b = zeros(m, 1);
    c = zeros(K, 1);
    U = randn(m, K)*sig;
    W = randn(m, m)*sig;
    V = randn(K, m)*sig;
    model = struct('b', b, 'c', c, 'U', U, 'W', W, 'V', V);
end

function result = synthesize(RNN, h0, x0, n)
    % Equations 1-4
end

function main
    book_fname = 'data/goblet_book.txt';
    book_data = load(book_fname);
    book_chars = unique(book_data);
    
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');
    
    K = size(book_chars, 2);
    
    for idx = 1:K
        ch = book_chars(idx);
        char_to_ind(ch) = idx;
        ind_to_char(idx) = ch;
    end
end
