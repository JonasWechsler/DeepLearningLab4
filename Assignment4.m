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

function model = init_model(b,c,U,W,V)
    model = struct('b', b, 'c', c, 'U', U, 'W', W, 'V', V);
end

function model = default_model(m, K)
    sig = 0.01;
    b = zeros(m, 1);
    c = zeros(K, 1);
    U = randn(m, K)*sig;
    W = randn(m, m)*sig;
    V = randn(K, m)*sig;
    model = init_model(b,c,U,W,V);
end

function L = ComputeLoss(RNN, X, Y)
    [P, ~, ~] = evaluate(RNN, X);
    
    YTP = sum(Y'.*P',2);
    L = sum(-1*arrayfun(@log, YTP));
end

function grad_RNN = gradient(RNN, X, Y)
    [P, H, A] = evaluate(RNN, X);
    
    K = size(X, 1);
    tau = size(X, 2);
    m = size(RNN.V,2);
    
    grad_O = -(Y-P).';
    
    grad_P = zeros(size(Y.'));
    for t = 1:tau
        y = Y(:,t);
        p = P(:,t);
        grad_P(t,:) = -y.'/(y.'*p);
    end

    grad_H = zeros(tau,m);
    grad_H(tau,:) = grad_O(tau,:)*RNN.V;
    grad_A = zeros(tau,m);
    grad_A(tau,:) = grad_H(tau,:)*diag(1-tanh(A(:,tau)).^2);
    
    for t = tau-1:-1:1
        grad_H(t,:) = grad_O(t,:)*RNN.V + grad_A(t+1,:)*RNN.W;
        grad_A(t,:) = grad_H(t,:)*diag(1-tanh(A(:,t)).^2);
    end
    
    grad_V = grad_O.'*H.';
    grad_c = sum(grad_O.',2);
    
    assert(isequal(size(grad_V),size(RNN.V)));
    assert(isequal(size(grad_c),size(RNN.c)));
    
    grad_RNN = struct('V',grad_V,'c',grad_c);
    %default_model(m,K);
    %grad_RNN.V = grad_V;
end

function [P, H, A] = evaluate(RNN, X)
    m = size(RNN.U, 1);
    h = zeros(m, 1);
    tau = size(X, 2);
    P = zeros(size(X));
    H = zeros(m, tau);
    A = zeros(m, tau);
    for idx=1:tau
        x = X(:,idx);
        A(:, idx) = RNN.W*h + RNN.U*x + RNN.b;
        h = tanh(A(:, idx));
        H(:, idx) = h;
        o = RNN.V*h + RNN.c;
        P(:, idx) = soft_max(o);
    end
end

function Y = synthesize(RNN, h0, x, n)
    [K, ~] = size(x);
    Y = zeros(K, n);
    h = h0;
    
    for t = 1:n
        a = RNN.W*h + RNN.U*x + RNN.b;
        h = tanh(a);
        o = RNN.V*h + RNN.c;
        p = soft_max(o);
        
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        
        x = zeros(K,1);
        x(ii) = 1;
        Y(ii, t) = 1;
        assert(isequal(Y(:,t), x));
    end
    
    
    % Equations 1-4
end

function [X, Y] = init
    book_fname = 'data/goblet_book.txt';
    book_data = load(book_fname);
    book_chars = unique(book_data);
    
    arch = default_architecture();
    
    global char_to_ind;
    global ind_to_char;
    
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');

    K = size(book_chars, 2);

    for idx = 1:K
        ch = book_chars(idx);
        char_to_ind(ch) = idx;
        ind_to_char(idx) = ch;
    end
    
    X_chars = book_data(1:arch.seq_length);
    Y_chars = book_data(2:arch.seq_length+1);
    X = zeros(K, arch.seq_length);
    Y = zeros(K, arch.seq_length);
    
    for idx = 1:size(X_chars,2)
        ch_idx = char_to_ind(X_chars(idx));
        X(ch_idx, idx) = 1;
        ch_idy = char_to_ind(Y_chars(idx));
        Y(ch_idy, idx) = 1;
    end
end

function main
    [X, Y] = init();
    K = size(X, 1);
    
    arch = default_architecture();
    RNN = default_model(arch.m, K);
    
    global ind_to_char;
    global char_to_ind;
    
    assert(isequal(size(X), [K, arch.seq_length]));
    assert(isequal(size(Y), [K, arch.seq_length]));
    
    h0 = zeros(arch.m, 1);
    
    x = zeros(K, 1);
    x(3) = 1;
    n = 5;
    r = synthesize(RNN, h0, x, n);
    
    for idx = 1:size(r,2)
        disp(ind_to_char(find(r(:,idx))));
    end
    
    disp(ComputeLoss(RNN, X, Y));
    disp(gradient(RNN, X, Y));
    tester_main();
end

function tester_main
    [X, Y] = init();
    K = size(X, 1);
    arch = default_architecture();
    RNN = default_model(arch.m, K);
    
    analytical_grad = gradient(RNN, X, Y);
    numerical_grad = ComputeGradsNum(X, Y, RNN, 1e-5);
    max_diff(analytical_grad, numerical_grad);
    %disp(ComputeGradsNum(X, Y, RNN, 1e-5));
end

function mu = soft_max(eta)
    tmp = exp(eta);
    tmp(isinf(tmp)) = 1e100;
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);
end

function max_diff(A,B)
    for f = fieldnames(A)'
        a = A.(f{1});
        b = B.(f{1});
        d = max(abs(a(:)-b(:)));
        fprintf("%s: %d ",f{1},d);
    end
    fprintf("\n");
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    %hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(RNN_try, X, Y);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(RNN_try, X, Y);
        grad(i) = (l2-l1)/(2*h);
    end
end

