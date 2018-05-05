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

function model = zero_model(m, K)
    b = zeros(m, 1);
    c = zeros(K, 1);
    U = zeros(m, K);
    W = zeros(m, m);
    V = zeros(K, m);
    model = init_model(b,c,U,W,V);
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

function L = ComputeLoss(RNN, X, Y, hprev)
    if ~exist('hprev','var')
        [P, ~, ~] = evaluate(RNN, X);
    else
        [P, ~, ~] = evaluate(RNN, X, hprev);
    end
    
    YTP = sum(Y'.*P',2);
    L = sum(-1*arrayfun(@log, YTP));
end

function [grad_RNN, hprev] = gradient(RNN, X, Y, h)
    [P, H, A] = evaluate(RNN, X, h);
    
    K = size(X, 1);
    tau = size(X, 2);
    m = size(RNN.V,2);
    
    hprev = H(:,tau);
    
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
    
    H_minus_one = [zeros(m, 1) H(:,1:tau-1)];
    grad_W = grad_A.'*H_minus_one.';
    grad_U = grad_A.'*X.';
    grad_b = sum(grad_A.',2);
    
    grad_RNN = init_model(grad_b, grad_c, grad_U, grad_W, grad_V);
end

function grad_RNN = clip(grad_RNN)
    for f = fieldnames(grad_RNN)'
        grad_RNN.(f{1}) = max(min(grad_RNN.(f{1}), 5), -5);
    end
end

function [P, H, A] = evaluate(RNN, X, h)
    m = size(RNN.U, 1);
    
    if ~exist("h", "var")
        h = zeros(m, 1);
    end

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

function Y = synthesize(RNN, n, x, h0)
    if ~exist('h0', 'var')
        h0 = zeros(size(RNN.b));
    end
    
    if ~exist('x', 'var')
        x = zeros(size(RNN.c));
        x(1) = 1;
    end
    
    [K, ~] = size(x);
    Y = zeros(K, n);
    h = h0;
    
    for t = 1:n
        [p, h, ~] = evaluate(RNN, x, h);
        
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        
        x = zeros(K,1);
        x(ii) = 1;
        Y(ii, t) = 1;
    end
end

function [RNN, M] = adagrad(RNN, M, RNN_grad, eta)
    for f_cell = fieldnames(RNN_grad)'
        f = f_cell{1};
        g = RNN_grad.(f);
        M.(f) = M.(f) + g.^2;
        denom = sqrt(M.(f) + eps).^-1;
        RNN.(f) = RNN.(f) - eta*denom.*g;
    end 
end

function [X, Y] = one_hot_encode(X_chars, Y_chars)
    global char_to_ind;
    K = size(char_to_ind,1);
    
    seq_length = size(X_chars,2);
    X = zeros(K, seq_length);
    Y = zeros(K, seq_length);
    
    for idx = 1:size(X_chars,2)
        ch_idx = char_to_ind(X_chars(idx));
        X(ch_idx, idx) = 1;
        ch_idy = char_to_ind(Y_chars(idx));
        Y(ch_idy, idx) = 1;
    end
end

function RNN = epoch(RNN, arch, book_data)
    global ind_to_char;
    e = 1;
    hprev = zeros(arch.m, 1);
    smooth_loss = 0;
    [m, K] = size(RNN.U);
    assert(arch.m == m);
    M = zero_model(m,K);
    
    iteration = 0;
    while e <= length(book_data)-arch.seq_length
        en = e + arch.seq_length;
        X_chars = book_data(e:en-1);
        Y_chars = book_data(e+1:en);
        [X, Y] = one_hot_encode(X_chars, Y_chars);
        
        [grad_RNN, hprev] = gradient(RNN, X, Y, hprev);
        grad_RNN = clip(grad_RNN);
        [RNN, M] = adagrad(RNN, M, grad_RNN, arch.eta);
        
        loss = ComputeLoss(RNN, X, Y);
        smooth_loss = 0.999*smooth_loss + 0.001*loss;
        
        if mod(iteration, 100) == 0
            fprintf("%d, %d\n", iteration, loss);
        end
        
        if mod(iteration, 500) == 0
                r = synthesize(RNN, 200, X(:, 1), hprev);
    
                for idx = 1:size(r,2)
                    fprintf('%s', ind_to_char(find(r(:,idx))));
                end
                fprintf('\n');
        end
        
        iteration = iteration + 1;
        e = e + arch.seq_length;
    end
end

function [book_data, K] = init(arch)
    book_fname = 'data/goblet_book.txt';
    book_data = load(book_fname);
    book_chars = unique(book_data);
    
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
end

function main
    arch = default_architecture();
    [book_data, K] = init(arch);
    
    RNN = default_model(arch.m, K);
    RNN = epoch(RNN, arch, book_data);
    RNN = epoch(RNN, arch, book_data);
        
    %{
    X_chars = book_data(1:arch.seq_length);
    Y_chars = book_data(2:arch.seq_length+1);
    [X, Y] = one_hot_encode(X_chars, Y_chars);
    
    global ind_to_char;
    global char_to_ind;
    
    assert(isequal(size(X), [K, arch.seq_length]));
    assert(isequal(size(Y), [K, arch.seq_length]));
    
    h0 = zeros(arch.m, 1);
    
    x = zeros(K, 1);
    x(3) = 1;
    n = 5;
    r = synthesize(RNN, n, x, h0);
    
    for idx = 1:size(r,2)
        disp(ind_to_char(find(r(:,idx))));
    end
    
    disp(ComputeLoss(RNN, X, Y));
    disp(gradient(RNN, X, Y));
    tester_main();
    %}
end

function tester_main
    arch = default_architecture();
    arch.m = 5;
    [X, Y] = init(arch);
    K = size(X, 1);
    
    for t=1:10
        RNN = default_model(arch.m, K);

        [analytical_grad, ~] = gradient(RNN, X, Y);
        numerical_grad = ComputeGradsNum(X, Y, RNN, 1e-4);
        max_diff(analytical_grad, numerical_grad);
    end
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
        %disp('Computing numerical gradient for')
        %disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(RNN_try, X, Y, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(RNN_try, X, Y, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end

