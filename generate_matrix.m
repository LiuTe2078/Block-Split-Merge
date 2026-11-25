function [A, Ustar, lambda] = generate_matrix(n, p, gap, mode)
%GENERATE_MATRIX SPD matrix with an exact leading block eigengap.
% [A,Ustar,lambda] = generate_matrix(n,p,gap,mode)
%   Eigenvalues are generated DIRECTLY on the 4x base scale.
%   lambda(p) = 8, lambda(p+1) = 8-gap.
%   mode='dense'    : explicit symmetric matrix A=U*diag(lambda)*U'.
%   mode='operator' : implicit orthogonal similarity using Givens layers.
%
    if nargin < 4 || isempty(mode), mode = 'dense'; end
    assert(isscalar(n) && n == floor(n) && n >= 3, 'n must be >=3.');
    assert(isscalar(p) && p == floor(p) && p >= 1 && p < n, '1<=p<n required.');
    scale = 4;
    floorEig = scale * 0.05;
    assert(isscalar(gap) && isfinite(gap) && gap > 0 && ...
        gap < scale*2-floorEig, 'Require 0 < gap < 7.8 for SPD construction.');

    if p == 1
        top = scale * 2.0;
    else
        top = scale * linspace(2.08, 2.0, p).';
    end
    boundary = top(end) - gap;
    assert(boundary > floorEig, 'Boundary must exceed the positive spectral floor.');

    tailCount = n - p - 1;
    tail = floorEig + (boundary-floorEig)*rand(tailCount,1);
    lambda = [top(:); boundary; sort(tail,'descend')];

    switch lower(mode)
        case 'dense'
            [U, ~] = qr(randn(n,n));
            A = (U .* (lambda.')) * U';
            A = (A + A')/2;
            Ustar = U(:,1:p);
        case 'operator'
            layers = max(10, 2*ceil(log2(n)));
            pair = cell(layers,1);
            cc = cell(layers,1);
            ss = cell(layers,1);
            for ell = 1:layers
                perm = randperm(n);
                pair{ell} = reshape(perm(1:2*floor(n/2)),2,[]);
                angles = 2*pi*rand(1,size(pair{ell},2));
                cc{ell} = cos(angles);
                ss{ell} = sin(angles);
            end
            A = @(X) apply_spd_implicit(X,lambda,pair,cc,ss);
            Ustar = apply_rotations(eye(n,p),pair,cc,ss,false);
        otherwise
            error('mode must be ''dense'' or ''operator''.');
    end
end

function Y = apply_spd_implicit(X,lambda,pair,cc,ss)
    Y = apply_rotations(X,pair,cc,ss,true);
    Y = bsxfun(@times,lambda,Y);
    Y = apply_rotations(Y,pair,cc,ss,false);
end

function Y = apply_rotations(X,pair,cc,ss,transposeFlag)
    Y = X;
    if transposeFlag
        order = numel(pair):-1:1;
    else
        order = 1:numel(pair);
    end
    for ell = order
        a = pair{ell}(1,:);
        b = pair{ell}(2,:);
        x = Y(a,:);
        y = Y(b,:);
        c = cc{ell}(:);
        s = ss{ell}(:);
        if transposeFlag
            Y(a,:) = bsxfun(@times,c,x) + bsxfun(@times,s,y);
            Y(b,:) = bsxfun(@times,-s,x) + bsxfun(@times,c,y);
        else
            Y(a,:) = bsxfun(@times,c,x) - bsxfun(@times,s,y);
            Y(b,:) = bsxfun(@times,s,x) + bsxfun(@times,c,y);
        end
    end
end
