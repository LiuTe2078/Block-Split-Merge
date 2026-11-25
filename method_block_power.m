function [X,it,status,info] = method_block_power(Afun,Q0,tol,maxIter,opts)
% Block power iteration for the p largest eigenpairs.
% tol is the absolute Ritz residual tolerance.
% MVs counts applications of A to individual vector columns.

    if nargin < 5, opts = struct(); end %#ok<NASGU>
    [~,p] = size(Q0);
    if ~isscalar(tol) || ~isfinite(tol) || tol < 0
        error('BlockPower:tol','tol must be a nonnegative finite scalar.');
    end
    if ~isscalar(maxIter) || maxIter < 0 || maxIter ~= floor(maxIter)
        error('BlockPower:maxIter','maxIter must be a nonnegative integer.');
    end

    [X,~] = qr(Q0,0);
    info = struct('MVs',0, 'ResidualHistory',[], 'RitzValues',[]);

    % Initial Rayleigh--Ritz extraction and online residual check.
    AX = applyA(X);
    [X,AX,theta,res] = local_rr(X,AX,p,'largest');
    info.ResidualHistory = max(res);
    info.RitzValues = theta;
    it = 0;
    status = 'maxIter';
    if max(res) <= tol
        status = 'converged';
        return;
    end

    for k = 1:maxIter
        % Classical block power subspace update: range(A*X).
        % AX is cached from the preceding Rayleigh--Ritz extraction.
        [Q,~] = qr(AX,0);

        % One new A*block per iteration; cache its rotated version.
        AQ = applyA(Q);
        [X,AX,theta,res] = local_rr(Q,AQ,p,'largest');

        it = k;
        info.ResidualHistory(end+1,1) = max(res);
        info.RitzValues = theta;
        if max(res) <= tol
            status = 'converged';
            return;
        end
    end

    function Y = applyA(V)
        Y = Afun(V);
        info.MVs = info.MVs + size(V,2);
    end
end

function [X,AX,theta,residuals,R] = local_rr(V,AV,p,target)
% Rayleigh--Ritz extraction from an orthonormal basis.
if nargin<4 || isempty(target), target='largest'; end
m=size(V,2);
if p>m, error('rr_extract:dimension','Need at least p basis vectors.'); end
H=(V'*AV+AV'*V)/2;
[Y,D]=eig(H);
d=real(diag(D));
if strcmpi(target,'largest'), [~,idx]=sort(d,'descend');
elseif strcmpi(target,'smallest'), [~,idx]=sort(d,'ascend');
else, error('rr_extract:target','Use largest or smallest.'); end
idx=idx(1:p); Y=Y(:,idx); theta=d(idx);
X=V*Y; AX=AV*Y;
R=AX-bsxfun(@times,X,theta');
residuals=sqrt(sum(abs(R).^2,1)).';
end
