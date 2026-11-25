function [X,it,status,info] = method_bsm(Afun,Q0,tol,maxIter,opts)
% Ritz-residual BSM for the p largest eigenpairs.
% tol is the absolute Ritz residual tolerance.
% MVs counts applications of A to individual vector columns.

if nargin < 5 || isempty(opts), opts = struct(); end
if isfield(opts,'epsSigma')
    epsSigma = opts.epsSigma;
else
    epsSigma = 0.02;
end
if ~(isscalar(epsSigma) && isreal(epsSigma) && ...
        epsSigma > 0 && epsSigma < 1)
    error('BSM:epsSigma','0 < epsSigma < 1 required.');
end

[~,p] = size(Q0);
[X,~] = qr(Q0,0);
info = struct('MVs',0,'ResidualHistory',[], ...
              'TauHistory',[],'RitzValues',[]);

% Initial Ritz extraction: its AX is cached for the next filter step.
AX = applyA(X);
[X,AX,theta,res,E] = local_rr(X,AX,p,'largest');
info.ResidualHistory = max(res);
info.RitzValues = theta;
it = 0;
status = 'maxIter';
if max(res) <= tol
    status = 'converged';
    return;
end

for k = 1:maxIter
    % theta and AX are supplied by the previous Ritz extraction.
    d = real(theta(:));
    if any(d <= 0) || any(~isfinite(d))
        status = 'nonpositive_ritz';
        return;
    end
    invSqrtD = 1 ./ sqrt(d);
    tt = invSqrtD * invSqrtD.';

    % E is cached from preceding Rayleigh--Ritz. No repeat residual.
    % In exact arithmetic X'*E = 0.
    F = E' * E;
    F = (F + F') / 2;

    % Original C = Theta^(-1/2)*(AX'*AX)*Theta^(-1/2),
    % but AX'*AX = diag(d.^2) + F in exact arithmetic.
    C = diag(d) + tt .* F;
    C = (C + C') / 2;
    alpha = min(real(eig(C)));

    % One A*block multiplication to form A*E DIRECTLY.
    % In exact arithmetic AE = A2X - AX*diag(d).
    % Directly applying A to E avoids near-cancellation in this difference.
    AE = applyA(E);

    % Original M = T*(AX'*A2X-Q2*(Theta\Q2))*T.
    % Equivalent residual formula (exact arithmetic):
    % M = T*(E'*A*E-F*Theta^(-1)*F)*T.
    invDF = bsxfun(@rdivide,F,d);       % diag(d)^(-1)*F
    M = tt .* (E' * AE - F * invDF);
    M = (M + M') / 2;
    [Z,Dm] = eig(M);
    [mu,idx] = max(real(diag(Dm)));
    if ~isfinite(mu)
        status = 'invalid_auxiliary_ritz';
        return;
    end

    % ||AX||_F^2 = sum(d.^2)+trace(F), since X'*E=0.
    % This is the original near-invariance safeguard in exact arithmetic.
    axNorm2 = sum(d.^2) + real(trace(F));
    if mu <= 1e-28 * max(1,axNorm2)
        tau = 0;
    else
        z = Z(:,idx);
        u = invSqrtD .* z;

        % r = A2X*u - AX*(K*u), K = Theta^(-1)*(AX'*AX).
        % With K = diag(d)+diag(1./d)*F and AE=A2X-AX*diag(d):
        r = AE*u - AX*((F*u)./d);
        corner = norm(r,2)^2 / mu;

        H = [C, sqrt(mu)*z; sqrt(mu)*z', corner];
        H = (H + H') / 2;
        beta = sort(real(eig(H)),'descend');
        gamma = beta(p) + beta(p+1) - alpha;
        tau = alpha + gamma - max(alpha,gamma/(1-epsSigma));
    end

    info.TauHistory(end+1,1) = tau;

    % EXACT same quadratic filter, now using only cached AX and AE:
    % A*(A-tau*I)*X = A*(AX-X*diag(d)) + AX*(diag(d)-tau*I).
    Yfilter = AE + AX .* (d.' - tau);
    [Q,~] = qr(Yfilter,0);

    % Ritz extraction after every filter update: next iterate consists of
    % individual Ritz vectors (not merely an arbitrary subspace basis).
    % Cache A*X through the already computed A*Q and Ritz rotation.
    AQ = applyA(Q);
    [X,AX,theta,res,E] = local_rr(Q,AQ,p,'largest');

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
H=V'*AV;
H=(H+H')/2;
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
