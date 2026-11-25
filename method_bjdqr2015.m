function [X,it,status,info]=method_bjdqr2015(Afun,Q0,tol,maxIter,opts)
% Roehrig-Zoellner et al. (2015), Algorithms 1--2, specialized to real
% symmetric A, B=I, largest p eigenpairs. Sequential MATLAB implementation.
% Default HARD Schur locking follows the paper. Option softLocking=true
% replaces locking by rechecking all p Ritz residuals each iteration.
% Batched independent MINRES: each correction equation keeps its own shift,
% Krylov recurrence and stopping tolerance; operator products are grouped.
% New correction directions are orthogonalized as a block using pivoted QR.
% Adaptive inner tolerances are enabled by default; no preconditioner, SIMD,
% parallel sparse kernels or oracle spectrum are used.
if nargin<5||isempty(opts),opts=struct();end
[n,p]=size(Q0);
b=min(p,max(1,round(optn('blockSize',3))));
mmax=min(n,max(p+b+2,round(optn('mmax',max(4*p,p+4*b)))));
mmin=max(p+b,min(mmax-b,round(optn('mmin',max(2*p,p+b)))));
mmin=min(mmin,mmax-b);
innerTol=optn('innerTol',0.15);
innerMaxit=max(1,round(optn('innerMaxit',30)));
rt=optn('rankTol',1e-12);
clusterTol=optn('clusterTol',1e-8);
softLocking=logical(optn('softLocking',false));
adaptiveInner=logical(optn('adaptiveInner',true));
innerTolMin=optn('innerTolMin',min(0.01,innerTol));
innerTolScale=optn('innerTolScale',0.5);
if ~(innerTolMin>0 && innerTolMin<=innerTol)
    error('BJDQR:innerTolMin','Require 0 < innerTolMin <= innerTol.');
end
if ~(innerTolScale>0 && isfinite(innerTolScale))
    error('BJDQR:innerTolScale','Require a positive innerTolScale.');
end
if ~(innerTol>0 && innerTol<1),error('BJDQR:innerTol','Require 0<innerTol<1.');end
if mmin<p||mmax<mmin+b,error('BJDQR:dimensions','Invalid mmin/mmax.');end
info=struct('MVs',0,'ResidualHistory',[],'SearchDims',[], ...
    'LockedHistory',[],'Restarts',0,'InnerCalls',0,'InnerBatches',0, ...
    'InnerFlags',[],'InnerTargetTols',[], ...
    'InnerIters',[],'InnerRelRes',[],'RandomFallbacks',0, ...
    'Breakdowns',0,'SchurCandidates',[],'ProjectedDims',[], ...
    'RitzValues',[],'StatusDetail','');
% Algorithm 1 (3--6): single-vector Arnoldi start to mmin dimensions.
v=Q0(:,1);v=v/norm(v);W=v;AW=mult(v);
while size(W,2)<mmin
    z=AW(:,end);
    z=twopass(z,W);
    nz=norm(z);
    if nz<rt*max(1,norm(AW(:,end)))
        info.Breakdowns=info.Breakdowns+1;
        z=twopass(randn(n,1),W);nz=norm(z);
        if nz<=rt,break;end
        info.RandomFallbacks=info.RandomFallbacks+1;
    end
    q=z/nz;W(:,end+1)=q;AW(:,end+1)=mult(q);
end
H=(W'*AW+AW'*W)/2;
l=0;it=0;status='maxIter';X=zeros(n,p);
while true
    m=size(W,2);
    if m<p, status='subspace_breakdown';return;end
    % Algorithm 1 (8--10): Ritz/Schur of UNLOCKED trailing part.
    if softLocking,l=0;end
    hatl=min([m,l+2*b,p+b-1]);
    [Ya,Da]=eig((H(l+1:m,l+1:m)+H(l+1:m,l+1:m)')/2);
    [da,order]=sort(real(diag(Da)),'descend');Ya=Ya(:,order);
    Y=blkdiag(eye(l),Ya);
    Tsch=(Y'*H*Y);Tsch=(Tsch+Tsch')/2;
    d=real(diag(Tsch));
    % Algorithm 1 (11--14): Schur vector/A image and Schur residual.
    Q=W*Y;QA=AW*Y;
    rSch=zeros(m,1);rEig=zeros(m,1);
    for j=l+1:m
        rj=QA(:,j)-Q(:,1:j)*Tsch(1:j,j);
        rSch(j)=norm(rj);
        rEig(j)=norm(QA(:,j)-d(j)*Q(:,j));
    end
    % Algorithm 1 (15--16): for Hermitian clustered Ritz values, place
    % the more accurate ones earlier WITHIN the degenerate Ritz cluster.
    j=l+1;
    while j<=hatl
        k=j;
        while k<hatl && abs(d(k)-d(k+1)) <= ...
                clusterTol*max([1,abs(d(k)),abs(d(k+1))])
            k=k+1;
        end
        if k>j
            [~,ord]=sort(rSch(j:k),'ascend');
            perm=1:m;perm(j:k)=j-1+ord;
            Y=Y(:,perm);Tsch=Tsch(perm,perm);
            d=d(perm);Q=Q(:,perm);QA=QA(:,perm);
            rSch=rSch(perm);rEig=rEig(perm);
        end
        j=k+1;
    end
    % A conventional Ritz extraction is used for the common *online*
    % stopping rule and for returning the leading p actual Ritz vectors.
    [X,~,theta,res]=local_rr(W,AW,p,'largest');
    info.ResidualHistory(end+1,1)=max(res);
    info.SearchDims(end+1,1)=m;
    info.LockedHistory(end+1,1)=l;
    info.SchurCandidates(end+1,1)=hatl;
    info.RitzValues=theta;
    if max(res)<=tol
        status='converged';info.StatusDetail='All p leading Ritz residuals <= tol.';return;
    end
    if it>=maxIter
        status='maxIter';info.StatusDetail='Maximum outer expansions reached.';return;
    end
    if ~softLocking
        % Algorithm 2 (17--26): lock CONSECUTIVE converged Schur pairs,
        % move their Schur vectors to the beginning of the stored basis.
        ec=0;
        for j=l+1:min(p,hatl)
            if rSch(j)<=tol,ec=ec+1;else,break;end
        end
        if ec>0
            W=Q;AW=QA;H=Tsch;
            l=l+ec;
            m=size(W,2);
            Y=eye(m);Q=W;QA=AW;
            d=real(diag(H));
            if l>=p
                [X,~,~,res]=local_rr(W,AW,p,'largest');
                if max(res)<=tol,status='converged';else,status='locked_verification_failed';end
                return;
            end
            if l>=hatl
                % All candidates from this extraction locked. The next
                % iteration increases hatl using the new locked count.
                continue;
            end
        end
        % Algorithm 2 (26): effective number of correction systems.
        nCorr=min([b,hatl-l,p-l]);
        selected=(l+1):(l+nCorr);
    else
        % Optional soft locking: re-activate a pair if its residual grows.
        selected=find(rEig(1:p)>tol);
        selected=selected(1:min(b,numel(selected)));
        nCorr=numel(selected);
    end
    if isempty(selected)
        status='no_active_pairs';return;
    end
    % Algorithm 2 (28--31): Schur thick restart preserving locked basis.
    if m+nCorr>mmax
        nsave=min(mmin,m);
        if ~isequal(Y,eye(m))
            C=Y(:,1:nsave);W=W*C;AW=AW*C;H=C'*H*C;
        else
            W=W(:,1:nsave);AW=AW(:,1:nsave);H=H(1:nsave,1:nsave);
        end
        H=(H+H')/2;
        m=nsave;info.Restarts=info.Restarts+1;
        % Recompute Ritz/Schur information after possible reordering.
        [Ya,Da]=eig((H(l+1:m,l+1:m)+H(l+1:m,l+1:m)')/2);
        [da,ord]=sort(real(diag(Da)),'descend');Ya=Ya(:,ord);
        Y=blkdiag(eye(l),Ya);Q=W*Y;QA=AW*Y;
        Tsch=Y'*H*Y;d=real(diag(Tsch));
        if ~softLocking
            nCorr=min([b,p-l,m-l]);selected=(l+1):(l+nCorr);
        else
            [~,~,~,res]=local_rr(W,AW,p,'largest');
            rEig=zeros(m,1);
            for jj=1:min(p,m)
                rEig(jj)=norm(QA(:,jj)-d(jj)*Q(:,jj));
            end
            selected=find(rEig(1:p)>tol);
            selected=selected(1:min(b,numel(selected)));
            nCorr=numel(selected);
        end
    end
    % Algorithm 2 (33--34): protect locked Schur vectors and all current
    % correction vectors; include an entire adjacent multiplicity cluster.
    if softLocking
        % Soft-locked wanted vectors are retained, but not permanently frozen.
        protect=unique([find(rEig(1:p)<=tol).', selected(:).']);
        U=Q(:,protect);
        ltilde=max(protect);
    else
        ltilde=max(selected);
        while ltilde<m && abs(d(ltilde)-d(ltilde+1)) <= ...
                clusterTol*max([1,abs(d(ltilde)),abs(d(ltilde+1))])
            ltilde=ltilde+1;
        end
        U=Q(:,1:ltilde);
    end
    info.ProjectedDims(end+1,1)=size(U,2);
    % Assemble independent projected correction systems first.  All columns
    % share the same protection basis U, but have different shifts and RHS.
    RHS=zeros(n,0);
    shifts=zeros(0,1);
    tolerances=zeros(0,1);
    for j=selected(:).'
        q=Q(:,j);lam=d(j);
        if softLocking
            rhs=-(QA(:,j)-lam*q);
        else
            % Schur residual, including coupling with locked Schur vectors.
            rhs=-(QA(:,j)-Q(:,1:j)*Tsch(1:j,j));
        end
        rhs=proj(rhs,U);
        rhsNorm=norm(rhs);
        if ~(isfinite(rhsNorm) && rhsNorm>rt)
            info.Breakdowns=info.Breakdowns+1;
            continue;
        end
        target=innerTol;
        if adaptiveInner
            % Same residual/shift-based inexact-JD heuristic as the original
            % optional branch, now enabled by default, per correction system.
            target=min(innerTol,max(innerTolMin, ...
                innerTolScale*rhsNorm/max(1,abs(lam))));
        end
        RHS(:,end+1)=rhs; %#ok<AGROW>
        shifts(end+1,1)=lam; %#ok<AGROW>
        tolerances(end+1,1)=target; %#ok<AGROW>
    end

    % Independent MINRES short recurrences advance synchronously. At each
    % inner step a single mult() is applied to all still-active vectors.
    Cnew=zeros(n,0);
    if ~isempty(shifts)
        info.InnerBatches=info.InnerBatches+1;
        info.InnerCalls=info.InnerCalls+numel(shifts); % equation count
        [T,flags,relres,innerIters]=solve_batched_minres( ...
            RHS,U,shifts,tolerances,innerMaxit);
        info.InnerFlags=[info.InnerFlags;flags];
        info.InnerRelRes=[info.InnerRelRes;relres];
        info.InnerIters=[info.InnerIters;innerIters];
        info.InnerTargetTols=[info.InnerTargetTols;tolerances];

        % Discard invalid / tiny solutions BEFORE block orthogonalization.
        keep=false(1,size(T,2));
        for jj=1:size(T,2)
            nt=norm(T(:,jj));
            keep(jj)=all(isfinite(T(:,jj))) && ...
                nt>rt*max(1,norm(RHS(:,jj)));
        end
        info.Breakdowns=info.Breakdowns+sum(~keep);
        [Cnew,nRejected]=orth_block(T(:,keep),W,rt,nCorr);
        info.Breakdowns=info.Breakdowns+nRejected;
    end

    % Algorithm 2 (39): fill any missing directions with random vectors.
    % Their orthogonalization is batched too, with rank-revealing QR.
    for attempt=1:3
        missing=nCorr-size(Cnew,2);
        if missing<=0,break;end
        [extra,~]=orth_block(randn(n,missing),[W,Cnew],rt,missing);
        if ~isempty(extra)
            Cnew=[Cnew,extra]; %#ok<AGROW>
            info.RandomFallbacks=info.RandomFallbacks+size(extra,2);
        end
    end
    if isempty(Cnew),status='correction_breakdown';return;end
    % Algorithm 2 (40--42): cached A images, incremental projected H.
    AC=mult(Cnew);HC=W'*AC;
    HCC=(Cnew'*AC+AC'*Cnew)/2;
    H=[H,HC;HC',HCC]; %#ok<AGROW>
    W=[W,Cnew];AW=[AW,AC]; %#ok<AGROW>
    it=it+1;
end
    function B=mult(Z)
        B=Afun(Z);info.MVs=info.MVs+size(Z,2);
    end
    function v=optn(name,default)
        if isfield(opts,name)&&~isempty(opts.(name)),v=opts.(name);else,v=default;end
    end
    function OpZ=jdop_batch(Z,U0,shiftVec)
        % Different shifts, one matrix-multiple-vector product.
        PZ=proj(Z,U0);
        OpZ=mult(PZ);
        OpZ=OpZ-bsxfun(@times,PZ,reshape(shiftVec,1,[]));
        OpZ=proj(OpZ,U0);
    end

    function [Tsol,flags,relres,nits]=solve_batched_minres(RHS,U0,shiftVec,tolVec,maxit)
        % Independent unpreconditioned MINRES recurrences sharing only the
        % batched operator evaluation; NOT a single shared block Krylov space.
        % MINRES tolerates indefinite projected shifted symmetric systems.
        [nloc,nSys]=size(RHS);
        Tsol=zeros(nloc,nSys);
        bnorm=sqrt(sum(abs(RHS).^2,1)).';
        beta=bnorm;
        oldb=zeros(nSys,1);
        r1=zeros(nloc,nSys);
        r2=RHS;
        cs=-ones(nSys,1);
        sn=zeros(nSys,1);
        dbar=zeros(nSys,1);
        epsln=zeros(nSys,1);
        phibar=bnorm;
        w=zeros(nloc,nSys);
        w2=zeros(nloc,nSys);
        flags=ones(nSys,1);  % 0: converged, 1: maxit, -2: breakdown
        nits=zeros(nSys,1);
        active=bnorm>0 & isfinite(bnorm);
        flags(~active)=-2;
        for itn=1:maxit
            idx=find(active).';
            if isempty(idx),break;end
            V=bsxfun(@rdivide,r2(:,idx),beta(idx).');
            AVbat=jdop_batch(V,U0,shiftVec(idx));
            for jj=1:numel(idx)
                sys=idx(jj);
                vj=V(:,jj);
                yj=AVbat(:,jj);
                if nits(sys)>0
                    if oldb(sys)<=0
                        flags(sys)=-2;active(sys)=false;continue;
                    end
                    yj=yj-(beta(sys)/oldb(sys))*r1(:,sys);
                end
                alpha=real(vj'*yj);
                yj=yj-(alpha/beta(sys))*r2(:,sys);
                if any(~isfinite(yj))
                    flags(sys)=-2;active(sys)=false;continue;
                end
                r1(:,sys)=r2(:,sys);
                r2(:,sys)=yj;
                oldb(sys)=beta(sys);
                beta(sys)=norm(yj);

                % MINRES plane rotations for the tridiagonal Lanczos system.
                oldeps=epsln(sys);
                delta=cs(sys)*dbar(sys)+sn(sys)*alpha;
                gbar=sn(sys)*dbar(sys)-cs(sys)*alpha;
                epsln(sys)=sn(sys)*beta(sys);
                dbar(sys)=-cs(sys)*beta(sys);
                gamma=hypot(gbar,beta(sys));
                if ~(isfinite(gamma) && gamma>realmin('double'))
                    flags(sys)=-2;active(sys)=false;continue;
                end
                cs(sys)=gbar/gamma;
                sn(sys)=beta(sys)/gamma;
                phi=cs(sys)*phibar(sys);
                phibar(sys)=sn(sys)*phibar(sys);
                wnew=(vj-oldeps*w2(:,sys)-delta*w(:,sys))/gamma;
                w2(:,sys)=w(:,sys);
                w(:,sys)=wnew;
                Tsol(:,sys)=Tsol(:,sys)+phi*wnew;
                nits(sys)=nits(sys)+1;
                if abs(phibar(sys))<=tolVec(sys)*bnorm(sys)
                    flags(sys)=0;active(sys)=false;
                elseif beta(sys)<=realmin('double')
                    flags(sys)=-2;active(sys)=false;
                end
            end
        end
        % True projected-system residuals, not only recurrence estimates.
        % One grouped matrix operation checks all computed corrections.
        residual=RHS-jdop_batch(Tsol,U0,shiftVec);
        relres=sqrt(sum(abs(residual).^2,1)).'./max(bnorm,realmin('double'));
        flags(~isfinite(relres))=-2;
        flags(isfinite(relres) & relres<=tolVec*(1+1e-10))=0;
        flags(flags==0 & relres>tolVec*(1+1e-10))=1;
    end
end
function z=proj(z,U)
if ~isempty(U),z=z-U*(U'*z);end
end
function z=twopass(z,W)
if isempty(W),return;end
for pass=1:2,z=z-W*(W'*z);end
end
function [Qnew,nRejected]=orth_block(Z,B,rankTol,maxKeep)
% Two-pass BLOCK projection against B followed by pivoted Householder QR.
% B is an orthonormal locked/search basis. QR pivoting reveals rank of Z.
[n,nCols]=size(Z);
Qnew=zeros(n,0);
if nCols==0,nRejected=0;return;end
valid=all(isfinite(Z),1);
Z=Z(:,valid);
if isempty(Z),nRejected=nCols;return;end
for pass=1:2
    if ~isempty(B),Z=Z-B*(B'*Z);end
end
% With three outputs MATLAB QR selects column pivoting (rank revealing).
[Q,R,~]=qr(Z,0);
d=abs(diag(R));
if isempty(d)
    r=0;
else
    r=sum(d>rankTol*max(1,norm(Z,'fro')));
end
r=min(r,maxKeep);
if r>0
    Qnew=Q(:,1:r);
    % Correct the small loss of orthogonality to the existing basis.
    if ~isempty(B),Qnew=Qnew-B*(B'*Qnew);end
    [Qnew,R2]=qr(Qnew,0);
    r2=sum(abs(diag(R2))>rankTol);
    Qnew=Qnew(:,1:r2);
end
nRejected=nCols-size(Qnew,2);
end

function [X,AX,theta,residuals,R] = local_rr(V,AV,p,target)
% Rayleigh--Ritz extraction from an orthonormal basis with cached A*V.
if nargin<4 || isempty(target), target='largest'; end
m=size(V,2);
if p>m, error('local_rr:dimension','Need at least p basis vectors.'); end
H=(V'*AV+AV'*V)/2;
[Y,D]=eig(H);
d=real(diag(D));
if strcmpi(target,'largest'), [~,idx]=sort(d,'descend');
elseif strcmpi(target,'smallest'), [~,idx]=sort(d,'ascend');
else, error('local_rr:target','Use largest or smallest.'); end
idx=idx(1:p); Y=Y(:,idx); theta=d(idx);
X=V*Y; AX=AV*Y;
R=AX-bsxfun(@times,X,theta');
residuals=sqrt(sum(abs(R).^2,1)).';
end
