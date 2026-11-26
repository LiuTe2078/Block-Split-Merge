%% Real SPD matrices
% max_j ||A*x_j - theta_j*x_j||_2 / ||A||_2 <= tol.
% IRBL and ChebDav use native relative tolerances; verify after timing.
clear; clc;
scriptDir=fileparts(mfilename('fullpath'));
if isempty(scriptDir),scriptDir=pwd;end
addpath(scriptDir,'-begin');

%% Experiment settings
matrixNames={'msc10848','smt','gyro'};
pList=[5 20];
tol=1e-5;
maxIter=2800;
numTimingRepeats=100;
maxSolverSeconds=180;
seed=2026;
assert(numTimingRepeats>=1 && fix(numTimingRepeats)==numTimingRepeats, ...
    'numTimingRepeats must be a positive integer.');

methodNames={'BSM','Block Power','LOBPCG','IRBL (2003)','BJDQR', ...
    'Block Chebyshev-Davidson'};
solvers=local_resolve_solvers(scriptDir);
assert(numel(methodNames)==numel(solvers),'Method labels and solver handles must match.');

% Reference eigenspaces are computed in memory, not stored on disk.
referenceCfg=struct('seed',seed,'referenceTol',1e-10, ...
    'referenceMaxIt',3000,'referenceResidualTol',1e-8);
outputCSV=fullfile(scriptDir,'real_mean_time.csv');

% One compact row per (matrix, target size, solver).
rowBlank=struct('Matrix','','N',NaN,'P',NaN,'Gap',NaN,'Rho',NaN, ...
    'Algorithm','','MeanTimeSeconds',NaN,'MeanMatvecColumns',NaN,'Successes',0, ...
    'Repetitions',numTimingRepeats,'AllConverged',false);
rows=repmat(rowBlank,0,1);

fprintf('Six methods | %s | p=%s | repeats=%d | tol=%.1e\n', ...
    strjoin(matrixNames,','),mat2str(pList),numTimingRepeats,tol);
fprintf('One output file: %s\n',outputCSV);

[irblCallback,irblCleanup]=local_irbl_callback();
try
for im=1:numel(matrixNames)
    name=matrixNames{im};
    filepath=local_locate_matrix(scriptDir,name);
    A=local_load_matrix(filepath);
    n=size(A,1);
    fprintf('\nLoaded %s | n=%d nnz=%d\n',name,n,nnz(A));

    % Reference computation is outside solver timing and never passed in.
    tRef=tic;
    [Uref,lambdaRef,referenceResidual]=local_reference_solver( ...
        A,max(pList)+1,referenceCfg);
    fprintf('Reference: %.2fs, max relative residual %.2e\n', ...
        toc(tRef),referenceResidual);

    % For SPD matrices, ||A||_2=lambda_max(A).
    normA=lambdaRef(1);
    assert(isfinite(normA)&&normA>0, ...
        'Norm reference must be positive: expected SPD matrix.');
    solverTol=tol*normA;
    fprintf('||A||_2 ~= %.12e, absolute solver threshold = %.12e\n', ...
        normA,solverTol);

    for p=pList
        rng(seed+p,'twister');
        [Q0,~]=qr(randn(n,p),0);
        Ustar=Uref(:,1:p);
        gap=lambdaRef(p)-lambdaRef(p+1);
        rho=lambdaRef(p+1)/lambdaRef(p);
        options=local_method_options(p);
        Afun=@(X) A*X;
        fprintf('\n--- %s, p=%d, gap=%.6g, rho=%.8f ---\n', ...
            name,p,gap,rho);

        for j=1:numel(solvers)
            repeats=cell(numTimingRepeats,1);
            for rep=1:numTimingRepeats
                one=local_run_one(solvers{j},Afun,Q0,Ustar, ...
                    lambdaRef(1:p),tol,normA,maxIter,options{j}, ...
                    seed+p,maxSolverSeconds,irblCallback);
                repeats{rep}=one;
                fprintf('  %-30s run %d/%d | %.4fs | MV=%7g | relres=%.2e | %s\n', ...
                    methodNames{j},rep,numTimingRepeats, ...
                    one.TimeSeconds,one.MVs,one.MaxNormRitzResidual,one.Status);
            end

            isOK=cellfun(@(r) r.Success,repeats);
            item=rowBlank;
            item.Matrix=name;
            item.N=n;
            item.P=p;
            item.Gap=gap;
            item.Rho=rho;
            item.Algorithm=methodNames{j};
            item.Successes=sum(isOK);
            item.AllConverged=all(isOK);
            if item.AllConverged
                item.MeanTimeSeconds=mean(cellfun(@(r) r.TimeSeconds,repeats));
                item.MeanMatvecColumns=mean(cellfun(@(r) r.MVs,repeats));
            else
                % Do not report a misleading mean if any repeat fails.
                item.MeanTimeSeconds=NaN;
            end
            rows(end+1,1)=item; %#ok<SAGROW>
            writetable(struct2table(rows),outputCSV);
            fprintf('  >> %-28s success=%d/%d, mean=%g s | mean MV=%g\n', ...
                item.Algorithm,item.Successes,item.Repetitions, ...
                item.MeanTimeSeconds,item.MeanMatvecColumns);
        end
    end
end
catch ME
    clear irblCleanup;
    rethrow(ME);
end
clear irblCleanup;
fprintf('\nFinished. Saved only: %s\n',outputCSV);


function options=local_method_options(p)
% Solver parameters.
optsBSM=struct('epsSigma',1e-8);
optsPower=struct();
optsLOB=struct();
optsIRBL=struct('BLSZ',3,'SIZINT',1, ...
    'ENDPT','MON','SIGMA','LE','DISPR',0,'ZERTYP','ML');
optsJD=struct('blockSize',3,'mmax',45,'mmin',15, ...
    'innerTol',0.15,'innerMaxit',30,'softLocking',false, ...
    'adaptiveInner',false,'clusterTol',1e-8);
optsJD.mmax=max(optsJD.mmax,3*p+12);
optsJD.mmin=min(optsJD.mmax-optsJD.blockSize, ...
    max(2*p,p+optsJD.blockSize));
optsCheb=struct('blk',3,'polym',20,'vimax',30,'displ',0);
options={optsBSM,optsPower,optsLOB,optsIRBL,optsJD,optsCheb};
end

function r=local_run_one(solver,Afun,Q0,Ustar,lambdaWanted, ...
    tol,normA,maxIter,opts,seed,maxSeconds,irblCallback)
% Time the solver and verify the result separately.
n=size(Q0,1);p=size(Q0,2);
r=struct('Matrix','','N',NaN,'P',NaN,'Algorithm','', ...
    'AppliedShift',0,'Run',NaN,'TimeSeconds',NaN, ...
    'MVs',NaN,'MaxAbsRitzResidual',NaN,'MaxNormRitzResidual',NaN, ...
    'MaxRelEigError',NaN, ...
    'TanTheta',NaN,'Success',false,'Status','not_started','StatusDetail','');
mvCount=0;
rng(seed,'twister');
isIRBL=strcmp(func2str(solver),'irbleigs');
isLOB=strcmp(func2str(solver),'lobpcg');
isCheb=strcmp(func2str(solver),'bchdav');
solverTol=tol*normA;
if isIRBL
    opts.K=p;
    opts.BLSZ=min(p,opts.BLSZ);
    opts.V0=Q0(:,1:opts.BLSZ);
    opts.TOL=tol;
    opts.MAXIT=maxIter;
    opts.FUNPAR=@counted_operator;
elseif isCheb
    opts=local_cheb_options(opts,p,Q0,tol,maxIter);
end
tRun=tic;
try
    if isIRBL
        [X,~,prg]=solver(irblCallback,n,opts);
        elapsed=toc(tRun);
        solverStatus='maxIter';
        if prg(1)==0, solverStatus='converged'; end
        info=struct('MVs',prg(2));
    elseif isLOB
        [X,~,flag]=solver(Q0,@(Z)-counted_operator(Z),solverTol,maxIter,0);
        elapsed=toc(tRun);
        solverStatus='maxIter';
        if flag==0, solverStatus='converged'; end
        info=struct('MVs',mvCount);
    elseif isCheb
        [vals,V,kconv]=solver(@(Z)-counted_operator(Z),n,p,opts);
        elapsed=toc(tRun);
        clear global A_operator;
        X=local_cheb_vectors(vals,V,kconv,p);
        solverStatus='maxIter';
        if kconv>=p, solverStatus='converged'; end
        info=struct('MVs',mvCount);
    else
        [X,~,solverStatus,info]=solver( ...
            @counted_operator,Q0,solverTol,maxIter,opts);
        elapsed=toc(tRun);
    end
catch ME
    r.TimeSeconds=toc(tRun);
    if isCheb, clear global A_operator; end
    r.MVs=mvCount;
    r.Status=['ERROR:' ME.identifier];
    r.StatusDetail=ME.message;
    return;
end
r.TimeSeconds=elapsed;
r.MVs=mvCount;
r.Status=char(solverStatus);
if isstruct(info)&&isfield(info,'StatusDetail')
    r.StatusDetail=char(info.StatusDetail);
end
% Verify the returned Ritz pairs.
if isequal(size(X),[n,p]) && all(isfinite(X(:))) && rank(X)==p
    try
        [Q,~]=qr(X,0);
        AX=Afun(Q);
        [~,~,ritz,res]=local_rr(Q,AX,p,'largest');
        r.MaxAbsRitzResidual=max(res);
        r.MaxNormRitzResidual=max(res)/normA;
        r.MaxRelEigError=max(abs(ritz(:)-lambdaWanted(:))./ ...
            max(1,abs(lambdaWanted(:))));
        % Principal angle (diagnostic).
        [~,r.TanTheta]=local_tangent_stop(Q,Ustar,tol);
    catch ME
        r.Status=['verification_error:' ME.identifier];
        r.StatusDetail=ME.message;
        return;
    end
end
if strcmp(r.Status,'converged')
    if ~isfinite(r.MaxNormRitzResidual) || ...
            r.MaxNormRitzResidual>tol*(1+1e-5)
        r.Status='normalized_ritz_residual_verification_failed';
    elseif ~isfinite(r.MaxRelEigError) || r.MaxRelEigError>1e-4
        r.Status='wrong_target_eigenvalues';
    end
end
r.Success=strcmp(r.Status,'converged');

    function Y=counted_operator(Z)
        if toc(tRun)>maxSeconds
            error('Demo:SolverTimeLimit', ...
                'Solver exceeded %.1f seconds before the next A*X call.',maxSeconds);
        end
        Y=Afun(Z);
        mvCount=mvCount+size(Z,2);
    end
end

function [done,value]=local_tangent_stop(Q,Ustar,tol)
% Exact tangent of the largest principal angle between equal-rank spaces.
p=size(Ustar,2);
if isempty(Q) || size(Q,1)~=size(Ustar,1) || size(Q,2)~=p ...
        || any(~isfinite(Q(:)))
    done=false; value=Inf; return;
end
[Q,~]=qr(Q,0);
C=Ustar'*Q;
if min(svd(C))<1e-13
    value=Inf;
else
    Z=(Q-Ustar*C)/C;
    G=Z'*Z;
    G=(G+G')/2;
    value=sqrt(max(0,max(real(eig(G))))); % p-by-p, not a tall SVD
end
done=isfinite(value) && value<=tol;
end

function file=local_locate_matrix(rootDir,name)
% Accept name.mtx/name.mat and duplicate-upload suffix name(1).mtx.
% Prefer canonical files if present, then scan the same folder.
    exts={'.mtx','.mat','.mtx.gz'};
    for k=1:numel(exts)
        candidate=fullfile(rootDir,[name exts{k}]);
        if isfile(candidate),file=candidate;return;end
    end
    entries=dir(rootDir);
    pattern=['^' lower(name) '(?:\s*\(\d+\))?\.(?:mtx|mat|mtx\.gz)$'];
    for j=1:numel(entries)
        if entries(j).isdir,continue;end
        if ~isempty(regexp(lower(entries(j).name),pattern,'once'))
            file=fullfile(entries(j).folder,entries(j).name);
            return;
        end
    end
    error('BSM:MissingData', ...
        'Cannot find %s.mtx/.mat or %s(1).mtx in: %s',name,name,rootDir);
end

function A=local_load_matrix(file)
% Accept original SuiteSparse Matrix Market or MAT formats (no transformation).
    fileLower=lower(file);
    if endsWith(fileLower,'.mtx')
        A=local_read_mtx(file);
    elseif endsWith(fileLower,'.mtx.gz')
        tmp=tempname;
        mkdir(tmp);
        cleanup=onCleanup(@() local_delete_temp(tmp)); %#ok<NASGU>
        unpacked=gunzip(file,tmp);
        if isempty(unpacked), error('BSM:BadMTX','Empty mtx.gz archive'); end
        A=local_read_mtx(unpacked{1});
    elseif endsWith(fileLower,'.mat')
        info=whos('-file',file);
        names={info.name};
        if any(strcmp(names,'Problem'))
            s=load(file,'Problem');
            if ~isstruct(s.Problem) || ~isfield(s.Problem,'A')
                error('BSM:BadMat','Problem.A not found.');
            end
            A=s.Problem.A;
        elseif any(strcmp(names,'A'))
            s=load(file,'A'); A=s.A;
        else
            error('BSM:BadMat','MAT file must contain A or Problem.A.');
        end
    else
        error('BSM:BadInput','Unsupported matrix format: %s',file);
    end
    if ~isnumeric(A) || ndims(A)~=2 || size(A,1)~=size(A,2) || ...
            ~isreal(A) || any(~isfinite(nonzeros(A)))
        error('BSM:BadMat','Invalid real square finite matrix.');
    end
    A=sparse(double(A));
    if norm(A-A',1)>1e-12*max(1,norm(A,1))
        error('BSM:BadMat','Matrix is not numerically symmetric.');
    end
end

function A=local_read_mtx(file)
% Minimal standalone reader for SuiteSparse real/integer coordinate MTX.
% Symmetric MTX may store one triangle or both; never rescale or shift.
    fid=fopen(file,'rt');
    if fid<0, error('BSM:BadMTX','Could not open %s',file); end
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    header=fgetl(fid);
    if ~ischar(header), error('BSM:BadMTX','Empty Matrix Market file'); end
    tokens=regexp(lower(strtrim(header)), ...
        '^%%matrixmarket\s+matrix\s+coordinate\s+(real|integer)\s+(symmetric|general|hermitian)$', ...
        'tokens','once');
    if isempty(tokens)
        error('BSM:BadMTX','Unsupported Matrix Market header: %s',header);
    end
    kind=tokens{2};
    while true
        line=fgetl(fid);
        if ~ischar(line),error('BSM:BadMTX','Missing Matrix Market size line');end
        line=strtrim(line);
        if ~isempty(line) && line(1)~='%', break; end
    end
    dims=sscanf(line,'%d %d %d');
    if numel(dims)~=3 || dims(1)~=dims(2) || dims(1)<1 || dims(3)<0
        error('BSM:BadMTX','Invalid square matrix size line: %s',line);
    end
    n=dims(1); nz=dims(3);
    contents=textscan(fid,'%f%f%f',nz,'CollectOutput',true,'CommentStyle','%');
    values=contents{1};
    if size(values,1)~=nz
        error('BSM:BadMTX','Truncated Matrix Market data: %s',file);
    end
    i=values(:,1); j=values(:,2); a=values(:,3);
    if any(~isfinite(values(:))) || any(i~=fix(i)) || any(j~=fix(j)) || ...
            any(i<1 | i>n) || any(j<1 | j>n)
        error('BSM:BadMTX','Invalid Matrix Market coordinates/values');
    end
    A=sparse(i,j,a,n,n);
    if strcmp(kind,'symmetric') || strcmp(kind,'hermitian')
        lowerCount=nnz(tril(A,-1));
        upperCount=nnz(triu(A,1));
        if lowerCount>0 && upperCount==0
            A=A+tril(A,-1)';
        elseif upperCount>0 && lowerCount==0
            A=A+triu(A,1)';
        elseif lowerCount>0 && upperCount>0 && ...
                norm(A-A',1)>1e-12*max(1,norm(A,1))
            error('BSM:BadMTX', ...
                'Symmetric MTX contains inconsistent upper/lower triangles');
        end
    end
end

function local_delete_temp(folder)
    if isfolder(folder),rmdir(folder,'s');end
end


function [U,lam,relMax]=local_reference_solver(A,k,cfg)
    % Robust reference computation. Its time is excluded from both solvers.
    lastProblem='not attempted';
    for attempt=1:2
        op=struct();
        op.tol=cfg.referenceTol;
        op.maxit=cfg.referenceMaxIt*attempt;
        op.p=min(size(A,1)-1,max(100,4*k+40*(attempt-1)));
        op.disp=0;
        rng(cfg.seed,'twister');
        op.v0=randn(size(A,1),1);
        try
            try
                [U,D,flag]=eigs(A,k,'largestreal',op);
            catch
                [U,D,flag]=eigs(A,k,'la',op);
            end
            [lam,order]=sort(real(diag(D)),'descend');
            U=real(U(:,order));
            R=A*U-U*diag(lam);
            rel=sqrt(sum(R.^2,1))'./max(1,abs(lam));
            relMax=max(rel);
            if flag==0 && relMax<=cfg.referenceResidualTol && ...
                    norm(U'*U-eye(k),'fro')<=1e-7
                return;
            end
            lastProblem=sprintf('flag=%d, residual=%.3e',flag,relMax);
        catch ME
            lastProblem=ME.message;
        end
    end
    error('Unreliable reference eigs after two attempts: %s',lastProblem);
end


function solvers=local_resolve_solvers(scriptDir)
files={'method_bsm','method_block_power','lobpcg', ...
       'irbleigs','method_bjdqr2015','bchdav'};
solvers=cell(1,numel(files));
for j=1:numel(files)
    stem=files{j};
    expectedFile=fullfile(scriptDir,[stem '.m']);
    assert(exist(expectedFile,'file')==2,'Missing %s.',expectedFile);
    resolvedFile=which(stem);
    if ispc
        sameFile=strcmpi(resolvedFile,expectedFile);
    else
        sameFile=strcmp(resolvedFile,expectedFile);
    end
    assert(sameFile,'MATLAB resolves %s to %s instead of %s.', ...
        stem,resolvedFile,expectedFile);
    solvers{j}=str2func(stem);
end
end

function [name,guard]=local_irbl_callback()
% IRBLEIGS takes a function name for operator input.
folder=tempname;
mkdir(folder);
[~,stem]=fileparts(folder);
name=['irbl_' regexprep(stem,'[^a-zA-Z0-9_]','_')];
file=fullfile(folder,[name '.m']);
guard=onCleanup(@()local_remove_callback(folder));
fid=fopen(file,'wt');
assert(fid>=0,'Cannot create temporary IRBL operator.');
fileGuard=onCleanup(@()fclose(fid));
fprintf(fid,'function Y=%s(X,n,blsz,Afun)\nY=Afun(X);\nend\n',name);
clear fileGuard;
addpath(folder,'-begin');
assert(strcmpi(which(name),file),'Cannot resolve IRBL operator.');
end

function local_remove_callback(folder)
if contains([path pathsep],[folder pathsep])
    rmpath(folder);
end
if isfolder(folder), rmdir(folder,'s'); end
end

function opts=local_cheb_options(opts,p,Q0,tol,maxIter)
opts.blk=min(p,opts.blk);
opts.vimax=max(opts.vimax,p+opts.blk);
opts.tol=tol;
opts.itmax=maxIter;
opts.v0=Q0;
end

function X=local_cheb_vectors(vals,V,kconv,p)
if kconv<p || size(V,2)<p || numel(vals)<p
    X=V;
    return;
end
[~,ix]=sort(real(vals(1:kconv)),'ascend');
X=V(:,ix(1:p));
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
