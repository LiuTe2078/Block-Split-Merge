%% Synthetic SPD matrices
% max_j ||A*x_j - theta_j*x_j||_2 / ||A||_2 <= tol.
% IRBL and ChebDav use native relative tolerances; verify after timing.
clear; clc;
scriptDir=fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir=pwd; end
addpath(scriptDir,'-begin');

%% Experiment settings
nList=[2048 4096 8192];
pList=[5 10 20];
gapList=[0.1 0.5 1];
tol=1e-5;
maxIter=2800;
numRuns=100;
seed=2026;
assert(numRuns>=1 && numRuns==fix(numRuns),'numRuns must be positive integer.');

%% Solver options
optsBSM=struct('epsSigma',0.02);
optsPower=struct();
optsLOB=struct();
optsIRBL=struct('BLSZ',3,'NBLS',11,'SIZINT',1, ...
    'ENDPT','MON','SIGMA','LE','DISPR',0,'ZERTYP','ML');
optsJD=struct('blockSize',3,'mmax',45,'mmin',15, ...
    'innerTol',0.15,'innerMaxit',30,'softLocking',false, ...
    'adaptiveInner',false,'clusterTol',1e-8);
optsCheb=struct('blk',3,'polym',20,'vimax',30, ...
    'vomax',45,'filter',1,'displ',0);

names={'BSM','Block Power','LOBPCG','IRBL (2003)','BJDQR', ...
    'Block Chebyshev-Davidson'};
solvers=local_resolve_solvers(scriptDir);
options={optsBSM,optsPower,optsLOB,optsIRBL,optsJD,optsCheb};
assert(numel(names)==numel(solvers),'Method labels and solver handles must match.');

%% Results
outCSV=fullfile(scriptDir,'synthetic_matrix_mean_time.csv');
blank=struct('N',NaN,'P',NaN,'Gap',NaN,'Method','', ...
    'MeanTimeSeconds',NaN,'StdTimeSeconds',NaN, ...
    'MeanMatvecColumns',NaN, ...
    'Successes',0,'Runs',numRuns,'AllConverged',false);
numCases=numel(nList)*numel(pList)*numel(gapList);
numMethods=numel(names);
summary=repmat(blank,numCases*numMethods,1);
rowCount=0;
caseCount=0;
rng(seed,'twister');
fprintf('SIX synthetic SPD solvers: %d cases x %d methods x %d runs\n', ...
    numCases,numMethods,numRuns);
fprintf('tol=%.1e: max ||Ritz residual||_2 / ||A||_2; maxIter=%d\n',tol,maxIter);
fprintf('ONE output CSV: %s\n',outCSV);

global MVprod
[irblCallback,irblCleanup]=local_irbl_callback();
try
for n=nList
    for p=pList
        for gap=gapList
            caseCount=caseCount+1;
            [Afun,~,lambdaExact]=local_generate_matrix_scale1(n,p,gap,'operator');
            lambdaExact=lambdaExact(:);
            assert(all(isfinite(lambdaExact)) && all(lambdaExact>0), ...
                'Generator did not return a positive finite SPD spectrum.');
            assert(abs(lambdaExact(p)-lambdaExact(p+1)-gap)<1e-9, ...
                'Unexpected boundary eigenvalue gap.');
            assert(abs(lambdaExact(p)-2)<1e-12 && min(lambdaExact)>=0.05, ...
                'Expected the direct scale-1 spectrum.');
            normA=lambdaExact(1);                 % Exact ||A||_2 for SPD.
            assert(normA>0 && isfinite(normA));
            tolAbs=tol*normA;                      % Absolute residual threshold.
            [Q0,~]=qr(randn(n,p),0);
            solverRngState=rng;                    % Same initial RNG for all.
            fprintf('\nCase %2d/%d: n=%d, p=%d, gap=%g, ||A||=%.6g, tolAbs=%.3g\n', ...
                caseCount,numCases,n,p,gap,normA,tolAbs);

            for im=1:numMethods
                thisOpts=options{im};
                if im==4
                    % Relative tolerance for IRBLEIGS.
                    thisOpts.K=p;
                    thisOpts.BLSZ=min(p,thisOpts.BLSZ);
                    thisOpts.V0=Q0(:,1:thisOpts.BLSZ);
                    thisOpts.TOL=tol;
                    thisOpts.MAXIT=maxIter;
                    thisOpts.FUNPAR=Afun;
                elseif im==5
                    thisOpts.mmax=max(thisOpts.mmax,3*p+12);
                    thisOpts.mmin=min(thisOpts.mmax-thisOpts.blockSize, ...
                        max(2*p,p+thisOpts.blockSize));
                elseif im==6
                    thisOpts=local_cheb_options(thisOpts,p,Q0,tol,maxIter);
                end

                times=nan(numRuns,1);
                mvs=nan(numRuns,1);
                success=false(numRuns,1);
                for run=1:numRuns
                    rng(solverRngState);
                    tRun=tic;
                    try
                        if im==4
                            [X,~,prg]=solvers{im}( ...
                                irblCallback,n,thisOpts);
                            times(run)=toc(tRun);
                            status='maxIter';
                            if prg(1)==0, status='converged'; end
                            info=struct('MVs',prg(2));
                        elseif im==3
                            [X,flag,mv]=local_lobpcg_run(solvers{im},Afun,Q0,tolAbs,maxIter);
                            times(run)=toc(tRun);
                            status='maxIter';
                            if flag==0, status='converged'; end
                            info=struct('MVs',mv);
                        elseif im==6
                            [vals,V,kconv]=solvers{im}(@(Z)-Afun(Z),n,p,thisOpts);
                            times(run)=toc(tRun);
                            info=struct('MVs',MVprod);
                            clear global A_operator;
                            X=local_cheb_vectors(vals,V,kconv,p);
                            status='maxIter';
                            if kconv>=p, status='converged'; end
                        else
                            [X,~,status,info]=solvers{im}( ...
                                Afun,Q0,tolAbs,maxIter,thisOpts);
                            times(run)=toc(tRun);
                        end
                        if isstruct(info) && isfield(info,'MVs')
                            mvs(run)=info.MVs;
                        end
                    catch ME
                        times(run)=toc(tRun);
                        if im==6, clear global A_operator; end
                        fprintf('  %-28s run %d/%d: ERROR (%s): %s\n', ...
                            names{im},run,numRuns,ME.identifier,ME.message);
                        continue;
                    end

                    resultStatus=char(status);
                    maxNormResidual=NaN;
                    if strcmp(resultStatus,'converged')
                        % Verify the Ritz residual and target eigenvalues.
                        try
                            if ~isequal(size(X),[n,p]) || ...
                                    any(~isfinite(X(:))) || rank(X)<p
                                resultStatus='invalid_eigenvector_block';
                            else
                                [Xcheck,~]=qr(X,0);
                                AXcheck=Afun(Xcheck);
                                [~,~,ritz,res]=local_rr( ...
                                    Xcheck,AXcheck,p,'largest');
                                maxNormResidual=max(res)/normA;
                                target=lambdaExact(1:p);
                                maxRelEigError=max(abs(ritz(:)-target(:))./ ...
                                    max(1,abs(target(:))));
                                if ~isfinite(maxNormResidual) || ...
                                        maxNormResidual>tol*(1+1e-5)
                                    resultStatus='normalized_residual_failed';
                                elseif ~isfinite(maxRelEigError) || ...
                                        maxRelEigError>1e-4
                                    resultStatus='wrong_target_eigenvalues';
                                end
                            end
                        catch ME
                            resultStatus=['verification_error:' ME.identifier];
                        end
                    end
                    success(run)=strcmp(resultStatus,'converged');
                    fprintf('  %-28s run %d/%d | %8.4fs | MV=%7g | relres=%.2e | %s\n', ...
                        names{im},run,numRuns,times(run), ...
                        mvs(run),maxNormResidual,resultStatus);
                end

                item=blank;
                item.N=n;item.P=p;item.Gap=gap;item.Method=names{im};
                item.Successes=sum(success);
                item.AllConverged=all(success);
                if item.AllConverged
                    item.MeanTimeSeconds=mean(times);
                    item.StdTimeSeconds=std(times);
                    item.MeanMatvecColumns=mean(mvs);  % NaN if not reported.
                end
                rowCount=rowCount+1;
                summary(rowCount)=item;
                writetable(struct2table(summary(1:rowCount),'AsArray',true),outCSV);
                fprintf('  >> %-26s success=%d/%d, mean=%g s, mean MV=%g\n', ...
                    names{im},item.Successes,numRuns,item.MeanTimeSeconds, ...
                    item.MeanMatvecColumns);
            end
            rng(solverRngState);  % Restore generator sequence independently.
        end
    end
end
catch ME
    clear irblCleanup;
    rethrow(ME);
end
clear irblCleanup;
fprintf('\nCompleted. Only one results file saved: %s\n',outCSV);


%% SPD matrix generator
function [A, Ustar, lambda] = local_generate_matrix_scale1(n, p, gap, mode)
% Prescribed spectrum: lambda(p)=2, lambda(p+1)=2-gap.
    if nargin < 4 || isempty(mode), mode = 'dense'; end
    assert(isscalar(n) && n == floor(n) && n >= 3, 'n must be >=3.');
    assert(isscalar(p) && p == floor(p) && p >= 1 && p < n, '1<=p<n required.');
    scale = 1;
    floorEig = scale * 0.05;
    assert(isscalar(gap) && isfinite(gap) && gap > 0 && ...
        gap < scale*2-floorEig, 'Require 0 < gap < 1.95 for SPD construction.');

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
opts.vomax=max(opts.vomax,p+2*opts.blk);
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

function [X,flag,mv]=local_lobpcg_run(solver,Afun,Q0,tolAbs,maxIter)
mv=0;
[X,~,flag]=solver(Q0,@counted_minus,tolAbs,maxIter,0);
    function Y=counted_minus(Z)
        Y=-Afun(Z);
        mv=mv+size(Z,2);
    end
end
