function f = interpFreq(cumPwr, cumF, pwrThresh)

nChan = size(cumPwr,2);
f = coder.nullcopy(zeros(1,nChan,'like',cumPwr(1)+cumF(1)+pwrThresh(1)));

for iChan = 1:nChan
  idx = find(pwrThresh(iChan)<=cumPwr(:,iChan),1,'first');
  if ~isempty(idx)
    % scalar inference for codegen
    idx1 = idx(1);
    if idx1==1
       idx1=2;
    end
    f(iChan) = signal.internal.linterp(cumF(idx1-1),cumF(idx1), ...
                 cumPwr(idx1-1,iChan),cumPwr(idx1,iChan),pwrThresh(iChan));
  else
    % codegen requires both conditional branches to have the same data type
    % for 'f'
    f(iChan) = nan(1,1,'like',cumPwr(1)+cumF(1)+pwrThresh(1));
  end
end

end