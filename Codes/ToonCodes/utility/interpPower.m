function p = interpPower(cumPwr, cumF, fsel)

idx = find(fsel<=cumF,1,'first');
if ~isempty(idx)
  % scalar inference for codegen
  idx1 = idx(1);
  if idx1==1
    p = signal.internal.linterp(cumPwr(1,:),cumPwr(2,:),cumF(1),cumF(2),fsel);
  else
    p = signal.internal.linterp(cumPwr(idx1,:),cumPwr(idx1-1,:), ...
                                cumF(idx1),cumF(idx1-1),fsel);
  end
else
  % codegen requires both conditional branches to have the same data type
  % for 'p'
  p = nan(1,size(cumPwr,2),'like',cumPwr(1)+cumF(1)+fsel(1));
end

end