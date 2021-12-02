function SaveData(Plt);

% Save PI function on disk
[fname,pname]=uiputfile('*.txt','Save Data As');
if pname ~= 0
 fid = fopen([pname,fname],'w');
 if fid ~= -1
  fprintf(fid,'%5.1f \t %4.4f \n',Plt);
  fclose(fid);
 else
  error('Could not create Data file');
 end;
end;

