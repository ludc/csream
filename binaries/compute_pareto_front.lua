import "csvigo"
import 'gnuplot'

function keptLines(read,index,opt)
  local lkept={}
  local pos=1
  for j=2,#read do
    local kept=true
    
    
    if (opt.dataset~="") then
      local ddataset=read[j][index["dataset"]]
      if (not (string.match(ddataset,opt.dataset))) then kept=false end
    end
    
    if (opt.size~="") then
      local dsize=read[j][index["size"]]
      if (dsize~=opt.size) then kept=false end
    end
    
    if (opt.N~="") then 
	    if (read[j][index["N"]]~=opt.N) then 
        
		kept=false 
    end
  end

    if (opt.cell~="") then 
	    if (read[j][index["cell"]]~=opt.cell) then 			
		kept=false end    
	end
  
    if (opt.ent_factor~="") then 
	    if (read[j][index["ent_factor"]]~=opt.ent_factor) then 			
		kept=false end    
	end
    
  
    if (opt.bias~="") then 
	    if (read[j][index["bias"]]~=opt.bias) then 			
		kept=false end    
	end
  
    if (opt.l1_discount~="") then 
	    if (read[j][index["l1_discount"]]~=opt.l1_discount) then 			
		kept=false end    
	end
    
    if (kept) then lkept[pos]=read[j]; pos=pos+1 end
  end    
  print("KEEPT = "..#lkept)
  return lkept
end

function keptBest(lkept,index)
   local best_accuracy={}
   local best_accuracy_index={}
   for i=1,#lkept do
     local sp=lkept[i][index["sparsity_train"]]
     local acc=lkept[i][index["accuracy_train"]]
     if(best_accuracy[sp]==nil) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i end
     if(best_accuracy[sp]<acc) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i  end
   end
   
   local llkept={}
   local pos=1
   for k,v in pairs(best_accuracy_index) do
     llkept[pos]=lkept[v]
     pos=pos+1
    end  
    return(llkept)
end

function keptBest_validation(lkept,index)
   local best_accuracy={}
   local best_accuracy_index={}
   for i=1,#lkept do
     local sp=lkept[i][index["sparsity_validation"]]
     local acc=lkept[i][index["accuracy_validation"]]
--     print(sp); print(acc);
     if(best_accuracy[sp]==nil) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i end
     if(best_accuracy[sp]<acc) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i  end
   end
   
   local llkept={}
   local pos=1
   for k,v in pairs(best_accuracy_index) do
     llkept[pos]=lkept[v]
     pos=pos+1
    end  
    return(llkept)
end


function keepColumn(t,i)
  local c=torch.Tensor(#t)
  local pos=1
  for j=1,#t do
    c[pos]=t[j][i]
    pos=pos+1
  end
  return c
end
  
 
--- Assumes that lines hve been sorted by increasing sparsity
function pareto(lkept,index)
 
   local llkept={}
   llkept[1]=lkept[1]
   local pos=2
   local sp=lkept[1][index["sparsity_train"]]
   local acc=lkept[1][index["accuracy_train"]]
   for j=2,#lkept do
     
     local ssp=lkept[j][index["sparsity_train"]]
     local sacc=lkept[j][index["accuracy_train"]]
     --print(ssp.." vs "..sp.." / "..sacc.." vs "..acc)
     if ((ssp<sp) and (sacc>acc)) then
       llkept[pos]=lkept[j]
       pos=pos+1
       sp=ssp
       acc=sacc
     end
    end
   return(llkept)
end
 
function pareto_validation(lkept,index,prec)
 
   local llkept={}
   llkept[1]=lkept[1]
   local pos=2
   local sp=lkept[1][index["sparsity_validation"]]
   local acc=lkept[1][index["accuracy_validation"]]
   for j=2,#lkept do
     
     local ssp=lkept[j][index["sparsity_validation"]]
     local sacc=lkept[j][index["accuracy_validation"]]
    -- print(ssp.." vs "..sp.." / "..sacc.." vs "..acc)
     if ((tonumber(ssp)<tonumber(sp)) and (tonumber(sacc)>tonumber(acc)*prec)) then
       llkept[pos]=lkept[j]
       pos=pos+1
       sp=ssp
       acc=sacc
     end
    end
   local ff=#llkept
   llkept[ff+1]={}
   for k,v in pairs(llkept[ff]) do
	llkept[ff+1][k]=v
   end
   llkept[ff+1][index["sparsity_validation"]]=0
   llkept[ff+1][index["sparsity_test"]]=0
   llkept[ff+1][index["sparsity_train"]]=0	
   return(llkept)
end
 
function doAll(read,index,opt)
  local lkept=keptLines(read,index,opt)
  print(#lkept.." lines kept.")
  if (#lkept==0) then return nil end
  ---- Remove lines with same sparisty (keeping the best)
 -- lkept=keptBest(lkept,index,opt)
  lkept=keptBest_validation(lkept,index,opt)
  print(#lkept.." lines kept.")
  ---- Sort lines by training_sparsity
  function sort_sparsity(a,b)
    return tonumber(a[index["sparsity_validation"]])>tonumber(b[index["sparsity_validation"]])
  end
  table.sort(lkept,sort_sparsity)

  ---- Keep pareto front
--  llkept=pareto(lkept,index)
  llkept=pareto_validation(lkept,index,opt.precision)
  print(#llkept.." lines kept in pareto front")
  return llkept
end
 
 
cmd=torch.CmdLine()
cmd:text()
cmd:option('--dataset', '', 'name of the dataset')
cmd:option('--file', '', 'name of the input CSV file')
cmd:option('--file2', '', 'name of the input CSV file')
cmd:option('--N', '', 'restrict to particular values of N')
cmd:option('--cell', '', 'restrict to particular cells')
cmd:option('--ent_factor', '', '')
cmd:option('--bias', '', '')
cmd:option('--l1_discount', '', '')
cmd:option('--output', '', '')
cmd:option('--computeSparsity', 'false', '')

cmd:option('--precision',1,'')
cmd:text()

local opt = cmd:parse(arg or {})
if not opt.silent then
   print(opt)
end

 
local read=csvigo.load{path=opt.file,separator='\t',verbose=false,mode='raw'}
 local index={}
 for i=1,#read[1] do
   index[read[1][i]]=i
 end
 print("Read "..#read)
   collectgarbage()
if (opt.file2~="") then
  local read2=csvigo.load{path=opt.file2,separator='\t',verbose=false,mode='raw'}
  local pos=#read+1
  for y=2,#read2 do
    local t={}      
    for yy=1,#read2[1] do
      
      local c=read2[1][yy]
      local ii=index[c]
      if (ii~=nil) then t[ii]=read2[y][yy] end
    end
    read[pos]=t
    pos=pos+1
  end
end
print("Read "..#read)
 
read2=nil
print("End fusion") 

 
 local distinct_values={}
 for k,_ in pairs(index) do
   distinct_values[k]={}
 end
 for i=2,#read do
   for j=1,#read[i] do
     local k=read[1][j]
     distinct_values[k][read[i][j]]=1
    end
end
print(distinct_values)
os.exit(1)

local sp={}
local acc={}

SIZES={"1","2","3","4","5","6","7","8","9","10","15","20"}
if (opt.output~="") then
  for v,k in pairs(SIZES) do
    opt.size=k
    local llkept=doAll(read,index,opt)
    if (llkept~=nil) then
      sp=keepColumn(llkept,index["sparsity_train"])
      acc=keepColumn(llkept,index["accuracy_train"])
      spv=keepColumn(llkept,index["sparsity_validation"])
      accv=keepColumn(llkept,index["accuracy_validation"])
      spt=keepColumn(llkept,index["sparsity_test"])
      acct=keepColumn(llkept,index["accuracy_test"])         
      io.output(opt.output..".size="..k)  
      for j=1,sp:size(1) do
        io.write(sp[j].." "..acc[j].." "..spv[j].." "..accv[j].." "..spt[j].." "..acct[j].."\n")
      end   
      io.close()
      io.output(io.stdout)
    end
  end
end

io.output(io.stdout)

LEVELS={0,0.25,0.5,0.75,1.0}
if (opt.computeSparsity=="true") then
  for v,k in pairs(SIZES) do
    print("SIZE = "..k)
   opt.size=k
   local llkept=doAll(read,index,opt)
    
  sp=keepColumn(llkept,index["sparsity_validation"])
  acc=keepColumn(llkept,index["accuracy_validation"])
  
  for _,k in pairs(LEVELS) do 
      io.stdout:write(" Sparsity="..k.." : ")
      local s=sp[1]
      local a=acc[1]
      if (s<=k) 
      then 
        io.stdout:write(" "..a.."("..s..")")
      else
        local pos=2
        local flag=true
        while(flag) do
          if (sp[pos]<k) then
            local ratio=(k-sp[pos])/(s-sp[pos])
            local aa=ratio*acc[pos]+(1-ratio)*a
            flag=false
            io.stdout:write(" "..aa)
          else
            a=acc[pos]
            s=sp[pos]
            pos=pos+1
            if (pos==sp:size(1)+1) then 
              io.stdout:write(" "..a.."("..s..")")
              flag=false
            end
          end
        end
      end
  end  
    print("")
end
end

local sss="sparsity_test"
local aaa="accuracy_test"


local pos=1
  toplot={}
for _,k in pairs(SIZES) do
  opt.size=k
  local llkept=doAll(read,index,opt)
  if (llkept~=nil) then
    sp[k]=keepColumn(llkept,index[sss])  
    acc[k]=keepColumn(llkept,index[aaa])
    local n=sp[k]:size(1)
    if (n>0) then
        print("KKEEPPP")
    toplot[#toplot+1]={"size "..k,sp[k],acc[k],"lines ls "..pos}
    pos=pos+1
    end
    
  end
end

  print(toplot)

  gnuplot.plot(toplot)

--gnuplot.plot({"test",spt,acct,"lines ls 2"},{"validation",spv,accv,"lines ls 3"})
