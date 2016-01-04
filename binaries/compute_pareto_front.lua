import "csvigo"
import 'gnuplot'
import 'lfs'

function clone(a)
  local r={}
  for k,v in pairs(a) do
    r[k]=v
  end
  return(r)
end

--- returns a sub table keeping only lines such that index_column=value
function filter(ttable,index_column,value)
  local retour={}
  local pos=1
  for i=1,#ttable do
    if (ttable[i][index_column]==value) then
      retour[pos]=clone(ttable[i])
      pos=pos+1
    end
  end
  return retour
end

---- Keep the best accuracy for each cost level
function keepBestAccuracy(ttable,column_cost,column_accuracy)
   local best_accuracy={}
   local best_accuracy_index={}
   for i=1,#ttable do
     local sp=tonumber(ttable[i][column_cost])
     local acc=tonumber(ttable[i][column_accuracy])     
     if(best_accuracy[sp]==nil) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i end
     if(best_accuracy[sp]<acc) then best_accuracy[sp]=acc; best_accuracy_index[sp]=i  end
   end
   
   local retour={}
   local pos=1
   for k,v in pairs(best_accuracy_index) do
     retour[pos]=clone(ttable[v])
     pos=pos+1
    end  
  return(retour)
end

function keepColumn(t,i)
  local c=torch.Tensor(#t)
  local pos=1
  for j=1,#t do
    c[pos]=tonumber(t[j][i])
    pos=pos+1
  end
  return c
end
  
--- Assumes that lines hve been sorted by increasing sparsity. 
function pareto(ttable,column_cost,column_accuracy)
 
  -- First sort retour from low cost to high cost
  function sort_cost(a,b)
    return tonumber(a[column_cost])<tonumber(b[column_cost])
  end
  table.sort(ttable,sort_cost)
  
  -- Second: filter
   local retour={}
   retour[1]=ttable[1]
   local pos=2
   local sp=tonumber(ttable[1][column_cost])
   local acc=tonumber(ttable[1][column_accuracy])
   for j=2,#ttable do     
     local ssp=tonumber(ttable[j][column_cost])
     local sacc=tonumber(ttable[j][column_accuracy])
     if (sacc>acc) then
       retour[pos]=clone(ttable[j])
       pos=pos+1
       sp=ssp
       acc=sacc
     end
    end
   return(retour)
end

 
cmd=torch.CmdLine()
cmd:text()
cmd:option('--files', '', 'name of the input CSV file separed by :')
cmd:option('--directory', '', 'name of the input directory (reading only .csv files)')
cmd:option('--filters', '', 'the filters separated by ":"')
cmd:option('--by', '', '')
cmd:option('--sparsity', 'false', '')
cmd:option('--output', '', '')
cmd:text()

local opt = cmd:parse(arg or {})
if not opt.silent then
   print(opt)
end

COST=COST."_"
if (opt.sparsity=="true") then COST="sparsity_" end

------ READING filters
local filters={}
do
  local pos=1
  if (opt.filters~="") then
    for w in string.gmatch(opt.filters,"[^:]+") do  
      local token=string.gmatch(w,"[^=]+")
      local key=token()
      local value=token()
      filters[pos]={}
      filters[pos].key=key
      filters[pos].value=value
      pos=pos+1
    end
  end
end
print("Filters are: ")
for _,v in pairs(filters) do
  print("\t"..v.key.." = "..v.value)
end

---------------- filenames... ------------------
local filenames={}
if (opt.files~='') then 
  for w in string.gmatch(opt.files,"[^:]+") do filenames[#filenames+1]=w end
else
  for file in lfs.dir(opt.directory) do    
    if (string.find(file,".csv$")) then 
      filenames[#filenames+1]=opt.directory.."/"..file
    end
  end
end

---------------- READING all files in the 'read' table --------------
local first=true
local read={}
local index={}
local size_index=0;
local pos=1
print("Reading "..#filenames.." files... ('#' are problems)")
do
    for _,w in pairs(filenames) do
      io.write('.'); io.flush()
      local rr=csvigo.load{path=w,separator='\t',verbose=false,mode='raw'}
      if ((rr==nil) or (#rr==0))  then io.write('#'); io.flush() 
      else
        
        --- Update index
        for i=1,#rr[1] do
          local k=rr[1][i]
          if (index[k]==nil) then 
            index[k]=size_index+1; size_index=size_index+1 
            --- filling missing values
          end
        end
              
        for line=2,#rr do
          local t={}      
          for column=1,#rr[1] do      
            local key=rr[1][column]            
            local ii=index[key]            
            t[ii]=rr[line][column]
          end
          
          local keep=true
          for _,f in pairs(filters) do
            local i=index[f.key]
            assert(i~=nil,"Unknown filter key "..f.key)
            if (t[i]~=f.value) then keep=false end
          end
          if (keep) then
            read[pos]=t;                     
            pos=pos+1 
          end
        end
      end
    end
   -- print("Reading "..w.." = "..#read)
end
io.write("\n");
print("Finding "..#read.." lines")
print("Columns are: ")
for k,_ in pairs(index) do
  io.write(k.." ")
end
io.write("\n")

------------------- Computing the distinct values for all columns
print("Computing distinct values for the columns....")
local distinct_values={}
for k,_ in pairs(index) do
  distinct_values[k]={}
end
for i=2,#read do
  for k,v in pairs(index) do
    if (read[i][v]~=nil) then
      distinct_values[k][read[i][v]]=1
    end
  end
end


-----------------------------------------------------------------------------------
---- Simple pareto
if (opt.by=="") then
    local nt=keepBestAccuracy(read,index[COST."_validation"],index["accuracy_validation"])
    nt=pareto(read,index[COST."_validation"],index["accuracy_validation"])  
    cost=keepColumn(nt,index[COST."_test"])
    accuracy=keepColumn(nt,index["accuracy_test"])
    gnuplot.plot({"all",cost,accuracy,"lines ls 1"})
else
  local td={}
  local name={}
  local cost={}
  local acc={}
  
  local pos=1
  local i=index[opt.by]; assert(i~=nil) 
  for v,_ in pairs(distinct_values[opt.by]) do
    print("Computing pareto curve for "..opt.by.." = "..v)
    local nt=filter(read,i,v)
    nt=keepBestAccuracy(nt,index[COST."_validation"],index["accuracy_validation"])
    nt=pareto(nt,index[COST."_validation"],index["accuracy_validation"])  
    cost[pos]=keepColumn(nt,index[COST."_test"])
    acc[pos]=keepColumn(nt,index["accuracy_test"])   
    name[pos]=opt.by.."="..v
  
    if (opt.output~="") then
      io.output(opt.output.."."..name[pos])
      io.write("#cost_alidation accuracy_validation\n")
      for jj=1,cost[pos]:size(1) do        
        io.write(cost[pos][jj].." "..acc[pos][jj].."\n");
      end      
      io.output(io.stdout)
    end
  
    td[pos]={name[pos],cost[pos],acc[pos],"linespoints ls "..pos} 
    pos=pos+1
   -- gnuplot.plot(td[#td])
  end  
  gnuplot.plot(td)
  
end  
os.exit(1)


