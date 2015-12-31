 require 'logroll'
 require 'json'
 
local ExperimentLogCSV = torch.class('csream.ExperimentLogCSV','csream.ExperimentLog'); 

function ExperimentLogCSV:fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function ExperimentLogCSV:JSONToCSV(json,prefix)
  local retour={}
  for k,v in pairs(json) do
      local nprefix=prefix..":"..k
      if (prefix=="") then nprefix=k end
      if (type(v)=="table") then
        for k2,v2 in pairs(self:JSONToCSV(v,nprefix)) do
          retour[k2]=v2
        end
      else
        retour[nprefix]=v
      end
  end
  return retour
end

function ExperimentLogCSV:__init(memory,directory,filename)
  csream.ExperimentLog.__init(self,memory)
  self.log=logroll.print_logger()
  self.filename=directory.."/"..filename
  if (filename=="now") then
    local t=os.time()
    t=os.date('%d_%b_%Y_%H:%M:%S',t)
    local ff=directory.."/"..t..".csv"
    local idx=0
    while(self:fileExists(ff)) do
      idx=idx+1
      ff=directory.."/"..t..".v"..idx..".csv"
    end
    self.filename=ff
  end
  self.parameters={log={filename=ff}}
  print(self.filename)
  self.of=io.open(self.filename, "w")
  self.of2=io.open(self.filename..".description","w")
  self.log.info("Initialization of a new experiment log. Output is "..self.filename)
  self.firstWritten=false  
end

function ExperimentLogCSV:newIteration()
  if (not csream.ExperimentLog.isEmpty(self)) then      
        local flat=self:JSONToCSV(self.currentjson,"")
        local flat_parameters=self:JSONToCSV(self.parameters,"")
        
        -- Affichage de la première ligne
        if (self.firstWritten==false) then	
          self.of:write("iteration")
          self.orderp={}   
          local pos=1
          for k,v in pairs(flat) do
            self.orderp[pos]=k
            self.of:write("\t"..k)
            pos=pos+1
          end
          for k,v in pairs(flat_parameters) do
            self.orderp[pos]=k
            self.of:write("\t"..k)
            pos=pos+1
          end
          self.number_of_arguments=pos-1
          self.of:write("\n")
        self.firstWritten=true
        end
        
        -- Affichage des arguments
        self.of:write(self.iteration-1)
        for n=1,self.number_of_arguments do
          local kk=self.orderp[n]
          if (flat[kk]==nil) then
            if (flat_parameters[kk]==nil) then
              self.of:write("\t")
            else
             self.of:write("\t"..flat_parameters[kk])
            end
          else
            self.of:write("\t"..flat[kk])
          end
        end
        self.of:write("\n")
  end
  csream.ExperimentLog.newIteration(self)
  self.of:flush()
end
 
function ExperimentLogCSV:addDescription(text)  
  if (torch.isTensor(text)) then
    print(text:size())
    print(text:size():size())
    if (text:size():size()==1) then
      self.of2:write("Tensor  (text:size(1)) =")
      for i=1,text:size(1) do
        self.of2:write(" "..text[i])
      end
      self.of2:write("\n")
    else
      print("Unable to save tensor in file")
      print(text)
    end
  else  
    self.of2:write(text)    
  end
end
