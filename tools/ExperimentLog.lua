local ExperimentLog = torch.class('csream.ExperimentLog'); 

function ExperimentLog:__init(memory)
  self.memory=memory
  self.jsons={}
  self.currentjson=nil
  self.iteration=0
  self.parameters={}
end

function ExperimentLog:addFixedParameters(arg)
    for k,v in pairs(arg) do
      self.parameters[k]=v
    end
end

function ExperimentLog:isEmpty()
  if (self.currentjson==nil) then return true end
  local nb=0
  for k,v in pairs(self.currentjson) do
    nb=nb+1
  end
  if (nb==0) then return true end
  return false
end

function ExperimentLog:newIteration()
  if (self.memory) then
    if (self.currentjson~=nil) then
      table.insert(self.jsons,self.currentjson)
    end
  end
  self.iteration=self.iteration+1
  self.currentjson={}
end

function ExperimentLog:addValue(key,value)
  self.currentjson[key]=value
end

function ExperimentLog:addDescription(text)  
end

function ExperimentLog:size()
  return table.getn(self.jsons)
end

function ExperimentLog:getColumn(name)
  local s=#self.jsons
  local c=torch.Tensor(s)
  for k,v in ipairs(self.jsons) do
    c[k]=v[name]
  end
  return c
end
