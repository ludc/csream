 require 'nn'
 require 'dpnn'
 
 local DREAM, parent = torch.class('csream.DREAM', 'nn.Module')

--- internal_cell : add or gru
function DREAM:__init(input_dimension,output_dimension,N,length,coef_l1,costs,internal_cell)
  parent.__init(self)
  
  self.length=length
  self.internal_cell=internal_cell
  assert((internal_cell=="add") or (internal_cell=="gru") or (internal_cell=="rnn"),"Internal cell is 'add' or 'gru' or 'rnn'")
  
  self.l1={}
  for tt=1,length do self.l1[tt]=coef_l1 end
  
  
  self.costs=costs
  self.N=N
  self.input_dimension=input_dimension
  self.output_dimension=output_dimension
  self.initial_state=torch.Tensor(self.N):fill(0)
  self:createModules()
  self:updateGModule()
  --os.exit(0)
end

function DREAM:createModules()
 self.constant_module=csream.MyConstant(0,self.N); 
 self.module_classify=nn.Linear(self.N,self.output_dimension)
 
 self.cells={}
 self.module_policies={}
 for i=1,self.length do
   self.module_policies[i]=nn.Linear(self.N,self.input_dimension)  
end

  if (self.internal_cell=="add") 
  then 
   local iin=nn.Identity()()
   local phi=nn.Linear(self.input_dimension,self.N)(iin)
   local h=nn.Identity()()
   local a=nn.CAddTable()({h,phi})    
   self.cell=nn.gModule({h,iin},{a})
  elseif (self.internal_cell=="rnn")
  then
    self.cell=csream.RNN():rnn_cell(self.input_dimension,self.N,self.N)
  else
   self.cell=csream.GRU():gru_cell(self.input_dimension,self.N)
  end
self.cells=csream.ModelsUtils():clone_many_times(self.cell,self.length)
end

function DREAM:updateGModule()
  local input_x=nn.Identity()()
  local initial_vector=self.constant_module(input_x)
  
  self.nmodule_policies={}
  for i=1,self.length do
    self.nmodule_policies[i]=nn.Sequential():add(self.module_policies[i]):add(nn.ReLU()):add(csream.MyL1Penalty(torch.mul(self.costs,self.l1[i])))
  end
  
  local zt=initial_vector
  for t=1,self.length do
      local alpha_t=self.nmodule_policies[t](zt)
      local x_t=nn.CMulTable()({alpha_t,input_x})      
      local transform=self.cells[t]({zt,x_t})
      zt=transform
  end
  
  local mm=self.module_classify(zt)
  self.module=nn.gModule({input_x},{mm})
end

function DREAM:reset(stdv,bias)
  for i=1,self.length do
    self.cells[i]:reset(stdv)
    
    self.module_policies[i]:reset(stdv)
    self.module_policies[i].bias:fill(bias)    
  end
  
  self.module_classify:reset(stdv)
end

---- Return the outputs of the policies
function DREAM:getPoliciesOutput()
  local outputs={}
  for i=1,self.length do 
    outputs[i]=self.nmodule_policies[i].output
  end
  return outputs
end


--- input is mu, sigma
function DREAM:updateOutput(input)
  self.module:updateOutput(input)
  self.output=self.module.output
  return self.output
end

function DREAM:updateGradInput(input, gradOutput)
  self.module:updateGradInput(input,gradOutput)
  self.gradInput=self.module.gradInput
  return(self.gradInput)
end

function DREAM:accGradParameters(input, gradOutput, scale)
  self.module:accGradParameters(input,gradOutput,scale)
end

function DREAM:zeroGradParameters()
   self.module:zeroGradParameters()
end

function DREAM:updateParameters(learningRate)
   self.module:updateParameters(learningRate)
end

-- we do not need to accumulate parameters when sharing
DREAM.sharedAccUpdateGradParameters = DREAM.accUpdateGradParameters

function DREAM:__tostring__()
return torch.type(self)
end
