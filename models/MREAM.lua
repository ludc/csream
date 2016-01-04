 require 'nn'
 require 'dpnn'
 
 local MREAM, parent = torch.class('csream.MREAM', 'nn.Module')

--- internal_cell : add or gru or rnn
--- input_dimension and output_dimension are the input and output dimensions of the vectors
--- N is the size of the latent space
--- length is the number of steps
--- coef_l1 is the value of the l1 (lambda) hyper-parameters
--- coef_l1 is the value of the l1 (lambda) hyper-parameters
--- costs is the table of costs for each feature
--- internal_cell is the nature of the dynamic cell (rnn, gru or add)
function MREAM:__init(input_dimension,output_dimension,N,length,coef_l1,costs,internal_cell)
  parent.__init(self)
  
  self.length=length
  self.internal_cell=internal_cell
  
  
  self.l1={}
  --- the l1 value is the same at all steps of the process
  for tt=1,length do self.l1[tt]=coef_l1 end
  
  self.weights=costs
  
  self.N=N
  self.input_dimension=input_dimension
  self.output_dimension=output_dimension
  self.initial_state=torch.Tensor(self.N):fill(0)
  
  --- Here, the value of the probability is between smoothing and 1-smoothing in order to avoid log(0) values during learning
  self.smoothing=0.01
  self.__mul=nn.MulConstant(1.0-2*self.smoothing);
  self.__add=nn.AddConstant(self.smoothing);
  
  self:createModules()
  self:updateGModule()
    
end

--- Creatin of the base modules 
function MREAM:createModules()
 self.constant_module=csream.MyConstant(0,self.N)
 
 self.module_classify=nn.Linear(self.N,self.output_dimension)
 
 self.cell=nil
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
  elseif (self.internal_cell=="gru") then 
   self.cell=csream.GRU():gru_cell(self.input_dimension,self.N)
  elseif (self.internal_cell=="rnn")  then
   self.cell=csream.RNN():rnn_cell(self.input_dimension,self.N,self.N)
  end
 
 self.cells=csream.ModelsUtils():clone_many_times(self.cell,self.length)
end

--- Creating of the whole module corresponding to the sequential model
function MREAM:updateGModule(cuda)
  local input_x=nn.Identity()()
  local initial_vector=self.constant_module(input_x)
  
  self.nmodule_policies={}
  for i=1,self.length do
    self.nmodule_policies[i]=nn.Sequential():add(self.module_policies[i]):add(nn.SoftMax()):add(self.__mul:clone()):add(self.__add:clone()):add(csream.MyL1Penalty(torch.mul(self.weights,self.l1[i]))):add(nn.ReinforceCategorical())
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
  if (cuda) then self.module:cuda() end
end

--- Reset modules parameters.  bias is used for the policy only and allow one to fix the initial sparsity of the models
function MREAM:reset(stdv,bias)
  self.cell:reset(stdv)
  for i=1,self.length do
    
    self.module_policies[i]:reset(stdv)
    self.module_policies[i].bias:fill(bias)    
  end
  
  self.module_classify:reset(stdv)
end

---- Return the outputs of the policies
function MREAM:getPoliciesOutput()
  local outputs={}
  for i=1,self.length do 
    outputs[i]=self.nmodule_policies[i].output
  end
  return outputs
end

---- Return the probabilities computes by the policy 
function MREAM:getPoliciesProbas()
  local outputs={}
  for i=1,self.length do 
    outputs[i]=self.nmodule_policies[i].modules[4].output  
  end
  return outputs
end

---- Reinforce over smapling modules
function MREAM:reinforce(reward)
  for i=1,self.length do
    self.nmodule_policies[i]:reinforce(reward)
  end
end


--- input is mu, sigma
function MREAM:updateOutput(input)
  self.module:updateOutput(input)
  self.output=self.module.output
  return self.output
end

function MREAM:updateGradInput(input, gradOutput)
  self.module:updateGradInput(input,gradOutput)
  self.gradInput=self.module.gradInput
  return(self.gradInput)
end

function MREAM:accGradParameters(input, gradOutput, scale)
  self.module:accGradParameters(input,gradOutput,scale)
end

function MREAM:zeroGradParameters()
   self.module:zeroGradParameters()
end

function MREAM:updateParameters(learningRate)
   self.module:updateParameters(learningRate)
end

-- we do not need to accumulate parameters when sharing
MREAM.sharedAccUpdateGradParameters = MREAM.accUpdateGradParameters

function MREAM:__tostring__()
return torch.type(self)
end
