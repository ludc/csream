require 'svm'
require 'nn'
require 'csream'



----- READ DATA from libSVM (using .train, .test and .validation files)
function readFromLibSVMs(filename,default_value)
  if (default_value==nil) then default_value=0 end
  print("Loading training file : "..filename..".train")
  local dataset=svm.ascread(filename..".train")
  
  local dim_max=0
  local nb_examples=0
  local categories_map={}
  local nb_categories=0
  for k,v in pairs(dataset) do
    local category=v[1]
    if (categories_map[category]==nil) then local s=(nb_categories+1); categories_map[category]=s; nb_categories=nb_categories+1 end
    local features=v[2][1]
    for i=1,features:size(1) do 
      local f=features[i]
      if (f>dim_max) then dim_max=f end
    end
  end
  print("Dimension input "..dim_max)
  print("Dimension output "..nb_categories)
  local examples={}
  for k,v in pairs(dataset) do
    local category=v[1]
    local index_category=categories_map[category]
    local y=torch.Tensor(nb_categories):fill(-1)
    y[index_category]=1
    examples[k]={}
    examples[k].y=y
    
    local features=v[2][1]
    local values=v[2][2]
    local x=torch.Tensor(dim_max):fill(default_value)
    for i=1,features:size(1) do 
      local f=features[i]
      local val=values[i]
      x[f]=val
    end
    examples[k].x=x
  end
   print("\tFound "..#examples.." training examples")
  
print("Loading testing file : "..filename..".test")
   dataset=svm.ascread(filename..".test")
   local test_examples={}

  for k,v in pairs(dataset) do
    local category=v[1]
    local index_category=categories_map[category]
    local y=torch.Tensor(nb_categories):fill(-1)
    y[index_category]=1
    test_examples[k]={}
    test_examples[k].y=y
    
    local features=v[2][1]
    local values=v[2][2]
    local x=torch.Tensor(dim_max):fill(default_value)
    for i=1,features:size(1) do 
      local f=features[i]
      local val=values[i]
      if (f<=dim_max) then x[f]=val end
    end
    test_examples[k].x=x
  end
  print("\tFound "..#test_examples.." testing examples")
  
 print("Loading validation file : "..filename..".validation")
   dataset=svm.ascread(filename..".validation")
   local val_examples={}

  for k,v in pairs(dataset) do
    local category=v[1]
    local index_category=categories_map[category]
    local y=torch.Tensor(nb_categories):fill(-1)
    y[index_category]=1
    val_examples[k]={}
    val_examples[k].y=y
    
    local features=v[2][1]
    local values=v[2][2]
    local x=torch.Tensor(dim_max):fill(default_value)
    for i=1,features:size(1) do 
      local f=features[i]
      local val=values[i]
      if (f<=dim_max) then x[f]=val end
    end
    val_examples[k].x=x
  end
  print("\tFound "..#val_examples.." validation examples")
  
  return({train=examples,test=test_examples,validation=val_examples})
  
end

function computeAccuracyAndSparsity(set,model,device,weights)
  local nb_ok=0
  local nb_total=0
  
  local sp_total=0
  local sp_levels={}
  local ent={}
  local total=torch.Tensor(set[1].x:size(1),set[1].x:size(2)):fill(0)
  
  if (device>0) then 
    total=total:cuda()
  end
  
  local cost=0
  for k,v in ipairs(set) do
    total:fill(0)
        local y=v.y
        local x=v.x
        local py=model:forward(x)
        local vmax,imax=py:max(2)
        for j=1,imax:size(1) do
          if (y[j][imax[j][1]]==1) then nb_ok=nb_ok+1 end
          nb_total=nb_total+1          
        end

        --- Sparsity
        local outputs=model:getPoliciesOutput()
        local probas=model:getPoliciesProbas()
          for i=1,#outputs do               
            if (sp_levels[i]==nil) then sp_levels[i]=0 end
            if (ent[i]==nil) then ent[i]=0 end
            sp_levels[i]=sp_levels[i]+torch.eq(outputs[i],0):sum()/total:size(2)
            
            total:add(outputs[i])
            
            local e=torch.log(probas[i]):cmul(probas[i])            
            
            e[torch.ne(e,e)]=0
            
            ent[i]=ent[i]+e:sum()/probas[i]:size(2)
          end
          sp_total=sp_total+torch.eq(total,0):sum() / (total:size(2))
          
          local c=torch.Tensor(total:size()):copy(torch.ne(total,0))
          assert(c:size(2)==weights:size(1))
          for j=1,total:size(1) do            
              local p=torch.cmul(weights,c[j]):sum()
              cost=cost+p
          end
--          if (sp_total/nb_total==1) then print(outputs[1]); os.exit(0); end
  end
  for i=1,#sp_levels do sp_levels[i]=sp_levels[i]/nb_total; ent[i]=ent[i]/nb_total end
  
  sp_total=sp_total/nb_total
  cost=cost/nb_total
  local accuracy=nb_ok/nb_total
  return({accuracy=accuracy,sparsity=sp_total,sparsities=sp_levels,entropies=ent,cost=cost})
end


--- Same function, but on CPU
function computeAccuracyAndSparsityCPU(set,model,device,weights)
    local nb_ok=0
  local nb_total=0
  
  local sp_total=0
  local sp_levels={}
  local total=torch.Tensor(set[1].x:size(1),set[1].x:size(2)):fill(0)
  
  if (device>0) then 
    total=total:cuda()
  end
local   cost=0
  for k,v in ipairs(set) do
    total:fill(0)
        local y=v.y
        local x=v.x
        if (device>0) then x=x:cuda(); y=y:cuda() end
        
        local py=model:forward(x)
        local vmax,imax=py:max(2)
        for j=1,imax:size(1) do
          if (y[j][imax[j][1]]==1) then nb_ok=nb_ok+1 end
          nb_total=nb_total+1          
        end

        --- Sparsity
        local outputs=model:getPoliciesOutput()
          for i=1,#outputs do    
            if (sp_levels[i]==nil) then sp_levels[i]=0 end
            sp_levels[i]=sp_levels[i]+torch.eq(outputs[i],0):sum()/total:size(2)
            
            total:add(outputs[i])
          end
          sp_total=sp_total+torch.eq(total,0):sum() / (total:size(2))
--          if (sp_total/nb_total==1) then print(outputs[1]); os.exit(0); end

          local c=torch.Tensor(total:size()):copy(torch.ne(total,0))
          assert(c:size(2)==weights:size(1))
          for j=1,total:size(1) do            
              local p=torch.cmul(weights,c[j]):sum()
              cost=cost+p
          end


  x=nil
  y=nil
  end
  for i=1,#sp_levels do sp_levels[i]=sp_levels[i]/nb_total end
  
  sp_total=sp_total/nb_total
  cost=cost/nb_total
  local accuracy=nb_ok/nb_total
  return({accuracy=accuracy,sparsity=sp_total,sparsities=sp_levels,cost=cost})
end




-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
cmd=torch.CmdLine()
cmd:text()
cmd:option('--learningRate', 0.01, 'learning rate')
cmd:option('--l1',0.0,'the starting l1 coefficient')
cmd:option('--maxL1',-1,'the l1 coefficient at the end of the process. -1 means that l1 will be constant during the process')
cmd:option('--costs','no','the cost of each feature (>=0) separated by a \':\'. If costs == "no" then a cost of one is associated with each features. ')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--saturateEpoch', -1, 'The epoch where the  l1 value will reach "maxL1"')
cmd:option('--burninEpoch', 0, 'Number of iterations where l1 is kept constant at the beginning of the process')

cmd:option('--evaluationEpoch', 10, 'Number of epochs after which an evaluation of the model is made on the training set')
cmd:option('--evaluationEpochTest', 10, 'Number of epochs after which an evaluation of the model is made on all the sets')

cmd:option('--dataset', '', 'input file (libSVM format)')
cmd:option('--default_value', 0, 'Default value for empty features when loading from libSVM files')

cmd:option('--uniform', 0.1, 'initialize model\'s parameters using uniform distribution between -uniform and uniform.')
cmd:option('--bias', 1, 'Value for the initialization of the bias of the linear policy layer. A value > 0 encourages the model to select all the features at the beginning of the process')
cmd:option('--size', 1, 'The number of steps')
cmd:option('--N', 10, 'The dimension of the latent space')
cmd:option('--cell', 'add', 'Nature of the internal dynamic module : add or rnn or gru')

cmd:option('--logPath', './log', 'The path to the log file')
cmd:option('--verbose', 'false', 'Verbose ? if true, the output is on the console. If false, the output will be produced as a csv file in the logPath directory')

cmd:option('--run', 1, 'index of the run (used if one runs many runs with the same parameters)')
cmd:option('--device', 0, 'GPU Device, 0 means that GPU will not be used')
cmd:option('--size_minibatch', 1, 'Size of the minibatches.')

cmd:option('--l1_discount', 1.0, 'The l1 discount for different steps of the process.')
cmd:option('--ent_factor', 1.0, '')
cmd:text()



local opt = cmd:parse(arg or {})
if not opt.silent then
   print(opt)
end

--- Initialization of the GPU if needed
if (opt.device>0) then 
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.device) 
end

--- Initialization of the hyper-parameters
if (opt.maxL1<0) then opt.maxL1=opt.l1 end
if (opt.saturateEpoch<0) then opt.saturateEpoch=opt.maxEpoch end
if (opt.evaluationEpochTest==0) then opt.evaluationEpochTest=opt.evaluationEpoch end

--- Creation of the log (console or file)
if (opt.verbose=='false') then log=csream.ExperimentLogCSV(false,opt.logPath,"now")
  else log=csream.ExperimentLogConsole(false) end
log:addFixedParameters(opt)

---- Reading the dataset
splitted=readFromLibSVMs(opt.dataset)
train_dataset=splitted.train
test_dataset=splitted.test
validation_dataset=splitted.validation

--- isize is the input dimension, and osize is the output dimension
local isize=train_dataset[1].x:size(1)
local osize=train_dataset[1].y:size(1)

local costs=nil

if (opt.costs=="linear") then
  local step=1.0/isize
  costs=torch.Tensor(isize)
  for i=1,isize do costs[i]=i*step end
elseif (opt.costs=="no") then
  costs=torch.Tensor(isize)
  for i=1,isize do costs[i]=1 end
else
  local yy=1
  local wei={}
  for w in string.gmatch(opt.costs,"[^:]+") do
    wei[yy]=tonumber(w)
    print("Cost for feature "..yy.." = "..wei[yy])
    yy=yy+1
  end
  costs=torch.Tensor(yy-1)
  for iui=1,yy-1 do costs[iui]=wei[iui] end
end
assert(isize==costs:size(1),"The number of costs is different than the number of input features...")

--- Buld the minibatches form the loaded datasets
print(" -- Building minibatches")
local train_batch_dataset={}
local test_batch_dataset={}
local validation_batch_dataset={}

do
	local sb=opt.size_minibatch
	local nb=#train_dataset
	
	
	for k=1,(nb/sb)+1 do
		local nx=torch.Tensor(sb,isize)
		local ny=torch.Tensor(sb,osize)
		for t=1,sb do 
			local idx=math.random(nb)
			nx[t]:copy(train_dataset[idx].x)
			ny[t]:copy(train_dataset[idx].y)
		end
		train_batch_dataset[k]={}
		train_batch_dataset[k].x=nx
		train_batch_dataset[k].y=ny
	end
	train_dataset=nil
	train_dataset=train_batch_dataset

nb=#test_dataset

	for k=1,(nb/sb)+1 do
		local nx=torch.Tensor(sb,isize)
		local ny=torch.Tensor(sb,osize)
		for t=1,sb do 
			local idx=math.random(nb)
			nx[t]:copy(test_dataset[idx].x)
			ny[t]:copy(test_dataset[idx].y)
		end
		test_batch_dataset[k]={}
		test_batch_dataset[k].x=nx
		test_batch_dataset[k].y=ny
	end
	test_dataset=nil
	test_dataset=test_batch_dataset

nb=#validation_dataset

	for k=1,(nb/sb)+1 do
		local nx=torch.Tensor(sb,isize)
		local ny=torch.Tensor(sb,osize)
		for t=1,sb do 
			local idx=math.random(nb)
			nx[t]:copy(validation_dataset[idx].x)
			ny[t]:copy(validation_dataset[idx].y)
		end
		validation_batch_dataset[k]={}
		validation_batch_dataset[k].x=nx
		validation_batch_dataset[k].y=ny
	end
	validation_dataset=nil
	validation_dataset=validation_batch_dataset
end


--- Building the model and initialization
model=csream.BREAM(isize,osize,opt.N,opt.size,opt.l1,costs,opt.cell) 
model:reset(opt.uniform,opt.bias)

---- Learning criterion
LOSS=nn.MSECriterion() 

---- Initialization of the regularization coefficients (one coefficient per step)
local L1=opt.l1
for i=1,opt.size do model.l1[i]=math.pow(opt.l1_discount,i-1)*L1; model.l1[i]=model.l1[i]/opt.size_minibatch end
model:updateGModule(opt.device>0) -- This function must be called when the model.l1 values are changed

local pairwise=nn.PairwiseDistance(2)
 
 ---Transfer on GPUs if needed (data+model)
if (opt.device>0) then
  print("Copying training data on GPU")
  for i=1,#train_dataset do 
   train_dataset[i].x=train_dataset[i].x:cuda()
   train_dataset[i].y=train_dataset[i].y:cuda()
  end
 print("Copying model on GPU")
 model:cuda()
 LOSS:cuda()
 pairwise:cuda()
end
 
local LR=opt.learningRate  

--- Evaluation of the model at the beginning of the learning process
--- entropies is the policy entropy allowing to know if the model is converging to a deterministic acquisition policy
acc_train=computeAccuracyAndSparsity(train_dataset,model,opt.device,costs)
acc_test=computeAccuracyAndSparsityCPU(test_dataset,model,opt.device,costs)
acc_val=computeAccuracyAndSparsityCPU(validation_dataset,model,opt.device,costs)
log:newIteration()
log:addValue("accuracy_train",acc_train.accuracy)
log:addValue("entropies_train",acc_train.entropies)
log:addValue("sparsity_train",acc_train.sparsity)
log:addValue("sparsities_train",acc_train.sparsities)
log:addValue("accuracy_test",acc_test.accuracy)
log:addValue("sparsity_test",acc_test.sparsity)
log:addValue("cost_train",acc_train.cost)
log:addValue("cost_test",acc_test.cost)
log:addValue("cost_validation",acc_val.cost)
log:addValue("accuracy_validation",acc_val.accuracy)
log:addValue("sparsity_validation",acc_val.sparsity)
log:addValue("l1_value",L1)
log:addValue("loss_value",0)
log:addValue("lr_value",LR)

local nb_training_examples=#train_dataset
local nb_max_sparsity=0
for iteration=1,opt.maxEpoch do
  LR=opt.learningRate
  
  local loss=0
  for k=1,nb_training_examples do
    --- One minibatch is sampled at random
    local idx=math.random(nb_training_examples)
    local x=train_dataset[idx].x
    local y=train_dataset[idx].y
    local nb_examples=train_dataset[idx].x:size(1)
    
    model:zeroGradParameters()
    local py=model:forward(x)    
    
    --- Compute the ||py-y||^2 for each example in the minibatch. This error will be used as a reward signal. The reward is rescaled w.r.t its average on the minibatch to reduce the variance
    local individual_loss=pairwise:forward({py,y})
    assert(nb_examples==individual_loss:size(1))
    local average_loss=individual_loss:sum()/nb_examples
    individual_loss:add(-average_loss):mul(opt.ent_factor) -- the opt.ent_factor is used if one wants to reduce the influence of the loss on the policy.
    
    local leloss=LOSS:forward(py,y)
    loss=loss+leloss
    
    --- Set the reward value on the policies modules
    model:reinforce(-individual_loss)
    
    --- Backpropagation
    local deltas=LOSS:backward(py,y)    
    model:backward(x,deltas)
    model:updateParameters(LR)
  end
  loss=loss/nb_training_examples
  
  --- Stop if loss==NaN
  if (loss~=loss) then os.exit(1) end
  
  if (opt.verbose=="true") then print("At "..iteration.." loss is "..loss) end
  
  --- Evaluate the quality of the model
  if (iteration%opt.evaluationEpoch==0) then
    acc_train=computeAccuracyAndSparsity(train_dataset,model,opt.device,costs)

    log:newIteration()
    log:addValue("accuracy_train",acc_train.accuracy)
    log:addValue("entropies_train",acc_train.entropies)
    log:addValue("cost_train",acc_train.cost)
    log:addValue("sparsity_train",acc_train.sparsity)
    log:addValue("sparsities_train",acc_train.sparsities)
    log:addValue("loss_value",loss)    
    log:addValue("l1_value",L1)
    log:addValue("lr_value",LR)
    print("At iteration "..iteration.." loss is "..loss.. " acc is "..acc_train.accuracy.." sp is "..acc_train.sparsity.." l1 is "..L1.." LR is "..LR)
    if (iteration%opt.evaluationEpochTest==0) then
        
        acc_test=computeAccuracyAndSparsityCPU(test_dataset,model,opt.device,costs)
        acc_val=computeAccuracyAndSparsityCPU(validation_dataset,model,opt.device,costs)

        log:addValue("accuracy_test",acc_test.accuracy)
        log:addValue("sparsity_test",acc_test.sparsity)
        log:addValue("accuracy_validation",acc_val.accuracy)
        log:addValue("sparsity_validation",acc_val.sparsity)
        log:addValue("cost_test",acc_test.cost)
        log:addValue("cost_validation",acc_val.cost)
    end
   
    --if sparsity is 1 during 10 evaluation, then stop the process since the model is not using any features...
    if (acc_train.sparsity>=0.999) then nb_max_sparsity=nb_max_sparsity+1 else nb_max_sparsity=0 end
    if (nb_max_sparsity>10) then os.exit(1) end
    
  end
  
  ---- Change the value of l1
  if (iteration>opt.burninEpoch) then 
    local stepl1=(opt.maxL1-opt.l1)/(opt.saturateEpoch-opt.burninEpoch)
    local ll=opt.l1+stepl1*(iteration-opt.burninEpoch)
    if (ll>opt.maxL1) then ll=opt.maxL1 end
    --print("L1 = "..ll)
    L1=ll
    for iii=1,opt.size do model.l1[iii]=math.pow(opt.l1_discount,iii-1)*L1; model.l1[iii]=model.l1[iii]/opt.size_minibatch end
    model:updateGModule(opt.device>0)
  end
 
end
  



