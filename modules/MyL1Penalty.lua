local MyL1Penalty, parent = torch.class('csream.MyL1Penalty','nn.Module')

--This module acts as an L1 latent state regularizer, adding the 
--[gradOutput] to the gradient of the L1 loss. The [input] is copied to 
--the [output]. 
--l1 weight is a vector of weights, on for each dimension of the input

function MyL1Penalty:__init(l1weight)
    parent.__init(self)
    self.l1weight = l1weight     
end
    
function MyL1Penalty:updateOutput(input)
    local m = self.l1weight 
    local loss = m*input:norm(1) 
    self.loss = loss  
    self.output = input 
    return self.output 
end

function MyL1Penalty:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    if (input:dim()==1) then
        assert(input:size(1)==self.l1weight:size(1))        
        self.gradInput:copy(input):sign():cmul(self.l1weight)    
        self.gradInput:add(gradOutput)  
    else
        assert(input:size(2)==self.l1weight:size(1))        
      local a=torch.sign(input)
        for i=1,input:size(1) do
          self.gradInput[i]:copy(a[i]:cmul(self.l1weight))
        end
    end

    return self.gradInput 
end

function MyL1Penalty:clone()
  return nn.MyL1Penalty(l1weight:clone())
end