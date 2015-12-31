------------------------------------------------------------------------
--[[ Constant ]]--
-- Outputs a constant value given an input.
-- If nInputDim is specified, uses the input to determine the size of 
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append constant inputs to
-- an input : nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity()) .
------------------------------------------------------------------------
local MyConstant, parent = torch.class("csream.MyConstant", "nn.Module")

function MyConstant:__init(value, outputDims)
   	parent.__init(self)
   self.outputDims=outputDims
   self.value = value
   self.output = torch.Tensor(outputDims):fill(value)
end

function MyConstant:updateOutput(input)
  if (input==nil) then self.output:resize(self.outputDims) 
	elseif (input:dim()~=self.output:dim()) then
	   if input:dim()==2 then
  	    self.output:resize(input:size(1),self.outputDims)
  	    self.output:fill(self.value)
   	 else
      	self.output:resize(self.outputDims)
	 	 end
  end

   return self.output
end

function MyConstant:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end

function MyConstant:cuda()
  self.output=self.output:cuda()
  self.gradInput=self.gradInput:cuda()
end
