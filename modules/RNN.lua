require 'nngraph'

local RNN = torch.class('csream.RNN'); 

--- output = TanH(W z_{t-1} + W' x_t)
-- LATENT, INPUT -> OUTPUT
function RNN:rnn_cell(input_size, latent_size,output_size)
  local os=latent_size
  if (output_size~=nil) then os=output_size end
  
  local input1=nn.Identity()()  
	local a=nn.Linear(latent_size,os)(input1)
  
  local input2=nn.Identity()()  
  local b=nn.Linear(input_size,os)(input2)
  local c=nn.CAddTable()({a,b})
  local d=nn.Tanh()(c)
  local le_module_dynamique=nn.gModule({input1,input2},{d})
  return le_module_dynamique
end
