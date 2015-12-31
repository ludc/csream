local GRU = torch.class('csream.GRU'); 



--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU:gru_cell(observation_size, latent_size)
    
  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, latent_size)(xv)
    local h2h = nn.Linear(latent_size, latent_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  
   local x = nn.Identity()()
   local prev_h=nn.Identity()()
  
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(observation_size, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(observation_size, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(latent_size, latent_size)(gated_hidden)
    local p1 = nn.Linear(observation_size, latent_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    return nn.gModule({prev_h,x}, {next_h})
end


function GRU:gru_cell_no_input(latent_size)
    
  function new_input_sum( hv)
    local h2h = nn.Linear(latent_size, latent_size)(hv)
    return h2h
  end

  
   local prev_h=nn.Identity()()
  
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(  prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum( prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(latent_size, latent_size)(gated_hidden)
    local hidden_candidate = nn.Tanh()(p2)
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    return nn.gModule({prev_h}, {next_h})
end