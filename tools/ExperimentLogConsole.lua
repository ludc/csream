 require 'logroll'
 require 'json'
 
local ExperimentLogConsole = torch.class('csream.ExperimentLogConsole','csream.ExperimentLog'); 

function ExperimentLogConsole:__init(memory)
  csream.ExperimentLog.__init(self,memory)
  self.log=logroll.print_logger()
end

function ExperimentLogConsole:newIteration()
  if (self.iteration==0) then print(self.parameters) end
  if (not csream.ExperimentLog.isEmpty(self)) then
        self.log.info("Iteration "..self.iteration)
        self.log.info(json.encode(self.currentjson))
  end
  csream.ExperimentLog.newIteration(self)
end


function ExperimentLogConsole:addDescription(text)  
  io.stdout:write(text)
end
