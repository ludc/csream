require 'torch'
csream={}

--- MODULES
include('GRU.lua')
include('RNN.lua')
include('ModelsUtils.lua')
include('MyConstant.lua')
include('MyL1Penalty.lua')

include('BREAM.lua')
include('DREAM.lua')
include('MREAM.lua')

include('ExperimentLog.lua')
include('ExperimentLogConsole.lua')
include('ExperimentLogCSV.lua')

return csream
