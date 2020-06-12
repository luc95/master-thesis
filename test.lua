require 'torch'

io.write("Hello world, from ", _VERSION, "!\n")

cmd = torch.CmdLine()
cmd:text()
cmd:option('-option', 1024, 'test option')
cmd:text()

opt = cmd:parse(arg or {})
print(opt)
