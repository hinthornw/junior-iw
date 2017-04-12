--[[
author: William Hinthorn
hinthorn at princeton dot edu
--]]
require 'torch'
require 'nn'
require 'image'

cmd = torch.CmdLine()
cmd:text('Adversarial network generation')
cmd:text()
cmd:text('Options')
cmd:option('-i', 'none', 'Input image file')
cmd:option('-g', 'false', 'Greyscale input')
cmd:option('-o', './images/mnist/transform/', 'Output directory')
cmd:option('-cuda', false,'CUDA support')
cmd:option('-gpu', 1,'GPU device number')
cmd:option('-seed',123,'Random seed')
cmd:text()
params = cmd:parse(arg)

input_image = params['i']
greyscale = params['g']
outPath = params['o']
cuda = params['cuda']
deviceNum = params['gpu']
seed = params['seed']
threads = 4
mnist = true
transform = true
if cuda then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(deviceNum)
end

torch.manualSeed(seed)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)

package.path = package.path .. ';./mnist/?.lua;./overfeat/?.lua;./lbfgsb/?.lua'

if mnist then
	require 'mnist_model'
	mnist = require 'mnist_utils'
else
	require 'overfeat'
end
--For sample adversarial image
size = image.getSize(input_image)
local img = image.load(input_image,size[1],'byte')

save_name = input_image:gsub(".png", "")
save_name = save_name:gsub(".*/", "")

io.write("Testing file - ", save_name)
--Strip labels from filename
i, j = 0, 0
i, j = save_name:find('[0-9]+')
true_label = save_name:sub(i, j)
i, j = save_name:find('[0-9]+', j+1)
adv_label = save_name:sub(i, j)
i, j = save_name:find('[0-9]+', j+1)
seed = save_name:sub(i, j)


print("True Label: #" .. true_label)
print("Adversarial Prediction: #" .. adv_label)

-- Make a modified prediction function
makePrediction = function(x)
	if cuda then
		return torch.max(net:forward(x):float(), 1)
	else 
		return torch.max(net:forward(x), 1)
	end
end

showResults = function(xform, param, prob, idx)
	local guess = label[idx:squeeze()]
	local validity
	local result
	if tonumber(true_label) == tonumber(guess) then
		validity = 'Correct Prediction: '
		result = 0
	elseif tonumber(guess) == tonumber(adv_label) then
		validity = 'Trained Prediction: '
		result = 1
	else
		validity = 'Unexpected Prediction: '
		result = -1
	end
	io.write(xform,"Param: (", param, ") ", validity, "(", guess, ")\n")-- with prob - ",
		 -- prob:squeeze() .. "\n")
	return result
end


--Test that the input image does indeed spoof the model
local toTest = img
if cuda then toTest = toTest:cuda():reshape(mnist.n) end
--Get label
local prob, idx = makePrediction(toTest)
local result = showResults("Rotate", i, prob, idx)

if result ~= 1 then print("Image correctly classified. Exitting."); os.exit();
else print("Image spoofs model. Conducting tests")
end


-- Rotate to +/- 45 degrees using simple interpolation
print('Rotating image')
for i = -45, 45, 5 do
	local imgRot = image.rotate(img, i * math.pi/180, mode)
	local toTest = imgRot
	if cuda then toTest = toTest:cuda():reshape(mnist.n) end
	--Get label
	local prob, idx = makePrediction(toTest)
	local result = showResults("Rotate", i, prob, idx)
	if result == -1 then
		image.save(outPath .. save_name .. '_rotated_' .. 'strange_' .. i .. '.png', imgRot)
	elseif result == 0 then
		image.save(outPath .. save_name .. '_rotated_' .. 'correct_' .. i .. '.png', imgRot)
	end
end


--Random noise
print('Adding Random Gaussian noise')
for i = 0, 255, 5 do
	local r = (torch.rand(size[2], size[3]) * i - (i/2)):floor()
	-- print(r)
	local imgNoise = img:float():add(r)
	-- print(imgNoise)
	local min = imgNoise:min()
	local max = imgNoise:max()
	--Renormalize values in [0, 255)
	imgNoise = imgNoise:add(-min):div((max-min)/255.0):floor():byte()
	local toTest = imgNoise
	if cuda then toTest = toTest:cuda():reshape(mnist.n) end
	--Get label
	local prob, idx = makePrediction(toTest)
	local result = showResults("Random Noise", i, prob, idx)
	if result == -1 then
		image.save(outPath .. save_name .. '_rand_noise_' .. 'strange_' .. i .. '.png', imgNoise)
	elseif result == 0 then
		image.save(outPath .. save_name .. '_rand_noise_' .. 'correct_' .. i .. '.png', imgNoise)
	end
end


--Try a negative image
print('Negative Scaled Image')
local toTest = -img + 255
local norm = toTest:clone()
if cuda then toTest = toTest:cuda():reshape(mnist.n) end
--Get label
local prob, idx = makePrediction(toTest)
local result = showResults("Negative Image", -1, prob, idx)
if result == -1 then
	image.save(outPath .. save_name .. '_negative_' .. 'strange_' .. i .. '.png', norm)
elseif result == 0 then
	image.save(outPath .. save_name .. '_negative_' .. 'correct_' .. i .. '.png', norm)
end



--Subtract adverserial mean from the adv image
print('Subtracting the mean of the adversarial spoofed label')

for i = 0, 0.5, 0.025 do 
	local toTest = img:float():add(-i*mnist.means[adv_label+1])
	--Renormalize
	local min = toTest:min()
	local max = toTest:max()
	--Renormalize values in [0, 255)
	toTest = toTest:add(-min):div((max-min)/255.0):floor():byte()
	local norm = toTest:clone()
	if cuda then toTest = toTest:cuda():reshape(mnist.n) end
	--Get label
	local prob, idx = makePrediction(toTest)
	local result = showResults("Subtract Adv Mean", i, prob, idx)
	if result == -1 then
		image.save(outPath .. save_name .. '_sub_adv_mean_' .. 'strange_' .. i .. '.png', norm)
	elseif result == 0 then
		image.save(outPath .. save_name .. '_sub_adv_mean_' .. 'correct_' .. i .. '.png', norm)
	end

end



--Apply a gaussian filter
print('Try adding gaussian blur')
local kern = image.gaussian()
print(kern)
toTest = img:float()
toTest = image.convolve(toTest, kern, 'same'):byte()
local norm = toTest:clone()
if cuda then toTest = toTest:cuda():reshape(mnist.n) end
--Get label
local prob, idx = makePrediction(toTest)
local result = showResults("Gaussian Blur", i, prob, idx)
if result == -1 then
	image.save(outPath .. save_name .. '_gaussian_' .. 'strange_' .. i .. '.png', norm)
elseif result == 0 then
	image.save(outPath .. save_name .. '_gaussian_' .. 'correct_' .. i .. '.png', norm)
end

--Apply a gaussian filter
print('Try adding laplacian blur')
local kern = image.laplacian(3, 0.25, 1, true)
toTest = norm:float()
toTest = image.convolve(toTest, kern, 'same'):byte()
local norm = toTest:clone()
if cuda then toTest = toTest:cuda():reshape(mnist.n) end
--Get label
local prob, idx = makePrediction(toTest)
local result = showResults("Laplacian Blur", i, prob, idx)

if result == -1 then
	image.save(outPath .. save_name .. '_laplacian_' .. 'strange_' .. i .. '.png', norm)
elseif result == 0 then
	image.save(outPath .. save_name .. '_laplacian_' .. 'correct_' .. i .. '.png', norm)
end


