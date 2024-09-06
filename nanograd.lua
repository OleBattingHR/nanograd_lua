-- nanograd implementation in lua

-- patches
if not table.unpack then
	table.unpack = unpack
end

-- kinda prototype like object
Var = { val = 0.0, parents = {}, grad = 0.0 }

function Var:new(val, parents)
	assert(type(val) == "number")
	local var = { val = val, parents = parents or {}, grad = 0.0 }
	setmetatable(var, self)
	self.__index = self
	return var
end

function Var:backprop(bp)
	self.grad = self.grad + bp
	for _, v in ipairs(self.parents) do
		local parent, grad = table.unpack(v)
		parent:backprop(grad * bp)
	end
end

function Var:update(lr, clamp)
	local grad = self.grad
	if grad < -clamp then
		grad = -clamp
	elseif grad > clamp then
		grad = clamp
	end
	self.val = self.val - grad * lr
	self.grad = 0
end

-- helper for printing
function Var:__tostring()
	return string.format("Var\t| val  = %+.3e\n\t| grad = %+.3e", self.val, self.grad)
end

-- addition operator overloading
function Var:__add(other)
	if type(other) == "table" then
		return Var:new(self.val + other.val, { { self, 1 }, { other, 1 } })
	elseif type(other) == "number" then
		return Var:new(self.val + other, { { self, 1 } })
	end
end

-- multiplication operator overloading
function Var:__mul(other)
	if type(other) == "table" then
		return Var:new(self.val * other.val, { { self, other.val }, { other, self.val } })
	elseif type(other) == "number" then
		return Var:new(self.val * other, { { self, other } })
	end
end

-- negation operator overloading
function Var:__unm()
	return Var:new(-1) * self
end

-- subtraction operator overloading
function Var:__sub(other)
	return self + -other
end

-- exponentiation operator overloading
function Var:__pow(other)
	if type(other) == "table" then
		return Var:new(self.val ^ other.val, {
			{ self, other.val * self.val ^ (other.val - 1) },
			{ other, self.val ^ other.val * math.log(self.val) },
		})
	elseif type(other) == "number" then
		return Var:new(self.val ^ other, { { self, other * self.val ^ (other - 1) } })
	end
end

-- division operator overloading
function Var:__div(other)
	return self * (other ^ -1)
end

-- criterion function
function l2loss(a, b)
	return (a - b) ^ 2
end

-- main test
math.randomseed(1)
local n_epochs = 3000
local n_samples = 100
local n_print = 100
local x_scale = 3
local lr = 0.001
local clamp = 10
local discount = 0.996
local c1 = Var:new(math.random())
local c3 = Var:new(math.random())
local c5 = Var:new(math.random())
local c7 = Var:new(math.random())

for epoch = 1, n_epochs do
	local loss = Var:new(0)
	for i = 1, n_samples do
		local x = Var:new((2 * i / n_samples - 1) * x_scale)
		local y = Var:new(math.sin(x.val))
		local y_hat = c1 * x + c3 * x ^ 3 + c5 * x ^ 5 + c7 * x ^ 7
		loss = loss + l2loss(y, y_hat)
	end
	loss = loss / n_samples
	loss:backprop(1)
	if epoch % n_print == 0 then
		print(string.format("%4d: %.3e (%.3e, %.3e)", epoch, loss.val, lr, clamp))
	end
	c1:update(lr, clamp)
	c3:update(lr, clamp)
	c5:update(lr, clamp)
	c7:update(lr, clamp)
	-- lr = lr * discount
	clamp = clamp * discount
end

print(c1)
print(c3)
print(c5)
print(c7)
