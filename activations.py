class para_act_relu(Function):
    
    @staticmethod
    def forward(ctx, input, a, b):

        ctx.save_for_backward(input, a, b)
        output = a * torch.relu(input) + b * torch.pow(input, 3) + (1 - a - b) * torch.tanh(input)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        
        input, a, b = ctx.saved_tensors
        
        grad_a = None
        grad_b = None
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
    
        if ctx.needs_input_grad[0]:
            grad_input = a * grad_input + (3 * b * torch.pow(input, 2) + (1 - a - b) * (1 - torch.tanh(input)**2)) * grad_output
               
        if ctx.needs_input_grad[1]:
            grad_a = (torch.relu(input) - torch.tanh(input)) * grad_output # + lam * (a - torch.mean(a))           
            grad_a[:, :] = torch.mean(grad_a, dim = 0)
                     
        if ctx.needs_input_grad[2]:
            grad_b = (torch.pow(input, 3) - torch.tanh(input)) * grad_output # + lam * (b - torch.mean(b))            
            grad_b[:, :] = torch.mean(grad_b, dim = 0)
                     
        return grad_input, grad_a, grad_b

class para_act_sigmoid_tanh(Function):
    
    @staticmethod
    def forward(ctx, input, a, b):
        
        act_e2 = b * input.clone() + 1/2
        act_e2_temp = act_e2.clone()        
        act_e2[ act_e2_temp > 1] = 1
        act_e2[ act_e2_temp < 0] = 0
        output = a * torch.sigmoid(input) + (1-a) * act_e2
        ctx.save_for_backward(input, a, b, act_e2)       

        return output
    
    @staticmethod
    def backward(ctx, grad_output, grad_pooling = False):
        
        input, a, b, act_e2 = ctx.saved_tensors
        
        grad_a = None
        grad_b = None
        
        act_e2_temp = b * input.clone() + 1/2
        grad_b_e2 = (1-a) * input
        grad_b_e2[ act_e2_temp < 0 ] = 0
        grad_b_e2[ act_e2_temp > 1 ] = 0
        
        grad_input_e2 = (1 - a) * b * torch.ones_like(input, dtype = torch.float) 
        grad_input_e2[ act_e2_temp < 0 ] = 0
        grad_input_e2[ act_e2_temp > 1 ] = 0
    
        if ctx.needs_input_grad[0]:
            grad_input = (a * torch.sigmoid(input) * (1 - torch.sigmoid(input)) + grad_input_e2) * grad_output
         
        if ctx.needs_input_grad[1]:
            grad_a = (torch.sigmoid(input) - act_e2) * grad_output

        if ctx.needs_input_grad[2]:
            grad_b = grad_b_e2 * grad_output
            
        if grad_pooling == True:
            grad_a = torch.mean(grad_a) * torch.ones(grad_a.shape)
            grad_b = torch.mean(grad_b) * torch.ones(grad_b.shape)
            grad_a = grad_a.type(torch.cuda.FloatTensor)
            grad_b = grad_b.type(torch.cuda.FloatTensor)
            
        return grad_input, grad_a, grad_b
        
class prelu(Function):
    
    @staticmethod
    def forward(ctx, input, a):
        
        act_e2 = input.clone()
        act_e2_a = a * input.clone()
        act_e2[ input < 0] = act_e2_a[ input < 0] # input.clone()
        output = act_e2
        ctx.save_for_backward(input, a, act_e2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output, grad_pooling = False):
        
        input, a, act_e2 = ctx.saved_tensors
        
        grad_a = None
    
        if ctx.needs_input_grad[0]:
            grad_input = 1 * torch.ones_like(input, dtype = torch.float)
            grad_a_0 = a * torch.ones_like(input, dtype = torch.float)
            grad_input[input < 0] = grad_a_0[input < 0]  
            grad_input = grad_input * grad_output
            
        if ctx.needs_input_grad[1]:
            grad_a = input.clone()
            grad_a[input > 0] = 0
            grad_a = grad_a * grad_output
            
        return grad_input, grad_a
    
class p_act_layer_prelu(nn.Module):
    
    def __init__(self, input_features):
        
        super(p_act_layer_prelu, self).__init__()
        self.input_features = input_features
        
        self.a = nn.Parameter(torch.Tensor(input_features))
        # self.rand_act = args['rand_act']
        self.a.data = torch.tensor([0.0]*input_features, dtype = torch.float)
        # print("p_act_layer_prelu!!!")

    def forward(self, input):
        # print("a", torch.mean(self.a))
        return prelu.apply(input, self.a)

class p_act_layer_sig_tanh(nn.Module):
    
    def __init__(self, input_features, layer_type = "sigmoid_slope"):
        
        
        super(p_act_layer_all, self).__init__()
        self.input_features = input_features
        self.layer_type = layer_type
        
        self.a = nn.Parameter(torch.Tensor(input_features))
        self.b = nn.Parameter(torch.Tensor(input_features))

        self.a.data = torch.tensor([1.0]*input_features, dtype = torch.float)
        self.b.data = torch.tensor([0.1]*input_features, dtype = torch.float)

    def forward(self, input):
     
        if self.layer_type == "sigmoid_slope":
            return para_act_sigmoid_tanh.apply(input, self.a, self.b), [torch.mean(self.a.data), torch.mean(self.b.data)], [torch.std(self.a.data), torch.std(self.b.data)]
        if self.layer_type == "tanh_slope":
            return para_act_tanh.apply(input, self.a, self.b)
             
class p_act_relu(nn.Module):
    
    def __init__(self, input_features, args):
        
        super(p_act_layer, self).__init__()
        self.input_features = input_features
        
        self.a = nn.Parameter(torch.Tensor(input_features))
        self.b = nn.Parameter(torch.Tensor(input_features))
        # self.lam = args['reg_act']
        self.rand_act = args['rand_act']

        if self.rand_act == True:
            self.a.data = 0.8 + torch.rand(input_features)/5
        else:
            self.a.data = torch.tensor([1.0]*input_features, dtype = torch.float)
            
        self.b.data = torch.tensor([0.0]*input_features, dtype = torch.float)
        
    def forward(self, input):

        return para_act_relu.apply(input, self.a, self.b)
