class image_optimiser():
  def __init__(self, model, lr = 0.1, step = 400, device = device):
    self.model = model
    self.model.eval()
    self.lr = lr
    self.step = step
    self.activation_dictionary = {}

  def activation_collector(self, layer_name):

    def data_hook(model, input, output):
      self.activation_dictionary[layer_name] = output
    return data_hook

  def maximiser(self, layer_index, layer_name, filter_index):
    hook = layer_index.register_forward_hook(self.activation_collector(layer_name))
    self.model.eval()
    image = torch.rand(1, 3, 28, 28, device = device, requires_grad = True)
    image = image.to(device)
    optimiser = torch.optim.Adam([image], lr = self.lr)

    for i in range(self.step):
      optimiser.zero_grad()
      output = model(image)
      loss = - self.activation_dictionary[layer_name][0,filter_index].mean() + 0.01*torch.norm(image)
      loss.backward()
      optimiser.step()

    hook.remove()

    return image.detach().cpu().numpy()
