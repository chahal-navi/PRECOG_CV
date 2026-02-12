""" For simplicity of implementation we will use the grad_cam only at the
 last convolutional layer's feature maps."""

class Grad_Cam():
  def __init__(self, model):
    self.model = model
    self.model.eval()
    self.gradient = None
    self.f_maps = None
    self.grad_hook()

  def grad_hook(self):

    def forwd(model, input, output):
      self.f_maps = output
      def backward_hook(grad):
        self.gradient = grad
      output.register_hook(backward_hook)
    self.model.con_layer3.register_forward_hook(forwd)

  def heatmap_gen(self, image, class_index):
    model_pred = self.model(image)
    self.model.zero_grad()
    feature_maps = self.f_maps
    class_activation = model_pred[0, class_index]
    class_activation.backward()
    weight = torch.mean(self.gradient, dim = (2, 3), keepdim = True)

    fmap_activation = F.relu(torch.sum(weight * feature_maps , dim=1, keepdim=True))
    fmap_activation = fmap_activation - fmap_activation.min()
    fmap_activation = fmap_activation / fmap_activation.max()
    mask = F.interpolate(fmap_activation, size=(28, 28), mode='bilinear', align_corners=False)
    mask = mask.squeeze().detach().cpu().numpy()
    image = einops.rearrange(image.squeeze(), 'c h w -> h w c').squeeze().cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    mask = numpy.uint8(255 * mask)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB) / 255.0
    superimposed = 0.6*mask_colored + 0.4 * image
    plt.imshow(superimposed)
    plt.show()



