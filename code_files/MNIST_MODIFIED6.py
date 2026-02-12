
""" Modified image dataset of colored images. This will return a dataset object
which can be passed on to dataloaders from torch. """

""" The color_map arg is a dictionary which record the colors to be assigned to the
 foreground of each class of image. """

## Optimised dataset using vectorisation and masks

class MNIST_Modified0_5(Dataset):

  def __init__(self, color_map, easy_hard_split, bg_texture_intensity, perlin):

    self.perlin = perlin
    self.split = easy_hard_split
    self.texture_intensity = bg_texture_intensity
    self.color_map = torch.stack([color_map[i] for i in range(6)])[:, :, None, None]
    self.t = transforms.Compose([transforms.ToTensor()])
    self.assigned_colors = []
    self.training_dataset_original = datasets.MNIST(
        root = '/content/data', train = True, download = True, transform = self.t
        )
    self.indices = self.training_dataset_original.targets <= 5
    self.data = self.training_dataset_original.data[self.indices]
    self.targets = self.training_dataset_original.targets[self.indices]
    self.color_img_flat_bg = self.color_maps()
    self.color_img_text_bg = self.background_perlin()

  def color_maps(self):

    splitter_mask = torch.rand(len(self.data)) < self.split #shape is just 60K
    self.assigned_colors = self.color_map[self.targets]  # 60k, 3, 1, 1 Shape
    self.assigned_colors[~splitter_mask] = self.color_map[torch.randint(0, 6, [len(self.data)-splitter_mask.sum().item()])]
    return (self.data[:, None, :, :]*self.assigned_colors/255)

  def background_perlin(self):

    noise = self.assigned_colors * torch.rand([len(self.data), 1, 28, 28]) * self.texture_intensity
    bg_noise = noise * (1 - self.data[:, None, :, :]/255)
    return self.assigned_colors * self.data[:, None, :, :]/255 + bg_noise

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.perlin:
      return self.color_img_text_bg[idx], self.targets[idx]
    else:
      return self.color_img_flat_bg[idx], self.targets[idx]
