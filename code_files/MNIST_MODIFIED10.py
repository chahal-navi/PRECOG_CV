""" Modified image dataset of colored images. This will return a dataset object
which can be passed on to dataloaders from torch. """

""" The color_map arg is a dictionary which record the colors to be assigned to the
 foreground of each class of image. """

## Optimised dataset using vectorisation and masks

class MNIST_Modified(Dataset):

  def __init__(self, color_map, easy_hard_split, bg_texture_intensity, perlin):

    self.perlin = perlin
    self.split = easy_hard_split
    self.texture_intensity = bg_texture_intensity
    self.color_map = torch.stack([color_map[i] for i in range(10)])[:, :, None, None]
    self.t = transforms.Compose([transforms.ToTensor()])
    self.assigned_colors = []
    self.training_dataset_original = datasets.MNIST(
        root = '/content/data', train = True, download = True, transform = self.t
        )
    self.color_img_flat_bg = self.color_maps()

    # Normalization of the dataset.

    mean = self.color_img_flat_bg.view(len(self.color_img_flat_bg), 3, -1).mean(dim=(0, 2))
    std = self.color_img_flat_bg.view(len(self.color_img_flat_bg), 3, -1).std(dim=(0, 2))
    self.color_img_flat_bg = (self.color_img_flat_bg - mean[None, :, None, None]) / std[None, :, None, None]
    self.color_img_text_bg = self.background_perlin()
    mean = self.color_img_text_bg.view(len(self.color_img_text_bg), 3, -1).mean(dim=(0, 2))
    std = self.color_img_text_bg.view(len(self.color_img_text_bg), 3, -1).std(dim=(0, 2))
    self.color_img_text_bg = (self.color_img_text_bg - mean[None, :, None, None]) / std[None, :, None, None]



  def color_maps(self):


    splitter_mask = torch.rand(len(self.training_dataset_original)) < self.split #shape is just 60K
    self.assigned_colors = self.color_map[self.training_dataset_original.targets]  # 60k, 3, 1, 1 Shape
    self.assigned_colors[~splitter_mask] = self.color_map[torch.randint(0, 10, [60000-splitter_mask.sum().item()])]
    return (self.training_dataset_original.data[:, None, :, :]*self.assigned_colors/255)

  def background_perlin(self):

    noise = self.assigned_colors * torch.rand([60000, 1, 28, 28]) * self.texture_intensity
    bg_noise = noise * (1 - self.training_dataset_original.data[:, None, :, :]/255)
    return self.assigned_colors * self.training_dataset_original.data[:, None, :, :]/255 + bg_noise

  def __len__(self):
    return len(self.training_dataset_original)

  def __getitem__(self, idx):
    if self.perlin:
      return self.color_img_text_bg[idx], self.training_dataset_original[idx][1]
    else:
      return self.color_img_flat_bg[idx], self.training_dataset_original[idx][1]
