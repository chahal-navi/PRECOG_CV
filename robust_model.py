def mask_creation(images, thresh):
  gray_img = images.mean(dim = 1, keepdim = True)

  mask = (gray_img > thresh)*1.0
  return mask

def test_loop(dataloader, model, loss_fn):

  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X = X.to(device)
      y = y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

      test_loss /= num_batches
      correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def new_loss_trainer(model, train_loader, test_dataloader, test_loss, lambda_color = 100000000000, epochs = 5, lambda_pen = 10000000000):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  model = model.to(device)
  model.train()
  for epoch in range(epochs):
    loss_t = 0
    correct = 0
    tot = 0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      images.requires_grad = True
      pred = model(images)
      loss1 = F.cross_entropy(pred, labels)

      gradient_wrt_image = torch.autograd.grad(loss1, images, create_graph = True, retain_graph = True, only_inputs = True)[0]
      bg_mask = 1 - mask_creation(images, 0.6)
      norm_bg_grad = (bg_mask * gradient_wrt_image).pow(2).sum(dim = (1, 2, 3)).mean()

      foreground_grads = gradient_wrt_image * mask_creation(images, 0.6)
      color_penalty = foreground_grads.std(dim=1, keepdim=True).mean()
      total_loss = loss1 + norm_bg_grad * lambda_pen + lambda_color * color_penalty
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      loss_t += total_loss.item()
      _, predicted = pred.max(1)
      tot += labels.size(0)
      correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch+1} | Acc: {100.*correct/tot:.2f}% | Loss: {loss_t/len(train_loader):.4f}")
    print(f" (Debug) CE: {loss1.item():.4f} | BG_Pen: {norm_bg_grad.item():.4f} | Color_Pen: {color_penalty.item():.4f}")
    test_loop(test_dataloader, model, test_loss)
  return model
