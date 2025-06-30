import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        else:
            self.skip= nn.Identity()

        num_groups = max(out_channels // 4, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm1 = nn.GroupNorm(num_groups = num_groups, num_channels = out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm2 = nn.GroupNorm(num_groups = num_groups, num_channels = out_channels)
        self.act2 = nn.SiLU()
        self.tembd = nn.Linear(time_embedding_dim, out_channels)

    def forward(self, x, time_embedding):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        t = self.tembd(time_embedding).view(-1, self.out_channels, 1, 1)
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.res1 = ResidualBlock(in_channels, out_channels, time_embedding_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_embedding_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_embedding):
        skip = x
        h = self.res1(x, time_embedding)
        h = self.res2(h, time_embedding)
        h = self.downsample(h)
        return h, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding =1)
        self.res1 = ResidualBlock(2*out_channels, out_channels, time_embedding_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_embedding_dim)
        
    def forward(self, x, time_embedding, skip):
        h = self.upsample(x)
        h = self.upconv(h)
        #print(" up h:",  h.shape, " skip:", skip.shape)
        h = torch.cat([h, skip], dim = 1)
        #print(" cat→", h.shape)
        h = self.res1(h, time_embedding)
        h = self.res2(h, time_embedding)
        return h


class UNet(nn.Module):
    def __init__(self,time_embedding_dim):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)

        self.down1 = DownBlock(64, 128, time_embedding_dim)
        self.down2 = DownBlock(128, 256, time_embedding_dim)
        self.down3 = DownBlock(256, 512, time_embedding_dim)

        self.bot1 = ResidualBlock(512, 512, time_embedding_dim)
        self.bot2 = ResidualBlock(512, 512, time_embedding_dim)

        self.up3 = UpBlock(512, 256, time_embedding_dim)
        self.up2 = UpBlock(256, 128, time_embedding_dim)
        self.up1 = UpBlock(128, 64, time_embedding_dim)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size = 1)
    
    def forward(self, x, t, time_embedding):

        h = self.initial_conv(x)
        #print("initial_conv: ", h.shape)

        h, skip1 = self.down1(h, time_embedding)
        #print("down1: ", h.shape)
        h, skip2 = self.down2(h, time_embedding)
        #print("down2: ", h.shape)
        h, skip3 = self.down3(h, time_embedding)
        #print("down3: ", h.shape)
        h = self.bot1(h, time_embedding)
        #print("bot1: ", h.shape)
        h = self.bot2(h, time_embedding)
        #print("bot2: ", h.shape)

        #print("skip3: ", skip3.shape)
        h = self.up3(h, time_embedding, skip3)
        #print("up3: ", h.shape)
        h = self.up2(h, time_embedding, skip2)
        #print("up2: ", h.shape)
        h = self.up1(h, time_embedding, skip1)
        #print("up1: ", h.shape)

        h = self.final_conv(h)
        return h



class Diffusion:
    def __init__(self, timesteps, B_start, B_end, time_embedding_dim: int = 128):
        self.timesteps = timesteps
        self.B_start = B_start
        self.B_end = B_end
        self.B_t = torch.linspace(B_start, B_end, timesteps)
        self.alpha_hat = torch.cumprod(1-self.B_t, dim = 0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
        self.time_embedding_dim = time_embedding_dim

        # --- pre-compute per-timestep quantities for sampling ---
        self.alpha_t = 1.0 - self.B_t                      # α_t = 1 − β_t
        self.sqrt_alpha_t = torch.sqrt(self.alpha_t)       # √α_t
        self.sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alpha_t)  # 1/√α_t
        self.sqrt_beta_t = torch.sqrt(self.B_t)            # √β_t
        # β_t / √(1− \bar{α}_t) term used in DDPM update
        self.beta_over_sqrt_one_minus_alphabar = self.B_t / self.sqrt_one_minus_alpha_hat

    def noise_scheduler(self, t):
        return self.alpha_hat[t]
    
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
        dataset = datasets.CIFAR10(root = "data", train=True, transform = transform, download = True)
        dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 2)
        return dataloader
    
    def show_images(self, images, labels=None):
        print("Image shape:", images.shape)            # should be [64, 3, 32, 32]
        print("Image min/max:", images.min(), images.max())  # should be close to -1 and 1
        if labels is not None:
            print("Label sample:", labels[:5])             # optional
        import matplotlib.pyplot as plt

        # Unnormalize from [-1,1] back to [0,1] for visualization
        image = (images[0] + 1) / 2
        image_np = image.permute(1, 2, 0).cpu().numpy()  # [C,H,W] → [H,W,C] for matplotlib
        # Clip to valid range
        image_np = np.clip(image_np, 0, 1)

        plt.imshow(image_np)
        if labels is not None:
            plt.title(f"Label: {labels[0]}")
        else:
            plt.title("Generated Image")
        plt.axis("off")
        plt.show()

    def show_images_grid(self, images, labels=None, num_images=16, show=True):
        import matplotlib.pyplot as plt
        
        # Take first 16 images
        images = images[:num_images]
        if labels is not None:
            labels = labels[:num_images]
        
        # Calculate grid dimensions
        grid_size = int(np.sqrt(num_images))
        
        # Create subplot grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        # Handle case where there's only one subplot
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_images):
            # Unnormalize from [-1,1] back to [0,1] for visualization
            image = (images[i] + 1) / 2
            image_np = image.permute(1, 2, 0).cpu().numpy()  # [C,H,W] → [H,W,C] for matplotlib
            # Clip to valid range
            image_np = np.clip(image_np, 0, 1)
            
            axes[i].imshow(image_np)
            if labels is not None:
                axes[i].set_title(f"Label: {labels[i].item()}")
            else:
                axes[i].set_title(f"Sample {i+1}")
            axes[i].axis("off")
        
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close()

    def forward_diffusion(self, x_0):
        ts = torch.randint(0, self.timesteps, (x_0.shape[0],), device = x_0.device)
        epsilon = torch.randn_like(x_0)
        # print(ts.shape)
        # print(x_0.shape)
        # print(epsilon.shape) 
        # print(self.sqrt_alpha_hat.view(-1,1,1,1).shape)
        # print(self.sqrt_one_minus_alpha_hat.shape)
        x_t = self.sqrt_alpha_hat[ts].view(-1,1,1,1) * x_0 + self.sqrt_one_minus_alpha_hat[ts].view(-1,1,1,1) * epsilon
        return x_t, ts, epsilon
    
    def get_timestep_embedding(self, ts):
        dim = torch.arange(self.time_embedding_dim //2, device = ts.device)
        freqs = 10000**(2*dim/self.time_embedding_dim)
        ts = ts.float().unsqueeze(1) / freqs
        embedding = torch.cat([torch.sin(ts), torch.cos(ts)], dim = 1)
        return embedding

    def training_loop(self, model, dataloader, optimizer, scheduler, device, loss_fn, epochs, sample_save_dir=None, loss_plot_path=None):
        model.train()
        model.to(device)
        self.B_t = self.B_t.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        self.sqrt_alpha_hat = self.sqrt_alpha_hat.to(device)
        self.sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat.to(device)
        # move sampling buffers
        self.alpha_t = self.alpha_t.to(device)
        self.sqrt_alpha_t = self.sqrt_alpha_t.to(device)
        self.sqrt_recip_alpha_t = self.sqrt_recip_alpha_t.to(device)
        self.sqrt_beta_t = self.sqrt_beta_t.to(device)
        self.beta_over_sqrt_one_minus_alphabar = self.beta_over_sqrt_one_minus_alphabar.to(device)
        
        # Prepare directories if saving samples
        import os
        if sample_save_dir:
            os.makedirs(sample_save_dir, exist_ok=True)

        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for x_0, _ in dataloader:
                # Move data to device
                x_0 = x_0.to(device)
                
                x_t, ts, epsilon = self.forward_diffusion(x_0)
                pred_noise = model(x_t, ts, self.get_timestep_embedding(ts))
                loss = loss_fn(pred_noise, epsilon)

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                # Step the scheduler after every parameter update
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss.item():.6f}")

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

            # Save samples grid
            if sample_save_dir:
                sample_path = os.path.join(sample_save_dir, f"samples_epoch_{epoch+1}.png")
                # Generate without displaying during training
                self.generate_samples(model, num_samples=16, img_size=(3,32,32), device=device, save_path=sample_path, show=False)

            # Record and plot loss curve
            epoch_losses.append(avg_loss)
            if loss_plot_path:
                self.save_loss_plot(epoch_losses, loss_plot_path)

    def reverse_diffusion(self, model, img_size=(3, 32, 32), device=None, num_samples=1):
        """DDPM sampling (ancestral) starting from pure Gaussian noise."""
        device = torch.device('cpu') if device is None else device
        model.eval()
        model.to(device)

        with torch.no_grad():
            x = torch.randn(num_samples, *img_size, device=device)

            for i in range(self.timesteps - 1, -1, -1):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                time_emb = self.get_timestep_embedding(t)

                eps_theta = model(x, t, time_emb)

                coef1 = self.sqrt_recip_alpha_t[i]
                coef2 = self.beta_over_sqrt_one_minus_alphabar[i]
                mean   = coef1 * (x - coef2 * eps_theta)

                if i > 0:
                    noise = torch.randn_like(x)
                    x = mean + self.sqrt_beta_t[i] * noise
                else:
                    x = mean  # final image

        return x
                
    def save_model(self, model, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'timesteps': self.timesteps,
            'B_start': self.B_start,
            'B_end': self.B_end,
        }, path)
    
    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_images(self, images, save_path, labels=None):
        """Save images to disk"""
        import matplotlib.pyplot as plt
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # If single image, save it directly
        if len(images.shape) == 3:  # Single image [C, H, W]
            image = (images + 1) / 2  # Unnormalize from [-1,1] to [0,1]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # Clip to valid range
            image_np = np.clip(image_np, 0, 1)
            plt.figure(figsize=(8, 8))
            plt.imshow(image_np)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:  # Multiple images [B, C, H, W]
            num_images = images.shape[0]
            grid_size = int(np.sqrt(num_images))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            # Handle case where there's only one subplot
            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i in range(num_images):
                image = (images[i] + 1) / 2
                image_np = image.permute(1, 2, 0).cpu().numpy()
                # Clip to valid range
                image_np = np.clip(image_np, 0, 1)
                axes[i].imshow(image_np)
                if labels is not None:
                    axes[i].set_title(f"Label: {labels[i]}")
                else:
                    axes[i].set_title(f"Sample {i+1}")
                axes[i].axis("off")
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        print(f"Images saved to: {save_path}")

    def generate_samples(self, model, num_samples=16, img_size=(3, 32, 32), device=None, save_path=None, show=True):
        """Generate multiple samples, optionally display, and optionally save to disk"""
        device = torch.device('cpu') if device is None else device
        samples = self.reverse_diffusion(model, img_size=img_size, device=device, num_samples=num_samples)

        # Display grid if requested
        if show:
            self.show_images_grid(samples, num_images=num_samples, show=True)

        # Save images if path is provided
        if save_path:
            self.save_images(samples, save_path)
        
        return samples

    def save_loss_plot(self, epoch_losses, save_path):
        """Save line plot of epoch average losses"""
        import matplotlib.pyplot as plt
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def main():
    ### constants
    timesteps = 1000
    B_start = 1e-4
    B_end = 2e-2
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    
    # Initialize diffusion process
    TIME_EMB_DIM = 256
    diffusion = Diffusion(timesteps, B_start, B_end, time_embedding_dim=TIME_EMB_DIM)
    
    # Model and training setup
    unet = UNet(time_embedding_dim=TIME_EMB_DIM)
    dataloader = diffusion.load_data()
    optimizer = torch.optim.Adam(unet.parameters(), lr=2e-4, weight_decay=1e-4)
    # Cosine annealing LR scheduler
    EPOCHS = 50
    total_steps = EPOCHS * len(dataloader)  # epochs * iterations per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    # Training
    print("Starting training...")
    diffusion.training_loop(
        unet,
        dataloader,
        optimizer,
        scheduler,
        device=device,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        sample_save_dir="epoch_samples",
        loss_plot_path="loss_curve.png"
    )
    
    # Save the trained model
    diffusion.save_model(unet, "diffusion_model.pth")
    print("Model saved to diffusion_model.pth")
    
    # Generate and display samples
    print("Generating samples (final)...")
    samples = diffusion.generate_samples(
        unet,
        num_samples=16,
        device=device,
        save_path="generated_samples.png",
        show=False
    )
    
    # Save individual samples
    print("Generating and saving individual samples...")
    for i in range(7):
        single_sample = diffusion.reverse_diffusion(unet, img_size=(3, 32, 32), device=device)
        diffusion.save_images(single_sample, f"sample_{i+1}.png")


    
    # diffusion = Diffusion(timesteps, B_start, B_end)
    # dataloader = diffusion.load_data()
    # images, labels = next(iter(dataloader))
    # x_t, ts, epsilon = diffusion.forward_diffusion(images)
    # print(diffusion.get_timestep_embedding(ts).shape)
    #diffusion.show_images_grid(x_t, ts)
    # diffusion.show_images(images, labels)

    ## testing residual block
    # fake batch
    # x = torch.rand(4, 32, 40, 40)     # [B, in_ch, H, W]
    # t = torch.rand(4, 128)            # [B, time_emb_dim]

    # # encoder: 32→64, halves spatial
    # down = DownBlock(32, 64, 128)
    # h, skip = down(x, t)
    # # h has shape [4, 64, 20, 20]
    # # skip has shape [4, 64, 40, 40]

    # # decoder: 64→32, doubles spatial
    # up = UpBlock(64, 64, 128)
    # out = up(h, t, skip)
    # # out should now be [4, 32, 40, 40]

    # print("h:",  h.shape)
    # print("skip:", skip.shape)
    # print("out:", out.shape)

    # x = torch.randn(4, 3, 32, 32)  # Fake CIFAR images
    # t = torch.randint(0, 1000, (4,))  # Random timesteps
    # time_embedding = diffusion.get_timestep_embedding(t)

    # unet = UNet(128)
    # out = unet(x, t, time_embedding)
    # print(out.shape)



if __name__ == "__main__":
    main()