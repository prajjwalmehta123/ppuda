import torch
import torch.nn.functional as F


def train_adaptive_model(model, meta_train_loader, meta_val_loader,
                         learning_rate=0.001, epochs=50, device="cuda"):
    # Move model to device
    model = model.to(device)

    # Freeze base encoder, train only adaptation layers
    for param in model.base_encoder.parameters():
        param.requires_grad = False

    # Set adaptation layers to training mode
    for param in model.adaptation_module.parameters():
        param.requires_grad = True

    # Optimization setup
    optimizer = torch.optim.Adam(model.adaptation_module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        tasks_processed = 0

        for batch_data in meta_train_loader:
            if isinstance(batch_data, list) and len(batch_data) == 2:
                task_batch, dataset_indices = batch_data
            else:
                task_batch = batch_data

            # Process batch
            batch_size = task_batch[0].size(0)
            meta_loss = 0
            correct = 0
            total = 0

            for i in range(batch_size):
                # Extract single episode
                if len(task_batch) == 4:  # support_imgs, support_labs, query_imgs, query_labs
                    support_imgs = task_batch[0][i].to(device)
                    support_labs = task_batch[1][i].to(device)
                    query_imgs = task_batch[2][i].to(device)
                    query_labs = task_batch[3][i].to(device)

                    # Forward pass for this episode
                    logits = model(support_imgs, support_labs, query_imgs,
                                   n_way=support_labs.max().item() + 1)

                    # Loss for this episode
                    loss = F.cross_entropy(logits, query_labs)
                    meta_loss += loss

                    # Track accuracy
                    pred = logits.argmax(dim=1)
                    correct += (pred == query_labs).sum().item()
                    total += query_labs.size(0)
            # Update weights
            optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # Gradient clipping
            optimizer.step()

            # Track metrics
            train_loss += meta_loss.item()
            train_acc += correct / total
            tasks_processed += 1

        # Calculate average training metrics
        avg_train_loss = train_loss / tasks_processed
        avg_train_acc = train_acc / tasks_processed

        # Validation
        model.eval()
        val_acc = evaluate(model, meta_val_loader, device)
        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_adaptive_model.pth')

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
              f"Train Acc {avg_train_acc:.4f}, Val Acc {val_acc:.4f}")

    return model


def evaluate(model, data_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in data_loader:
            # Unpack data
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                task_batch, _ = batch_data
            else:
                task_batch = batch_data

            # Process each episode in the batch
            batch_size = task_batch[0].size(0)

            for i in range(batch_size):
                support_imgs = task_batch[0][i].to(device)
                support_labs = task_batch[1][i].to(device)
                query_imgs = task_batch[2][i].to(device)
                query_labs = task_batch[3][i].to(device)

                # Forward pass
                logits = model(support_imgs, support_labs, query_imgs,
                               n_way=support_labs.max().item() + 1)

                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == query_labs).sum().item()
                total += query_labs.size(0)

    return correct / total


if __name__ == '__main__':
    from ppuda.task.task_adaptation import TaskAdaptiveEncoder, TaskAdaptationModule
    from torch.utils.data import DataLoader
    from ppuda.task.task_encoder import TaskEncoder, initialize_with_ghn

    # Create and test the full model
    base_encoder = TaskEncoder()
    base_encoder = initialize_with_ghn(
        base_encoder,
        ghn_checkpoint_path="/Users/prajjwalmehta/Desktop/projects/ppuda/checkpoints/ghn2_cifar100.pt",
        device='cpu',
    )

    adaptation_module = TaskAdaptationModule(feature_dim=512, adaptation_dim=64)
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)

    # Before doing full training, test the training procedure on a small batch
    from torch.utils.data import DataLoader


    # Create simple mock data loaders
    class MockMetaDataset:
        def __init__(self, n_episodes=100):
            self.n_episodes = n_episodes

        def __len__(self):
            return self.n_episodes

        def __getitem__(self, idx):
            n_way, k_shot = 5, 1
            support_images = torch.randn(n_way * k_shot, 3, 84, 84)
            support_labels = torch.arange(n_way).repeat_interleave(k_shot)
            query_images = torch.randn(n_way * 5, 3, 84, 84)
            query_labels = torch.arange(n_way).repeat_interleave(5)
            return support_images, support_labels, query_images, query_labels


    meta_train_dataset = MockMetaDataset(n_episodes=10)
    meta_val_dataset = MockMetaDataset(n_episodes=5)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=1)
    meta_val_loader = DataLoader(meta_val_dataset, batch_size=1)

    # Test training for 2 epochs
    model = TaskAdaptiveEncoder(base_encoder, adaptation_module)
    train_adaptive_model(model, meta_train_loader, meta_val_loader, epochs=2,device='cpu')