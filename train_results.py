import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 👉 CHANGE THIS to your actual tensorboard event file
log_file = "logs/DQN_1/events.out.tfevents.1770269856.LAPTOP-IF7NTFEN.24040.0"

# Load tensorboard data
ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

# Print available tags (optional)
print("Available scalar tags:", ea.Tags()["scalars"])

# Extract loss data
loss_events = ea.Scalars("train/loss")

# Get timesteps and loss values
steps = [e.step for e in loss_events]
loss_values = [e.value for e in loss_events]

# Plot
plt.figure(figsize=(10,5))
plt.plot(steps, loss_values)
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("Training Loss vs Timesteps")
plt.show()
