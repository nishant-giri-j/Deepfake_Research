import os

# Define your main project root name
root_dir = "Deepfake_Research"

# List of all sub-directories you need
folders = [
    f"{root_dir}/data/FaceForensics++/original_sequences/youtube/c23/videos",
    f"{root_dir}/data/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos",
    f"{root_dir}/data/FaceForensics++/manipulated_sequences/Face2Face/c23/videos",
    f"{root_dir}/data/FaceForensics++/manipulated_sequences/FaceSwap/c23/videos",
    f"{root_dir}/data/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos",
    f"{root_dir}/data/real",
    f"{root_dir}/data/fake",
    f"{root_dir}/checkpoints",
    f"{root_dir}/logs",
    f"{root_dir}/results"
]

# Create them
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\nFolder structure ready! Now copy your downloaded videos into the FaceForensics++ folders.")