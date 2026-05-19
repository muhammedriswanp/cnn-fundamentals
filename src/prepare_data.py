import os
import shutil #copying files
import random

random.seed(42)

# source folders
cat_src = "data/Cat"
dog_src = "data/Dog"

# destination
base = "data/cats_vs_dogs"
for split in ["train", "val"]:
    for cls in ["cat", "dog"]:
        os.makedirs(f"{base}/{split}/{cls}", exist_ok=True)

def split_and_copy(src, cls, train_ratio=0.8):
    files = os.listdir(src)     #Gets all filenames.
    random.shuffle(files)
    split = int(len(files) * train_ratio)
    train_files = files[:split]
    val_files   = files[split:]

    for f in train_files:
        shutil.copy(f"{src}/{f}", f"{base}/train/{cls}/{f}")
    for f in val_files:
        shutil.copy(f"{src}/{f}", f"{base}/val/{cls}/{f}")

    print(f"{cls} → train: {len(train_files)}, val: {len(val_files)}")

split_and_copy(cat_src, "cat")
split_and_copy(dog_src, "dog")
print("Done!")