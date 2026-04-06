# Cloud Frame Stacking Workflow

Documentation for preprocessing NFS Most Wanted dataset in cloud to avoid local storage/upload bottlenecks.

## Problem

Local frame stacking explodes data size: **2.9 GB raw footage → 45 GB processed** (16× expansion).

**Cause:** 
- uint8→float32 conversion: 4× bigger
- 4-frame stacking: 4× bigger
- Total: 4 × 4 = 16× expansion

Initial approach (upload 45 GB locally) = 11+ hours, risky if connection drops (restart from zero).

## Solution: Preprocess in Cloud

Upload raw data only (~3 GB per session), preprocess on cheap CPU in cloud, then attach preprocessed data to GPU for training.

**Why it works:** Frame stacking is I/O-bound CPU work, not GPU work. No compute needed.

## Workflow

### Step 1: Create Block Volume
- Verda console → Storage → Block Volumes
- Size: 100+ GB (for all sessions)
- Storage type: HDD (cheap for idle, $0.02/GB/month)
- Note the region (e.g., `eu-west-1`)

### Step 2: Rent CPU Instance
- Verda console → Deploy Instance
- Select: **CPU-only instance** (8 cores, 32 GB RAM sufficient)
- **Same region as block volume**
- Cost: ~$0.03/hour
- Attach block volume during setup

### Step 3: Mount Block Volume on CPU Instance
```bash
ssh -i ~/.ssh/your_key root@cpu-instance-ip

# Mount
sudo mkdir -p /mnt/data
sudo mkfs.ext4 /dev/vdb
sudo mount /dev/vdb /mnt/data
sudo chown $USER:$USER /mnt/data

# Verify
df -h /mnt/data
```

### Step 4: Upload Raw Footage
From local machine:
```bash
scp -r -i ~/.ssh/your_key data/captures/ root@cpu-instance-ip:/mnt/data/
```

**Time estimate:** 2-4 hours for 9 sessions @ 3 GB each (depends on internet speed)

### Step 5: Run Preprocessing
On CPU instance:
```bash
# Install dependencies
apt-get update && apt-get install -y python3-pip
pip3 install --break-system-packages numpy

# Clone repo and run
git clone --branch integration_and_preprocessing https://github.com/pdeckers98/most_wanted_ai.git /opt/code
cd /opt/code

# Run preprocessing
python3 src/preprocessing/pipeline.py \
  --captures-dir /mnt/data/captures \
  --output-dir /mnt/data/processed
```

**Time estimate:** 10-20 minutes per session (CPU-bound, not GPU)

### Step 6: Verify Output
```bash
ls -lh /mnt/data/processed/
du -sh /mnt/data/processed/
cat /mnt/data/processed/session_*/dataset_meta.json
```

### Step 7: Unmount and Shut Down CPU Instance
On CPU instance:
```bash
sudo umount /mnt/data
exit
```

In Verda console:
- Click "Shutdown" on the CPU instance (don't delete)
- Detach block volume from CPU instance

### Step 8: Rent GPU and Attach Volume
- Verda console → Deploy Instance
- Select: **H100 80GB or RTX 5090 32GB** (based on your choice)
- **Same region as block volume**
- During setup: "Add existing storage" → select your preprocessed block volume
- **Before attaching to GPU:** Switch block volume from HDD to NVMe in console (no re-upload needed)

### Step 9: Mount and Train
On GPU instance:
```bash
ssh -i ~/.ssh/your_key root@gpu-instance-ip

# Mount
sudo mkdir -p /data
sudo mount /dev/vdb /data

# Verify preprocessed data is there
ls -la /data/processed/

# Clone repo and train
git clone --branch integration_and_preprocessing https://github.com/pdeckers98/most_wanted_ai.git /opt/code
cd /opt/code
pip3 install --break-system-packages torch numpy
python3 src/agent/train.py --data-dir /data/processed --output-dir /data/checkpoints
```

## Storage Optimization

### Reduce Processed Data Size by 75%

Instead of storing frames as **float32** (4 bytes/pixel), keep as **uint8** (1 byte/pixel) on disk:

1. In `src/preprocessing/pipeline.py`, modify `normalize_frames()`:
   ```python
   def normalize_frames(frames):
       # Don't convert to float32, keep as uint8
       return frames  # uint8 [0-255]
   ```

2. Update `_build_sequences_to_memmap()` to use `dtype=np.uint8`:
   ```python
   memmap_file = np.lib.format.open_memmap(
       out_path, mode='w+', dtype=np.uint8,  # Changed from float32
       shape=(n_sequences, stack_size, h, w)
   )
   ```

3. In training DataLoader, normalize on-the-fly:
   ```python
   frames = frames.float() / 255.0  # Convert to float32 [0, 1] during training
   ```

**Result:** 12 GB per session instead of 46 GB → 4× faster upload, 4× cheaper storage.

## Cost Breakdown (9 sessions)

| Component | Time/Size | Cost |
|-----------|-----------|------|
| CPU instance (preprocessing) | 2-3 hours | $0.09-0.15 |
| Block volume HDD (idle) | 500 GB × 1 month | $10 |
| Block volume NVMe (training) | 500 GB × 0.1 hour | $5 |
| GPU instance (H100 training) | 1-2 hours | $10-20 |
| **Total** | | **$25-45** |

## Key Constraints

- **Region lock:** Instance and block volume must be in same region
- **One-at-a-time:** Block volume can only attach to one instance
- **Shutdown for detach:** Must shut down (not delete) CPU instance before detaching volume
- **Connection safety:** `scp` doesn't resume; use `rsync -P` for reliability on large transfers

## Monitoring Progress

In a separate terminal, watch preprocessing in real-time:
```bash
ssh -i ~/.ssh/your_key root@cpu-instance-ip
watch -n 5 'du -sh /mnt/data/processed/'
```

This updates every 5 seconds showing current output size.

## Troubleshooting

**`pip install` fails with "externally-managed-environment":**
```bash
pip3 install --break-system-packages package_name
```

**`git clone` authentication fails:**
Make repo public on GitHub, or set up SSH keys on the instance:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_key -N ""
# Add public key to GitHub → Settings → SSH Keys
git clone git@github.com:username/repo.git
```

**Block volume won't mount:**
- Verify volume is attached (Verda console)
- Check region matches instance
- Ensure you ran `mkfs.ext4 /dev/vdb` (one-time format)
