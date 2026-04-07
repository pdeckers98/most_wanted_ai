# Cloud Setup

Instructions for setting up training on rented cloud GPU/CPU instances (Verda, Vast.ai, etc.).

## Repository

**Git repo URL**: `https://github.com/pdeckers98/most_wanted_ai.git`

**Branch for training/preprocessing**: `inference`

## SSH Key Caching (One-time per session)

Always cache your SSH key first to avoid typing the passphrase repeatedly:

```bash
eval $(ssh-agent -s)
ssh-add ~/.ssh/private_key_03_04_26
```

Enter passphrase once. All subsequent commands will use the cached key.

## Cloud Instance Setup

### 1. Mount Block Volume

```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "lsblk"
```

Find the block volume (usually `/dev/vdb`). Then:

```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "mkfs.ext4 /dev/vdb && mkdir -p /mnt/data && mount /dev/vdb /mnt/data && df -h"
```

### 2. Clone Repository

```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "cd /mnt/data && git clone https://github.com/pdeckers98/most_wanted_ai.git && cd most_wanted_ai && git checkout inference"
```

### 3. Install Python and pip

**Rented CPUs do NOT have pip installed by default.**

```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "apt-get update && apt-get install -y python3 python3-pip"
```

### 4. Install Dependencies

**Important**: Add `--break-system-packages` when installing on rented CPUs (Verda, etc). This flag overrides Python's externally-managed-environment restriction.

For GPU instances:
```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "cd /mnt/data/most_wanted_ai && pip3 install -r requirements.txt --break-system-packages"
```

For CPU-only instances (torch CPU version):
```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "cd /mnt/data/most_wanted_ai && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages && pip3 install -r requirements.txt --break-system-packages"
```

## Running Preprocessing

Frame stacking with steering filter and 20k target:

```bash
ssh -i ~/.ssh/private_key_03_04_26 root@<IP> "cd /mnt/data/most_wanted_ai && python3 -m src.preprocessing.pipeline --captures-dir /mnt/data/captures --output-dir /mnt/data/processed --filter-steering --target-frames 20000"
```

See `TRAINING.md` for full training pipeline.

## Common Gotchas

- **pip not found**: Install with `apt-get install -y python3 python3-pip`
- **Can't install packages**: Add `--break-system-packages` flag (rented instances only)
- **Git clone fails with "No such device or address"**: Network/DNS issue with instance. Try HTTPS instead of SSH.
- **SSH key passphrase every command**: Run `ssh-add` once at the start of the session to cache it.
