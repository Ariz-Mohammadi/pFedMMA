# PFEDMMA Reproduction & Fix Report

## 1. Objective
Reproduce pFedMMA results on OxfordPets (16-shot) and fix issues in evaluation and checkpoint loading.

---

## 2. Issues Identified

### 2.1 Evaluation Mismatch
Original metrics:
- Global test acc
- Global test error
- Global test macro_f1

These correspond to:
- Mean of per-client accuracies (unweighted)

Not directly equal to:
- Local / Base / Novel / HM (paper metrics)

---

### 2.2 Dataset Split Behavior

DATASET.SUBSAMPLE_CLASSES:
- base → first half of classes
- new → second half of classes

Applied to:
- train, val, test

---

### 2.3 Incorrect Training Attempt
Running:
DATASET.SUBSAMPLE_CLASSES new + USEALL False

Result:
ValueError: num_samples=0

Reason:
No training data for novel classes.

---

## 3. Correct Experimental Pipeline

### Step 1: Train (BASE only)
- SUBSAMPLE_CLASSES = base
- USEALL = False

### Step 2: Evaluate BASE
- --eval-only
- SUBSAMPLE_CLASSES = base
- USEALL = True

### Step 3: Evaluate NEW
- --eval-only
- SUBSAMPLE_CLASSES = new
- USEALL = True

### Step 4: Compute HM
HM = 2 * Base * Novel / (Base + Novel)

---

## 4. Checkpoint Loading Bug

### Problem
Evaluation expected:
fedadapter_best.pt

But training saved:
save.pt

### Fix
Dynamic checkpoint loading:

1. Try:
   - save.pt
   - fedadapter_best.pt
2. Else:
   - pick latest *.pt file

### Patch
import glob

ckpt_dir = args.model_dir if args.model_dir else args.output_dir

preferred = [
    os.path.join(ckpt_dir, "save.pt"),
    os.path.join(ckpt_dir, "fedadapter_best.pt"),
]

model_path = next((p for p in preferred if os.path.isfile(p)), None)

if model_path is None:
    candidates = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found")
    model_path = max(candidates, key=os.path.getmtime)

checkpoint = torch.load(model_path, map_location="cpu")

---

## 5. Federated Aggregation

- Method: Weighted FedAvg
- Weight: client dataset size

Aggregated:
- shared_adapter parameters

Local (not aggregated):
- text_adapter
- visual_adapter

---

## 6. Status

| Component | Status |
|----------|-------|
| Training | OK |
| Aggregation | Verified |
| Evaluation | Fixed |
| Checkpoint Loading | Fixed |

---

## 7. Next Steps
- Run base eval
- Run new eval
- Compute HM
- Compare with paper
