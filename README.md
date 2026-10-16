# Setup


```powershell
pip install -r requirements.txt
```

Chỉnh sửa nội dung đường dẫn đến data phù hợp trong configs/folders_patchcore.yaml và configs/folders_cflow_ensemble.yaml

```powershell
python -m imiad.run --config configs/real_iad_patchcore.yaml
python -m imiad.run --config configs/real_iad_cflow_ensemble.yaml
python -m imiad.run --config configs/real_iad_cflow_input_tiling.yaml
```

