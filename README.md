# IMIAD

Project chạy các thí nghiệm phát hiện bất thường bằng PatchCore và CFlow trên ảnh độ phân giải cao. Cấu hình được đọc từ YAML; mỗi lần chạy tạo một thư mục kết quả riêng, lưu metric, điểm dự đoán và các ảnh cần xem lại.

## Cài đặt và chạy

```powershell
pip install -r requirements.txt
```

Sửa đường dẫn dữ liệu trong YAML trước khi chạy. Các cấu hình mẫu nằm trong `configs/`, gồm Real-IAD và dữ liệu tổ chức theo thư mục. Ví dụ:

```powershell
python -m imiad.run --config configs/real_iad_patchcore.yaml
python -m imiad.run --config configs/real_iad_cflow_input_tiling.yaml
```

Với dữ liệu riêng, sửa các đường dẫn `train_normal_dir`, `test_normal_dir`, `test_anomaly_dir` và `mask_dir` trong `configs/folders_patchcore.yaml` hoặc `configs/folders_cflow_ensemble.yaml`. Tên file CFlow có chữ `ensemble` hiện chỉ là tên cấu hình cũ: nội dung file đang bật `input_tiling` và tắt `tiled_ensemble`.

Các tham số thường cần chỉnh:

| Tham số | Vai trò |
| --- | --- |
| `data.max_train`, `data.max_test` | Giới hạn số ảnh train/test. Test được lấy mẫu theo tỷ lệ các loại lỗi trong tập gốc. |
| `data.image_size: null` | Giữ kích thước ảnh gốc để chia tile trước khi resize từng tile. |
| `data.batch_size` | Số ảnh train trong một lượt tải dữ liệu; khi test, code xử lý một ảnh mỗi lượt. |
| `input_tiling.tile_size`, `stride` | Kích thước tile và bước trượt trên ảnh gốc. `stride` nhỏ hơn `tile_size` tạo vùng chồng lấp. |
| `input_tiling.tile_resize` | Kích thước tile đưa vào backbone sau khi chia; bằng `tile_size` thì bỏ qua bước resize. |
| `patchcore.tile_batch_size`, `cflow.tile_batch_size` | Số tile tối đa đưa qua mô hình trong một lượt xử lý. |

## Kết quả của mỗi lần chạy

Kết quả nằm trong `runs/<run_id>/` theo `experiment.output_dir`:

- `metrics.json`: I-AUROC, P-AUROC, AUPRO, thời gian train, FPS inference, trạng thái và tiến độ.
- `predictions.csv`: đường dẫn ảnh test, nhãn thật và điểm bất thường ở cấp ảnh.
- `progress.log`, `config_resolved.yaml`: tiến độ và cấu hình thực tế của lần chạy.
- `visualizations/`: khi `evaluation.visualization.enabled: true`, lưu toàn bộ ảnh dự đoán sai (FP/FN) và tối đa `correct_anomalies` ảnh bất thường dự đoán đúng (TP, mặc định 5). Mỗi trường hợp có ảnh đối chiếu gồm ảnh gốc, heatmap, overlay, mask thật; heatmap và overlay cũng được lưu riêng. `manifest.csv` liệt kê các file này.

Nhãn dự đoán để chọn ảnh xem lại được xác định bằng ngưỡng điểm cấp ảnh `evaluation.visualization.image_threshold`. Nếu để `null`, code suy ra ngưỡng từ phân vị `train_quantile` của điểm trên ảnh train bình thường (mặc định 0,99); bước này cần thêm một lượt inference trên tập train. Đây là ngưỡng phục vụ phân tích lỗi, không phải ngưỡng đã được đánh giá trên tập validation độc lập. Heatmap được chuẩn hóa màu theo phân vị 1% và 99% **riêng cho từng ảnh**; vì vậy màu giữa hai ảnh không biểu thị cùng một thang điểm tuyệt đối. `max_side` giới hạn kích thước ảnh xuất ra.

Để giảm RAM khi tính metric pixel, anomaly map và mask của ảnh test được lưu tạm dưới dạng `.npy` rồi đọc từng ảnh; các file tạm được xóa sau khi chạy. `evaluation.save_maps: true` giữ lại anomaly map gốc trong `maps/`. Với ảnh lớn, nên đặt `evaluation.pixel_auroc_bins` (ví dụ `4096`) để tính P-AUROC xấp xỉ bằng histogram; để `null` sẽ dùng cách tính chính xác và có thể cần rất nhiều RAM.

## Benchmark tham khảo trên Real-IAD

Báo cáo thí nghiệm sử dụng category `phone_battery`, ảnh độ phân giải 5000 × 3648 và tile 1000 × 912. Hai trường hợp được so sánh là không chồng lấp (0%; `stride: [1000, 912]`) và chồng lấp 25% (`stride: [750, 684]`). I-AUROC, P-AUROC và AUPRO càng cao càng tốt; thời gian train càng thấp càng tốt; FPS inference càng cao càng tốt.

PatchCore dùng `coreset_ratio = 0.01`, `num_neighbors = 1`:

| Chồng lấp | I-AUROC | P-AUROC | AUPRO | Train (giây) | Inference (FPS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.902576 | 0.994369 | 0.980774 | 320.721 | 0.201748 |
| 25% | 0.923108 | 0.998687 | 0.987850 | 668.673 | 0.152837 |

CFlow dùng `condition_vector = 128`, `coupling_blocks = 8`, `clamp_alpha = 1.9`, `fiber_batch_size = 64`, `learning_rate = 0.0001` và `epochs = 10`:

| Chồng lấp | I-AUROC | P-AUROC | AUPRO | Train (giây) | Inference (FPS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.724638 | 0.526983 | 0.154032 | 821.062 | 0.450430 |
| 25% | 0.829308 | 0.779545 | 0.373015 | 6047.778 | 0.315084 |

Trong kết quả này, PatchCore đạt cả ba metric độ chính xác cao hơn CFlow ở cùng mức chồng lấp và train nhanh hơn. CFlow inference nhanh hơn khoảng 2,23 lần ở mức 0% và 2,06 lần ở mức 25%, theo chính các giá trị FPS trong bảng. Chồng lấp 25% cải thiện độ chính xác của cả hai mô hình nhưng làm tăng thời gian train và giảm FPS; tác động lên thời gian train của CFlow đặc biệt lớn (821 lên 6048 giây).

Các bảng là số liệu từ báo cáo thí nghiệm, không phải kết quả được tạo lại bởi README này. Báo cáo không ghi rõ số ảnh train/test, phần cứng hoặc phiên bản code, nên cần cố định các điều kiện đó trước khi so sánh với một lần chạy mới. Các YAML mẫu hiện có cũng cần được kiểm tra và chỉnh để khớp chính xác cấu hình benchmark trước khi tái lập kết quả.

