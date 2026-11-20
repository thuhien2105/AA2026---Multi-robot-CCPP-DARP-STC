# Flow của pipeline Multi-robot CCPP

Toàn bộ flow do `mcpp/planner.py::plan()` gọi lần lượt:

```
Bản đồ + vị trí k robot
  → [0] Đọc bản đồ, dựng lưới thô 2×2
  → [1] Chia vùng: ô nào giao cho robot nào        (DARP / Voronoi / k-means)
  → [1b] Local search: làm đều kích thước vùng    (chỉ các method *_ls)
  → [2] Dựng cây khung cho từng vùng              (lược ngang / dọc)
  → [3] Đi vòng quanh cây ra tour khép kín        (STC, dài đúng 4 × số ô)
  → [4] Đo metrics                                (makespan, turns, coverage...)
```

## Bảng các bước

| # | Bước | File / hàm | Input | Output | Câu hỏi bước này trả lời | Ảnh hưởng makespan | Ảnh hưởng số lần rẽ |
|---|---|---|---|---|---|---|---|
| 0 | Đọc bản đồ | `grid.py` | File `.map` MovingAI | Lưới mịn + lưới thô (2×2) | Ô nào đi được? | — | — |
| 1 | Chia vùng | `darp.py::darp` (hoặc `partition.py::voronoi_partition`, `kmeans_partition`) | Lưới thô, vị trí k robot | Ma trận `assign`: mỗi ô thuộc robot nào | Ô nào giao cho robot nào? | **Quyết định chính** | Gián tiếp (độ trơn của biên) |
| 1b | Local search *(chỉ `_ls`)* | `local_search.py::improve_partition` | `assign` từ bước 1 | `assign` đều hơn | Chuyển ô nào để vùng lớn nhất nhỏ lại? | **Giảm thêm** | Ít (qua `nbr_count`) |
| 2 | Dựng cây khung | `stc.py::spanning_tree` (gọi trong `plan_region`) | Tập ô của 1 vùng | Cây nối mọi ô, không vòng | Nối các ô theo thứ tự nào? | **Không** | **Quyết định chính** |
| 3 | Đi vòng quanh cây | `stc.py::stc_tour` | Cây + ô xuất phát | Tour khép kín, dài đúng `4 × số ô` | Robot đi từng bước ra sao? | Không (độ dài cố định) | Theo cây |
| 4 | Đo kết quả | `metrics.py::evaluate` | Tour của k robot | makespan, turns, coverage, conflicts... | Kế hoạch tốt đến đâu? | — | — |

## Công thức quan trọng

Vì tour STC của một vùng có `s` ô thô luôn dài đúng `4s` bước, bất kể hình dạng cây:

```
makespan          = 4 × (số ô của vùng LỚN NHẤT)      → nên chia càng đều càng tốt
makespan (dị thể) = 4 × max_i (số ô_i / tốc độ_i)     → chia theo tỉ lệ tốc độ
lower bound       = 4n / Σ tốc độ_i                   (n = số ô thô)
```

Nhờ vậy bài toán tách làm hai phần độc lập:

- **Chia vùng đều** (bước 1, 1b) thì **makespan thấp**.
- **Chọn cây tốt** (bước 2) thì **ít rẽ**. Độ dài không đổi, nhưng robot thực tế chạy nhanh hơn vì mỗi lần rẽ tốn thời gian (`time_makespan`).

## Ví dụ: bản đồ 4×4 ô thô, 2 robot

**Bước 1: chia vùng**

```
A A A A
A A A A        Robot 1: 8 ô
B B B B        Robot 2: 8 ô
B B B B
```

**Bước 2: cây lược ngang cho vùng A**

```
A─A─A─A
│
A─A─A─A
```

**Bước 3: tour** dài 4 × 8 = 32 bước cho mỗi robot, nên makespan = 32. Đây cũng là mức tối ưu: 64 ô mịn chia cho 2 robot.

| Cách chia | Vùng 1 | Vùng 2 | Makespan |
|---|---|---|---|
| Lệch | 12 ô | 4 ô | 4 × 12 = **48** |
| Đều (DARP) | 8 ô | 8 ô | 4 × 8 = **32** |

## Các phương pháp (`method` của `plan()`)

| `method` | Bước 1 | Bước 1b | Bước 2–3 |
|---|---|---|---|
| `voronoi` | Voronoi | ✗ | STC |
| `kmeans` | k-means + sửa liên thông | ✗ | STC |
| `darp` | DARP | ✗ | STC |
| `darp_ls` | DARP | ✓ | STC |
| `voronoi_ls` | Voronoi | ✓ | STC |
| `mmrtc` / `mmrtc_dedup` | Cây phủ Even et al. (vùng được phép chồng nhau) | ✗ | STC |

## Số lần rẽ theo hình dạng cây

Số lần rẽ của tour bên trong mỗi ô thô chỉ phụ thuộc vào các cạnh cây chạm vào ô đó:

| Dạng ô trong cây | Số lần rẽ |
|---|---|
| Bậc 2, thẳng (trái–phải hoặc trên–dưới) | 0 |
| Lá (bậc 1) | 2 |
| Bậc 2, góc vuông | 2 |
| Bậc 3 (chữ T) | 2 |
| Bậc 4 (chữ thập) | 4 |

```
turns = 2·(|V| − #ô_thẳng) + 2·#ô_chữ_thập
```

Công thức này được suy tay từ `stc_tour`, chưa kiểm chứng bằng test.

Hiện tại `plan_region(strategy="adaptive")` chỉ thử hai cây lược (ngang, dọc) rồi giữ cây ít rẽ hơn. Chưa có bước local search trên cây để giảm thêm số lần rẽ.
