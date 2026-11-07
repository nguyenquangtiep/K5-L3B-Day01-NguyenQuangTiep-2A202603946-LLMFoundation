# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603946
- **Họ và tên:**Nguyễn Quang Tiệp
- **SEED** (in ở ô đầu notebook):6878

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.975   0.908      0.940        0.947   
2. BoW + MLP (deep learning)         0.944   0.977      0.960        0.974   
3a. Encoder (MiniLM) + kNN           0.866   0.966      0.913        0.974   
3b. Encoder (MiniLM) + LogReg        0.954   0.954      0.954        0.974   
3c. Encoder (MiniLM) + MLP head      0.965   0.954      0.960        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.356      38726.592       0.046   
2. BoW + MLP (deep learning)         9.212      36711.278       0.154   
3a. Encoder (MiniLM) + kNN          98.482         16.566      85.615   
3b. Encoder (MiniLM) + LogReg      100.339         19.606      83.514   
3c. Encoder (MiniLM) + MLP head    100.018         19.077     108.065   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.060           0.0  
2. BoW + MLP (deep learning)          0.206           0.0  
3a. Encoder (MiniLM) + kNN          148.473           0.0  
3b. Encoder (MiniLM) + LogReg       153.324           0.0  
3c. Encoder (MiniLM) + MLP head     204.263           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

* Dự đoán ml_f1: Tôi dự đoán Logistic Regression chỉ đạt F1 khoảng 0.70. Thực tế: F1 test = 0.940, lệch +0.240. Nguyên nhân: Tôi đã đánh giá thấp khả năng của Bag-of-Words kết hợp Logistic Regression trên tập SMS thường. Các từ như call, txt, free, reply giúp mô hình phân biệt nhiều trường hợp; tuy nhiên điểm cao này chưa bảo đảm mô hình xử lý tốt spam ngụy trang.

* Dự đoán emb_tricky: Tôi dự đoán Encoder + LogReg nhận diện 6 SMS ngụy trang tốt hơn BoW. Thực tế: cả hai đều đúng 5/6. Chúng cùng đoán tin “Last chance! Your account rewards expire tonight, confirm details to keep them” là ham, trong khi nhãn là spam. Nguyên nhân: embedding có khả năng biểu diễn ngữ nghĩa nhưng bộ phân loại gắn lên vẫn được huấn luyện từ dữ liệu và không chắc phát hiện mọi kiểu dụ xác nhận thông tin; lợi thế của embedding không thể kết luận từ chỉ 6 mẫu.

* Dự đoán ml_lr: Tôi dự đoán lr=0.001 làm loss giảm ít và lr=50 làm loss giảm rất nhanh. Thực tế: hướng dự đoán về loss cuối là đúng (lần lượt 0.628 và 0.007), nhưng đồ thị lr=50 xuất hiện một đỉnh loss khoảng 2.7 lúc ban đầu rồi mới giảm mạnh. Điểm chưa dự đoán đúng: learning rate lớn có thể khiến cập nhật vượt quá điểm tối ưu, dẫn tới tăng/dao động loss; trong bài này về sau vẫn hội tụ. Đây là sai lệch về hình dạng đường loss, không phải một dự đoán hoàn toàn ngược kết quả.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất khi dùng cùng cột F1 SUB làm mốc so sánh: giảm 0.579. Theo nhóm, mô hình chỉ đoán đúng 12% nhóm Thao túng/lừa đảo, 17% nhóm Né từ khóa và 20% nhóm Spam giọng tự nhiên. BoW + MLP còn có tỷ lệ đúng 0% ở nhóm né từ khóa. Trong khi đó, MiniLM + kNN đạt F1 = 0.746 trên tập khó, cao nhất trong 5 mô hình đã chạy, nhưng p95 = 148.473 ms/tin, không đạt SLA 50 ms.

## 5. Thử thách cá nhân

* SEED: 6878; đề #3: “DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?”

* Cách làm: 

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
* Hỏi AI: “Hướng dẫn tôi hoàn thành các mục 2–7 với notebook day01_lab.ipynb.”
* Nhận được: AI đọc các output, lập bảng kết quả, so sánh F1 tập khó, diễn giải nhóm case sai và hướng dẫn cách làm thử thách #3.
* Tôi sửa: Cần tự đối chiếu với output ô bảng Phần 5, ô stress test, biểu đồ lr=50 và chạy thử thách #3 trước khi nộp. Không dùng số liệu giả cho LLM hoặc thử thách chưa chạy.
* Điều tôi hiểu ra hôm nay là F1 cao trên tập test thường không đồng nghĩa mô hình đáng tin trên các kiểu spam khác phân phối: BoW + LogReg giảm từ F1 SUB 0.947 xuống 0.368 ở bộ 50 case khó và khi chọn mô hình cho sản phẩm thật cần xét cả chất lượng, tốc độ và hậu quả của việc chặn nhầm tin hợp lệ.