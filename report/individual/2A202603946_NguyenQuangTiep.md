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

ml_f1: Tôi dự đoán F1 khoảng 0.70, nhưng khi chạy, BoW + Logistic Regression đạt 0.940 trên tập test. Tôi đã đoán thấp hơn kết quả thực tế. Sau khi xem những từ có trọng số lớn như call, txt, free, tôi thấy mô hình có thể dựa vào các từ thường gặp trong spam để phân loại khá tốt trên tập test này.

emb_tricky: Trước khi chạy, trong ô predict tôi chỉ ghi Encoder + LogReg, chưa giải thích rõ hơn. Kết quả trên 6 tin nhắn: BoW và Encoder + LogReg đều đúng 5/6. Cả hai đều đoán nhầm tin về phần thưởng tài khoản sắp hết hạn thành ham. Như vậy, với 6 tin này, tôi chưa thấy Encoder + LogReg tốt hơn BoW.

ml_lr: Tôi dự đoán learning rate 0.001 làm loss giảm ít, còn learning rate 50 làm loss giảm nhanh. Kết quả cho thấy lr=50 có lúc làm loss tăng mạnh ban đầu rồi mới giảm. Tôi hiểu thêm rằng learning rate lớn không phải lúc nào cũng giúp huấn luyện ổn định.

## 4. Phân tích 50 case khó

BoW + LogReg có mức giảm F1 lớn nhất khi so sánh F1 SUB 40 (0.947) với F1 trên 50 case khó (0.368), tức giảm 0.579. Mô hình này chỉ đúng 12% ở nhóm thao túng/lừa đảo, 17% ở nhóm né từ khóa và 20% ở nhóm spam giọng tự nhiên. MiniLM + kNN đạt F1 cao nhất trên tập khó (0.746), nhưng p95 148.473 ms/tin vượt yêu cầu 50 ms/tin.
- Case #12 – spam giọng tự nhiên: Nhãn thật là spam, nhưng cả 5 mô hình đều dự đoán ham. Tin nhắn bắt đầu như lời chào của người quen rồi dẫn sang một trang hẹn hò. Tôi nghĩ mô hình dễ nhầm vì nội dung giống lời nhắn cá nhân, không có những từ spam rõ ràng.
- Case #41 – OTP hợp lệ: Nhãn thật là ham, nhưng cả 5 mô hình đều dự đoán spam. Tin nhắn có mã OTP và lời cảnh báo liên hệ ngân hàng. Những dấu hiệu này giống một số tin nhắn lừa đảo nên mô hình có thể chặn nhầm.
Tôi rút ra rằng điểm F1 trên test thường cao không có nghĩa mô hình sẽ xử lý tốt mọi tin nhắn. Tuy nhiên, tập 50 case khó do người viết ra cũng chưa đủ để đại diện cho tất cả SMS ngoài đời.

## 5. Thử thách cá nhân

* SEED: 6878; đề #3: “DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?”

* Cách làm: 

## 6. Phương án cho sản phẩm thật

Tôi chọn BoW + MLP để tiếp tục thử nghiệm, vì trong kết quả đã chạy, mô hình đạt F1 test 0.960 và p95 0.206 ms/tin, thấp hơn giới hạn 50 ms/tin. Yêu cầu 10 triệu SMS/ngày tương đương khoảng 10.000.000 / 86.400 ≈ 116 tin/giây trung bình. Trong notebook, mô hình xử lý cả lô khoảng 36.711 tin/giây, nhưng số đo trên Colab chưa chứng minh nó sẽ đạt tốc độ đó trên hệ thống thực tế.

Tôi chấp nhận đánh đổi rằng mô hình vẫn có thể bỏ sót spam ngụy trang và chặn nhầm tin bình thường. Trên 50 case khó, F1 của BoW + MLP chỉ còn 0.410. Nếu triển khai thật, cần kiểm thử thêm, đặc biệt với tin OTP hợp lệ và spam giả dạng lời nhắn cá nhân. Có thể cân nhắc kiểm tra bổ sung những trường hợp mô hình chưa chắc chắn, nhưng tôi chưa triển khai cách này trong lab.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
* Hỏi AI: Cách hoàn thành báo cáo dựa trên notebook, Cách phân tích dự đoán và 50 case khó, Cách làm thử thách #3.
* Nhận được: Chỉ vị trí bảng số liệu, gợi ý cách giải thích, Gợi ý so sánh F1 và các case #12, #41, Giải thích yêu cầu, công thức đếm tham số.
* Tôi sửa: 	Tôi đưa các chỉ số có trong output Phần 5 vào bảng báo cáo, Tôi đối chiếu các ví dụ với output trong notebook; các giải thích được diễn đạt lại ngắn hơn
* Điều tôi hiểu ra hôm nay là tôi chưa chú ý rằng một mô hình có điểm test cao vẫn có thể sai nhiều khi gặp các tin nhắn khác với dữ liệu quen thuộc. Tôi cũng hiểu rằng khi chọn mô hình không chỉ nhìn F1 mà còn phải xem thời gian dự đoán và các trường hợp phân loại sai.