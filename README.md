🤖 Tối Ưu Hóa Cắt Thép 1D Sử Dụng Mạng Nơ-ron Đồ Thị (GNN) và Tối ưu hóa Bầy kiến (ACO)
Dự án này triển khai một phương pháp lai ghép tiên tiến để giải quyết Bài toán Cắt Thép một chiều (1D Cutting Stock Problem). 
Mục tiêu là tìm ra kế hoạch cắt các thanh thép có chiều dài tiêu 
chuẩn thành các sản phẩm theo yêu cầu sao cho tổng lượng vật liệu lãng phí là ít nhất.

Điểm đặc biệt của dự án là sự kết hợp giữa Mạng Nơ-ron Đồ thị (GNN) để học các mối quan hệ phức tạp giữa các sản phẩm và thuật toán Tối ưu hóa Bầy kiến (ACO) để tìm kiếm giải pháp tối ưu.

--------------------------------------------------------------------------------------------------------------------------------

🚀 Cách Hoạt Động
Hệ thống hoạt động theo một cơ chế lai ghép thông minh:

Học hỏi từ Dữ liệu (GNN):

Một Mạng Nơ-ron Đồ thị (cụ thể là GATConv) được huấn luyện dựa trên các mẫu cắt hiệu quả trong quá khứ (lưu trong du_lieu_cat.csv) hoặc từ dữ liệu được tạo ngẫu nhiên.

Mục tiêu của GNN là học và tạo ra một "ma trận gợi ý" (heuristic matrix). Ma trận này biểu thị xác suất hai sản phẩm bất kỳ nên được cắt cùng nhau trên một thanh thép.

Tìm kiếm Tối ưu (ACO):

Thuật toán Tối ưu hóa Bầy kiến (ACO) được sử dụng để xây dựng các kế hoạch cắt hoàn chỉnh.

Các "con kiến" trong thuật toán sẽ lựa chọn các sản phẩm để đưa vào một mẫu cắt. Quyết định của chúng được dẫn dắt bởi hai yếu tố:

Mùi Pheromone: Dấu vết do các "con kiến" thành công ở các thế hệ trước để lại, cho biết những cặp sản phẩm nào đã từng tạo ra kết quả tốt.

Gợi ý từ GNN: Ma trận heuristic do GNN cung cấp, giúp các con kiến đưa ra những lựa chọn thông minh hơn ngay từ đầu.

Qua nhiều thế hệ, thuật toán sẽ hội tụ về giải pháp có độ lãng phí thấp nhất.

Tự Cải tiến:

Sau khi tìm được kế hoạch cắt tối ưu cho một đơn hàng, các mẫu cắt hiệu quả trong kế hoạch đó sẽ được lưu lại vào tệp du_lieu_cat.csv.

Điều này giúp mô hình GNN ngày càng "thông minh" hơn trong những lần chạy tiếp theo, vì nó được học từ chính những kết quả tốt nhất mà nó đã tìm ra.

--------------------------------------------------------------------------------------------------------------------------------

📁 Cấu trúc Thư mục và Tệp
Để dự án hoạt động, bạn cần có các tệp sau trong cùng một thư mục:

/your_project_folder
  |
  |-- model.GNN-ACO.py         # File mã nguồn chính của chương trình
  |-- don_hang.csv             # INPUT: File chứa thông tin đơn hàng cần xử lý
  |
  |-- gnn_model.pt             # OUTPUT: File lưu trọng số của mô hình GNN đã huấn luyện
  |-- du_lieu_cat.csv          # OUTPUT & INPUT: Kho dữ liệu các mẫu cắt hiệu quả
  |-- cutting.log              # OUTPUT: File ghi lại nhật ký hoạt động của chương trình
1. File Đầu vào (don_hang.csv)
Đây là file CSV chứa danh sách các sản phẩm cần cắt. File phải có 3 cột: ten_san_pham, chieu_dai, so_luong.

Ví dụ:
  ten_san_pham,chieu_dai,so_luong
  SP-A,23.5,50
  SP-B,17.0,80
  SP-C,42.1,35
2. File Đầu ra / Dữ liệu học
gnn_model.pt: Trọng số của mô hình GNN sẽ được tự động lưu vào file này sau lần huấn luyện đầu tiên. Ở những lần chạy sau, chương trình sẽ tải mô hình từ file này thay vì huấn luyện lại (trừ khi file bị xóa).

du_lieu_cat.csv: Chứa các mẫu cắt tốt nhất được tìm thấy. Dữ liệu trong file này được dùng để huấn luyện GNN.

cutting.log: Ghi lại các thông tin, cảnh báo hoặc lỗi xảy ra trong quá trình thực thi.

--------------------------------------------------------------------------------------------------------------------------------

🛠️ Cài đặt Môi trường
Để chạy dự án, bạn cần cài đặt các thư viện Python cần thiết.

Tạo một môi trường ảo (khuyến khích):

Bash

python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
Cài đặt các thư viện: Dự án yêu cầu các thư viện PyTorch và PyTorch Geometric. Hãy cài đặt chúng trước theo hướng dẫn trên trang chủ của chúng để đảm bảo tương thích với hệ thống của bạn.

PyTorch Installation

PyTorch Geometric Installation

Sau đó, cài đặt các thư viện còn lại:

Bash

  pip install pandas numpy tqdm
or 
  pip install requirement.txt

--------------------------------------------------------------------------------------------------------------------------------

⚙️ Cách Sử dụng
Chuẩn bị file don_hang.csv: Đảm bảo file này tồn tại trong cùng thư mục và có đúng định dạng như đã mô tả.

Chạy chương trình: Mở terminal hoặc command prompt, điều hướng đến thư mục dự án và chạy lệnh:

Bash

python model.GNN-ACO.py
Xem kết quả:

Lần chạy đầu tiên: Chương trình sẽ mất một chút thời gian để huấn luyện mô hình GNN.

Các lần chạy sau: Chương trình sẽ tải mô hình đã được huấn luyện và chạy nhanh hơn.

Kết quả chi tiết về kế hoạch cắt, hiệu suất, và sản lượng sẽ được in ra màn hình một cách trực quan.

--------------------------------------------------------------------------------------------------------------------------------

📜 Luồng Hoạt động của Chương trình
Khi bạn chạy file model.GNN-ACO.py, nó sẽ thực hiện các bước sau:

Đọc Đơn hàng: Tải dữ liệu từ don_hang.csv.

Chuẩn bị GNN:

Kiểm tra xem file gnn_model.pt có tồn tại không.

Nếu có: Tải trọng số mô hình đã được huấn luyện.

Nếu không:

Tạo dữ liệu huấn luyện từ du_lieu_cat.csv (nếu có) và bổ sung bằng các mẫu ngẫu nhiên.

Huấn luyện mô hình GNN từ đầu.

Lưu mô hình đã huấn luyện vào gnn_model.pt.

Khởi tạo Solver GNN-ACO:

Sử dụng GNN để tính toán ma trận heuristic.

Khởi tạo các tham số cho thuật toán ACO (số lượng kiến, tốc độ bay hơi, v.v.).

Giải bài toán: Chạy thuật toán ACO qua nhiều thế hệ để tìm ra kế hoạch cắt tốt nhất.

In Báo cáo: Hiển thị kết quả chi tiết ra màn hình, bao gồm:

Bảng tóm tắt tổng quan (tổng lãng phí, hiệu suất).

Bảng chi tiết các mẫu cắt và số lần lặp lại.

Bảng so sánh sản lượng yêu cầu và sản lượng thực tế.

Cập nhật Kho dữ liệu: Lưu các mẫu cắt tối ưu vừa tìm được vào du_lieu_cat.csv để cải thiện mô hình cho các lần chạy trong tương lai.
