# 🤖 Tối Ưu Hóa Cắt Thép 1D Sử Dụng Mạng Nơ-ron Đồ Thị (GNN) và Thuật Toán Bầy Kiến (ACO)

## 🧠 Giới thiệu
Dự án này triển khai một **phương pháp lai ghép tiên tiến (Hybrid Approach)** để giải quyết **Bài toán Cắt Thép Một Chiều (1D Cutting Stock Problem)**.  
Mục tiêu là tìm ra **kế hoạch cắt các thanh thép tiêu chuẩn** thành sản phẩm theo yêu cầu **với lượng vật liệu lãng phí tối thiểu**.

Điểm đặc biệt của dự án là sự kết hợp giữa:
- 🧩 **Mạng Nơ-ron Đồ Thị (Graph Neural Network – GNN)**: học mối quan hệ phức tạp giữa các sản phẩm.  
- 🐜 **Thuật toán Tối ưu hóa Bầy Kiến (Ant Colony Optimization – ACO)**: tìm kiếm giải pháp tối ưu toàn cục dựa trên pheromone và heuristic.

---

## 🚀 Cơ Chế Hoạt Động

### 🧠 1. Học từ dữ liệu (GNN)
- Sử dụng **Graph Attention Network (GATConv)** huấn luyện từ các mẫu cắt hiệu quả (trong `du_lieu_cat.csv`) hoặc dữ liệu ngẫu nhiên.
- GNN sinh ra **ma trận gợi ý (heuristic matrix)** biểu diễn xác suất hai sản phẩm nên được cắt cùng nhau.

### 🐜 2. Tối ưu hóa tìm kiếm (ACO)
- Thuật toán **Ant Colony Optimization (ACO)** xây dựng các mẫu cắt hoàn chỉnh.
- Mỗi "con kiến" chọn sản phẩm dựa trên:
  - **Mùi pheromone**: dấu vết của các phương án cắt thành công trước đó.
  - **Gợi ý từ GNN**: ma trận heuristic giúp định hướng thông minh hơn.
- Sau nhiều thế hệ, ACO hội tụ về **phương án có độ lãng phí thấp nhất**.

### 🔁 3. Tự cải tiến
- Các mẫu cắt hiệu quả được lưu lại trong `du_lieu_cat.csv` → mô hình GNN ngày càng thông minh hơn ở những lần chạy sau.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```bash
/your_project_folder
│
├── model.GNN-ACO.py          # 🧩 Mã nguồn chính: thực hiện huấn luyện GNN và tối ưu ACO
├── don_hang.csv              # 📥 INPUT: Danh sách sản phẩm cần cắt
│
├── gnn_model.pt              # 💾 OUTPUT: Trọng số mô hình GNN đã huấn luyện
├── du_lieu_cat.csv           # 📚 Dữ liệu các mẫu cắt hiệu quả (vừa là đầu vào, vừa là kết quả)
├── cutting.log               # 🪶 Nhật ký chạy chương trình (log file)
│
├── requirements.txt          # 📦 Danh sách thư viện cần thiết
└── README.md                 # 📖 File mô tả dự án

```
# ⚙️ Hướng Dẫn Sử Dụng & Cấu Hình Dự Án GNN–ACO Cutting Optimization

## 📊 1. Dữ Liệu Đầu Vào & Đầu Ra

### 🗂️ File **don_hang.csv**
Chứa thông tin đơn hàng cần cắt, gồm các cột:
- **ten_san_pham**: Tên sản phẩm
- **chieu_dai**: Chiều dài mỗi sản phẩm (đơn vị cùng với thanh thép)
- **so_luong**: Số lượng cần cắt

**Ví dụ:**
```csv
ten_san_pham,chieu_dai,so_luong
SP-A,23.5,50
SP-B,17.0,80
SP-C,42.1,35

```
## 🛠️ Cài Đặt Môi Trường  
---

### 1️⃣ Yêu cầu hệ thống / System Requirements  
**🇻🇳**  
- Python >= 3.8  
- pip >= 21.0  
- (Tùy chọn) GPU hỗ trợ CUDA nếu bạn muốn huấn luyện nhanh hơn  

---

### 2️⃣ Tạo môi trường ảo / Create a virtual environment  
**🇻🇳** (Khuyến khích để tránh xung đột thư viện)  
**🇬🇧** (Recommended to prevent library conflicts)  

```bash
python -m venv venv
```
# 🐧 Linux / macOS
```bash
source venv/bin/activate
```
# 🪟 Windows
```bash
venv\Scripts\activate
```
## 📜 Luồng Hoạt Động Của Chương Trình  

Dưới đây là quy trình hoạt động tổng thể của hệ thống **GNN + ACO** trong việc tối ưu hóa cắt thép 1D:

---

### 🔁 Quy trình tổng quát:

1️⃣ **Đọc dữ liệu đầu vào**  
   - Đọc file **`don_hang.csv`** chứa danh sách sản phẩm, chiều dài, và số lượng cần cắt.  

---

2️⃣ **Kiểm tra sự tồn tại của mô hình GNN** (`gnn_model.pt`)  
   - 🔍 **Nếu có:**  
     → Tải mô hình GNN đã được huấn luyện trước đó.  
   - ⚙️ **Nếu không có:**  
     → Tạo dữ liệu huấn luyện (từ `du_lieu_cat.csv` hoặc dữ liệu ngẫu nhiên).  
     → Huấn luyện mô hình GNN mới và lưu lại vào **`gnn_model.pt`**.  

---

3️⃣ **Sinh ma trận heuristic (gợi ý cắt)**  
   - Mô hình **GNN** học từ dữ liệu các mẫu cắt hiệu quả để sinh ra **ma trận heuristic**,  
     biểu thị khả năng hai sản phẩm nên được cắt cùng nhau.  

---

4️⃣ **Thuật toán Tối ưu hóa Bầy kiến (ACO)**  
   Sử dụng hai nguồn thông tin để xây dựng mẫu cắt tối ưu:  
   - 🐜 **Pheromone (vết mùi):** Dấu vết của các lời giải tốt trước đó.  
   - 🧭 **Heuristic từ GNN:** Gợi ý thông minh giúp hướng dẫn quá trình tìm kiếm.  

---

5️⃣ **Tìm kiếm kế hoạch cắt tối ưu**  
   - Các "con kiến" trong thuật toán sẽ dần dần xây dựng các **mẫu cắt** khả thi.  
   - Sau nhiều thế hệ lặp lại, thuật toán hội tụ và tìm ra kế hoạch cắt có **lượng lãng phí thấp nhất**.  

---

6️⃣ **In báo cáo tổng hợp kết quả**  
   - 🧾 Hiển thị các thông tin chính:
     - Hiệu suất sử dụng vật liệu.  
     - Tổng lượng thép lãng phí.  
     - Danh sách chi tiết các mẫu cắt tối ưu.  
   - Ghi lại toàn bộ log hoạt động vào **`cutting.log`**.  

---

7️⃣ **Cập nhật dữ liệu học cho GNN**  
   - Lưu lại các mẫu cắt tốt nhất vào file **`du_lieu_cat.csv`**.  
   - Lần chạy sau, GNN sẽ học từ dữ liệu này để **cải thiện độ chính xác** và **rút ngắn thời gian tìm kiếm**.  

---

### 🔄 Sơ đồ tóm tắt quy trình

```mermaid
flowchart TD
A[1️⃣ Đọc don_hang.csv] --> B[2️⃣ Kiểm tra gnn_model.pt]
B -->|Có| C[Tải mô hình GNN]
B -->|Không| D[Huấn luyện GNN mới]
C & D --> E[3️⃣ Sinh ma trận heuristic]
E --> F[4️⃣ Chạy thuật toán ACO]
F --> G[5️⃣ Tìm kế hoạch cắt tối ưu]
G --> H[6️⃣ In báo cáo & ghi log]
H --> I[7️⃣ Lưu mẫu cắt vào du_lieu_cat.csv]
I --> J[Hoàn thành quy trình]
