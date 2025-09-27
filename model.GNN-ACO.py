import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from collections import Counter
import time
import textwrap
import pandas as pd
import os
import logging
import csv
from tqdm import tqdm

# --- Phần 0: Cấu hình Logging ---
# Ghi lại các thông tin, cảnh báo và lỗi trong quá trình chạy vào file cutting.log
logging.basicConfig(filename='cutting.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Phần 1: Định nghĩa và các hàm liên quan đến GNN ---
class GNN_Heuristic(nn.Module):
    """
    Định nghĩa kiến trúc Mạng Nơ-ron Đồ thị (GNN).
    Mạng này học mối quan hệ giữa các sản phẩm để đưa ra gợi ý (heuristic)
    cho thuật toán ACO về việc sản phẩm nào nên được cắt cùng nhau.
    """
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(GNN_Heuristic, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, embedding_dim, heads=1)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, x, edge_index, edge_label_index=None):
        # Lan truyền tiến qua các lớp GATConv với hàm kích hoạt elu và dropout
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        # Nếu đang trong quá trình huấn luyện, tính toán xác suất cho các cạnh
        if edge_label_index is not None:
            # Ghép nối các vector embedding của các cặp nút
            edge_features = torch.cat([x[edge_label_index[0]], x[edge_label_index[1]]], dim=-1)
            # Trả về xác suất cạnh (0 đến 1) qua lớp linear và hàm sigmoid
            return torch.sigmoid(self.fc(edge_features))
        
        # Nếu đang trong quá trình suy luận, trả về vector embedding của các nút
        return x

def update_and_save_patterns(plan, items, file_path='du_lieu_cat.csv'):
    """
    Cập nhật file csv chứa các mẫu cắt tối ưu.
    Lưu dưới dạng các tên sản phẩm phân cách bởi dấu phẩy.
    """
    if not plan:
        return
    print(f"\n💾 Cập nhật kho dữ liệu mẫu cắt tại '{file_path}'...")
    length_to_name = {v: k for k, v in items.items()}
    existing_patterns = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Bỏ qua header
            try: next(reader)
            except StopIteration: pass
            for row in reader:
                if row: existing_patterns.add(row[0])
    except FileNotFoundError:
        print(f"   -> File '{file_path}' chưa tồn tại, sẽ được tạo mới.")
        pass

    new_patterns_to_add = set()
    for p_info in plan:
        pattern_names = [length_to_name.get(length, 'UNKNOWN') for length in p_info['pattern']]
        canonical_form = ",".join(sorted(pattern_names))
        
        if canonical_form not in existing_patterns:
            new_patterns_to_add.add(canonical_form)

    if new_patterns_to_add:
        try:
            # Mở file ở chế độ 'a' (append), tạo nếu chưa có
            is_new_file = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if is_new_file:
                    writer.writerow(['pattern_names_csv'])
                
                for pattern_str in sorted(list(new_patterns_to_add)):
                     writer.writerow([pattern_str])
            print(f"   -> Đã thêm {len(new_patterns_to_add)} mẫu cắt mới vào kho dữ liệu.")
        except IOError as e:
            logging.error(f"Không thể ghi vào file {file_path}: {e}")
            print(f"LỖI: Không thể ghi vào file {file_path}.")
    else:
        print("   -> Không có mẫu cắt mới nào để thêm.")

def generate_training_data(items, demands, stock_length, num_samples=2000, historical_patterns_path='du_lieu_cat.csv'):
    """
    Tạo dữ liệu huấn luyện GNN từ file lịch sử và dữ liệu ngẫu nhiên.
    Đọc dữ liệu từ cấu trúc mới (phân cách bởi dấu phẩy).
    """
    item_lengths_arr = np.array(list(items.values()))
    n_items = len(items)
    item_names = list(items.keys())
    name_to_idx = {name: i for i, name in enumerate(item_names)}

    # Chuẩn hóa đặc trưng của nút (chiều dài và sản lượng)
    norm_lengths = item_lengths_arr / stock_length
    max_demand = np.max(list(demands.values())) if demands else 1
    norm_demands = np.array([demands.get(name, 0) for name in item_names]) / max_demand if max_demand > 0 else np.zeros(n_items)
    node_features = np.vstack([norm_lengths, norm_demands]).T
    
    edge_list = [[i, j] for i in range(n_items) for j in range(n_items) if i != j]
    edge_index_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data_list = []
    all_source_patterns = []

    # Đọc dữ liệu lịch sử
    try:
        with open(historical_patterns_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Bỏ qua header
            historical_patterns = []
            for row in reader:
                if not row: continue
                pattern_names = row[0].split(',')
                pattern_indices = [name_to_idx[name] for name in pattern_names if name in name_to_idx]
                if pattern_indices:
                    historical_patterns.append(pattern_indices)
            all_source_patterns.extend(historical_patterns)
        if historical_patterns:
            print(f"   -> Đã đọc {len(historical_patterns)} mẫu cắt từ '{historical_patterns_path}' để làm dữ liệu huấn luyện.")
    except (FileNotFoundError, StopIteration):
        print(f"   -> Không tìm thấy hoặc file '{historical_patterns_path}' trống. Sẽ dùng dữ liệu ngẫu nhiên.")
    except Exception as e:
        print(f"   -> Lỗi khi đọc file lịch sử {historical_patterns_path}: {e}. Bỏ qua.")

    # Sinh ngẫu nhiên dữ liệu để bổ sung
    num_random_samples = max(0, num_samples - len(all_source_patterns))
    if num_random_samples > 0:
        print(f"   -> Sẽ tạo thêm {num_random_samples} mẫu ngẫu nhiên để đủ {num_samples} mẫu.")
        for _ in range(num_random_samples):
            remaining_length = stock_length
            pattern = []
            possible_indices = list(range(n_items))
            np.random.shuffle(possible_indices)
            for idx in possible_indices:
                if item_lengths_arr[idx] <= remaining_length and demands.get(item_names[idx], 0) > 0:
                    pattern.append(idx)
                    remaining_length -= item_lengths_arr[idx]
            all_source_patterns.append(pattern)
            
    # Tạo đối tượng Data cho tất cả các mẫu
    for pattern in all_source_patterns:
        edge_labels = torch.zeros(len(edge_list), dtype=torch.float)
        if len(pattern) > 1:
            for i in range(len(pattern)):
                for j in range(i + 1, len(pattern)):
                    u, v = pattern[i], pattern[j]
                    try:
                        edge_idx = edge_list.index([u, v]) if [u, v] in edge_list else edge_list.index([v, u])
                        edge_labels[edge_idx] = 1.0
                    except ValueError:
                        continue
        data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index_tensor, edge_label_index=edge_index_tensor, edge_label=edge_labels)
        data_list.append(data)
    
    return data_list

def train_gnn(model, data_list, epochs=50, lr=0.01):
    """
    Hàm huấn luyện mô hình GNN với hiển thị tiến trình chi tiết.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    
    print("\nBắt đầu huấn luyện GNN...")
    if not data_list:
        print("CẢNH BÁO: Không có dữ liệu để huấn luyện. Bỏ qua bước huấn luyện.")
        return model

    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        # Sử dụng tqdm để tạo thanh tiến trình cho mỗi epoch
        data_iterator = tqdm(data_list, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=100)
        
        for data in data_iterator:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_label_index).squeeze()
            
            if out.shape != data.edge_label.shape:
                 logging.warning(f"Bỏ qua mẫu do lỗi shape: out: {out.shape}, label: {data.edge_label.shape}")
                 continue

            loss = criterion(out, data.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            data_iterator.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(data_list) if len(data_list) > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs} hoàn thành | Loss trung bình: {avg_loss:.4f} | Thời gian: {epoch_duration:.2f} giây")

    total_training_time = time.time() - training_start_time
    
    print("-" * 60)
    print(f"✅ Huấn luyện GNN hoàn tất!")
    print(f"   -> Tổng thời gian huấn luyện: {total_training_time:.2f} giây.")
    print("-" * 60)
    
    return model

def save_gnn_model(model, path='gnn_model.pt'):
    """Lưu trọng số của mô hình GNN đã huấn luyện."""
    torch.save(model.state_dict(), path)
    print(f"Đã lưu mô hình GNN tại '{path}'")

# --- Phần 2: Định nghĩa Solver lai ghép GNN-ACO ---
class GNN_ACO_Solver:
    """
    Lớp chính để giải bài toán, kết hợp GNN và ACO.
    - GNN: Cung cấp ma trận heuristic (gợi ý).
    - ACO: Sử dụng ma trận heuristic và pheromone để xây dựng các giải pháp.
    """
    def __init__(self, stock_length, items, demands, gnn_model, aco_params):
        self.stock_length = stock_length
        self.item_lengths = np.array(list(items.values()))
        self.item_indices = {name: i for i, name in enumerate(items.keys())}
        self.index_to_name = {i: name for name, i in self.item_indices.items()}
        self.demands_initial = np.array([demands.get(self.index_to_name[i], 0) for i in range(len(items))])
        self.n_items = len(items)
        self.params = aco_params
        
        self.gnn_model = gnn_model
        self.heuristic_matrix = self._calculate_heuristic_info()
        self.pheromone_matrix = np.ones((self.n_items, self.n_items))
        
        self.best_overall_plan = None
        self.best_overall_waste = float('inf')
        self.best_overall_fitness = float('inf')

    def _create_graph_from_items(self):
        """Tạo đối tượng đồ thị từ dữ liệu bài toán hiện tại cho GNN."""
        norm_lengths = self.item_lengths / self.stock_length
        max_demand = np.max(self.demands_initial)
        norm_demands = self.demands_initial / max_demand if max_demand > 0 else np.zeros_like(self.demands_initial, dtype=float)
        node_features = np.vstack([norm_lengths, norm_demands]).T
        
        edge_list = [[i, j] for i in range(self.n_items) for j in range(self.n_items) if i != j]
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def _calculate_heuristic_info(self):
        """Sử dụng GNN để suy luận và tạo ra ma trận heuristic."""
        print("\nGNN đang tính toán ma trận heuristic...")
        graph = self._create_graph_from_items()
        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(graph.x, graph.edge_index).cpu().numpy()
        
        heuristic_matrix = np.dot(embeddings, embeddings.T)
        min_val, max_val = np.min(heuristic_matrix), np.max(heuristic_matrix)
        if max_val == min_val:
            return np.ones_like(heuristic_matrix)
        return (heuristic_matrix - min_val) / (max_val - min_val + 1e-9)

    def _build_one_pattern(self, demands):
        """Một con kiến xây dựng một mẫu cắt."""
        pattern_indices = []
        remaining_length = self.stock_length
        
        possible_first = np.where((demands > 0) & (self.item_lengths <= remaining_length))[0]
        if len(possible_first) == 0:
            return [], []
        
        first_item_idx = np.random.choice(possible_first)
        pattern_indices.append(first_item_idx)
        remaining_length -= self.item_lengths[first_item_idx]
        
        while True:
            last_item_idx = pattern_indices[-1]
            possible_next = np.where((demands > 0) & (self.item_lengths <= remaining_length))[0]
            
            if len(possible_next) == 0:
                break
                
            pheromones = self.pheromone_matrix[last_item_idx, possible_next]
            heuristics = self.heuristic_matrix[last_item_idx, possible_next]
            probabilities = (pheromones ** self.params['alpha']) * (heuristics ** self.params['beta'])
            
            prob_sum = np.sum(probabilities)
            if prob_sum <= 1e-9 or not np.all(np.isfinite(probabilities)):
                probabilities = np.ones(len(possible_next)) / len(possible_next)
            else:
                probabilities /= prob_sum
            
            next_item_idx = np.random.choice(possible_next, p=probabilities)
            pattern_indices.append(next_item_idx)
            remaining_length -= self.item_lengths[next_item_idx]
        
        pattern_lengths = [self.item_lengths[i] for i in pattern_indices]
        return pattern_lengths, pattern_indices

    def _calculate_pattern_repeats(self, pattern_indices, demands):
        """Tính số lần lặp lại tối đa cho một mẫu cắt để không vượt quá sản lượng yêu cầu."""
        if not pattern_indices:
            return 0
        
        counts = Counter(pattern_indices)
        max_repeats_options = [int(demands[idx] // count) for idx, count in counts.items() if demands[idx] > 0 and count > 0]
        
        return max(1, min(max_repeats_options)) if max_repeats_options else 0

    def _construct_full_cutting_plan(self):
        """Một con kiến xây dựng một kế hoạch cắt hoàn chỉnh để đáp ứng tất cả yêu cầu."""
        remaining_demands = self.demands_initial.copy()
        cutting_plan = []
        produced_counts = np.zeros(self.n_items)
        
        while np.sum(remaining_demands[remaining_demands > 0]) > 0:
            new_pattern_lengths, new_pattern_indices = self._build_one_pattern(remaining_demands)
            
            if not new_pattern_lengths:
                break 
            
            repeats = self._calculate_pattern_repeats(new_pattern_indices, remaining_demands)
            
            if repeats == 0:
                for idx in set(new_pattern_indices):
                    remaining_demands[idx] = 0 
                continue

            cutting_plan.append({'pattern': new_pattern_lengths, 'repeats': repeats, 'indices': new_pattern_indices})
            counts_in_pattern = Counter(new_pattern_indices)
            for item_idx, count in counts_in_pattern.items():
                produced_counts[item_idx] += count * repeats
                remaining_demands[item_idx] = max(0, remaining_demands[item_idx] - count * repeats)

        for item_idx in range(self.n_items):
            shortfall = self.demands_initial[item_idx] - produced_counts[item_idx]
            if shortfall > 0:
                item_length = self.item_lengths[item_idx]
                if item_length <= self.stock_length:
                    items_per_bar = self.stock_length // item_length
                    num_bars_needed = int(np.ceil(shortfall / items_per_bar))
                    
                    cutting_plan.append({
                        'pattern': [item_length] * int(items_per_bar),
                        'repeats': num_bars_needed,
                        'indices': [item_idx] * int(items_per_bar)
                    })
                    produced_counts[item_idx] += items_per_bar * num_bars_needed

        total_waste = sum([(self.stock_length - sum(p['pattern'])) * p['repeats'] for p in cutting_plan])
        
        unmet_demands = np.sum(np.maximum(0, self.demands_initial - produced_counts))
        penalty = self.params.get('penalty_weight', 1e6) * unmet_demands
        fitness = total_waste + penalty
        
        return cutting_plan, total_waste, fitness

    def _update_pheromones(self, all_ant_plans):
        """Cập nhật ma trận pheromone sau mỗi thế hệ."""
        self.pheromone_matrix *= (1 - self.params['evaporation_rate'])
        
        for plan, waste, fitness in all_ant_plans:
            if plan is None:
                continue
            
            deposit_amount = 1.0 / (fitness + 1.0)
            
            for pattern_info in plan:
                indices = pattern_info['indices']
                for i in range(len(indices) - 1):
                    for j in range(i + 1, len(indices)):
                        start_node, end_node = indices[i], indices[j]
                        self.pheromone_matrix[start_node, end_node] += deposit_amount
                        self.pheromone_matrix[end_node, start_node] += deposit_amount

    def solve(self):
        """Hàm chính để chạy thuật toán GNN-ACO."""
        print("\n🚀 Bắt đầu giải bài toán bằng GNN-ACO...")
        for gen in range(self.params['max_generations']):
            all_ant_plans = []
            for _ in range(self.params['num_ants']):
                plan, waste, fitness = self._construct_full_cutting_plan()
                all_ant_plans.append((plan, waste, fitness))
            
            best_in_gen_plan, _, best_in_gen_fitness = min(all_ant_plans, key=lambda x: x[2])
            
            if best_in_gen_fitness < self.best_overall_fitness:
                self.best_overall_fitness = best_in_gen_fitness
                self.best_overall_plan = best_in_gen_plan
                # Tính lại waste chính xác cho giải pháp tốt nhất (vì fitness có thể chứa penalty)
                self.best_overall_waste = sum([(self.stock_length - sum(p['pattern'])) * p['repeats'] for p in self.best_overall_plan])
                
                is_valid = (self.best_overall_fitness - self.best_overall_waste) < 1.0
                valid_str = "✅ Hợp lệ" if is_valid else "❌ Chưa hợp lệ"
                print(f"Thế hệ {gen+1:02d}: 🔥 Giải pháp mới! Lãng phí: {self.best_overall_waste:.2f} (Fitness: {self.best_overall_fitness:.2f}) - {valid_str}")

            self._update_pheromones(all_ant_plans)
            
            if (gen + 1) % 10 == 0:
                print(f"   -> Hoàn thành thế hệ {gen+1}/{self.params['max_generations']}. Fitness tốt nhất hiện tại: {self.best_overall_fitness:.2f}")
        
        print("\n✅ Giải xong!")
        return self.best_overall_plan, self.best_overall_waste

# --- Phần 3: Hàm in kết quả ---
def print_beautiful_results(plan, waste, stock_length, execution_time, items, demands):
    """In kết quả cuối cùng ra màn hình một cách trực quan."""
    stt_col_width, pattern_col_width, repeats_col_width, waste_col_width = 4, 45, 10, 15
    total_width = stt_col_width + pattern_col_width + repeats_col_width + waste_col_width + 13

    print("\n" + "="*total_width)
    print("||" + " BÁO CÁO KẾT QUẢ TỐI ƯU HÓA CẮT THÉP ".center(total_width - 4) + "||")
    print("="*total_width)

    if not plan:
        print("\nKhông tìm thấy giải pháp nào.".center(total_width))
        print("="*total_width)
        return

    # Gộp các mẫu cắt giống hệt nhau
    unique_patterns = {}
    for p_info in plan:
        pattern_tuple = tuple(sorted(p_info['pattern'], reverse=True))
        if pattern_tuple not in unique_patterns:
            unique_patterns[pattern_tuple] = 0
        unique_patterns[pattern_tuple] += p_info['repeats']

    total_bars = sum(unique_patterns.values())
    total_stock_material = total_bars * stock_length
    total_used_material = sum(sum(p) * r for p, r in unique_patterns.items())
    actual_waste = total_stock_material - total_used_material
    efficiency = (total_used_material / total_stock_material) * 100 if total_stock_material > 0 else 0
    
    print("\n📊 [ BẢNG TÓM TẮT TỔNG QUAN ]\n")
    print(f"   - Tổng số thanh thép gốc sử dụng   : {total_bars} thanh")
    print(f"   - Hiệu suất sử dụng vật liệu       : {efficiency:.2f} %")
    print(f"   - Tổng lượng lãng phí              : {actual_waste:.2f}")
    print(f"   - Thời gian thực thi toàn bộ       : {execution_time:.4f} giây")

    print("\n📋 [ BẢNG CHI TIẾT KẾ HOẠCH CẮT ]\n")
    header = f"| {'STT':^{stt_col_width}} | {'MẪU CẮT (CÁC SẢN PHẨM)':^{pattern_col_width}} | {'LẶP LẠI':^{repeats_col_width}} | {'LÃNG PHÍ/THANH':^{waste_col_width}} |"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    sorted_patterns = sorted(unique_patterns.items(), key=lambda item: sum(item[0]), reverse=True)
    for i, (pattern, count) in enumerate(sorted_patterns):
        pattern_str = f"({', '.join(map(str, pattern))})"
        waste_per_bar = stock_length - sum(pattern)
        wrapped_lines = textwrap.wrap(pattern_str, width=pattern_col_width - 2)
        
        first_line = wrapped_lines[0] if wrapped_lines else ''
        print(f"| {i+1:<{stt_col_width}} | {first_line:<{pattern_col_width}} | {count:^{repeats_col_width}} | {waste_per_bar:^{waste_col_width}.2f} |")
        
        for line in wrapped_lines[1:]:
            print(f"| {'':<{stt_col_width}} | {line:<{pattern_col_width}} | {'':<{repeats_col_width}} | {'':<{waste_col_width}} |")
    print(separator)

    print("\n📦 [ BẢNG TỔNG HỢP SẢN LƯỢNG ]\n")
    prod_header = f"| {'SẢN PHẨM':<20} | {'YÊU CẦU (Cần)':^15} | {'SẢN XUẤT (Cắt được)':^20} | {'CHÊNH LỆCH':^15} |"
    prod_separator = "-" * len(prod_header)
    print(prod_separator)
    print(prod_header)
    print(prod_separator)

    length_to_name = {v: k for k, v in items.items()}
    produced_counts = {name: 0 for name in demands.keys()}
    for pattern_tuple, repeats in unique_patterns.items():
        counts_in_pattern = Counter(pattern_tuple)
        for length, num_in_pattern in counts_in_pattern.items():
            if length in length_to_name:
                produced_counts[length_to_name[length]] += num_in_pattern * repeats
            
    for name, required in sorted(demands.items(), key=lambda x: items[x[0]], reverse=True):
        produced = produced_counts.get(name, 0)
        diff = produced - required
        status = "✅" if diff >= 0 else "❌"
        print(f"| {name:<20} | {required:^15} | {produced:^20} | {f'{diff:+.0f}':^15} {status}|")
    print(prod_separator)
    print("\n" + "="*total_width)

# --- Phần 4: Hàm Main để chạy chương trình ---
if __name__ == '__main__':
    script_start_time = time.time()

    print("Đang đọc đơn hàng từ file don_hang.csv...")
    try:
        order_df = pd.read_csv('don_hang.csv')
        if any(order_df['chieu_dai'] <= 0) or any(order_df['chieu_dai'] > 100.0):
            raise ValueError("Chiều dài sản phẩm phải lớn hơn 0 và không quá 100.0.")
        if any(order_df['so_luong'] < 0):
            raise ValueError("Số lượng yêu cầu không được là số âm.")
        print("Đọc đơn hàng thành công!")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'don_hang.csv'. Vui lòng tạo file này.")
        exit()
    except ValueError as e:
        print(f"LỖI DỮ LIỆU: {e}")
        exit()

    # --- Khởi tạo các biến toàn cục ---
    ITEMS = pd.Series(order_df.chieu_dai.values, index=order_df.ten_san_pham).to_dict()
    DEMANDS = pd.Series(order_df.so_luong.values, index=order_df.ten_san_pham).to_dict()

    STOCK_LENGTH = 100.0
    ACO_PARAMS = {
        'num_ants': 20,
        'max_generations': 100,
        'alpha': 1.0,           # Tầm quan trọng của pheromone
        'beta': 3.0,            # Tầm quan trọng của heuristic (GNN)
        'evaporation_rate': 0.5,# Tốc độ bay hơi
        'penalty_weight': 1e6   # Trọng số phạt nếu không đủ sản lượng
    }

    print("\n" + "="*60)
    print(" BÀI TOÁN TỐI ƯU HÓA CẮT THÉP 1D (GNN-ACO)".center(60))
    print(f"Chiều dài thanh thép gốc: {STOCK_LENGTH}".center(60))
    print("="*60)

    # --- Xử lý mô hình GNN ---
    gnn_model = GNN_Heuristic(in_channels=2, hidden_channels=32, embedding_dim=16)
    model_path = 'gnn_model.pt'
    cutting_data_path = 'du_lieu_cat.csv'

    # Logic tải hoặc huấn luyện GNN
    # Nếu muốn huấn luyện lại từ đầu, hãy xóa file 'gnn_model.pt'
    if os.path.exists(model_path):
        print(f"\n✅ Đã tìm thấy mô hình GNN tại '{model_path}'. Đang tải...")
        try:
            gnn_model.load_state_dict(torch.load(model_path))
            gnn_model.eval()
            print("   -> Tải mô hình thành công. Bỏ qua bước huấn luyện.")
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}. Sẽ tiến hành huấn luyện lại.")
            if os.path.exists(model_path): os.remove(model_path)
            training_data = generate_training_data(ITEMS, DEMANDS, STOCK_LENGTH)
            gnn_model = train_gnn(gnn_model, training_data, epochs=50)
            save_gnn_model(gnn_model, model_path)
    else:
        print(f"\n⚠️ Không tìm thấy mô hình GNN tại '{model_path}'.")
        training_data = generate_training_data(ITEMS, DEMANDS, STOCK_LENGTH)
        gnn_model = train_gnn(gnn_model, training_data, epochs=50)
        save_gnn_model(gnn_model, model_path)

    # --- Khởi tạo và chạy Solver ---
    solver = GNN_ACO_Solver(STOCK_LENGTH, ITEMS, DEMANDS, gnn_model, ACO_PARAMS)
    best_plan, best_waste = solver.solve()
    
    script_end_time = time.time()
    execution_time = script_end_time - script_start_time

    # --- In kết quả ---
    print_beautiful_results(best_plan, best_waste, STOCK_LENGTH, execution_time, ITEMS, DEMANDS)
    
    # --- Cập nhật kho dữ liệu ---
    update_and_save_patterns(best_plan, ITEMS, cutting_data_path)