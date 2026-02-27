import pandas as pd
import requests
import os
import wget
from tqdm import tqdm

# 1. Đường dẫn file CSV đầu vào của bạn
# (Sửa lại tên file này cho khớp với file bạn đang có)
input_csv = 'toy_dataset/davis_with_uniprot.csv' 
output_csv = 'dataset_ready_for_3d.csv'

df = pd.read_csv(input_csv)

# 2. Tạo thư mục lưu file PDB NGAY TRONG REPO hiện tại
save_dir = "./alphafold_pdb"
os.makedirs(save_dir, exist_ok=True)

# 3. Lấy danh sách UniProt ID độc nhất (bỏ qua giá trị rỗng)
unique_uniprot_ids = df['UniProt_ID'].dropna().unique()

print(f"ĐANG TẢI CẤU TRÚC 3D CHO {len(unique_uniprot_ids)} PROTEIN TỪ ALPHAFOLD...")
found_pdb = 0
missing_pdb = 0

for uid in tqdm(unique_uniprot_ids):
    # Đường dẫn file nội bộ: ví dụ "./alphafold_pdb/Q13689.pdb"
    file_path = os.path.join(save_dir, f"{uid}.pdb")
    
    # Bỏ qua nếu file đã được tải từ trước (giúp bạn chạy lại code mà không bị tải lại từ đầu)
    if os.path.exists(file_path):
        found_pdb += 1
        continue
        
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            download_url = data[0]['pdbUrl'] 
            wget.download(download_url, out=file_path, bar=None)
            found_pdb += 1
        else:
            missing_pdb += 1
    except Exception as e:
        missing_pdb += 1

print("\nĐã tải xong file 3D!")
print(f" -> Thành công: {found_pdb} file")
print(f" -> Không tìm thấy: {missing_pdb} file")

# 4. Tạo cột 'pdb_path' trong dataframe
# Ghi thẳng đường dẫn tương đối vào CSV
df['pdb_path'] = df['UniProt_ID'].apply(
    lambda x: os.path.join(save_dir, f"{x}.pdb") if pd.notna(x) else None
)

# 5. Lưu lại thành file CSV mới
df.to_csv(output_csv, index=False)
print(f"\n✅ Đã lưu dataset hoàn chỉnh tại: {output_csv}")