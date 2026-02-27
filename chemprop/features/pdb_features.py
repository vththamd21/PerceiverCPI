import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import distance_matrix

def get_calpha_distance_matrix(pdb_path, max_length=1024):
    """Đọc file PDB, lấy tọa độ C-alpha và tạo ma trận khoảng cách 2D."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception:
        return np.zeros((max_length, max_length))

    ca_coords = []
    # Chỉ lấy mô hình (Model) đầu tiên trong file PDB
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
        break 
    
    if len(ca_coords) == 0:
        return np.zeros((max_length, max_length))

    ca_coords = np.array(ca_coords)
    dist_matrix = distance_matrix(ca_coords, ca_coords)
    
    # Cắt bớt nếu chuỗi quá dài so với max_length
    if dist_matrix.shape[0] > max_length:
        dist_matrix = dist_matrix[:max_length, :max_length]
        
    # Padding bằng số 0 để đưa về kích thước cố định N x N
    padded_matrix = np.zeros((max_length, max_length))
    curr_len = dist_matrix.shape[0]
    padded_matrix[:curr_len, :curr_len] = dist_matrix
    
    return padded_matrix