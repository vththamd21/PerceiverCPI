import csv
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
# import seaborn as sns
import statistics
from scipy.stats import norm
import numpy as np

def new_compound():
    index = 0
    for i in range(5):
        input_file = r'dataset_ready_for_3d.csv'
        train_file = r'toy_dataset/newcomp_3D/%sdavis_train_newcomp.csv'%i
        test_file = r'toy_dataset/newcomp_3D/%sdavis_test_newcomp.csv'%i
        with open(input_file) as csv_file_a:
            with open(train_file, mode='w',newline='') as result1:
                with open(test_file, mode='w',newline='') as result:
                        a = csv.reader(csv_file_a, delimiter=',')
                        or_headers = next(a)
                        csv_readera = list(a)
                        result = csv.writer(result, delimiter=',')
                        result1 = csv.writer(result1, delimiter=',')
                        headers = []
                        smile = []
                        for row1 in csv_readera:
                            headers.append(row1[3])
                            smile.append(row1[0])
                        headers_pro = sorted(list(set(headers)))
                        # print(headers_pro)
                        smiles = sorted(list(set(smile)))
                        thresh_hold = 0.2
                        number_smiles_test = round(thresh_hold*len(smiles))
                        # print(number_smiles_test)
                        b =  index + int(number_smiles_test)
                        smiles_test = smiles[index : b]
                        index = b
                        print(index)
                        all_headers = or_headers
                        result.writerow([i for i in all_headers])
                        result1.writerow([i for i in all_headers])

                        for smile in smiles_test:
                            for row in csv_readera:
                                if row[0] == smile:
                                    row_new = [*range(len(or_headers))]
                                    row_new[0] = smile
                                    row_new[1] = row[1]
                                    row_new[2] = row[2]
                                    row_new[3] = row[3]
                                    row_new[4] = row[4]
                                    row_new[5] = row[5]
                                    result.writerow(row_new)
                        for smile in smiles:
                            if smile not in smiles_test:
                                for row in csv_readera:
                                    if row[0] == smile:
                                        row_new = [*range(len(or_headers))]
                                        # row_new.append(smile)
                                        row_new[0] = smile
                                        row_new[1] = row[1]
                                        row_new[2] = row[2]
                                        row_new[3] = row[3]
                                        row_new[4] = row[4]
                                        row_new[5] = row[5]
                                        result1.writerow(row_new)
                        print('done!')
def new_protein():
    index = 0
    for i in range(5):
        input_file = r'dataset_ready_for_3d.csv'
        train_file = r'toy_dataset/newprot_3D/%sdavis_train_newprot.csv'%i
        test_file = r'toy_dataset/newprot_3D/%sdavis_test_newprot.csv'%i
        
        with open(input_file) as csv_file_a:
            with open(train_file, mode='w',newline='') as result:
                with open(test_file, mode='w',newline='') as result1:
                    a = csv.reader(csv_file_a, delimiter=',')
                    # print(len(list(a)))
                    or_headers = next(a)
                    csv_readera = list(a)
                    # print(len(list(csv_readera)))
                    result = csv.writer(result, delimiter=',')
                    result1 = csv.writer(result1, delimiter=',')
                    headers = []
                    smile = []
                    for row1 in csv_readera:
                        headers.append(row1[3])
                        smile.append(row1[0])
                    headers_pro = sorted(list(set(headers)))
                    # print(headers_pro.index('YES'))
                    # input()
                    smiles = sorted(list(set(smile)))
                    thresh_hold = 0.2
                    number_proteins_test = round(thresh_hold*len(headers_pro))
                    # print(number_proteins_test)
                    b =  index + int(number_proteins_test)
                    prots_test = headers_pro[index : b]
                    index = b
                    print(index)
                    all_headers = or_headers
                    result.writerow([i for i in all_headers])
                    result1.writerow([i for i in all_headers])
                    for prot in prots_test:
                        for row in csv_readera:
                            if row[3] == prot:
                                row_new = [*range(len(or_headers))]
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                row_new[3] = row[3]
                                row_new[4] = row[4]
                                row_new[5] = row[5]
                                result1.writerow(row_new)
                    prots_train = [prot for prot in headers_pro if prot not in prots_test]
                    for prot in prots_train:
                        for row in csv_readera:
                            if row[3] == prot:
                                row_new = [*range(len(or_headers))]
                                # row_new.append(smile)
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                row_new[3] = row[3]
                                row_new[4] = row[4]
                                row_new[5] = row[5]
                                result.writerow(row_new)
                    print('done!')

def newcompound_newprotein():
    index_compound = 0
    index_protein = 0
    for i in range(5):
        input_file = r'dataset_ready_for_3d.csv'
        train_file = r'toy_dataset/newpair_3D/%davis_train_newnew.csv'%i
        test_file = r'toy_dataset/newpair_3D/%davis_test_newnew.csv'%i
        
        with open(input_file) as csv_file_a:
            with open(train_file, mode='w',newline='') as result:
                with open(test_file, mode='w',newline='') as result1:
                    a = csv.reader(csv_file_a, delimiter=',')
                    or_headers = next(a)
                    csv_readera = list(a)
                    result = csv.writer(result, delimiter=',')
                    result1 = csv.writer(result1, delimiter=',')
                    headers = []
                    smile = []
                    for row1 in csv_readera:
                        headers.append(row1[1])
                        smile.append(row1[0])

                    headers_pro = sorted(list(set(headers)))
                    smiles = sorted(list(set(smile)))
                    # print(len(smiles))
                    # print(len(headers_pro))
                    # input()
                    thresh_hold = 0.2

                    number_proteins_test = round(thresh_hold*len(headers_pro))
                    # print(number_proteins_test)
                    protein_max =  index_protein + int(number_proteins_test)
                    prots_test = headers_pro[index_protein : protein_max]
                    index_protein = protein_max
                    print(index_protein)
                    number_smiles_test = round(thresh_hold*len(smiles))
                    compound_max =  index_compound + int(number_smiles_test)
                    smiles_test = smiles[index_compound : compound_max]
                    index_compound = compound_max
                    print(index_compound)
                    all_headers = or_headers
                    # print(or_headers)
                    # input()

                    result.writerow([i for i in all_headers])
                    result1.writerow([i for i in all_headers])

                    for prot in prots_test:
                        for row in csv_readera:
                            if row[1] == prot and row[0] in smiles_test:
                                row_new = [*range(len(or_headers))]
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                row_new[3] = row[3]
                                row_new[4] = row[4]
                                row_new[5] = row[5]
                                result1.writerow(row_new)

                    prots_train = [prot for prot in headers_pro if prot not in prots_test]
                    smiles_train = [smile for smile in smiles if smile not in smiles_test]
                    for prot in prots_train:
                        for row in csv_readera:
                            if row[1] == prot and row[0] in smiles_train:
                                row_new = [*range(len(or_headers))]
                                # row_new.append(smile)
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                row_new[3] = row[3]
                                row_new[4] = row[4]
                                row_new[5] = row[5]
                                result.writerow(row_new)
                    print('done!')
    return None
def make_val_set():
    path = r'C:\Users\DMIS_Quang\Desktop\project\dataset\kiba_davis_deeppurpose'
    list_files = os.listdir(path)
    # print(list_files[1][7:7+5])
    for file in list_files:
        if file.endswith('train_newprot.csv'):
            input_file = os.path.join(path, file)
            output_file = os.path.join(path, file[0:5]+'_val_newprot.csv')
            output_file1 = os.path.join(path, file[0:5]+'_1_train_newprot.csv')
            print(output_file)
            index = 0
            with open(input_file) as csv_file_a:
                with open(output_file, mode='w',newline='') as result:
                    with open(output_file1, mode='w',newline='') as result1:
                        a = csv.reader(csv_file_a, delimiter=',')
                        # print(len(list(a)))
                        or_headers = next(a)
                        csv_readera = list(a)
                        print(len(csv_readera))
                        random.shuffle(csv_readera)
                        result = csv.writer(result, delimiter=',')
                        result1 = csv.writer(result1, delimiter=',')
                        headers = []
                        smile = []
                        test_index = round(len(csv_readera)*0.2)
                        csv_readera[0:test_index]
                        print(len(csv_readera[0:test_index]))
                        csv_readera[0:test_index]
                        result.writerow([i for i in or_headers])
                        result1.writerow([i for i in or_headers])
                        result.writerows(csv_readera[0:test_index])
                        result1.writerows(csv_readera[test_index:])
                        # input()
                    
def check_data():
    data_path = r'C:\Users\DMIS_Quang\Desktop\project\dataset\idg_challenge_dtc_bdb_train\dtc_bdb_ic50_filtered_fullsequence.csv'
    with open(data_path) as csv_file_a:
        a = csv.reader(csv_file_a, delimiter=',')
        or_headers = next(a)
        csv_readera = list(a)
        headers = []
        smile = []
        affinity = []
        for row1 in csv_readera:
            headers.append(row1[1])
            smile.append(row1[0])
            affinity.append(float(row1[2]))
        headers_pro = sorted(list(set(headers)))
        smiles = sorted(list(set(smile)))
        print(len(headers_pro))
        print(len(smiles))
        print(len(csv_readera))

        x_axis = np.asarray(sorted(affinity))
        mean = statistics.mean(x_axis)
        sd = statistics.stdev(x_axis)
          
        plt.plot(x_axis, norm.pdf(x_axis, mean, sd),label = 'Data')
        plt.show()
def check_dup():
    data_path1 = r'toy_dataset/davis.csv'
    data_path2 = r'toy_dataset/pdbbind_data.csv'
    output = r'toy_dataset/cross_domain/pdb_data_test.csv'
    df_davis = pd.read_csv(data_path1)
    df_pdb = pd.read_csv(data_path2)
    # print(len(list(set(df_davis['smiles']))))
    # print(len(list(set(df_pdb['smiles']))))
    for smile_pdb in list(set(list(df_pdb['smiles']))):
        if smile_pdb in list(set(df_davis['smiles'])):
            print(smile_pdb)

    a = [smile for smile in list(set(list(df_pdb['smiles']))) if smile in list(set(df_davis['smiles']))]
    print(a)
    i=0
    sequence_list_pdb = list(set(list(df_pdb['sequence'])))
    proper_list = []

    for sequence_pdb in  list(set(df_davis['sequence'])):
        if sequence_pdb in list(set(df_pdb['sequence'])):
            print(sequence_pdb)
            i = i+1
        else:
            proper_list.append(sequence_pdb)
    # proper_list = [sequence for sequence in list(df_davis['sequence']) if sequence not in list(set(df_pdb['sequence'])) ]
    # print(proper_list)


    with open(output, mode='w',newline='') as result:
        with open(data_path1) as read_csv:
            result = csv.writer(result, delimiter=',')
            a = csv.reader(read_csv, delimiter=',')
            or_headers = next(a)
            csv_readera = list(a) 
            for row in csv_readera:
                if row[1] in proper_list:
                    row_new = [*range(len(or_headers))]
                    # row_new.append(smile)
                    row_new[0] = row[0]
                    row_new[1] = row[1]
                    row_new[2] = row[2]
                    row_new[3] = row[3]
                    result.writerow(row_new)


    print(i)

def newcompound_newproteingpcr():
    index_compound = 0
    index_protein = 0
    for i in range(5):
        input_file = r'C:\Users\DMIS_Quang\Desktop\project\dataset\GPCR_binary_classification\GPCR_Data.csv'
        train_file = r'C:\Users\DMIS_Quang\Desktop\project\dataset\GPCR_binary_classification\%sgpcr_train_newnew.csv'%i
        test_file = r'C:\Users\DMIS_Quang\Desktop\project\dataset\GPCR_binary_classification\%sgpcr_test_newnew.csv'%i

        with open(input_file) as csv_file_a:
            with open(train_file, mode='w',newline='') as result:
                with open(test_file, mode='w',newline='') as result1:
                    a = csv.reader(csv_file_a, delimiter=',')
                    or_headers = next(a)
                    csv_readera = list(a)
                    result = csv.writer(result, delimiter=',')
                    result1 = csv.writer(result1, delimiter=',')
                    headers = []
                    smile = []
                    for row1 in csv_readera:
                        headers.append(row1[1])
                        smile.append(row1[0])

                    headers_pro = sorted(list(set(headers)))
                    smiles = sorted(list(set(smile)))
                    print(len(smiles))
                    print(len(headers_pro))
                    # input()
                    thresh_hold = 0.2

                    number_proteins_test = round(thresh_hold*len(headers_pro))
                    print(number_proteins_test)
                    protein_max =  index_protein + int(number_proteins_test)
                    prots_test = headers_pro[index_protein : protein_max]
                    index_protein = number_proteins_test

                    number_smiles_test = round(thresh_hold*len(smiles))
                    compound_max =  index_compound + int(number_smiles_test)
                    smiles_test = smiles[index_compound : compound_max]
                    index_compound = number_smiles_test

                    all_headers = or_headers
                    # print(or_headers)
                    # input()

                    result.writerow([i for i in all_headers])
                    result1.writerow([i for i in all_headers])

                    for prot in prots_test:
                        for row in csv_readera:
                            if row[1] == prot and row[0] in smiles_test:
                                row_new = [*range(len(or_headers))]
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                # row_new[3] = row[3]
                                result1.writerow(row_new)

                    prots_train = [prot for prot in headers_pro if prot not in prots_test]
                    smiles_train = [smile for smile in smiles if smile not in smiles_test]
                    for prot in prots_train:
                        for row in csv_readera:
                            if row[1] == prot and row[0] in smiles_train:
                                row_new = [*range(len(or_headers))]
                                # row_new.append(smile)
                                row_new[0] = row[0]
                                row_new[1] = row[1]
                                row_new[2] = row[2]
                                # row_new[3] = row[3]

                                result.writerow(row_new)
                    print('done!')

def check_dup_fixed():
    data_path1 = r'toy_dataset/davis.csv'
    data_path2 = r'toy_dataset/pdbbind_data.csv'
    output = r'toy_dataset/cross_domain/pdb_data_test.csv'
    
    # 1. Đọc dữ liệu
    df_davis = pd.read_csv(data_path1)
    df_pdb = pd.read_csv(data_path2)
    
    # Lưu ý: Tìm đúng tên cột chứa protein sequence trong các file
    # (Tuỳ thuộc vào script tạo file trước đó, cột này có thể tên là 'sequence', 'sequences', hoặc 'Target_seq')
    seq_col_pdb = 'sequences' if 'sequences' in df_pdb.columns else 'sequence'
    seq_col_davis = 'Target_seq' if 'Target_seq' in df_davis.columns else 'sequence'
    
    # 2. Lấy danh sách protein của Davis (tập train)
    davis_seqs = set(df_davis[seq_col_davis].dropna())
    
    # 3. Tạo tập Test từ PDBBind bằng cách giữ lại các protein KHÔNG NẰM TRONG Davis
    df_pdb_test = df_pdb[~df_pdb[seq_col_pdb].isin(davis_seqs)]
    
    # Mở rộng (Tùy chọn): Nếu bạn muốn lọc luôn cả các hợp chất (SMILES) bị trùng:
    smiles_col_davis = 'smiles' if 'smiles' in df_davis.columns else df_davis.columns[1] # Thường cột smiles ở vị trí index 1
    davis_smiles = set(df_davis[smiles_col_davis].dropna())
    df_pdb_test = df_pdb_test[~df_pdb_test['smiles'].isin(davis_smiles)]
    
    # 4. Lưu ra file kết quả
    df_pdb_test.to_csv(output, index=False)
    print(f"Hoàn tất! Tập test mới có {len(df_pdb_test)} dòng (Lọc từ {len(df_pdb)} dòng gốc).")

#check_dup_fixed()
# def make_davis_cross_domain_split():
#     # Điều chỉnh lại đường dẫn này cho khớp với thư mục chứa dữ liệu trên máy của bạn
#     # Ví dụ định dạng đường dẫn: '/Users/username/Desktop/project/dataset/davis.csv'
#     data_path = r'toy_dataset/davis.csv'
#     train_output = r'toy_dataset/cross_domain/davis_train.csv'
#     val_output = r'toy_dataset/cross_domain/davis_val.csv'

#     print(f"Đang đọc dữ liệu từ {data_path}...")
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print("Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn!")
#         return

#     # Xáo trộn ngẫu nhiên dữ liệu
#     df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

#     # Cắt 20% cho tập validation
#     val_size = int(len(df_shuffled) * 0.2)
#     df_val = df_shuffled[:val_size]
#     df_train = df_shuffled[val_size:]

#     # Lưu ra file CSV
#     df_train.to_csv(train_output, index=False)
#     df_val.to_csv(val_output, index=False)
    
#     print(f"Hoàn tất chia tập Davis!")
#     print(f"- Tập Train: {len(df_train)} dòng -> {train_output}")
#     print(f"- Tập Validation: {len(df_val)} dòng -> {val_output}")
import pandas as pd

def make_davis_cross_domain_split():
    # Điều chỉnh lại đường dẫn
    data_path = r'toy_dataset/davis.csv'
    train_output = r'toy_dataset/cross_domain/davis_train.csv'
    val_output = r'toy_dataset/cross_domain/davis_val.csv'

    print(f"Đang đọc dữ liệu từ {data_path}...")
    df = pd.read_csv(data_path)

    # 1. KHAI BÁO TÊN CỘT GỐC TRONG FILE DAVIS CỦA BẠN
    # (Hãy thay đổi các chuỗi này nếu file gốc của bạn có tên cột khác, 
    # ví dụ: 'SMILES', 'Target_seq', 'y')
    col_smiles = 'smiles'     
    col_sequence = 'sequence' 
    col_label = 'label'       

    # 2. Đổi tên cột về chuẩn (nếu tên gốc đã chuẩn thì bước này không ảnh hưởng)
    df = df.rename(columns={
        col_smiles: 'smiles',
        col_sequence: 'sequence',
        col_label: 'label'
    })

    # 3. CHỈ GIỮ LẠI ĐÚNG 3 CỘT CẦN THIẾT
    df = df[['smiles', 'sequence', 'label']]

    # 4. Xáo trộn ngẫu nhiên dữ liệu
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Cắt 20% cho tập validation
    val_size = int(len(df_shuffled) * 0.2)
    df_val = df_shuffled[:val_size]
    df_train = df_shuffled[val_size:]

    # 6. Lưu ra file CSV
    df_train.to_csv(train_output, index=False)
    df_val.to_csv(val_output, index=False)
    
    print(f"Hoàn tất chia tập Davis!")
    print(f"- Tập Train: {len(df_train)} dòng (chỉ gồm 3 cột) -> {train_output}")
    print(f"- Tập Validation: {len(df_val)} dòng (chỉ gồm 3 cột) -> {val_output}")

if __name__ == '__main__':
    
    #new_protein()
    #new_compound()
    #newcompound_newprotein()
    # make_val_set()
    # check_data()
    #check_dup()
    # newcompound_newproteingpcr()
    # make_val_set()
    #check_dup_fixed()
    make_davis_cross_domain_split()