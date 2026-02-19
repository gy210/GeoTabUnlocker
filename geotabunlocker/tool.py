import os
import pandas as pd


def excel_to_csv(excel_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(excel_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names
    
    print(f"发现 {len(sheet_names)} 个工作表: {sheet_names}")
    t = ''
    
    for sheet_name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0)
        df['编号'] = range(1, len(df) + 1)
        df['保存时间'] = None

        df.columns = df.columns.str.replace(' ', '', regex=False)
        other_cols = [col for col in df.columns if col not in ['编号', '保存时间']]
        df = df[['编号'] + other_cols + ['保存时间']]
        
        safe_sheet_name = "".join(c.replace(' ', '') if c.isalnum() or c in " _-" else "_" for c in str(sheet_name))
        csv_filename = f"{safe_sheet_name}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig 支持 Excel 正确显示中文
        t += safe_sheet_name + '，'
        
    print(f"✅ 已保存: {len(sheet_names)} 个 CSV")
    print(t)



def csvs_to_excel(csv_dir, output_excel_path):
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    
    csv_files = [
        f for f in os.listdir(csv_dir)
        if f.lower().endswith('.csv')
    ]
    
    if not csv_files:
        raise ValueError(f"目录 {csv_dir} 中没有找到 CSV 文件！")
    
    print(f"发现 {len(csv_files)} 个 CSV 文件: {csv_files}")

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            sheet_name = os.path.splitext(csv_file)[0]
            
            safe_sheet_name = sheet_name[:31]
            for char in ['\\', '/', '?', '*', '[', ']', ':']:
                safe_sheet_name = safe_sheet_name.replace(char, '_')
            
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
    print(f"🎉 合并完成！Excel 文件已保存至: {output_excel_path}")



if __name__ == '__main__':
    excel_to_csv("./excel_data/同位素相关数据表.xlsx", output_dir="./excel_data/csv")