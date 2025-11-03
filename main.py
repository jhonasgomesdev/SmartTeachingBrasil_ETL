from etl.extract import extract_data
from etl.transform import flatten
from etl.load import load_dfs

def main():
    print("🚀 Iniciando pipeline ETL...")
    raw = extract_data()
    print(f"🔗 Extraídos {len(raw)} alunos")
    dfs = flatten(raw)
    for k,v in dfs.items():
        print(f"  - {k}: {len(v)}")
    load_dfs(dfs)
    print("🏁 ETL concluído")

if __name__ == "__main__":
    main()