import sys
import os
import re

# 依赖检测
def check_dirs_and_dependencies():
    REQUIRED_DIRS = ["history_papers", "current_papers"]
    for d in REQUIRED_DIRS:
        if not os.path.isdir(d):
            print(f"[未找到目录] 请在当前目录下创建 `{d}` 文件夹，并放入PDF文件后再运行本程序。")
            sys.exit(1)

    try:
        import PyPDF2
        import jieba
        import pandas
        import sklearn
    except ImportError:
        print("依赖库未安装！请先运行:\n    pip install PyPDF2 jieba scikit-learn pandas")
        sys.exit(1)


# 提取PDF的摘要内容
def extract_abstract_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    try:
        reader = PdfReader(pdf_path)
        text = ''.join(page.extract_text() or '' for page in reader.pages)
        # 常见“摘要”正则
        pattern = re.compile(r"(摘要[:：\s].+?)(关键词|Abstract|ABSTRACT|Introduction|引言|一、)", re.S)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        # 不行就用前500字
        return text[:500].strip()
    except Exception as e:
        print(f"[提取失败] {pdf_path}: {e}")
        return ""

# 中文分词预处理
def preprocess_text(text):
    import jieba
    words = jieba.lcut(text)
    return ' '.join(words)

# 读取摘要库
def build_abstract_lib(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    abstracts = []
    filenames = []
    for file in files:
        abs_text = extract_abstract_from_pdf(os.path.join(directory, file))
        if abs_text:
            abstracts.append(preprocess_text(abs_text))
            filenames.append(file)
    return filenames, abstracts

# 主查重函数
def check_for_plagiarism(history_dir, current_dir, threshold=0.95, output_csv="plagiarism_report.csv"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    history_files, history_abstracts = build_abstract_lib(history_dir)
    current_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".pdf")]

    result_rows = []
    for curr_file in current_files:
        abs_curr = extract_abstract_from_pdf(os.path.join(current_dir, curr_file))
        if not abs_curr:
            continue
        abs_curr_proc = preprocess_text(abs_curr)
        corpus = history_abstracts + [abs_curr_proc]
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        sim_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
        sim_scores = sim_matrix[0]
        for idx, sim in enumerate(sim_scores):
            if sim >= threshold:
                result_rows.append({
                    "current_paper": curr_file,
                    "history_paper": history_files[idx],
                    "similarity": f"{sim*100:.2f}%"
                })

    if result_rows:
        df = pd.DataFrame(result_rows)
        df.to_csv(output_csv, index=False)
        print(f"查重完成，发现疑似抄袭，共 {len(result_rows)} 条记录，结果已保存到 {output_csv}")
    else:
        print("查重完成，未检测到高度相似（抄袭）摘要。")

if __name__ == "__main__":
    check_dirs_and_dependencies()
    check_for_plagiarism("history_papers", "current_papers")