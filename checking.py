#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于【语义向量（Sentence-BERT）】的中文论文摘要查重工具

目录结构：
  history_papers/   # 历史论文 PDF
  current_papers/   # 待检测论文 PDF

输出：
  plagiarism_report.csv
  history_embeddings.npy
  history_files.json

依赖：
  pip install PyPDF2 sentence-transformers scikit-learn pandas jieba
"""

import os
import sys
import re
import json
import numpy as np

# =====================
# 依赖检测
# =====================

def check_dirs_and_dependencies():
    REQUIRED_DIRS = ["history_papers", "current_papers"]
    for d in REQUIRED_DIRS:
        if not os.path.isdir(d):
            print(f"[错误] 未找到目录 `{d}`，请创建后再运行")
            sys.exit(1)

    try:
        import PyPDF2
        import sentence_transformers
        import sklearn
        import pandas
        import jieba
    except ImportError:
        print("依赖未安装，请执行：\n"
              "pip install PyPDF2 sentence-transformers scikit-learn pandas jieba")
        sys.exit(1)


# =====================
# PDF 摘要提取
# =====================

def extract_abstract_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    try:
        reader = PdfReader(pdf_path)
        text = ''.join(page.extract_text() or '' for page in reader.pages)

        # 基础清洗
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\s+", " ", text)

        # 中英文摘要匹配
        pattern = re.compile(
            r"(摘要[:：\s].+?)(关键词|Abstract|ABSTRACT|Introduction|引言|一、)",
            re.S)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

        # fallback：前 600 字
        return text[:600].strip()
    except Exception as e:
        print(f"[PDF解析失败] {pdf_path}: {e}")
        return ""


# =====================
# 文本清洗（语义模型用）
# =====================

def preprocess_text(text):
    # 去掉参考文献之后内容
    text = re.sub(r"参考文献.*", "", text, flags=re.S)
    # 去除多余符号
    text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9，。；：,. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================
# 构建历史摘要库
# =====================

def load_history_abstracts(history_dir):
    files = [f for f in os.listdir(history_dir) if f.lower().endswith('.pdf')]
    texts = []
    valid_files = []

    for f in files:
        abs_text = extract_abstract_from_pdf(os.path.join(history_dir, f))
        if abs_text:
            texts.append(preprocess_text(abs_text))
            valid_files.append(f)

    return valid_files, texts


# =====================
# 主查重逻辑（语义向量）
# =====================

def semantic_plagiarism_check(
    history_dir="history_papers",
    current_dir="current_papers",
    threshold=0.85,
    top_k=3,
    output_csv="plagiarism_report.csv"
):
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMB_FILE = "history_embeddings.npy"
    META_FILE = "history_files.json"

    print("[INFO] 加载语义模型……")
    model = SentenceTransformer("shibing624/text2vec-base-chinese")

    # ---------- 历史库 ----------
    if os.path.exists(EMB_FILE) and os.path.exists(META_FILE):
        print("[INFO] 使用缓存的历史向量")
        history_embeddings = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            history_files = json.load(f)
    else:
        print("[INFO] 构建历史摘要向量（首次运行较慢）")
        history_files, history_texts = load_history_abstracts(history_dir)
        history_embeddings = model.encode(
            history_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        np.save(EMB_FILE, history_embeddings)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(history_files, f, ensure_ascii=False, indent=2)

    # ---------- 当前论文 ----------
    results = []
    current_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]

    for curr in current_files:
        abs_text = extract_abstract_from_pdf(os.path.join(current_dir, curr))
        if not abs_text:
            continue

        curr_text = preprocess_text(abs_text)
        curr_emb = model.encode(curr_text, normalize_embeddings=True)

        sims = cosine_similarity([curr_emb], history_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        for rank, idx in enumerate(top_idx, start=1):
            if sims[idx] >= threshold:
                results.append({
                    "current_paper": curr,
                    "history_paper": history_files[idx],
                    "similarity": round(float(sims[idx]) * 100, 2),
                    "rank": rank
                })

    # ---------- 输出 ----------
    if results:
        df = pd.DataFrame(results)
        df.sort_values(by=["current_paper", "similarity"], ascending=False, inplace=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[完成] 发现疑似高相似摘要 {len(results)} 条，已保存到 {output_csv}")
    else:
        print("[完成] 未发现超过阈值的相似摘要")


# =====================
# main
# =====================

if __name__ == "__main__":
    check_dirs_and_dependencies()
    semantic_plagiarism_check()
