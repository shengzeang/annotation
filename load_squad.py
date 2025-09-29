# 下载SQuAD v1.1官方数据集
import os
import json
import urllib.request

def download_squad(save_path="squad_train.json"):
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    if not os.path.exists(save_path):
        print(f"Downloading SQuAD v1.1 to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print(f"{save_path} already exists.")

def load_squad_to_qa_list(squad_path="squad_train.json", max_samples=200):
    with open(squad_path, encoding="utf-8") as f:
        squad = json.load(f)
    qa_list = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa["is_impossible"] if "is_impossible" in qa else False:
                    continue
                question = qa["question"]
                answer = qa["answers"][0]["text"] if qa["answers"] else ""
                qa_list.append({
                    "id": qa["id"],
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "text": f"Question: {question}\nContext: {context}"
                })
                if len(qa_list) >= max_samples:
                    return qa_list
    return qa_list

if __name__ == "__main__":
    download_squad()
    qa_data = load_squad_to_qa_list()
    print(qa_data[:2])
