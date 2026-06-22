import json

def main():
    # Read the 1000 new topics
    with open('data/generated_topics_manual.json', 'r', encoding='utf-8') as f:
        topics_manual = json.load(f)
        manual_topics_set = set(t.get('topik', '') for t in topics_manual)

    # Read the 2500 dataset topics
    dataset_topics_set = set()
    with open('data/t5-gemma-2-chat-instruct-dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if 'topik' in obj:
                    dataset_topics_set.add(obj['topik'])

    # Check for overlap
    overlap = manual_topics_set.intersection(dataset_topics_set)
    print(f"Jumlah topik di file 1000 (baru): {len(manual_topics_set)}")
    print(f"Jumlah topik di file 2500 (lama): {len(dataset_topics_set)}")
    print(f"Jumlah TOPIK YANG OVERLAP (BENTROK): {len(overlap)}")

    if overlap:
        print("\nContoh 5 topik yang bertabrakan:")
        for t in list(overlap)[:5]:
            print(f"- {t}")
    else:
        print("\nAMAN! Tidak ada satu pun topik yang tumpang tindih.")

if __name__ == "__main__":
    main()
