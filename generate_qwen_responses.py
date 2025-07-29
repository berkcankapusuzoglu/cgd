import os
from multiprocessing import Process
import time
from tqdm import tqdm
from typing import Callable
from openai import OpenAI
from pathlib import Path
from typing import Callable, List, Dict, Any
import argparse
from datasets import load_dataset
# import ray

def read_json_file(file):
    with open(file, "r") as r:
        response = r.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        response = "[" + response + "]"
        return json.loads(response)


def process_chunk(model_name: str,
                  start_idx: int,
                  end_idx: int,
                  input_path: str,
                  output_dir: str,
                  prompt_func: Callable,
                  process_id: int):
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{start_idx + 1}-{end_idx}.json')

    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = {item['idx']: item for item in json.load(f)}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_data = data[start_idx:end_idx]
    base_url="http://localhost:8008/v1"
    
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY"
    )

    results = []

    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        if item['idx'] in existing_results:
            results.append(existing_results[item['idx']])
            continue

        try:
            messages = prompt_func(item)

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
                top_p=0.95
            )
            item['model_output'] = completion.choices[0].message.content
            results.append(item)

            if len(results) % 10000 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['idx']}: {str(e)}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(model_name: str,
                          input_path: str,
                          output_path: str,
                          temp_output_dir: str,
                          start_idx: int,
                          end_idx: int,
                          prompt_func: Callable,
                          num_processes: int = 1):
    os.makedirs(temp_output_dir, exist_ok=True)
    if input_path in ["WebInstruct-CFT-50K"]:
        dataset = load_dataset("TIGER-Lab/WebInstruct-CFT", input_path)
        data_list = dataset['train'].to_list()
        data = []
        for each in data_list:
            input_str = each["input"]
            question = input_str.split("\n\nSolution:\n")[0]
            if question.startswith("Question:\n"):
                question = question[len("Question:\n"):]
            answer = input_str.split("\n\nSolution:\n")[1]
            data.append({"question": question, "answer": answer})
    elif input_path in ["webinstructsub_chunk1"]:
        file = "/llm-experiments-no-cache/beco/LLaMA-Factory/data/webinstructsub_chunk1.json"
        data_list = read_json_file(file)[0]
        # data_list = data_list[:10]
        data_list = data_list[start_idx:end_idx]
        data = []
        for each in data_list:
            conv = each["conversations"]
            question = conv[0]['content'].split(" Answer:")[0]
            if question.startswith("Question: "):
                question = question[len("Question: "):]
            answer = conv[1]['content']
            data.append({"question": question, "answer": answer})
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[start_idx:end_idx]
    
    total_items = len(data)
    print("len(data)", len(data))
    # add idx
    for i, each in enumerate(data):
        if "idx" not in each:
            data[i]["idx"] = i
    with open(input_path, "w") as fo:
        fo.write(json.dumps(data, indent=4))

    chunk_size = total_items // num_processes + 1

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else total_items

        p = Process(
            target=process_chunk,
            args=(model_name, start_idx, end_idx, input_path, temp_output_dir, prompt_func, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed!")
    print("Merging results")
    output_data = []
    for file in os.listdir(temp_output_dir):
        if not file.endswith(".json"):
            continue
        
        with open(os.path.join(temp_output_dir, file), "r") as fi:
            curr_data = json.load(fi)
            output_data += curr_data
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


def ex_prompt_func(item):
    chat_prompt = [
        {
            "role": "user",
            "content": item["question"]
        }
    ]
    return chat_prompt


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process large dataset with multiple processes')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help='Teacher model to use (default: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--input_path', type=str, default="WebInstruct-CFT-50K",
                        help='Path to the input dataset file 50K of WebInstruct dataset (default: WebInstruct-CFT-50K)')
    parser.add_argument('--output_path', type=str, default="WebInstruct-CFT-50K.json",
                        help='Path to store output files (default: ./WebInstruct-CFT-50K.json)')
    parser.add_argument('--num_processes', type=int, default=64,
                        help='Number of processes to use (default: 64)')
    parser.add_argument('--temp_output_dir', type=str, default="temp_output_dir",
                        help='Path to the temp saved files(default: temp_output_dir)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting idx to slice from the dataset (default: 0)')
    parser.add_argument('--end_idx', type=int, default=50000,
                        help='Ending idx to slice from the dataset (default: 50000)')
    args = parser.parse_args()

    process_large_dataset(
        model_name=args.model_name,
        input_path=args.input_path,
        output_path=args.output_path,
        temp_output_dir=args.temp_output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_func=ex_prompt_func,
        num_processes=args.num_processes
    )