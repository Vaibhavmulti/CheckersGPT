# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset  # huggingface datasets
import pickle
import torch

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

HUMAN_DATA_PATH = "data/checkers_games/OCA_2.0.pdn"
DATA_PATH = "data/checkers_games"


if __name__ == "__main__":
    # dataset = load_dataset("csv", data_files={"train": "pgn.csv"}) # For local testing

    #dataset_path = "adamkarvonen/chess_games"
    #file_path = "lichess_6gb_blocks.zip"
    # file_path = "smaller_pgn_file_blocks.zip"

    #CheckersGPT/data/checkers_games/OCA_2.0.pdn
    with open(HUMAN_DATA_PATH, "r") as file:
        pdn_content = file.read()
        moves = pdn_content.split('\n\n')

    # Filter out the games from the file
    with open(os.path.join(DATA_PATH, "filter.txt"), "w") as file:
        for move in moves:
            move = re.sub(r'\[.*?\]\n?', '', move)  # Upper Game stats
            move = re.sub(r'\n\s*', ' ', move) # Moves have newlines for beautifying
            move = re.sub(r'\{.*?\}', '', move)  # In moves there are some human interpretable comments
            file.write(move+'\n')
    
    
    # Define the paths for your input and output files
    input_file_path = os.path.join(DATA_PATH, "filter.txt")
    output_file_path = os.path.join(DATA_PATH, "desired_output.txt")

    # Open the input file in read mode and the output file in write mode
    #firstline = True
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        # Read each line from the input file
        # if firstline:
        #     firstline = False
        #     outfile.write("transcript\n")
        for line in infile:
            # Write the modified line to the output file
            outfile.write(';' + line)


    # Define the path to your text file and the output CSV file
    input_file_path = os.path.join(DATA_PATH, "desired_output.txt")
    output_csv_path = os.path.join(DATA_PATH, "output.csv")

    # Initialize a list to hold all transcripts
    transcripts = []

    # Read each line from the text file
    with open(input_file_path, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace and check if line is not empty
            cleaned_line = line.strip()
            if cleaned_line:
                # Add line to transcripts list
                transcripts.append(cleaned_line)

    # Create a DataFrame
    df = pd.DataFrame(transcripts, columns=['transcript'])


    def trim_to_400(text):
        if len(text) > 400:
            return text[:400]
        else:
            return text
    
    def pad_to_400(text):
        if len(text) < 400:
            padding_length = 400 - len(text)
            return text + ' ' * padding_length
        else:
            return text

    
    # Calculate lengths of each string in the 'transcripts' column
    # df['length'] = df['transcript'].apply(len)

    # # Calculate minimum, maximum, and average length
    # min_length = df['length'].min()
    # max_length = df['length'].max()
    # avg_length = df['length'].mean()
    # median_length = df['length'].median()

    # print("Minimum length:", min_length)
    # print("Maximum length:", max_length)
    # print("Average length:", avg_length)
    # print("Median length:", median_length)

    df['transcript'] = df['transcript'].apply(trim_to_400)
    df = df[df['transcript'].apply(len) >= 100]
    # Apply the function to pad the strings in the 'transcripts' column
    df['transcript'] = df['transcript'].apply(pad_to_400)
    
    # df2 = pd.read_csv('/home/vaibhav/nanogpt/CheckersGPT/data/checkers_games/top_rows.csv')
    # df2['transcript'] = df2['transcript'].apply(trim_to_400)
    # df2 = df2[df2['transcript'].apply(len) >= 100]
    # # Apply the function to pad the strings in the 'transcripts' column
    # df2['transcript'] = df2['transcript'].apply(pad_to_400)
    
    # df['length'] = df['transcript'].apply(len)

    # # Calculate minimum, maximum, and average length
    # min_length = df['length'].min()
    # max_length = df['length'].max()
    # avg_length = df['length'].mean()
    # median_length = df['length'].median()

    # print("Minimum length:", min_length)
    # print("Maximum length:", max_length)
    # print("Average length:", avg_length)
    # print("Median length:", median_length)

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    # df1 = pd.read_csv(output_csv_path)
    
    
    # # Concatenate the two DataFrames
    # merged_df = pd.concat([df1, df2])

    # # Save the merged DataFrame to a new CSV file
    # merged_df.to_csv(output_csv_path, index=False)


    with open(os.path.join(DATA_PATH, "desired_output.txt"), 'r') as f:
        data = f.read()
        dataset = data.split('\n')
    print(f"length of dataset in characters: {len(data):,}")
    
    # df_final = pd.read_csv(output_csv_path)
    # # If you want to process the entire CSV as a single string
    # data = df_final.to_csv(index=False, header=False)
    # #data = df_final.to_string(index=False)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

        
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    





    # Load the dataset
    #dataset = load_dataset(dataset_path, data_files=file_path)
    
    #mod
    # dataset_dict = {'transcript': dataset}
    # dataset = Dataset.from_dict(dataset_dict)
    dataset = load_dataset("csv", data_files={"train": os.path.join(DATA_PATH, "output.csv")})
    #dataset = load_dataset("csv", data_files="output.csv")
    # by default only contains the 'train' split, so create a test split
    #split_dataset = dataset.train_test_split(test_size=0.01, seed=2357, shuffle=True)
    
    split_dataset = dataset["train"].train_test_split(
        test_size=0.1, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val
    
    print(split_dataset["val"])

    #print(split_dataset['train']['transcript'][0])
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })
    # train_max = -1
    # train_min = 100000
    # train_avg = 0

    # for _ in split_dataset['train']['transcript']:
    #     length = len(_)
    #     if length>train_max:
    #         train_max = length
    #     elif length<train_min:
    #         train_min = length
    #     train_avg+=length

    # for _ in split_dataset['val']['transcript']:
    #     length = len(_)
    #     if length>train_max:
    #         train_max = length
    #     elif length<train_min:
    #         train_min = length
    #     train_avg+=length
    # train_avg/=len(split_dataset['train']['transcript'])

    
    # print(f"max _length {train_max}")
    # print(f"min _length {train_min}")
    # print(f"avg _length {train_avg}")

    
    our_pickle = {'vocab_size':17, 'itos':itos, 'stoi':stoi}
    # Pickle the dictionary
    with open(os.path.join(DATA_PATH, "meta.pkl"), 'wb') as f:
        pickle.dump(our_pickle, f)

    #we now want to tokenize the dataset. Using meta.pkl in the same directory as this file
    # meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    # with open(meta_path, "rb") as f:
    #     meta = pickle.load(f)

    # stoi = meta["stoi"]
    # itos = meta["itos"]

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint8, mode='r')
    # print(split_dataset["val"][0])
    # print(len(split_dataset["val"]["transcript"][0]))

    #For verifying that all games are 1024 tokens long
    # count = 0
    # for game in split_dataset["train"]["transcript"]:
    #     if len(game) != 1024:
    #         count+=1
    #         #print("Oh no")
    #         #print(len(game))
    #         #print(game)
    # print(count)
    # print(stoi)

    column_name = "transcript"

    def process(example):
        ids = np.array([stoi[c] for c in example[column_name]], dtype=dtype)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # print(tokenized["val"]["ids"])

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        #filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        filename = os.path.join(DATA_PATH, f"{split}.bin")
        
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        print(arr.shape)
        if split == "train":
            total_batches = 1024
        else:
            total_batches = 227

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            # print(batch[0])
            arr_batch = np.concatenate(batch["ids"])
            # print(arr_batch)
            # print(arr_batch.shape)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

