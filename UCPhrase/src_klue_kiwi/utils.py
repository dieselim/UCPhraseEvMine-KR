import gc
import json
import string
import orjson
import torch
import pickle
import shutil
import time
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from termcolor import colored
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PUNCS = set(string.punctuation) - {'-'}

@lru_cache(maxsize=100000)

def get_device(gpu):
    if gpu == 'True':
        if torch.cuda.is_available():
            print("CUDA is available")
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def mean(nums):
    return sum(nums) / len(nums)


def get_batches(input_list, batch_size):
    return [input_list[i: i + batch_size] for i in range(0, len(input_list), batch_size)]


def get_possible_spans(word_idxs, num_wordpieces, max_word_gram, max_subword_gram):
    possible_spans = []
    num_words = len(word_idxs)
    max_gram = min(max_word_gram, num_words)
    for len_span in range(max_gram, 1, -1):
        for i in range(num_words - len_span + 1):
            l_idx = word_idxs[i]
            r_idx = word_idxs[i + len_span] - 1 if i + len_span < num_words else num_wordpieces - 1
            if r_idx - l_idx + 1 <= max_subword_gram:
                possible_spans.append((l_idx, r_idx))
    return possible_spans


class Log:
    @staticmethod
    def info(message):
        # 콘솔에 출력할 때 인코딩 문제를 피하기 위해 올바른 인코딩 방식을 사용합니다.
        print(colored(message.encode('utf-8').decode('utf-8'), 'green'))


class String:
    @staticmethod
    def removeprefix(s: str, prefix: str) -> str:
        return s[len(prefix):] if s.startswith(prefix) else s[:]

    def removesuffix(s: str, suffix: str) -> str:
        return s[:-len(suffix)] if suffix and s.endswith(suffix) else s[:]


class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError


# class Json(IO):
#     @staticmethod
#     def load(path):
#         with open(path) as rf:
#             data = json.load(rf)
#         return data

#     @staticmethod
#     def loads(jsonline):
#         return json.loads(jsonline)

#     @staticmethod
#     def dump(data, path):
#         with open(path, 'w') as wf:
#             json.dump(data, wf, indent=4, ensure_ascii=False)

#     @staticmethod
#     def dumps(data):
#         return json.dumps(data, ensure_ascii=False)


class OrJson(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            data = orjson.loads(rf.read())
        return data

    @staticmethod
    def loads(jsonline):
        return orjson.loads(jsonline)

    @staticmethod
    def dump(data, path):
        with open(path, 'w') as wf:
            wf.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS).decode('utf-8'))

    @staticmethod
    def dumps(data):
        return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode()


Json = OrJson


class JsonLine(IO):
    @staticmethod
    def load(path, use_tqdm=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if use_tqdm:
            lines = tqdm(lines, ncols=100, desc='Load JsonLine')
        return [json.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [json.dumps(d, ensure_ascii=False) for d in instances]
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines))


class OrJsonLine(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            lines = rf.read().splitlines()
        return [orjson.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [orjson.dumps(d, option=orjson.OPT_NON_STR_KEYS).decode() for d in instances]
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines))


class TextFile(IO):
    @staticmethod
    def load(path):
        with open(path, encoding='utf-8') as rf:
            text = rf.read()
        return text

    @staticmethod
    def readlines(path, skip_empty_line=False):
        with open(path, encoding='utf-8') as rf:
            lines = rf.read().splitlines()
        if skip_empty_line:
            return [l for l in lines if l]
        return lines

    @staticmethod
    def dump(text, path):
        with open(path, 'w', encoding='utf-8') as wf:
            wf.write(text)

    @staticmethod
    def dumplist(target_list, path):
        with open(path, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join([str(o) for o in target_list]) + '\n')


class Pickle:
    @staticmethod
    def load(path, max_retries=5, retry_delay=5):
        attempt = 0
        while attempt < max_retries:
            try:
                with open(path, 'rb') as rf:
                    gc.disable()
                    data = pickle.load(rf)
                    gc.enable()
                return data
            except EOFError as e:
                print(f"EOFError: {e}")
                return None
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    print(f"An error occurred while loading the file: {e}. Retrying {attempt}/{max_retries} after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to load the file after {max_retries} attempts: {e}")
                    return None

    @staticmethod
    def dump(data, path):
        try:
            with open(path, 'wb') as wf:
                gc.disable()
                pickle.dump(data, wf, protocol=4)
                gc.enable()
        except Exception as e:
            print(f"An error occurred while dumping the data: {e}")
    
    # @staticmethod
    # def batch_dump(instances, dirpath, num_files=10):
    #     assert type(instances) == list
    #     dirpath = Path(dirpath)
    #     if dirpath.exists():
    #         shutil.rmtree(dirpath)
    #     dirpath.mkdir(exist_ok=True)
    #     num_instances = len(instances)
    #     batch_size = num_instances // num_files
    #     threads = []
    #     print('start batch dumping...', end='')
    #     time1 = time.perf_counter()
    #     for i in range(0, num_instances, batch_size):
    #         filepath = dirpath / str(len(threads))
    #         thread = multiprocessing.Process(target=Pickle.dump, args=(instances[i: i + batch_size], filepath))
    #         threads.append(thread)
    #     for t in threads:
    #         t.start()
    #     for t in threads:
    #         t.join()
    #     time2 = time.perf_counter()
    #     print(f'OK in {time2-time1:.1f} secs')
    
    @staticmethod
    def batch_dump(instances, dirpath, max_file_size=2 * 1024 * 1024 * 1024):
        assert isinstance(instances, list)
        dirpath = Path(dirpath)
        if dirpath.exists():
            shutil.rmtree(dirpath)
        dirpath.mkdir(exist_ok=True)

        threads = []
        semaphore = threading.BoundedSemaphore(value=1)  # Limit the number of concurrent threads to 3
        print('Start batch dumping...', end='')
        time1 = time.perf_counter()

        current_batch = []
        current_batch_size = 0
        file_index = 0

        def save_batch(batch, filepath):
            with semaphore:
                try:
                    Pickle.dump(batch, filepath)
                finally:
                    del batch
                    gc.collect()  # Explicitly call garbage collector to free up memory

        for instance in instances:
            serialized_instance = pickle.dumps(instance)
            instance_size = len(serialized_instance)
            
            if current_batch_size + instance_size > max_file_size and current_batch:
                # Dump current batch to file
                filepath = dirpath / str(file_index)
                thread = threading.Thread(target=save_batch, args=(current_batch, filepath))
                threads.append(thread)
                thread.start()

                current_batch = []
                current_batch_size = 0
                file_index += 1

                # Ensure the memory is freed up
                del serialized_instance
                gc.collect()

            current_batch.append(instance)
            current_batch_size += instance_size

        # Dump any remaining instances in the last batch
        if current_batch:
            filepath = dirpath / str(file_index)
            thread = threading.Thread(target=save_batch, args=(current_batch, filepath))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        time2 = time.perf_counter()
        print(f'OK in {time2 - time1:.1f} secs')
            
    @staticmethod
    def load_all_data(dirpath):
        dirpath = Path(dirpath)
        all_data = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(Pickle.load, filepath) for filepath in dirpath.iterdir() if filepath.is_file()]
            for future in as_completed(futures):
                data = future.result()
                if data is not None:
                    all_data.extend(data)
        return all_data


class Process:
    @staticmethod
    def par(func, iterables, num_processes, desc=''):
        pool = multiprocessing.Pool(processes=num_processes)
        pool_func = pool.imap(func=func, iterable=iterables)
        pool_func = tqdm(pool_func, total=len(iterables), ncols=100, desc=desc)
        # results = list(pool_func)
        results = [r for r in pool_func]
        pool.close()
        pool.join()
        return results


if __name__ == '__main__':
    print(OrJson.dumps({1: 2, 3: 'sheaf'}))
