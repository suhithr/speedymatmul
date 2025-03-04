import re
from dataclasses import dataclass

"""
Compares 2 files that we print out and sees if there is any difference in the mapping of the
memory addresses from the threadId
"""
@dataclass
class ThreadData:
    threadId: int
    C_coordinates: tuple
    A_coordinates: tuple
    B_coordinates: tuple

def parse_line(line):
    pattern = (r"threadId: (\d+), tIdx.x, tIdx.y: \(\d+, \d+\), C \((\d+), (\d+)\), "
               r"A \[(\d+), (\d+)\.\.(\d+)\], B\[(\d+)\.\.(\d+), (\d+)\]")
    match = re.match(pattern, line)
    
    if match:
        threadId = int(match.group(1))
        C_coordinates = (int(match.group(2)), int(match.group(3)))
        A_coordinates = (int(match.group(4)), int(match.group(5)), int(match.group(6)))
        B_coordinates = (int(match.group(7)), int(match.group(8)), int(match.group(9)))
        
        return ThreadData(threadId, C_coordinates, A_coordinates, B_coordinates)
    
    return None

def read_file(filepath):
    thread_data_list = []
    with open(filepath, 'r') as file:
        for line in file:
            parsed_data = parse_line(line.strip())
            if parsed_data:
                thread_data_list.append(parsed_data)
    return thread_data_list

def compare_files(file1, file2):
    data1 = {td.threadId: td for td in read_file(file1)}
    data2 = {td.threadId: td for td in read_file(file2)}
    
    all_thread_ids = set(data1.keys()).union(data2.keys())
    
    for thread_id in sorted(all_thread_ids):
        if thread_id not in data1:
            continue
            print(f"ThreadId {thread_id} is missing in {file1}")
        elif thread_id not in data2:
            print(f"ThreadId {thread_id} is missing in {file2}")
        else:
            obj1 = data1[thread_id]
            obj2 = data2[thread_id]
            
            differences = []
            if obj1.C_coordinates != obj2.C_coordinates:
                differences.append(f"C_coordinates mismatch: {obj1.C_coordinates} vs {obj2.C_coordinates}")
            if obj1.A_coordinates != obj2.A_coordinates:
                differences.append(f"A_coordinates mismatch: {obj1.A_coordinates} vs {obj2.A_coordinates}")
            if obj1.B_coordinates != obj2.B_coordinates:
                differences.append(f"B_coordinates mismatch: {obj1.B_coordinates} vs {obj2.B_coordinates}")
            
            if differences:
                print(f"Differences in ThreadId {thread_id}:")
                for diff in differences:
                    print(f"  - {diff}")
# Example usage:
# compare_files("file1.txt", "file2.txt")

if __name__ == '__main__':
    compare_files("my_coalesc_output", "siboehm_coalesc_output")