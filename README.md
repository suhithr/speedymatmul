Learning how to write a fast matmul kernel while in batch at the Recurse Center.

Profiling Command:
sudo ncu -o <output_file> --launch-skip 49 -f --set full ./sgemm <kernel> --profile