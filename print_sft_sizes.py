from models.encoders.sft.size_dynamics import PipelineDynamics


in_size = [224]
patch_size = [4]

ctx_sizes = [[8]]
qry_sizes = [[4]]

# ctx_sizes = [[7], [8], [8], [8]]
# qry_sizes = [[4], [4], [4], [4]]


pdy = PipelineDynamics(in_size, patch_size, ctx_sizes, qry_sizes)
n = len(ctx_sizes)
if n == 1:
    n = pdy.get_bottleneck_position()
print(pdy.get_sizes_str(0, n))
if pdy.recurrent:
    print("\n" + '~' * 10 + "\n")
    print(f"bottleneck position: {n}")