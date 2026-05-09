import sys

fp = '/home/dvdai/miniconda3/envs/cu130/lib/python3.12/site-packages/sglang/jit_kernel/csrc/elementwise/activation.cuh'
with open(fp) as f:
    s = f.read()

old_true = '''      const auto kernel = select_kernel<true>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);'''
new_true = '''      if (type == "silu") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kSiLU, true>, params);
      } else if (type == "gelu") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kGELU, true>, params);
      } else if (type == "gelu_tanh") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kGELUTanh, true>, params);
      } else {
        Panic("unsupported activation type: ", type);
      }'''

old_false = '''      const auto kernel = select_kernel<false>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);'''
new_false = '''      if (type == "silu") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kSiLU, false>, params);
      } else if (type == "gelu") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kGELU, false>, params);
      } else if (type == "gelu_tanh") {
        LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(activation_kernel<ActivationKind::kGELUTanh, false>, params);
      } else {
        Panic("unsupported activation type: ", type);
      }'''

if old_true not in s:
    sys.exit('old_true not found in file')
if old_false not in s:
    sys.exit('old_false not found in file')

s = s.replace(old_true, new_true)
s = s.replace(old_false, new_false)

with open(fp, 'w') as f:
    f.write(s)
print('patched OK')
