
# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 计算可训练参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params:,}")


# 使用 fvcore
from fvcore.nn import FlopCountAnalysis

# input_image 是你的模拟输入，例如一张 1024x1024 的图片
flops = FlopCountAnalysis(model, input_image)
total_gflops = flops.total() / 1e9
print(f"总 GFLOPs: {total_gflops:.2f}")