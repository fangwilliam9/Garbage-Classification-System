

### Use [Mobilenet V2](https://www.geeksforgeeks.org/what-is-mobilenet-v2/)

采用预训练模型Mobilenet V2

> 模型设计：如采用自定义模型或预训练模型（VGG、ResNet、Transformer等）；

A powerful and efficient CNN(Convolutional Neural Network) architecture, designed for mobile and embedded嵌入式 vision视觉 applications.

### What

MobileNetV2 is a convolutional neural network architecture optimized for mobile and embedded vision applications. 

It improves upon the original MobileNet by introducing inverted residual blocks and linear bottlenecks, resulting in higher accuracy and speed while maintaining low computational costs. 

MobileNetV2 is widely used for tasks like <u>image classification</u>, <u>object detection</u>, and <u>semantic语义的 segmentation</u> on mobile and edge devices.

### Why

Key Features of MobileNet V2

1. **Inverted Residuals**: One of the most notable features of MobileNet V2 is the use of inverted residual blocks. Unlike traditional residual blocks that connect layers of the same depth, inverted residuals connect layers with different depths, allowing for more efficient information flow and reducing computational complexity.
2. **Linear Bottlenecks**: MobileNet V2 introduces linear bottlenecks between the layers. These bottlenecks help preserve the information by maintaining low-dimensional representations, which minimizes information loss and improves the overall accuracy of the model.
3. **Depthwise Separable Convolutions**: Similar to MobileNet V1, MobileNet V2 employs depthwise separable convolutions to reduce the number of parameters and computations. This technique splits the convolution into two separate operations: depthwise convolution and pointwise convolution, significantly reducing computational cost.
4. **ReLU6 Activation Function**: MobileNet V2 uses the ReLU6 activation function, which clips the ReLU output at 6. This helps prevent numerical instability in low-precision computations, making the model more suitable for mobile and embedded devices.

### Advantages of MobileNet V2

1. **Efficiency**: MobileNet V2 ==significantly reduces== the number of parameters and computational cost ==through== the use of <u>depthwise separable convolutions</u> and <u>inverted residuals</u>, making it highly suitable for mobile and embedded applications. 通过使用深度可分离卷积和反转残差显著减少了参数数量和计算成本
2. **Performance**: Despite its efficiency, MobileNet V2 achieves high accuracy on various benchmarks, including ImageNet classification, COCO object detection, and VOC image segmentation.
3. **Flexibility**: The architecture supports various <u>width multipliers</u> and <u>input resolutions</u>, allowing for a trade-off 权衡 between model size, computational cost, and accuracy to meet different application requirements.
4. **Scalability可拓展性**: MobileNet V2 can be easily scaled to different performance points by adjusting the <u>width multiplier</u> and <u>input image size</u>, making it versatile for a wide range of use cases.
5. **Compatibility高兼容性**: The architecture is compatible with common deep learning frameworks and can be implemented efficiently using standard operations, facilitating integration into existing workflows and deployment on various hardware platforms.



### Architecture

Architecture of MobileNet V2

The MobileNet V2 architecture is designed to provide high performance while maintaining efficiency for ==mobile and embedded applications==. Below, we break down the architecture in detail, using the schematic示意图 of the MobileNet V2 structure as a reference.

#### 1. Initial Layers初始层

- **Input Layer**: The model takes an RGB image of fixed size (224x224 pixels) as input.
- **First Convolutional Layer**: This layer applies a standard convolution with a stride of 2 to downsample下采样 the input image. This operation increases the <u>number of channels通道数</u> to 32.
   步长为2的标准卷积，对图像进行下采样，此操作将通道数增加到32

#### 2. Inverted Residual Blocks倒立残差块

The core component of MobileNet V2 is the inverted residual block, which consists of three main layers:

- **Expansion Layer**: A 1x1 convolution that increases the number of channels (also known as the expansion factor). This layer is followed by the ReLU6 activation function, which introduces non-linearity.
   扩展层：增加通道数量的1x1卷积（也称为扩展因子）。这一层之后是ReLU6激活函数，引入了非线性。
- **Depthwise Convolution**: A depthwise convolution layer that performs spatial convolution independently over each channel. This layer is also followed by ReLU6.
   深度卷积：在每个通道上独立执行空间卷积的深度卷积层。这一层后面还有ReLU6。
- **Projection Layer**: A 1x1 convolution that projects the expanded channels back to a lower dimension. This layer does not use an activation function, hence it is linear.
   投影层：一个1x1卷积，将扩展的通道投影回较低的维度。这一层不使用激活函数，因此它是线性的。

Each inverted residual block has a shortcut connection that skips over the depthwise convolution and connects directly from the input to the output, allowing for better gradient flow during training. This connection only exists when the input and output dimensions match.
每个反向残差块都有一个快捷连接，跳过深度卷积，直接从输入连接到输出，在训练过程中允许更好的梯度流。此连接仅在输入和输出维度匹配时存在。

