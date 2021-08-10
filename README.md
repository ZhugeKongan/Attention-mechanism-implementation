# Attention-mechanism-implementation
**pytorch for Self-attention、Non-local、SE、SK、CBAM、DANet**

   According to the different application domains of the attention mechanism, that is, the different ways and positions of attention weights are applied, the article divides the attention mechanism into spatial domain, channel domain and hybrid domain, and introduces some advanced aspects of these different attentions. Attention model, carefully analyzed their design methods and application fields, and finally proved the effectiveness of these attention mechanisms and the improvement of the results brought by CV tasks with experimental methods.
   
1. Spatial attention method

1.1 Self-Attention

![image](https://user-images.githubusercontent.com/49756674/125032931-3e17f700-e0c1-11eb-894d-75999d04612a.png)

1.2 Non-local Attention

![image](https://user-images.githubusercontent.com/49756674/125032957-4839f580-e0c1-11eb-90d8-0b96f6ed2d6d.png)

2. Channel domain attention method

2.1 SENet

![image](https://user-images.githubusercontent.com/49756674/125032984-52f48a80-e0c1-11eb-82c0-be84d9f17941.png)

2.2 SKNet

![image](https://user-images.githubusercontent.com/49756674/125033021-5e47b600-e0c1-11eb-8ec4-11d91ecbebee.png)


3. Hybrid domain attention method

3.1 CBAM

![image](https://user-images.githubusercontent.com/49756674/125029553-6e10cb80-e0bc-11eb-86b0-426866a92e77.png)
![image](https://user-images.githubusercontent.com/49756674/125029579-779a3380-e0bc-11eb-9940-c2235aad367f.png)

3.2 DANet

![image](https://user-images.githubusercontent.com/49756674/125029618-88e34000-e0bc-11eb-9c45-4b598eb6c1c8.png)

4. RESULT
    For each set of experiments, we use Resnet18 as the Baseline, training 160 epoch, the initial learning rate is 0.1, 80 epoch is adjusted to 0.01, and 160 epoch is adjusted to 0.001. The batch size is set to 128, and the SGD optimizer with momentum is experimented. When reading the input, first perform random cropping and random flipping data enhancement. In particular, in order to maximize the attention effect, we all perform a warm-up operation of 1 epoch at the beginning of the experiment, and take the average of the best 5 epochs as the final result.
 ![image](https://user-images.githubusercontent.com/49756674/125030308-833a2a00-e0bd-11eb-9023-213436e4746e.png)


**reference**
[Self-Attention](http://arxiv.org/abs/1706.03762)
[ Non-local Attention](http://arxiv.org/abs/1711.07971)
[SENet](http://arxiv.org/abs/1709.01507)
[SKNet](http://arxiv.org/abs/1903.06586)
[CBAM](http://arxiv.org/abs/1807.06521)
[DANet](http://ieeexplore.ieee.org/document/8953974)
