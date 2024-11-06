# controllable text generation works published this year  (papers from icml, emnlp, acl and naacl,  but no publication on nips and colm)

- **Improving Open-Ended Text Generation via Adaptive Decoding**  
 This study introduces adaptive decoding, a mechanism that dynamically empowers language models to ascertain a sensible candidate set during generation. Specifically, we introduce an entropy-based metric called confidence and conceptualize determining the optimal candidate set as a confidence-increasing process. ![](https://img.shields.io/badge/ICML-orange)
  
- **Successor Features for Efficient Multi-Subject Controlled Text Generation**  
 This method leverages the concept of successor features to decouple the dynamics of LLMs from task-specific rewards. By employing successor features, our method proves to be memory-efficient and computationally efficient for both training and decoding, especially when dealing with multiple target subjects. ![](https://img.shields.io/badge/ICML-orange)
  
- **Model-Based Minimum Bayes Risk Decoding for Text Generation**  
  We propose a variant of MBR that uses the model probability itself as the estimate of the probability distribution instead of the Monte Carlo estimate. ![](https://img.shields.io/badge/ICML-orange)

- **Benchmarking and Improving Compositional Generalization of Multi-aspect Controllable Text Generation**  
  We propose CompMCTG, a benchmark encompassing diverse multi-aspect labeled datasets. We observe that existing MCTG works generally confront a noticeable performance drop in compositional testing. To mitigate this issue, we introduce a training framework incorporating meta-learning.![](https://img.shields.io/badge/ACL-orange)

- **Multi-Aspect Controllable Text Generation with Disentangled Counterfactual Augmentation**  
  We alleviate the issue of imbalanced attribute correlations during training using counterfactual feature vectors in the attribute latent space by disentanglement.![](https://img.shields.io/badge/ACL-orange)

- **Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor**  
  ScoPE modifies the context at the token level during the generation process of a backbone language model. ![](https://img.shields.io/badge/ACL-orange)
  
- **FreeCtrl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation**  
  We propose FreeCtrl, a learning free approach that dynamically adjusts the weights of selected feedforward neural network (FFN) vectors to steer the outputs of large language models (LLMs). ![](https://img.shields.io/badge/ACL-orange)


- **RSA-Control: A Pragmatics-Grounded Lightweight Controllable Text Generation Framework**  
  RSA-Control directs the generation process by recursively reasoning between imaginary speakers and listeners, enhancing the likelihood that target attributes are correctly interpreted by listeners amidst distractors. Additionally, we introduce a self-adjustable rationality parameter, which allows for automatic adjustment of control strength based on context. ![](https://img.shields.io/badge/EMNLP-orange)


- **Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding**  
*This work was rejected by ICLR 2024, and I also acknowledge the weaknesses highlighted in the feedback.* We propose formalizing text generation as a future-constrained generation problem to minimize undesirable behaviors and enforce faithfulness to instructions. The estimation of future constraint satisfaction, accomplished using LLMs, guides the text generation process. ![](https://img.shields.io/badge/EMNLP-orange)


- **Plug-in Language Model: Controlling Text Generation with a Simple Regression Model**  
PiLM leverages reinforcement learning to utilize black box tools directly, adjusting the latent state to control text generation. However, performing backpropagation during the inference phase is time-consuming for PiLM. By replacing backpropagation with a simple regression model, PiLM can achieve an inference time comparable to that of the original LLM. ![](https://img.shields.io/badge/NAACL-orange)


(I don't include papers accepted into the finding of NLP conferences) 


