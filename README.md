# Robustness-Learning-Notes

## 1908--
* Vision
  * [HXY] AutoAugment: Learning Augmentation Strategies from Data ([CVPR'19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)) Briefly introduce the RL technique in archtecture search.
  
## 190820
* Uncertainty
  * [HXY] On Calibration of Modern Neural Networks ([ICML'17](https://arxiv.org/abs/1706.04599)) ([Non-official Code](https://github.com/gpleiss/temperature_scaling)). 
    1) Metrics: ECE, MCE. 
    2) Methods: histogram binning and its variants, Platt scaling/Temperature scaling. 
    3) Factors of miscalibrated: model capacity, regularization. 
    4) NLL
  * [HXY] To Trust Or Not To Trust A Classifier ([NIPS'18](https://arxiv.org/abs/1805.11783)) ([scikit-learn Code]( https://github.com/google/TrustScore)). leverage KNN to measure whether the classification result is trustworthy.
* Privacy
  * [HXY] Scalable Private Learning with PATE ([ICLR'18](https://arxiv.org/abs/1802.08908)) ([Official Tensorflow Code](https://github.com/tensorflow/privacy)). 
     1) Differential privacy. 
     2) PATE: disjoint data for teachers, Laplacian noise added to each class of the teachers' vote histogram, leverage public unlabeled data to train student. 
     3) Scalable PATE: Gaussian noise (more centralized in more classes case), only samples with strong consensus in teachers are leaked to the student,  samples where student confidently agrees with teachers are discarded (can't help learning).   
* Meta-learning  
  - [JYZ] Learning to learn by gradient descent by gradient descent ([NIPS'16](https://arxiv.org/abs/1606.04474))([code](https://github.com/deepmind/learning-to-learn))  
* Adversarial  
  - [JYZ] Decision-based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models (First boundary attack) ([arxiv](https://arxiv.org/abs/1712.04248)) ([code](https://github.com/greentfrapp/boundary-attack))  
  - [JYZ] You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle. ([arxiv](https://arxiv.org/abs/1905.00877)) ([code](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once))  

## 190812
* Vision
  * [YJC] Meta-SR: A Magnification-Arbitrary Network for Super-Resolution ([CVPR'19](https://arxiv.org/abs/1903.00875)): very bad writing...
  * [YJC] Gradient Harmonized Single-stage Detector ([AAAI'19](https://arxiv.org/abs/1811.05181)): it seems a good and effective paper. Surprisingly effective on my dataset.
  * [YJC] Deep Layer Aggregation ([CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf))
  * [HXY] Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units ([ICML'16](https://arxiv.org/abs/1603.05201))
* Uncertainty
    - [HXY] Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples
 ([ICLR'18](https://arxiv.org/abs/1711.09325)) ([Official PyTorch Code](https://github.com/alinlab/Confident_classifier))
    - [HXY] Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples ([NIPS'18](https://arxiv.org/abs/1810.11580)) ([Official Caffe Code](https://github.com/AmIAttribute/AmI))
    * [YJC] Direct Uncertainty Prediction for Medical Second Opinions ([ICML'19](https://arxiv.org/abs/1807.01771)) ([Supp](http://proceedings.mlr.press/v97/raghu19a/raghu19a-supp.pdf)): it is not an easy-to-follow paper, though the core idea is very simple... it needs ground truth disagreement. It does not well leverage the classification and uncertainty prediction task (joint training drops the performance). Anyway, it is not easy to follow, thus I did not read all the experiment part.

## 190806
* Ambiguity
    - [YJC] A Probabilistic U-Net for Segmentation of Ambiguous Images ([NIPS'18](https://arxiv.org/abs/1806.05034)) (code: [official re-implementation (TensorFlow)](https://github.com/SimonKohl/probabilistic_unet), [PyTorch](https://github.com/stefanknegt/probabilistic_unet_pytorch))
    - [YJC] PHiSeg: Capturing Uncertainty in Medical Image Segmentation ([MICCAI'19](https://arxiv.org/abs/1906.04045)) ([code](https://github.com/baumgach/PHiSeg-code)): good results, enhanced Prob-UNet. Multi-scale prior encoding for ambiguous segmentation. 
* Uncertainty
    - [HXY] Using Pre-Training Can Improve Model Robustness and Uncertainty ([ICML'19](https://arxiv.org/abs/1901.09960)) ([Official PyTorch Code](https://github.com/hendrycks/pre-training))
    - [HXY] ML-LOO: Detecting Adversarial Examples with Feature Attribution ([arxiv](https://arxiv.org/abs/1906.03499))
    - [HXY] Hierarchical Novelty Detection for Visual Object Recognition ([CVPR'18](https://arxiv.org/abs/1804.00722)) ([Official PyTorch Code](https://github.com/kibok90/cvpr2018-hnd))
    - [HXY] Out-of-Distribution Detection using Multiple Semantic Label Representations
 ([NIPS'18](http://arxiv.org/abs/1808.06664)) ([Official PyTorch Code](https://github.com/MLSpeech/semantic_OOD))
    - [HXY] OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations ([CVPR'19](http://arxiv.org/abs/1903.08550)) ([Official MXNet Code](https://github.com/PramuPerera/OCGAN)) 
* Adversarial
    - [JYZ] Efﬁcient Decision-based Black-box Adversarial Attacks on Face Recognition ([CVPR'19](https://arxiv.org/abs/1904.04433v1))  
    - [JYZ] HopSkipJumpAttack: A Query-Efficient Decision-Based Attack ([arxiv](https://arxiv.org/abs/1904.02144)) ([code](https://github.com/Jianbo-Lab/HSJA))  
    - [JYZ] Towards Query-efficient Black-box ... (review)  
    - [JYZ] Sensitive-Sample Fingerprinting of Deep Neural Networks ([CVPR'19](http://openaccess.thecvf.com/content_CVPR_2019/html/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.html))  
