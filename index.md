# Abstract

Artificial neural networks (ANNs) have greatly advanced the field of video and image processing. These machine learning algorithms have been proven to excel at tasks such as object detection and handwritten digit recognition. This is shown in Wan et al.’s (2013) approach that performs handwritten digit recognition on the MNIST data set with 0.21\% error, and Kolesnikov et al.’s (2019) approach that performs image classification on the CIFAR10 data set with 99.37\% accuracy [1][2]. 

Although ANNs have achieved incredibly high levels of accuracy on these tasks when simulated in traditional computing environments, there is an increasing demand for performing these tasks in real time on embedded computer systems with low power consumption. For example, an autonomous unmanned aerial vehicle running on a battery power supply may employ an ANN to assist with collision avoidance. For tasks such as this, researchers believe that spiking neural networks (SNNs) are a suitable alternative because of their energy efficiency and event-driven architectures [3].

SNNs are networks of neurons that communicate information through short pulses of data called spikes. A spiking neuron will only output a spike to other neurons once a specific threshold of spikes have been received, thus making them more energy efficient than ANNs. Spiking neuron models are also capable of processing temporal information, which leads researchers to believe that they are more capable of processing spiking event data from devices such as dynamic vision sensors [3]. However, more research is required on SNNs to determine the best neural models, encoding methods and training techniques for their use in image processing applications.

Our group expands on the research of Deng et al. (2020) in comparing ANNs to SNNs for image classification tasks [4]. Deng et al. (2020) compare ANNs to SNNs by (1) comparing their accuracy in classifying the MNIST and CIFAR10 benchmarks, (2) comparing each of the networks’ memory cost for storing weights, and (3) comparing cost of performing computations with each network [4]. Our group reimplements the tests performed using different neural models, encoding methods, and training techniques to study how these factors affect the SNN model accuracy.

These experiments are performed using the PyTorch and BindsNET python packages [5]. PyTorch is a framework that allows researchers to quickly develop ANN models to test, while BindsNet is a recently developed extension of PyTorch that provides a way to create and train SNN models. The ANN model will be tested against the traditional MNIST and CIFAR10 benchmarks [6][7], while the SNN models will be tested against the developed NMNIST and CIFAR10-DVS data sets [8][9].

### Project Resources
- [Project Report]()
- [Project Video](https://www.youtube.com/watch?v=yVP_vmSdnkg)
- [Project Presentation](./Spiking Neural Networks for Image Classification.pdf)
- [Source Code](https://github.com/oshears/adv-ml-2020-snn-project)

### Project Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/yVP_vmSdnkg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Pages
- [Problem Definition](./problem.md)
- [Motivation](./motivation.md)
- [Related Works](./related.md)
- [Methods](./methods.md)
- [Results](./results.md)
- [References](./references.md)