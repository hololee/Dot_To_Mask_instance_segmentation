## An Instance Segmentation using Object Center Masks

This project is published on Smart Media Journal.  
[`An Instance Segmentation using Object Center Masks`](https://www.koreascience.or.kr/article/JAKO202018955009100.page)

### Abstract
In this paper, we propose a network model composed of Multi path Encoder-Decoder branches that can recognize each instance from the image. The network has two branches, Dot branch and Segmentation branch for finding the center point of each instance and for recognizing area of the instance, respectively. In the experiment, the CVPPP dataset was studied to distinguish leaves from each other, and the center point detection branch(Dot branch) found the center points of each leaf, and the object segmentation branch(Segmentation branch) finally predicted the pixel area of each leaf corresponding to each center point. In the existing segmentation methods, there were problems of finding various sizes and positions of anchor boxes (N > 1k) for checking objects. Also, there were difficulties of estimating the number of undefined instances per image. In the proposed network, an effective method finding instances based on their center points is proposed.
  
  
### Structure  
1. Yellow path predict center point on each instance.
2. Purple path embed each instance center information and added with yellow path embedded feature.
3. Blue path predict each instance.  

![whole_structure](https://github.com/hololee/Dot_To_Mask_instance_segmentation/blob/master/whole_structure.png?raw=true)

### Merge outputs  
Each instance information is merged together, and remove noise.
  
![merge](https://github.com/hololee/Dot_To_Mask_instance_segmentation/blob/master/sum.png?raw=true)

### Result
![output](https://github.com/hololee/Dot_To_Mask_instance_segmentation/blob/master/output_show.png?raw=true)



