# Assignment 4  : Report


# K - Nearest Neighbours
  . For each test point calculate the ecludian distance to every train point. 
  
  . Find the K nearest points and find the max occuring label.
 #### Conclusions
   . Ecludian distance is more Accurate than Mahattan distance.
   
   . It is performing better for K = 11 and getting same results as increasing from 10

Accuracy for different K's

| K   | Accuracy | 
| ------- | ----- | 
| 1     | 67.234    |
| 3 | 67.435     |  
| 5    | 68.71     |  
| 10 | 70.30    |
| 11 | 71.7    | 
| 15 | 71.5    | 
| 20 | 71.7    |


# Decision Tree 

We have implemented decision tree on the lines of ID3 algorithm , initially we have used random columns and compared the values between these two columns, where if a value in column1 is less than the value in column2 , it will end up in left split and whereas a value in column 1 greater than or equal to value in column2 ,then it will end up in right split.Once these splits are done , we caluclate the entropy of each split and caluclate the gain by subtracting the weighted entropy from the parent entropy .Th entropy claculation is based on the lecture_15 slides where we used it for Sunburn example .We do this for 5 iterations and identify the maximum gain and use that for the final split. <br><br>

Once the final split is done , we caluclate the purity , which is basically the the percentage of an orientation in the split and we consider the orientation with maxium number of samples in the split ,if this purity % is greater than 70 we , create a leaf node for this split and continue caluclating the purity for the right split. If at all the purity is less than 70 percentage we continue splitting the 
left split data and continue this process untill we reach the desired maximum depth of the tree. <br><br>

![alt text](https://github.iu.edu/cs-b551-fa2019/rparvat-nakopa-pvajja-a4/blob/master/5.PNG)

Once we have reached the maximum depth of the tree , we are adding that split and it's maximum orientation in that split/sample as the leaf node.While generating the tree node , we keep track of the maximum purity ,orientations and the data on which it has achieved the maximum purity .One the whole tree is built ,we traverse the tree and reach these leaf nodes and find out the orientation for that split and find the number of correctly classified samples and do the same for all the leaf nodes and sum them up and find out the percentage of the classfied samples from the train .We have recieved the train accuracy based on different heights ..ranging from 56 percentage to 64 percentage points.While our test accuracy is often quite low, partly because the way the decision tree is structured and it was challenging to recursively traverse the tree and update the orientations and data .<br><br>

Th accuracy might improve had we figured out better structuring the decision tree and also find the right splits ,we did try other split approches like taking mean of correspondng columns and checking if the the value is greater than or less than the first column and make splits accordings , similary comparing with the right column , we also tried normalizing the 3rd image and finding the splits, that was quite challenging and couldn't invest much time into it given the time contraints ,so  comparitively it performed better than the other two approaches.

# Neural Network 

Had invested a lot of time in Scaling and weight initialization.

### When we use np.random.uniform to generate weights ::
C:\Users\rutvi\Anaconda3\python.exe C:/Users/rutvi/Elem_AI/rparvat-nakopa-pvajja-a4/Neural_Net.py

Results & error (0, 0.0001, 108347.78319844665, 64.14971873647771)

Results & error (1, 0.0001, 107965.39570697554, 70.71073128515793)

Results & error (2, 0.0001, 109113.01711406083, 73.34216789268714)

Results & error (3, 0.0001, 110759.28921210479, 74.91075292081351)

Results & error (4, 0.0001, 113410.19684881487, 75.60309389874513)

The accuracy is  ::  75.60309389874513

Test Accuracy  0.7338282078472959

Overall Time taken is :: 31.320658206939697


### When we used np.random.randn to generate weights :: 
C:\Users\rutvi\Anaconda3\python.exe C:/Users/rutvi/Elem_AI/rparvat-nakopa-pvajja-a4/Neural_Net.py

Results & error (0, 0.0001, 107653.10549085277, 71.35439203807876)

Results & error (1, 0.0001, 109210.1648054338, 73.98312418866291)

Results & error (2, 0.0001, 112568.96815547996, 74.95672868887927)

Results & error (3, 0.0001, 115070.3683960239, 75.61120726958028)

Results & error (4, 0.0001, 118482.73325817016, 76.1007139766335)

The accuracy is  ::  76.1007139766335

Test Accuracy  0.7200424178154825

Overall Time taken is :: 29.924535036087036

### For 20,000 Training samples ::
C:\Users\rutvi\Anaconda3\python.exe C:/Users/rutvi/Elem_AI/rparvat-nakopa-pvajja-a4/Neural_Net.py\

Results :: (0, 0.0001, 55416.78645420024, 69.61500000000001)

Results :: (1, 0.0001, 56526.06059470263, 72.555)

Results :: (2, 0.0001, 57365.63945405599, 74.29)

Results :: (3, 0.0001, 58034.479166365665, 75.015)

Results :: (4, 0.0001, 58873.606899039456, 75.86500000000001)

The Training  accuracy is  ::  75.86500000000001

Test Accuracy  0.7126193001060446

Overall Time taken is :: 16.509839296340942

### For 1000 Training samples ::
C:\Users\rutvi\Anaconda3\python.exe C:/Users/rutvi/Elem_AI/rparvat-nakopa-pvajja-a4/Neural_Net.py

Results :: (0, 0.0001, 2066.903957531449, 30.5)

Results :: (1, 0.0001, 2057.7764314765373, 38.7)

Results :: (2, 0.0001, 2049.4668305488603, 46.7)

Results :: (3, 0.0001, 2041.4493094755055, 53.2)

Results :: (4, 0.0001, 2033.7827278814489, 57.099999999999994)

The Training  accuracy is  ::  57.099999999999994

Test Accuracy  0.7126193001060446

Overall Time taken is :: 2.0006768703460693

## We even tried using Sigmoid , but we got the best results using ::Relu & Softmax

### Relu - f(x) = max(x,0)






