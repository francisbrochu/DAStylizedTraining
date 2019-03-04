# DAStylizedTraining
Domain-Adversarial Training on Stylized and Regular Image Datasets

This project is for the Winter 2019 GLO-7030 (Université Laval) Deep Learning class.

This project is intended to reproduce and extend somewhat the ICLR article [ImageNet-trained CNNs are biaised towards texture; increasing shape biais improves accuracy and robustness.](https://openreview.net/pdf?id=Bygh9j09KX)

The data used for this project is not included in this git. All datasets were found and downloaded on Kaggle.com (see links below). 
Any unlabeled test set was discarded. The datasets containing validation sets and train sets were merged. 
20% of each dataset was split into a test set (randomly, stratified by class).
20% of the remaning images were split into validation sets (again, randomly and stratified by class).
The remaining images form the training set.

The datasets were then style-transferred by the [Stylized ImageNet git](https://github.com/rgeirhos/Stylized-ImageNet).

[Food101 (101 000 images)](https://www.kaggle.com/dansbecker/food-101) 
[DogBreedIdentification (10 200 images)](https://www.kaggle.com/c/dog-breed-identification)
[DogsVsCats (25 000 images)](https://www.kaggle.com/c/dogs-vs-cats)
[Dice (16 000 images)](https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images)
