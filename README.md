## Realtime sudoku solver

Simple and easy project explaining how to use OpenCV with Java and DL4J to detect sudoku puzzle, extract the digits and solve it.

### Training network
If you want to train your own network you would need to run `DataClassifier.java`. Data set can be found in the
resource folder. You can also add your own. If number of digits change make sure that you update `N_SAMPLES_TRAINING`
property.
Project comes with already pretrained network that can be found on following path
```
resources/models/trained.tar
```

### Youtube
Video tutorial for this project can be found on following link
[Youtube](https://www.youtube.com/watch?v=4OtEY5toG_w&list=PLXy8DQl3058MBCLLy1e0oYvWkOzvIKjm3)